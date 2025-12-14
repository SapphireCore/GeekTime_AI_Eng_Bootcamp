# main.py
# -*- coding: utf-8 -*-
"""
Homework 2 - ImageOCRReader for LlamaIndex (PaddleOCR)
======================================================
Deliverables (single-file):
- ImageOCRReader(BaseReader): image -> OCR text -> LlamaIndex Document
- Batch load (files / directory)
- Optional visualization: draw OCR boxes and save
- Integration demo: build VectorStoreIndex and run a basic query validation
  * Offline-first: if no LLM/Embedding is configured, falls back to mock components.

Recommended run:
  python main.py --download-samples --data-dir ./data/ocr_images
  python main.py --data-dir ./data/ocr_images --lang ch --visualize --query "图片中提到了什么文字？"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

try:
    import numpy as np
except Exception as e:
    raise ImportError("numpy is required. Please install it first: pip install numpy") from e

try:
    from llama_index.core.readers.base import BaseReader
    from llama_index.core.schema import Document
except Exception as e:
    raise ImportError(
        "llama-index is required. Please install it first: pip install llama-index"
    ) from e

# PaddleOCR (required by assignment)
try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception as e:
    raise ImportError(
        "PaddleOCR is not installed. Please run:\n"
        "  pip install \"paddlepaddle<=2.6\"  (or paddlepaddle-gpu<=2.6)\n"
        "  pip install \"paddleocr<3.0\""
    ) from e


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class OCRBlock:
    text: str
    conf: float
    bbox: Optional[List[List[float]]] = None  # 4 points: [[x,y],...]


class ImageOCRReader(BaseReader):
    """
    使用 PP-OCR 从图像中提取文本并返回 LlamaIndex Document。

    Design notes:
    - PaddleOCR 模型在 __init__ 阶段加载：避免批量处理时重复初始化，提高性能。
    - 输出采用“文本块 + 置信度”格式（可读、可调试），同时将关键统计信息下沉到 metadata。
    """

    def __init__(
        self,
        lang: str = "ch",
        use_gpu: bool = False,
        ocr_version: str = "PP-OCRv5",
        use_angle_cls: bool = True,
        **kwargs,
    ):
        """
        Args:
            lang: OCR 语言 ('ch', 'en', 'fr', etc.)
            use_gpu: 是否使用 GPU（若环境/驱动不满足会失败）
            ocr_version: 模型版本标识（用于 metadata；真实加载由 PaddleOCR 内部决定）
            use_angle_cls: 是否启用方向分类（对倾斜文字更稳，但略增耗时）
            **kwargs: 透传 PaddleOCR 其余参数
        """
        super().__init__()
        self.lang = lang
        self.use_gpu = use_gpu
        self.ocr_version = ocr_version

        # PaddleOCR v2.x 常用参数：use_angle_cls / lang / use_gpu
        # v3.x 也兼容 predict API。我们统一做兼容封装。
        self._ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            use_gpu=use_gpu,
            **kwargs,
        )

    def load_data(self, file: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """
        从单个或多个图像文件中提取文本，返回 Document 列表。

        Args:
            file: 图像路径字符串 或 路径列表
        Returns:
            List[Document]
        """
        files = self._normalize_inputs(file)
        docs: List[Document] = []

        for p in files:
            blocks = self._run_ocr(p)
            if not blocks:
                # 工程处理：无文本时也返回 Document（避免“静默丢数据”），但在 metadata 标记 empty
                doc = Document(
                    text="",
                    metadata=self._build_metadata(
                        image_path=str(p),
                        num_blocks=0,
                        avg_confidence=0.0,
                        empty=True,
                    ),
                )
                docs.append(doc)
                continue

            full_text = self._format_blocks(blocks)
            avg_conf = float(np.mean([b.conf for b in blocks])) if blocks else 0.0

            doc = Document(
                text=full_text,
                metadata=self._build_metadata(
                    image_path=str(p),
                    num_blocks=len(blocks),
                    avg_confidence=avg_conf,
                    empty=False,
                ),
            )
            docs.append(doc)

        return docs

    def load_data_from_dir(
        self,
        dir_path: Union[str, Path],
        recursive: bool = False,
        limit: Optional[int] = None,
    ) -> List[Document]:
        """
        （附加挑战）批量处理目录。

        Args:
            dir_path: 图片目录
            recursive: 是否递归子目录
            limit: 最多处理多少张（用于快速测试）
        """
        dirp = Path(dir_path)
        if not dirp.exists() or not dirp.is_dir():
            raise FileNotFoundError(f"Directory not found: {dirp}")

        files = list(iter_image_files(dirp, recursive=recursive))
        if limit is not None:
            files = files[:limit]
        return self.load_data(files)

    def visualize_ocr(
        self,
        image_path: Union[str, Path],
        out_path: Union[str, Path],
        thickness: int = 2,
    ) -> None:
        """
        （附加挑战）可视化 OCR 检测框并保存图片（OpenCV required）。

        Args:
            image_path: 输入图片
            out_path: 输出图片路径
        """
        try:
            import cv2  # type: ignore
        except Exception as e:
            raise ImportError("OpenCV is required for visualization: pip install opencv-python") from e

        img_path = Path(image_path)
        blocks = self._run_ocr(img_path)
        if not blocks:
            raise ValueError(f"No OCR blocks found for visualization: {img_path}")

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to read image with cv2: {img_path}")

        for b in blocks:
            if not b.bbox:
                continue
            pts = np.array(b.bbox, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=thickness)

        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(outp), img)

    # -------------------------
    # Internal helpers
    # -------------------------

    def _normalize_inputs(self, file: Union[str, Path, List[Union[str, Path]]]) -> List[Path]:
        if isinstance(file, (str, Path)):
            files = [Path(file)]
        else:
            files = [Path(x) for x in file]
        # validate ext
        valid: List[Path] = []
        for p in files:
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")
            if p.suffix.lower() not in SUPPORTED_EXTS:
                raise ValueError(f"Unsupported image extension: {p} (supported: {sorted(SUPPORTED_EXTS)})")
            valid.append(p)
        return valid

    def _run_ocr(self, image_path: Path) -> List[OCRBlock]:
        """
        PaddleOCR 兼容层：
        - v2.x: ocr.ocr(img_path, cls=True) -> [[ [bbox, (text, conf)], ... ]]
        - v3.x: ocr.predict(img_path) -> list[OCRResult] (iterable), each can be printed/saved
        """
        # Prefer predict if available (as in assignment reference), else fallback to ocr()
        if hasattr(self._ocr, "predict"):
            try:
                res = self._ocr.predict(str(image_path))
                return parse_paddle_predict_result(res)
            except Exception:
                # fallback to ocr API
                pass

        try:
            result = self._ocr.ocr(str(image_path), cls=True)
        except Exception as e:
            raise RuntimeError(f"OCR failed for {image_path}: {e}") from e

        return parse_paddle_ocr_result(result)

    def _format_blocks(self, blocks: List[OCRBlock]) -> str:
        lines = []
        for i, b in enumerate(blocks, start=1):
            lines.append(f"[Text Block {i}] (conf: {b.conf:.2f}): {b.text}")
        return "\n".join(lines)

    def _build_metadata(self, image_path: str, num_blocks: int, avg_confidence: float, empty: bool) -> Dict[str, object]:
        return {
            "image_path": image_path,
            "ocr_model": self.ocr_version,
            "language": self.lang,
            "num_text_blocks": int(num_blocks),
            "avg_confidence": float(avg_confidence),
            "empty": bool(empty),
        }


def parse_paddle_ocr_result(result: object) -> List[OCRBlock]:
    """
    Parse PaddleOCR(v2) output:
      result = [ [ [bbox, (text, conf)], ... ] ]
    """
    blocks: List[OCRBlock] = []
    if not result:
        return blocks

    # Most common: result[0] list of lines
    try:
        lines = result[0]  # type: ignore[index]
    except Exception:
        return blocks

    if not lines:
        return blocks

    for line in lines:
        try:
            bbox = line[0]
            text = line[1][0]
            conf = float(line[1][1])
            blocks.append(OCRBlock(text=str(text), conf=conf, bbox=bbox))
        except Exception:
            # best-effort: skip malformed line
            continue
    return blocks


def parse_paddle_predict_result(predict_result: object) -> List[OCRBlock]:
    """
    Parse PaddleOCR(v3) predict output:
      predict_result is iterable; each item often provides .json or can be cast to dict-like.

    We implement a tolerant parser:
    - If object has "to_json" / "json" / "save_to_json" etc, try to serialize
    - Else try to treat as list/dict
    """
    blocks: List[OCRBlock] = []
    if predict_result is None:
        return blocks

    # Many versions return list-like
    items = list(predict_result) if isinstance(predict_result, Iterable) and not isinstance(predict_result, (str, bytes)) else [predict_result]  # type: ignore[arg-type]

    for item in items:
        # Try to extract raw dict
        raw = None
        if hasattr(item, "json"):
            try:
                raw = item.json  # may be dict or property
            except Exception:
                raw = None

        if raw is None and hasattr(item, "to_json"):
            try:
                raw = item.to_json()
            except Exception:
                raw = None

        # If still None, try __dict__
        if raw is None:
            try:
                raw = item.__dict__
            except Exception:
                raw = None

        # Now parse common shapes
        # Expected: raw may contain "rec_texts", "rec_scores", "dt_polys" etc.
        if isinstance(raw, dict):
            texts = raw.get("rec_texts") or raw.get("texts") or []
            scores = raw.get("rec_scores") or raw.get("scores") or []
            polys = raw.get("dt_polys") or raw.get("polys") or raw.get("boxes") or []

            if texts and scores:
                for i in range(min(len(texts), len(scores))):
                    bbox = polys[i] if i < len(polys) else None
                    blocks.append(OCRBlock(text=str(texts[i]), conf=float(scores[i]), bbox=bbox))
                continue

        # Fallback: if item resembles v2 ocr output (rare), try that parser
        maybe_blocks = parse_paddle_ocr_result([item])
        blocks.extend(maybe_blocks)

    return blocks


def iter_image_files(dir_path: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        for p in dir_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                yield p
    else:
        for p in dir_path.glob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                yield p


# -------------------------
# LlamaIndex integration (offline-first)
# -------------------------

def build_index_and_query(documents: List[Document], query: str) -> str:
    """
    Build VectorStoreIndex and query.
    Offline-first strategy:
    - If user has configured any real LLM/embedding elsewhere, LlamaIndex Settings will pick it up.
    - Otherwise, fallback to MockEmbedding/MockLLM (if available).
    - If mock components not available, fallback to "retrieve only + heuristic answer" mode.
    """
    from llama_index.core import VectorStoreIndex

    # Try to setup mock components (no API key required)
    llm, embed = try_get_mock_components()

    if llm is not None or embed is not None:
        from llama_index.core import Settings
        if llm is not None:
            Settings.llm = llm
        if embed is not None:
            Settings.embed_model = embed

    index = VectorStoreIndex.from_documents(documents)

    # Preferred: standard query_engine.query
    try:
        query_engine = index.as_query_engine()
        resp = query_engine.query(query)
        return str(resp)
    except Exception:
        # Fallback: retrieve and do a simple heuristic synthesis
        try:
            retriever = index.as_retriever(similarity_top_k=3)
            nodes = retriever.retrieve(query)
            context = "\n".join([n.get_content() for n in nodes])
            return heuristic_answer(query, context)
        except Exception as e:
            return f"[ERROR] Query failed even in fallback mode: {e}"


def try_get_mock_components():
    """
    Attempt to import mock LLM/Embedding from LlamaIndex.
    Returns: (llm, embed) possibly None.
    """
    llm = None
    embed = None

    # MockLLM
    try:
        from llama_index.core.llms.mock import MockLLM  # type: ignore

        # MockLLM returns deterministic completions; we keep it minimal.
        llm = MockLLM()
    except Exception:
        llm = None

    # MockEmbedding
    try:
        from llama_index.core.embeddings.mock import MockEmbedding  # type: ignore

        embed = MockEmbedding(embed_dim=1536)
    except Exception:
        embed = None

    return llm, embed


def heuristic_answer(query: str, context: str) -> str:
    """
    Extremely lightweight synthesizer:
    - If user asks for date, extract date-like strings.
    - Else return top relevant lines from context.
    """
    q = query.lower()

    # date patterns (CN + ISO-like)
    patterns = [
        r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?",
        r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
    ]
    if "日期" in query or "date" in q:
        hits = []
        for pat in patterns:
            hits.extend(re.findall(pat, context))
        hits = list(dict.fromkeys(hits))
        if hits:
            return "检测到的日期候选：\n" + "\n".join(f"- {h}" for h in hits[:10])
        return "未在检索上下文中识别到明确日期。"

    # username extraction
    if "用户名" in query or "username" in q:
        m = re.search(r"(username\s*[:：]\s*)([A-Za-z0-9_\-\.]+)", context, flags=re.IGNORECASE)
        if m:
            return f"识别到用户名：{m.group(2)}"
        return "未在检索上下文中识别到明确用户名。"

    # default: return some context as answer
    lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
    return "基于检索到的内容（节选）：\n" + "\n".join(lines[:15])


# -------------------------
# Sample images (download links)
# -------------------------

SAMPLE_IMAGES = [
    # 1) scanned document (public domain, contains clear text)
    {
        "filename": "scan_document_quick_brown_fox.png",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/2e/Boston_Journal_1885-02-09_%28quick_brown_fox%29.png",
        "category": "scan_document",
        "license_note": "Public domain (published before 1930, per Wikimedia Commons file page).",
    },
    # 2) UI screenshot (contains code/editor text)
    {
        "filename": "ui_screenshot_text.png",
        "url": "https://upload.wikimedia.org/wikipedia/commons/6/64/Text_screenshot.png",
        "category": "ui_screenshot",
        "license_note": "CC BY-SA 4.0 (per Wikimedia Commons file page).",
    },
    # 3) natural scene (road sign with Japanese text)
    {
        "filename": "scene_road_sign_jp.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Road_sign_no_stopping_or_parking.JPG",
        "category": "natural_scene",
        "license_note": "CC BY-SA 4.0 (per Wikimedia Commons file page).",
    },
]


def download_samples(data_dir: Union[str, Path]) -> None:
    """
    Download 3 required categories of images into data_dir.
    Uses urllib only (no external dependencies).
    """
    import urllib.request

    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for item in SAMPLE_IMAGES:
        out_path = out_dir / item["filename"]
        if out_path.exists():
            continue
        try:
            urllib.request.urlretrieve(item["url"], str(out_path))
        except Exception as e:
            raise RuntimeError(f"Failed to download {item['url']} -> {out_path}: {e}") from e


# -------------------------
# CLI / main
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Homework2: ImageOCRReader for LlamaIndex using PaddleOCR")
    p.add_argument("--data-dir", type=str, default="./data/ocr_images", help="Image directory (default: ./data/ocr_images)")
    p.add_argument("--files", type=str, nargs="*", default=None, help="Optional explicit image files; overrides directory scan")
    p.add_argument("--recursive", action="store_true", help="Recursively scan directory for images")
    p.add_argument("--limit", type=int, default=None, help="Max number of images to OCR (for quick test)")
    p.add_argument("--lang", type=str, default="ch", help="OCR language: ch/en/fr/...")
    p.add_argument("--use-gpu", action="store_true", help="Use GPU for OCR (requires proper CUDA/driver)")
    p.add_argument("--ocr-version", type=str, default="PP-OCRv5", help="OCR model version tag in metadata")
    p.add_argument("--visualize", action="store_true", help="Save visualization images with OCR boxes (requires opencv-python)")
    p.add_argument("--viz-dir", type=str, default="./data/ocr_viz", help="Output directory for visualization images")
    p.add_argument("--query", type=str, default=None, help="Optional query to validate index")
    p.add_argument("--download-samples", action="store_true", help="Download 3 sample images into --data-dir")
    p.add_argument("--dump-docs-json", type=str, default=None, help="Dump Document list (text+metadata) to a JSON file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Step 0: download sample images if requested
    if args.download_samples:
        print("[Step 0] Downloading sample images...")
        download_samples(args.data_dir)
        print(f"[OK] Sample images downloaded to: {Path(args.data_dir).resolve()}")
        print("[Note] Licenses are from Wikimedia Commons; keep attribution if you redistribute.")
        return

    # Step 1: collect images
    data_dir = Path(args.data_dir)
    if args.files:
        image_files = [Path(x) for x in args.files]
    else:
        if not data_dir.exists():
            raise FileNotFoundError(f"data-dir does not exist: {data_dir}. You can run --download-samples first.")
        image_files = list(iter_image_files(data_dir, recursive=args.recursive))

    if args.limit is not None:
        image_files = image_files[: args.limit]

    if not image_files:
        raise RuntimeError(
            f"No images found. Put images under {data_dir} or pass --files. "
            f"Supported extensions: {sorted(SUPPORTED_EXTS)}"
        )

    print(f"[Step 1] Found {len(image_files)} image(s).")
    for p in image_files:
        print(f"  - {p}")

    # Step 2: OCR -> Documents
    print("[Step 2] Running OCR and building Documents...")
    reader = ImageOCRReader(
        lang=args.lang,
        use_gpu=args.use_gpu,
        ocr_version=args.ocr_version,
        use_angle_cls=True,
        # For assignment reference: disable extra pipelines by default (safe baseline)
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    documents = reader.load_data(image_files)
    print(f"[OK] Produced {len(documents)} Document(s).")

    # Optional: visualize
    if args.visualize:
        print("[Step 2.1] Saving OCR box visualization...")
        viz_dir = Path(args.viz_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)
        for img in image_files:
            out_img = viz_dir / f"{img.stem}.viz{img.suffix}"
            try:
                reader.visualize_ocr(img, out_img)
                print(f"  - saved: {out_img}")
            except Exception as e:
                print(f"  - skipped {img} (visualize failed): {e}")

    # Optional: dump docs json for report/debug
    if args.dump_docs_json:
        payload = [{"text": d.text, "metadata": d.metadata} for d in documents]
        outp = Path(args.dump_docs_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Dumped documents to: {outp}")

    # Step 3: basic validation via LlamaIndex index + query
    if args.query:
        print("[Step 3] Building VectorStoreIndex and querying...")
        answer = build_index_and_query(documents, args.query)
        print("\n===== QUERY =====")
        print(args.query)
        print("===== ANSWER =====")
        print(answer)

    # Step 4: show a short preview
    print("\n[Preview] First 2 documents:")
    for i, d in enumerate(documents[:2], start=1):
        txt_preview = (d.text[:240] + "...") if len(d.text) > 240 else d.text
        print(f"\n--- Document {i} ---")
        print(txt_preview)
        print(f"metadata={d.metadata}")


if __name__ == "__main__":
    main()
