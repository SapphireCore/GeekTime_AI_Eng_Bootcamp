# file: main.py
"""
Milvus FAQ 检索系统
- LlamaIndex 构建索引
- Milvus（优先 Milvus Lite / 本地文件）作为向量库
- 语义切分 + 重叠（Semantic Splitter + chunk overlap）
- FastAPI 提供 RESTful API（/api/query, /api/update-index）
- 支持热更新（re-index）

运行方式：
1) 启动 API：
   python main.py serve --host 0.0.0.0 --port 8000

2) 本地命令行检索（无需启动 API）：
   python main.py cli --q "第一次玩剧本杀需要准备什么？"

3) 触发热更新（API 方式）：
   POST http://localhost:8000/api/update-index

目录约定：
- data/faqs.csv  (FAQ 数据)
- vector_store/  (Milvus Lite 数据文件等)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# 基础配置
# -----------------------------
DATA_DIR = os.getenv("FAQ_DATA_DIR", "data")
FAQ_FILE = os.getenv("FAQ_FILE", os.path.join(DATA_DIR, "faqs.csv"))

# Milvus Lite：使用本地文件作为持久化（无需独立服务）
# 此处以常见 Milvus Lite 形式配置。
VECTOR_DIR = os.getenv("VECTOR_DIR", "vector_store")
MILVUS_URI = os.getenv("MILVUS_URI", os.path.join(VECTOR_DIR, "milvus_faq.db"))
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "faq_collection")

# Embedding 维度：本作业为了“可离线运行”，使用 MockEmbedding（确定性 hash 向量）
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

# LlamaIndex：切片大小与重叠（语义切分为主，chunk_* 作为兜底/全局默认）
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

# 检索 TopK
TOP_K = int(os.getenv("TOP_K", "3"))

# -----------------------------
# LlamaIndex / FastAPI 依赖
# -----------------------------
# 为了保证“可运行”，若环境缺 Milvus 依赖，自动降级为内存向量库（仍保留 Milvus 代码路径）。
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except Exception as e:
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore
    uvicorn = None  # type: ignore

try:
    from llama_index.core import VectorStoreIndex, StorageContext, Document
    from llama_index.core.settings import Settings
    from llama_index.core.node_parser import SemanticSplitterNodeParser
except Exception as e:
    raise RuntimeError(
        "缺少 llama-index 依赖。请先安装：pip install llama-index"
    ) from e

# Milvus vector store（优先）
_MILVUS_AVAILABLE = True
try:
    from llama_index.vector_stores.milvus import MilvusVectorStore
except Exception:
    _MILVUS_AVAILABLE = False

# 内存向量库（降级）
try:
    from llama_index.core.vector_stores.simple import SimpleVectorStore
except Exception:
    SimpleVectorStore = None  # type: ignore

# Embedding：尽量使用 LlamaIndex BaseEmbedding；若版本差异，提供兼容实现
try:
    from llama_index.core.embeddings import BaseEmbedding
except Exception:
    BaseEmbedding = object  # type: ignore


# -----------------------------
# 可离线运行的 Mock Embedding
# -----------------------------
class DeterministicHashEmbedding(BaseEmbedding):
    """
    说明：
    - 提供一个“确定性 hash -> 向量”的 Mock Embedding，保证：
      1) 可离线运行
      2) 同一文本每次向量一致
      3) 维度固定，满足 Milvus 存储要求

    注意：
    - 该 Embedding 不具备真实语义理解能力，检索效果仅用于演示/功能验收。
    """

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim

    @classmethod
    def class_name(cls) -> str:
        return "DeterministicHashEmbedding"

    def _text_to_vec(self, text: str) -> List[float]:
        # 使用 sha256 生成稳定伪随机数流，再映射到 [-1,1]
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # 扩展字节流到足够长度
        needed = self.dim * 2  # 2 bytes per element
        buf = (h * ((needed // len(h)) + 1))[:needed]
        vec = []
        for i in range(0, needed, 2):
            val = int.from_bytes(buf[i : i + 2], "big")  # 0..65535
            f = (val / 32767.5) - 1.0  # approx [-1,1]
            vec.append(float(f))
        return vec

    # 兼容不同 llama-index 版本的方法名
    def get_text_embedding(self, text: str) -> List[float]:
        return self._text_to_vec(text)

    def get_text_embedding_batch(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        return [self._text_to_vec(t) for t in texts]

    def get_query_embedding(self, query: str) -> List[float]:
        return self._text_to_vec(query)


# -----------------------------
# FAQ 数据加载与文档构建
# -----------------------------
@dataclass
class FAQItem:
    question: str
    answer: str


def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)


def load_faq_csv(path: str) -> List[FAQItem]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAQ 文件不存在：{path}。请先创建 data/faqs.csv")
    items: List[FAQItem] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "question" not in reader.fieldnames or "answer" not in reader.fieldnames:
            raise ValueError("CSV 必须包含表头：question,answer")
        for row in reader:
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if q and a:
                items.append(FAQItem(question=q, answer=a))
    if not items:
        raise ValueError("FAQ 数据为空。请在 data/faqs.csv 中至少放入 1 条 QA。")
    return items


def build_documents(items: List[FAQItem]) -> List[Document]:
    """
    将每个 FAQ 条目构造成一个 Document。
    文本格式保持稳定，便于从 source_nodes 解析回 question/answer。
    """
    docs: List[Document] = []
    for it in items:
        text = f"问题: {it.question}\n答案: {it.answer}"
        docs.append(Document(text=text, metadata={"question": it.question}))
    return docs


# -----------------------------
# Index 管理（单例缓存）
# -----------------------------
class IndexManager:
    def __init__(self) -> None:
        self._index: Optional[VectorStoreIndex] = None
        self._query_engine = None
        self._vector_backend: str = "unknown"

    def _create_vector_store(self, overwrite: bool) -> Tuple[Any, StorageContext, str]:
        """
        优先使用 MilvusVectorStore（Milvus Lite）。
        若环境无 Milvus 依赖，则降级 SimpleVectorStore（内存）。
        """
        if _MILVUS_AVAILABLE:
            vector_store = MilvusVectorStore(
                uri=MILVUS_URI,
                collection_name=COLLECTION_NAME,
                dim=EMBED_DIM,
                overwrite=overwrite,
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            return vector_store, storage_context, "milvus"
        else:
            if SimpleVectorStore is None:
                raise RuntimeError(
                    "Milvus 不可用且 SimpleVectorStore 不可用。请安装 llama-index-vector-stores-milvus 或升级 llama-index。"
                )
            vector_store = SimpleVectorStore()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            return vector_store, storage_context, "memory"

    def initialize(self) -> None:
        if self._index is not None and self._query_engine is not None:
            return

        ensure_dirs()
        items = load_faq_csv(FAQ_FILE)
        docs = build_documents(items)

        # 全局 Settings（chunk 参数作为兜底）
        Settings.chunk_size = CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP

        # 离线 Embedding（可替换为真实 Embedding）
        embed_model = DeterministicHashEmbedding(dim=EMBED_DIM)
        Settings.embed_model = embed_model

        # 语义切分（会使用 embed_model）
        splitter = SemanticSplitterNodeParser.from_defaults(
            embed_model=embed_model,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        _, storage_context, backend = self._create_vector_store(overwrite=False)

        # 尝试从现有向量库加载；若失败则从文档构建
        try:
            self._index = VectorStoreIndex.from_vector_store(vector_store=storage_context.vector_store)  # type: ignore
            self._vector_backend = backend
        except Exception:
            self._index = VectorStoreIndex.from_documents(
                docs,
                storage_context=storage_context,
                transformations=[splitter],
            )
            self._vector_backend = backend

        self._query_engine = self._index.as_query_engine(similarity_top_k=TOP_K)

    def query(self, question: str) -> List[Dict[str, Any]]:
        self.initialize()
        assert self._query_engine is not None

        resp = self._query_engine.query(question)
        out: List[Dict[str, Any]] = []

        # LlamaIndex Response：source_nodes 保存检索到的节点
        if not getattr(resp, "source_nodes", None):
            return out

        for node in resp.source_nodes:
            text = node.get_text() if hasattr(node, "get_text") else str(node)
            score = float(node.get_score() or 0.0) if hasattr(node, "get_score") else 0.0

            # 从“问题: ...\n答案: ...”解析
            q, a = self._parse_qa(text)
            out.append({"question": q, "answer": a, "score": score})

        # 按 score 降序（稳妥）
        out.sort(key=lambda x: x["score"], reverse=True)
        return out

    def update_index(self) -> Dict[str, Any]:
        """
        热更新：清空集合并重建索引
        """
        ensure_dirs()
        items = load_faq_csv(FAQ_FILE)
        docs = build_documents(items)

        Settings.chunk_size = CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP

        embed_model = DeterministicHashEmbedding(dim=EMBED_DIM)
        Settings.embed_model = embed_model

        splitter = SemanticSplitterNodeParser.from_defaults(
            embed_model=embed_model,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        _, storage_context, backend = self._create_vector_store(overwrite=True)
        self._index = VectorStoreIndex.from_documents(
            docs,
            storage_context=storage_context,
            transformations=[splitter],
        )
        self._query_engine = self._index.as_query_engine(similarity_top_k=TOP_K)
        self._vector_backend = backend

        return {
            "status": "success",
            "message": "索引已成功热更新（re-index 完成）。",
            "backend": self._vector_backend,
            "count": len(items),
        }

    @staticmethod
    def _parse_qa(text: str) -> Tuple[str, str]:
        # 允许切分后多段文本，尽量鲁棒解析
        q = ""
        a = ""
        if "答案:" in text:
            parts = text.split("答案:", 1)
            left = parts[0].strip()
            right = parts[1].strip()
            q = left.replace("问题:", "").strip()
            a = right
        else:
            q = text.strip()
            a = "（未解析到答案字段）"
        return q, a

    @property
    def backend(self) -> str:
        return self._vector_backend


INDEX_MANAGER = IndexManager()


# -----------------------------
# FastAPI：数据模型与路由
# -----------------------------
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    score: float


def create_app() -> Any:
    if FastAPI is None:
        raise RuntimeError("缺少 fastapi/uvicorn 依赖。请先安装：pip install fastapi uvicorn pydantic")

    app = FastAPI(
        title="Milvus FAQ 检索系统（单文件版）",
        version="1.0.0",
        description="基于 LlamaIndex + Milvus 的 FAQ 检索系统，支持热更新与 RESTful API。",
    )

    @app.get("/")
    def root():
        return {
            "message": "Milvus FAQ 检索系统已启动。请访问 /docs 查看接口文档。",
            "faq_file": FAQ_FILE,
            "vector_backend": INDEX_MANAGER.backend or "unknown",
        }

    @app.post("/api/query", response_model=List[QueryResponse])
    def query_faq(req: QueryRequest):
        q = (req.question or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="问题不能为空")

        results = INDEX_MANAGER.query(q)
        # 可按阈值过滤（演示用，默认不强制）
        return [QueryResponse(**r) for r in results]

    @app.post("/api/update-index")
    def update_index():
        try:
            return INDEX_MANAGER.update_index()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"索引更新失败: {str(e)}")

    return app


# -----------------------------
# CLI：命令行检索 / 服务启动
# -----------------------------
def run_cli(query: str) -> int:
    results = INDEX_MANAGER.query(query)
    payload = {
        "query": query,
        "top_k": TOP_K,
        "backend": INDEX_MANAGER.backend,
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_serve(host: str, port: int) -> int:
    app = create_app()
    # 启动信息
    print("=== Milvus FAQ 检索系统启动 ===")
    print(f"- FAQ_FILE: {FAQ_FILE}")
    print(f"- VECTOR_DIR: {VECTOR_DIR}")
    print(f"- MILVUS_URI: {MILVUS_URI}")
    print(f"- COLLECTION: {COLLECTION_NAME}")
    print(f"- EMBED_DIM: {EMBED_DIM}")
    print(f"- CHUNK_SIZE/OVERLAP: {CHUNK_SIZE}/{CHUNK_OVERLAP}")
    print(f"- TOP_K: {TOP_K}")
    print(f"- BACKEND: {'milvus' if _MILVUS_AVAILABLE else 'memory (fallback)'}")
    print("=============================")

    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Milvus FAQ 检索系统（LlamaIndex + Milvus, 单文件可提交版）")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_cli = sub.add_parser("cli", help="命令行检索（不启动 API）")
    p_cli.add_argument("--q", required=True, help="用户问题")

    p_serve = sub.add_parser("serve", help="启动 FastAPI 服务")
    p_serve.add_argument("--host", default="0.0.0.0", help="监听地址")
    p_serve.add_argument("--port", type=int, default=8000, help="监听端口")

    p_update = sub.add_parser("reindex", help="本地触发热更新（不通过 API）")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    # 启动时预热（可选）：确保索引可用
    # 对服务模式：避免首个请求阻塞；对 CLI：保证输出稳定
    if args.cmd in ("serve", "cli"):
        INDEX_MANAGER.initialize()

    if args.cmd == "cli":
        return run_cli(args.q)

    if args.cmd == "serve":
        return run_serve(args.host, args.port)

    if args.cmd == "reindex":
        res = INDEX_MANAGER.update_index()
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
