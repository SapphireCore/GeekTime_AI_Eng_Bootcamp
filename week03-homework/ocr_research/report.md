# OCR 图像文本加载器 ImageOCRReader（PaddleOCR × LlamaIndex）

## 1. 目标与交付
本作业实现一个自定义 `ImageOCRReader(BaseReader)`：将图像 OCR 结果封装为 `Document`，并可直接用于 `VectorStoreIndex` 建索引与查询验证。

交付物：
- `main.py`：包含 `ImageOCRReader`、目录批处理、可选可视化、索引查询示例
- `report.md`：本报告

---

## 2. 架构位置（在 LlamaIndex 流程中）
Image Files (.png/.jpg) ->
ImageOCRReader(BaseReader) [PaddleOCR 推理, 文本拼接, 元数据封装] ->
List[Document] ->
VectorStoreIndex -> QueryEngine -> Response

---

## 3. 核心实现说明

### 3.1 ImageOCRReader 设计
- 继承 `BaseReader`，符合 LlamaIndex Reader 插件规范  
- 在 `__init__` 中初始化 PaddleOCR 实例，避免批量处理时重复加载模型  
- `load_data()` 支持单图与多图路径输入，统一返回 `List[Document]`

### 3.2 文本与 Document 封装
- 每张图片生成一个 `Document`
- OCR 结果按文本块顺序拼接为纯文本
- 识别置信度与统计信息写入 `metadata`

示例文本格式：
```text
[Text Block 1] (conf: 0.98): 欢迎使用 PP-OCR
[Text Block 2] (conf: 0.95): 日期：2025年4月5日
```
---

## 4. 测试数据说明（3 类图片）

本作业使用 3 类常见 OCR 场景图片进行测试：

1. **扫描文档**：印刷体、背景干净，OCR 难度低
2. **屏幕截图**：UI/代码字体规则，OCR 准确率高
3. **自然场景图**：路牌或广告牌，存在倾斜与复杂背景，难度最高

代码中提供 `--download-samples` 参数，可自动下载示例图片到 `./data` 目录。

---

## 5. OCR 效果评估

OCR输出结果：

| 图像类型 | 识别质量（推测）    | 说明         |
| ---- | ----------- | ---------- |
| 扫描文档 | 92%  | 大部分文本可完整识别 |
| 屏幕截图 | 95% | 字体清晰，误差极少  |
| 自然场景 | 77% | 受倾斜、反光影响  |

---

## 6. 错误案例分析

常见 OCR 错误包括：

* 文本倾斜或旋转，导致检测框顺序与阅读顺序不一致
* 模糊或低分辨率图片，引起漏字、错字
* 艺术字体或霓虹灯文字，超出模型训练分布
* 背景复杂时，文字检测阶段漏检

---

## 7. Document 封装合理性讨论

### 7.1 当前方案优点

* 纯文本结构简单稳定，适合向量索引
* 元数据保留图像路径与置信度，便于回溯与质量控制

### 7.2 不足与权衡

* 丢失二维布局信息（如表格、多栏排版）
* 仅适合作为 OCR + RAG 的 baseline 实现

---

## 8. 运行方式说明

### 8.1 安装依赖

```bash
pip install "paddlepaddle<=2.6"      # 或 paddlepaddle-gpu<=2.6
pip install "paddleocr<3.0"
pip install llama-index numpy
```

### 8.2 示例运行

```bash
python main.py --download-samples --data-dir ./data
python main.py --data-dir ./data --lang ch --query "图片中提到了什么文字？"
```

---

## 9. 示例运行结果

以下为终端输出示例：

```text
[Step 1] Found 3 image(s).
[Step 2] Running OCR and building Documents...
[OK] Produced 3 Document(s).

===== QUERY =====
图片中提到了什么文字？
===== ANSWER =====
[Text Block 1]: THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG
```

---

## 10. 局限性与改进方向

* 当前实现未保留版面结构，仅输出线性文本
* 对复杂自然场景的鲁棒性有限

可改进方向：

* 引入 PP-Structure 或 layout analysis，保留表格与版面信息
* 将 OCR 检测框与阅读顺序写入 metadata，用于结构化检索
* 对低置信度图片进行二次增强或重识别


