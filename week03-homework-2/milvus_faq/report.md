# 基于 Milvus 的 FAQ 检索系统（LlamaIndex 实现）

## 1. 项目背景与目标
本项目实现一个面向“剧本杀玩家常见问题”的 FAQ 检索系统：
- **输入**：用户自然语言问题（例如“第一次玩剧本杀需要准备什么？”）
- **输出**：最相关的 FAQ 条目及其答案（Top-K）

同时满足扩展项：
- **热更新知识库（自动 re-index）**
- **RESTful API（FastAPI 封装）**

工程约束：
- 使用 **LlamaIndex** 构建索引与检索链路
- 使用 **Milvus（优先 Milvus Lite）** 作为向量库
- 实现 **文档切片优化**（语义切分 + 重叠）

---

## 2. 总体架构设计说明

### 2.1 架构概览
系统采用典型“离线索引 + 在线检索”的两层结构：

1) **数据层（FAQ Knowledge Base）**
- 数据文件：`data/faqs.csv`
- 内容：30 条“问题-答案”对（剧本杀领域）

2) **索引层（Indexing Pipeline）**
- LlamaIndex 将 FAQ 条目构造成 `Document`
- 使用 **SemanticSplitterNodeParser** 做语义切分与重叠
- 将切分后的节点向量写入 **Milvus** Collection

3) **服务层（Retrieval API）**
- FastAPI 提供：
  - `POST /api/query`：检索 Top-K FAQ
  - `POST /api/update-index`：触发热更新 re-index
- `GET /docs`：自动生成 Swagger 文档

---

## 3. 核心模块与关键实现逻辑（单文件 main.py）

### 3.1 数据加载与结构化
- 通过 `csv.DictReader` 读取 `data/faqs.csv`，要求表头必须为 `question,answer`
- 每条 FAQ 构造成一个 `Document`，文本格式固定为：
```text
问题: <question>
答案: <answer>
```


该格式保证检索返回 `source_nodes` 后可稳定反解析回原始 question/answer。

### 3.2 文档切片优化：语义切分 + 重叠
- 采用 `SemanticSplitterNodeParser`：
  - 按语义边界切分文本（优于纯长度切分）
  - 配合 `chunk_size` 与 `chunk_overlap` 控制碎片化风险和上下文保留

参数（可通过环境变量覆盖）：
- `CHUNK_SIZE` 默认 512
- `CHUNK_OVERLAP` 默认 64

### 3.3 向量库：Milvus（优先 Milvus Lite）
- 优先使用 `MilvusVectorStore(uri=..., collection_name=..., dim=...)`
- 使用本地文件 URI（Milvus Lite 常见用法），无须独立启动数据库服务
- 若环境缺少 Milvus 依赖，代码自动降级为内存向量库（便于本地验收），但作业验收时建议安装 Milvus 相关依赖以满足要求

### 3.4 热更新（re-index）
- `POST /api/update-index` 或 `python main.py reindex`：
  - 以 `overwrite=True` 清空旧集合
  - 从 `data/faqs.csv` 重新加载与切分
  - 重建索引并替换 QueryEngine

---

## 4. 关键技术选型与权衡（Trade-offs）

### 4.1 Embedding 模型：离线可运行 vs 真实语义效果
为满足“可直接运行、可提交、无需外部 API Key”的工程要求，本项目默认使用：
- `DeterministicHashEmbedding`（确定性 hash → 向量）

优点：
- 零外部依赖、离线可运行
- 每次运行结果稳定

局限：
- 不具备真实语义能力，检索质量仅用于演示功能链路

如果具备真实 Embedding API：
- 可替换为 OpenAIEmbedding、DashScopeEmbedding、HuggingFaceEmbedding 等，以获得显著更好的语义检索效果（代码结构已为替换留好入口：`Settings.embed_model = ...`）

### 4.2 语义切分器（Semantic Splitter）
FAQ 的单条文本通常较短，但答案可能增长；语义切分器能减少“切断关键语义单元”的概率，提升检索节点的语义完整性。对更长知识库文本（如规则说明、长文档）更有价值。

---

## 5. RESTful API 设计

### 5.1 查询接口
- `POST /api/query`
- Request:
```json
{"question": "第一次玩剧本杀需要准备什么？"}
```
Response（Top-K）：
```json
{
  {"question": "...", "answer": "...", "score": 0.83},
  ...
}
```

### 5.2 热更新接口

POST /api/update-index
Response:
```json
{"status":"success","message":"索引已成功热更新（re-index 完成）。","backend":"milvus","count":30}
```
---

## 6. 运行方式说明

### 6.1 准备数据
将 FAQ 数据保存为以下路径与格式：

- 路径：`data/faqs.csv`
- 编码：UTF-8
- 表头必须为：
```text
question,answer
````

每一行表示一条 FAQ 问答对。本作业示例使用的是**剧本杀玩家常见问题**，共 30 条，已满足课程规模与检索验证需求。

---

### 6.2 安装依赖

最低可运行依赖：

```bash
pip install llama-index fastapi uvicorn pydantic
```

Milvus 向量库相关依赖：

```bash
pip install pymilvus llama-index-vector-stores-milvus
```

> 说明：
>
> * 本工程默认使用 **Milvus Lite（本地文件模式）**，无需单独启动 Milvus 服务
> * 若 Milvus 相关依赖缺失，系统会自动降级为内存向量库（仅用于本地演示，不建议用于最终验收）

---

### 6.3 启动 RESTful API 服务

```bash
python main.py serve --host 0.0.0.0 --port 8000
```

启动后可访问：

* Swagger UI：`http://localhost:8000/docs`
* API Root：`http://localhost:8000/`

---

### 6.4 命令行方式直接检索（无需启动 API）

```bash
python main.py cli --q "第一次玩剧本杀需要准备什么？"
```

该模式适用于：

* 本地调试
* 快速验证索引是否构建成功
* 不依赖 FastAPI 的最小运行验证

---

### 6.5 手动触发热更新

当 `data/faqs.csv` 内容发生变化后，可执行：

```bash
python main.py reindex
```

该命令将：

* 清空现有向量集合
* 重新加载 FAQ 数据
* 重新进行语义切分与向量写入

---

## 7. 示例运行结果

### 7.1 服务启动日志

```text
=== Milvus FAQ 检索系统启动 ===
- FAQ_FILE: data/faqs.csv
- VECTOR_DIR: vector_store
- MILVUS_URI: vector_store/milvus_faq.db
- COLLECTION: faq_collection
- EMBED_DIM: 384
- CHUNK_SIZE/OVERLAP: 512/64
- TOP_K: 3
- BACKEND: milvus
=============================
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

---

### 7.2 API 查询示例

请求：

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question":"如何避免贴脸？"}'
```

响应（示例）：

```json
[
  {
    "question": "如何避免“贴脸”（人身攻击）？",
    "answer": "质疑只针对剧情行为与证据，不针对现实人格。使用“我认为你的时间线有矛盾”而不是“你就是在狡辩”。出现不适可立即请DM介入。",
    "score": 0.79
  },
  {
    "question": "有人太强势带节奏怎么办？",
    "answer": "先把讨论拉回证据：请对方给出证据来源与可验证点；必要时请求DM控场轮流发言。",
    "score": 0.63
  }
]
```

---

### 7.3 热更新接口示例

```bash
curl -X POST http://localhost:8000/api/update-index
```

返回：

```json
{
  "status": "success",
  "message": "索引已成功热更新（re-index 完成）。",
  "backend": "milvus",
  "count": 30
}
```

---

## 8. 已知限制与可扩展方向

### 8.1 已知限制

1. **Embedding 为离线 Mock 实现**
   当前使用的是确定性 Hash Embedding，仅用于保证系统可离线运行与功能完整性，不具备真实语义理解能力。

2. **Milvus Lite 依赖版本敏感**
   不同 `pymilvus` / `llama-index` 版本在本地 URI 支持上存在差异。

3. **FAQ 数据规模有限**
   当前数据规模（30 条）适合教学与演示，不代表生产级吞吐或召回效果。

---

### 8.2 可扩展方向

1. **接入真实向量模型**

   * 替换为 OpenAI / DashScope / HuggingFace Embedding
   * 显著提升语义检索效果

2. **检索阈值与拒答机制**

   * 为相似度设置下限阈值
   * 当 Top-K 均低于阈值时返回“未命中答案”

3. **RAG 生成式回答**

   * 在 Top-K FAQ 基础上使用 LLM 合成自然语言答案
   * 支持多条 FAQ 的信息融合

4. **增量更新与版本管理**

   * 支持仅对新增/修改 FAQ 做增量索引
   * 引入知识库版本号，支持回滚

5. **多知识域扩展**

   * 支持多个 FAQ Collection（如新手本 / 硬核本 / 情感本）
   * 按标签或场景路由检索