# 融合文档检索（RAG）+ 图谱推理（KG）的多跳问答系统（客服场景）实验报告

---

## 1. 项目背景与目标

本项目实现一个**融合文档检索（RAG）与知识图谱推理（KG）的多跳问答系统**，用于客服业务场景下的复杂问题回答。  
系统不仅依赖非结构化文本（客诉记录、客服画像），还结合结构化关系（案例—产品—客服—能力），从而避免“仅靠 LLM 猜测”的不可靠方式。

**典型问题示例：**
- “C-2025-004 这个客诉由谁负责？为什么？”
- “这个误扣费案例应该由哪个客服 Agent 处理？”

**核心目标：**
1. 使用 **Milvus + LlamaIndex** 实现客诉文档的向量化检索  
2. 使用 **Neo4j** 构建客服知识图谱并支持多跳推理  
3. 实现 **RAG + KG 融合问答流程**  
4. 设计 **联合评分与 Guardrail**，防止错误传播  
5. 输出 **可解释推理路径（Reasoning Path）**  

---

## 2. 数据设计与文件组织

系统使用两类文本数据文件，均为**可读、可审计**的原始文本：

### 2.1 客诉案例（RAG 主体）
- 文件：`data/complaints.txt`
- 数量：10 条
- 内容：用户信息、产品、严重级别、问题描述、关键词

用途：
- 构建向量索引（Milvus）
- 为问答提供事实性证据

### 2.2 客服 Agent 画像
- 文件：`data/agents.txt`
- 数量：5 名
- 内容：等级、技能、负责产品线、权限

用途：
- 构建知识图谱节点
- 作为 RAG 的补充检索上下文

---

## 3. 系统总体架构设计

### 3.1 逻辑流程

```

用户问题
↓
实体抽取（CASE_ID / 错误码 / 用户ID）
↓
RAG 文档检索（Milvus）
↓
知识图谱多跳推理（Neo4j + Cypher）
↓
联合评分（RAG + KG + 一致性）
↓
Guardrail 判断
↓
LLM 汇总生成最终回答

```

### 3.2 模块划分

| 模块 | 说明 |
|----|----|
| RAGStore | 文档切分、向量化、Milvus 检索 |
| GraphStore | Neo4j 建图、Cypher 查询 |
| MultiHopQA | 多跳编排与联合评分 |
| MockLLM | 可运行的 LLM 模拟实现 |
| FastAPI | 对外查询接口 |

---

## 4. 知识图谱建模与多跳推理

### 4.1 图谱 Schema 设计

**节点类型：**
- `Case(id, severity, channel)`
- `User(id, name)`
- `Product(name)`
- `Agent(id, name, level)`
- `Skill(name)`

**关系类型：**
- `(Case)-[:RAISED_BY]->(User)`
- `(Case)-[:ABOUT_PRODUCT]->(Product)`
- `(Case)-[:ASSIGNED_TO {confidence, why}]->(Agent)`
- `(Agent)-[:HAS_SKILL]->(Skill)`
- `(Agent)-[:SUPPORTS_PRODUCT]->(Product)`

### 4.2 多跳查询示例（Cypher）

```cypher
MATCH (cs:Case {id:$cid})-[r:ASSIGNED_TO]->(ag:Agent)
RETURN ag.id AS agent_id,
       ag.name AS agent_name,
       r.confidence AS kg_confidence,
       r.why AS why
LIMIT 3
```

说明：

* 不让 LLM 生成 Cypher，避免不稳定性
* 图谱返回结构化、可解释结果

---

## 5. RAG（Milvus + LlamaIndex）设计

### 5.1 文档切分与索引

* 文档来源：`complaints.txt`、`agents.txt`
* 切分策略：SentenceSplitter

  * chunk_size = 512
  * overlap = 80
* 存储：Milvus Collection（`cs_rag_docs`）

### 5.2 检索结果结构

```json
{
  "doc_id": "case::C-2025-004",
  "text": "...",
  "rag_score": 0.82
}
```

`rag_score` 被统一归一化到 `[0,1]`，用于后续联合评分。

---

## 6. RAG 与 KG 融合机制（关键技术点）

### 6.1 联合评分公式

```text
joint_score =
  0.45 * rag_conf
+ 0.45 * kg_conf
+ 0.10 * consistency
```

* `rag_conf`：RAG Top-1 相似度
* `kg_conf`：图谱关系置信度
* `consistency`：RAG 与 KG 的一致性命中

阈值：

* `joint_score ≥ 0.62`
* 或 `kg_conf ≥ 0.85`

---

### 6.2 错误传播防控（Guardrails）

**潜在风险：**

* 图谱关系错误
* 文档检索噪声

**防护措施：**

1. 一致性校验（case_id / agent_name）
2. 阈值门控（联合评分）
3. 失败时输出**保守回答 + 补充信息建议**

---

## 7. LLM 使用方式与 Prompt 设计

### 7.1 LLM 使用策略

* 默认使用 `MockLLM`

  * 不依赖外部 API
  * 可复现、可提交
* 工程上可无缝替换为真实 LLM（如 OpenAI / Qwen）

### 7.2 Prompt 设计原则

* 不生成结构化事实（交由 KG）
* 只负责：

  1. 实体抽取
  2. 证据汇总与行动建议生成

---

## 8. 运行方式说明（How to Run）

### 8.1 构建图谱与索引

```bash
python main.py --mode build
```

### 8.2 启动服务

```bash
python main.py --mode serve
```

访问：

```
http://127.0.0.1:8000/docs
```

### 8.3 命令行问答

```bash
python main.py --mode cli \
  --question "C-2025-004 这个客诉由谁负责？为什么？"
```

---

## 9. 示例运行结果（节选）

```json
{
  "final_answer": "建议由【林晓】负责处理...",
  "reasoning_path": [
    "实体抽取 -> C-2025-004",
    "RAG 命中 case::C-2025-004",
    "KG 命中 Agent 林晓（置信度 0.93）",
    "联合评分通过（0.88）"
  ]
}
```

---

## 10. 已知限制与可扩展方向

### 已知限制

* Milvus / Neo4j 未启动时会降级为本地模式
* 当前聚焦“案例负责人类问题”

### 可扩展方向

1. 自动 Issue / 错误码抽取写入图谱
2. SOP / 工单节点加入图谱，实现“行动推荐”
3. 联合评分权重学习（替代人工权重）
4. 支持多案例对比、多 Agent 推荐