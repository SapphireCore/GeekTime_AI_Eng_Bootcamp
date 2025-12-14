# 多代理协作写作流程记录

**任务主题：** 写一篇介绍RAG的理论与研究前沿的文章

**风格：** 通俗但专业

**目标长度：** 1400

## 研究代理输出（Research Report）

# 研究资料（Mock）

## 核心概念
- **RAG（Retrieval-Augmented Generation）**：在生成前检索外部知识，将检索证据注入上下文以降低幻觉、提升可追溯性。
- **向量检索**：通过 embedding 将文本映射到向量空间，用 ANN 索引实现相似度检索。
- **重排（Re-ranking）**：用更强模型对候选文档二次排序，提升相关性与答案一致性。

## 关键技术/机制
1. Chunking：语义切片与窗口策略影响召回与噪声。
2. Retriever：BM25 + Dense / Hybrid 是常见组合。
3. Reranker：cross-encoder / LLM rerank。
4. Context Construction：去重、压缩、引用标注。
5. Grounded Generation：基于证据生成，必要时拒答。
6. Evaluation：检索/生成分层评估与端到端评估结合。

## 代表性论文/系统（示例）
- 2020: REALM（Google）— 端到端检索增强预训练
- 2021: RAG（Meta）— 检索增强生成框架
- 2023-2024: Long-context + RAG 结合的工程范式（多家）

## 工程实践要点
- 先把评估体系做扎实：Recall@k、nDCG、Faithfulness、Answerability
- 建议做 Hybrid retrieval + rerank
- 引入引用与证据片段对齐（span-level）
- 防提示注入与数据外泄：输入消毒、内容隔离、allowlist 工具
- 在线监控：无答案率、引用覆盖率、延迟、成本
- 数据闭环：用户反馈→标注→增量索引/对齐

## 风险与误区
- 只看最终答案不看证据质量
- 过度堆 k 值导致噪声放大
- 评估集泄漏（与索引同源）
- 忽视时效性与版本管理
- 仅靠“长上下文”替代检索导致成本暴涨

## 参考链接
- https://ai.meta.com/
- https://arxiv.org/

## 撰写代理输出（Draft）

# RAG：从理论到工程落地的主流范式与研究前沿（Mock）
见sample_output.md。

## 审核代理输出（Review Suggestions）
见sample_output.md。

## 润色代理输出（Final Article）
见sample_output.md。

---

# 异常处理日志
无。
