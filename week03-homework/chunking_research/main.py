# main.py
# -*- coding: utf-8 -*-
"""
作业一：探索 LlamaIndex 中的句子切片检索及其参数影响分析
----------------------------------------------------------------
实现说明：
- 单一主文件，可直接运行、可提交
- 假定 ./data 目录下已存在markdown / text 文档
- 本文件仅负责：
  1) 读取文档并做 strip 处理
  2) 使用不同切片策略构建索引
  3) 统一评估检索与生成效果
  4) 输出对比结果表

运行方式：
    python main.py
"""

import os
from dotenv import load_dotenv
import pandas as pd

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    PromptTemplate,
)
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)

from llama_index.core.node_parser import (
    SentenceSplitter,
    TokenTextSplitter,
    SentenceWindowNodeParser,
    MarkdownNodeParser,
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor


# =========================
# 评估 Prompt（LLM-as-Judge）
# =========================
EVAL_TEMPLATE_STR = (
    "我们提供了一个标准答案和一个由模型生成的回答。请你判断模型生成的回答在语义上是否与标准答案一致、准确且完整。\n"
    "请只回答 '是' 或 '否'。\n\n"
    "标准答案：\n{ground_truth}\n\n"
    "模型生成的回答：\n{generated_answer}\n"
)
EVAL_PROMPT = PromptTemplate(EVAL_TEMPLATE_STR)


def evaluate_splitter(
    splitter,
    documents,
    question: str,
    ground_truth: str,
    splitter_name: str,
):
    """
    对某一种切片策略进行完整评估：
    - 构建索引
    - 检索上下文
    - 生成回答
    - 判断准确性
    - 人工输入冗余度评分
    """
    print(f"\n========== 开始评估：{splitter_name} ==========")

    # 1. 切片
    nodes = splitter.get_nodes_from_documents(documents)

    # 2. 构建向量索引
    index = VectorStoreIndex(nodes)

    # 3. 构建 Query Engine
    if isinstance(splitter, SentenceWindowNodeParser):
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            node_postprocessors=[
                MetadataReplacementPostProcessor(
                    target_metadata_key="window"
                )
            ],
        )
    else:
        query_engine = index.as_query_engine(similarity_top_k=5)

    # 4. 检索上下文
    retrieved_nodes = query_engine.retrieve(question)
    retrieved_context = "\n\n".join(
        [node.get_content() for node in retrieved_nodes]
    )

    # 5. 判断上下文是否包含答案关键信息
    contains_answer = "是" if ground_truth[:15] in retrieved_context else "否"

    # 6. 生成回答
    response = query_engine.query(question)
    generated_answer = str(response)

    # 7. 使用 LLM 进行自动评判
    judge = Settings.llm.predict(
        EVAL_PROMPT,
        ground_truth=ground_truth,
        generated_answer=generated_answer,
    )
    answer_correct = "是" if "是" in judge else "否"

    # 8. 人工评估上下文冗余度
    print("\n--- 检索到的上下文 ---\n")
    print(retrieved_context)
    redundancy = input(
        f"\n请为【{splitter_name}】的上下文冗余度评分 (1=低冗余, 5=高冗余)："
    )
    while redundancy not in ["1", "2", "3", "4", "5"]:
        redundancy = input("请输入 1~5 之间的整数：")

    # 9. 存储结果
    if not hasattr(evaluate_splitter, "results"):
        evaluate_splitter.results = []

    evaluate_splitter.results.append(
        {
            "切片策略": splitter_name,
            "上下文包含答案": contains_answer,
            "回答准确": answer_correct,
            "上下文冗余度(1-5)": int(redundancy),
            "回答摘要": generated_answer.replace("\n", " ")[:120] + "...",
        }
    )

    print(f"========== 完成评估：{splitter_name} ==========")


def print_summary():
    """打印最终结果对比表"""
    print("\n\n========== 最终对比结果 ==========")
    df = pd.DataFrame(evaluate_splitter.results)
    print(df.to_markdown(index=False))


def main():
    # =========================
    # 1. 环境与模型配置
    # =========================
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")

    Settings.llm = OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        is_chat_model=True,
    )

    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        embed_batch_size=6,
        embed_input_length=8192,
    )

    # =========================
    # 2. 加载数据（Markdown/Text）
    # =========================
    documents = SimpleDirectoryReader("./data").load_data()
    for doc in documents:
        doc.text = doc.text.strip()

    # =========================
    # 3. 评估问题与标准答案
    # =========================
    question = "剧本杀对新手玩家有哪些核心特点和体验价值？"
    ground_truth = (
        "剧本杀是一种融合角色扮演、推理和社交互动的沉浸式游戏。"
        "它强调通过阅读剧本、交流信息和逻辑推理来推动故事发展，"
        "不仅锻炼思考能力，也满足社交和情感体验需求。"
    )

    # =========================
    # 4. 不同切片策略评估
    # =========================
    evaluate_splitter(
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        documents,
        question,
        ground_truth,
        "SentenceSplitter",
    )

    evaluate_splitter(
        TokenTextSplitter(chunk_size=128, chunk_overlap=4),
        documents,
        question,
        ground_truth,
        "TokenTextSplitter",
    )

    evaluate_splitter(
        SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        ),
        documents,
        question,
        ground_truth,
        "SentenceWindowNodeParser",
    )

    evaluate_splitter(
        MarkdownNodeParser(),
        documents,
        question,
        ground_truth,
        "MarkdownNodeParser",
    )

    print_summary()


if __name__ == "__main__":
    main()
