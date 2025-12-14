"""
融合文档检索（RAG）+ 图谱推理（KG）的多跳问答系统（客服场景）

目标能力
- 文档检索：Milvus 向量库（优先），读取 ./data/complaints.txt 与 ./data/agents.txt
- 图谱推理：Neo4j（构建 Case/User/Product/Issue/Agent 关系），支持多跳 Cypher 查询
- 多跳问答：RAG -> KG -> LLM（可选真实 LLM；无 API 时走 Mock，保证可运行）
- 联合评分：RAG 相似度 + KG 置信度 + 一致性校验（降低错误传播）
- 可解释性：返回 reasoning_path（推理路径 + 证据 + 校验结果）

运行方式（示例）
1) 构建图谱并建立向量索引：
   python main.py --mode build
2) 启动 API：
   python main.py --mode serve
3) 直接命令行问答：
   python main.py --mode cli --question "C-2025-004 这个客诉由谁负责？为什么？"

依赖（建议）
pip install fastapi uvicorn neo4j pandas python-dotenv llama-index pymilvus

说明
- Milvus / Neo4j 若不可用，程序会降级到本地简化实现（仍可跑通流程），但报告中需说明真实环境要求。
"""

from __future__ import annotations

import os
import re
import json
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# 可选依赖：Neo4j
# ----------------------------
try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover
    GraphDatabase = None  # type: ignore

# ----------------------------
# 可选依赖：LlamaIndex
# ----------------------------
try:
    from llama_index.core import Document, VectorStoreIndex, StorageContext  # type: ignore
    from llama_index.core.schema import NodeWithScore  # type: ignore
    from llama_index.core import Settings  # type: ignore
    from llama_index.core.node_parser import SentenceSplitter  # type: ignore
except Exception:  # pragma: no cover
    Document = None  # type: ignore
    VectorStoreIndex = None  # type: ignore
    StorageContext = None  # type: ignore
    NodeWithScore = None  # type: ignore
    Settings = None  # type: ignore
    SentenceSplitter = None  # type: ignore

# ----------------------------
# 可选依赖：Milvus + LlamaIndex MilvusVectorStore
# ----------------------------
MILVUS_AVAILABLE = True
try:
    from pymilvus import connections, utility  # type: ignore
    from llama_index.vector_stores.milvus import MilvusVectorStore  # type: ignore
except Exception:  # pragma: no cover
    MILVUS_AVAILABLE = False

# ----------------------------
# 可选依赖：FastAPI
# ----------------------------
try:
    from fastapi import FastAPI, HTTPException  # type: ignore
    from pydantic import BaseModel, Field  # type: ignore
    import uvicorn  # type: ignore
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore
    HTTPException = None  # type: ignore
    BaseModel = None  # type: ignore
    Field = None  # type: ignore
    uvicorn = None  # type: ignore


# ============================================================
# 配置
# ============================================================

@dataclass
class AppConfig:
    data_dir: str = "data"
    complaints_path: str = os.path.join("data", "complaints.txt")
    agents_path: str = os.path.join("data", "agents.txt")

    # Milvus
    milvus_host: str = os.getenv("MILVUS_HOST", "127.0.0.1")
    milvus_port: str = os.getenv("MILVUS_PORT", "19530")
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "cs_rag_docs")
    milvus_dim: int = int(os.getenv("MILVUS_DIM", "1536"))  # 取决于 embedding 模型；Mock 模式不强依赖
    milvus_overwrite: bool = True

    # Neo4j
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    neo4j_user: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # Index persist（本地 fallback）
    local_index_dir: str = os.getenv("LOCAL_INDEX_DIR", "local_vector_index")

    # 系统参数
    rag_top_k: int = 4
    joint_score_threshold: float = 0.62
    consistency_min_hits: int = 1  # 至少 1 条一致性命中，否则降级回答
    verbose: bool = True


# ============================================================
# 轻量 Mock LLM / Embedding（无 Key 也可跑通）
# ============================================================

class MockLLM:
    """
    一个可控、可复现的 Mock LLM：
    - 用于实体抽取（case_id / 用户ID / 产品/错误码）
    - 用于最终回答的“模板化生成”
    """

    def complete(self, prompt: str) -> "MockLLMResponse":
        # 简化：根据 prompt 内部包含的关键片段做规则返回
        if "只返回名称" in prompt or "提取" in prompt:
            # 尝试抽取 CASE_ID
            m = re.search(r"(C-\d{4}-\d{3})", prompt)
            if m:
                return MockLLMResponse(m.group(1))
            # 抽取用户ID
            m = re.search(r"(U\d{4})", prompt)
            if m:
                return MockLLMResponse(m.group(1))
            # 抽取错误码
            m = re.search(r"([A-Z]{2,4}-\d{2,3})", prompt)
            if m:
                return MockLLMResponse(m.group(1))
            return MockLLMResponse("UNKNOWN")
        # 最终回答：尽量生成“专业客服归因 + 下一步”
        return MockLLMResponse(self._render_answer(prompt))

    def _render_answer(self, prompt: str) -> str:
        # 从 prompt 中抓取“图谱结果”与“文档证据”
        kg = self._extract_section(prompt, "知识图谱查询结果")
        rag = self._extract_section(prompt, "相关文档信息")
        question = self._extract_section(prompt, "用户问题")

        # 兜底解析
        agent = "（未找到）"
        reason = "（缺少足够证据）"
        next_steps = "建议补充更多信息后再处理。"

        # 从 KG JSON 中抓 agent
        m = re.search(r"'agent_name':\s*'([^']+)'", kg)
        if m:
            agent = m.group(1)
        else:
            m = re.search(r"agent_name[\"']?\s*[:=]\s*[\"']([^\"']+)[\"']", kg)
            if m:
                agent = m.group(1)

        # 从 RAG 里抓关键词
        if "P0" in rag or "安全" in rag or "入侵" in rag:
            reason = "该案例涉及 P0 安全/高风险问题，需要具备网络与安全响应能力的高级工程支持。"
            next_steps = "建议立即指导用户修改管理员密码、恢复出厂并升级固件，同时启动安全应急工单并核查异常登录来源。"
        elif "误扣费" in rag or "退款" in rag or "订阅" in rag or "RSK-" in rag:
            reason = "该案例属于计费/订阅/风控协同问题，需要对账与规则核查经验。"
            next_steps = "建议先核对订阅状态与扣费流水，必要时触发退款并发起计费核查工单或临时白名单放行。"
        elif "反锁" in rag or "上门" in rag or "换新" in rag or "机械" in rag:
            reason = "该案例为硬件售后与上门处理类问题，需要具备调度与换新/赔付流程权限。"
            next_steps = "建议立即安排上门，优先恢复可用性；评估是否走快速换新通道并对用户损失按政策处理。"
        elif "积分" in rag or "活动" in rag:
            reason = "该案例涉及会员权益/活动规则核对，需要核对活动口径与积分流水。"
            next_steps = "建议核对活动时间窗与订单条件，补发差额积分并对规则口径做书面说明。"
        elif "安装" in rag or "补偿券" in rag or "服务商" in rag:
            reason = "该案例属于履约/安装服务投诉，需要协调服务商与补偿发放。"
            next_steps = "建议重新预约上门并核对补偿券发放记录，对服务商进行投诉登记与考核。"
        else:
            reason = "根据现有证据可初步定位责任归属，但建议补充日志/截图以进一步确认。"
            next_steps = "建议收集日志、错误码、复现步骤，并按标准工单流程升级。"

        return (
            f"问题：{question.strip()}\n\n"
            f"结论：建议由【{agent}】负责处理。\n"
            f"原因：{reason}\n\n"
            f"建议下一步：{next_steps}\n"
        )

    @staticmethod
    def _extract_section(prompt: str, title: str) -> str:
        # 以 --- 标记的块抽取
        # 兼容：--- 标题 ---\n内容\n\n
        pat = rf"---\s*{re.escape(title)}\s*---\n(.*?)(\n---|\Z)"
        m = re.search(pat, prompt, flags=re.S)
        return (m.group(1).strip() if m else "")


class MockLLMResponse:
    def __init__(self, text: str):
        self.text = text


class MockEmbedder:
    """
    可复现的“伪 embedding”：对文本做 hash -> 向量（仅用于本地 fallback，不建议真实使用）
    """
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # 扩展到 dim
        vals = []
        for i in range(self.dim):
            vals.append((h[i % len(h)] / 255.0) * 2 - 1)
        return vals


# ============================================================
# 数据解析（从文本构建结构化对象）
# ============================================================

@dataclass
class ComplaintCase:
    case_id: str
    user_id: str
    user_name: str
    channel: str
    product: str
    severity: str
    raw_text: str
    keywords: List[str]


@dataclass
class AgentProfile:
    agent_id: str
    name: str
    level: str
    skills: List[str]
    products: List[str]
    raw_text: str


def ensure_data_files(cfg: AppConfig) -> None:
    """
    为避免“缺失文件导致无法运行”，如果 data 文件不存在则创建一个最小可运行样例。
    注意：正式提交建议使用你提供/生成的 data/*.txt 文件。
    """
    os.makedirs(cfg.data_dir, exist_ok=True)
    if not os.path.exists(cfg.complaints_path):
        with open(cfg.complaints_path, "w", encoding="utf-8") as f:
            f.write("[CASE_ID: C-2025-000]\n用户: 测试用户（用户ID: U0000）\n渠道: App\n产品: 测试产品\n时间: 2025-12-01\n严重级别: P3\n客诉摘要: 测试文本\n用户诉求: 测试诉求\n关键词: 测试\n")
    if not os.path.exists(cfg.agents_path):
        with open(cfg.agents_path, "w", encoding="utf-8") as f:
            f.write("[AGENT_ID: A-00]\n姓名: 测试客服\n级别: L1\n擅长领域: 测试\n负责产品线: 测试\n升级权限: 无\n备注: 测试\n")


def parse_complaints(text: str) -> List[ComplaintCase]:
    blocks = re.split(r"\n\s*\n(?=\[CASE_ID:)", text.strip(), flags=re.M)
    cases: List[ComplaintCase] = []
    for b in blocks:
        case_id = _find_field(b, r"\[CASE_ID:\s*(C-\d{4}-\d{3})\]")
        user_name = _find_field(b, r"用户:\s*([^\n（]+)")
        user_id = _find_field(b, r"用户ID:\s*(U\d{4})")
        channel = _find_field(b, r"渠道:\s*([^\n]+)")
        product = _find_field(b, r"产品:\s*([^\n（]+)")
        severity = _find_field(b, r"严重级别:\s*([^\n]+)")
        keywords_line = _find_field(b, r"关键词:\s*([^\n]+)", default="")
        keywords = [k.strip() for k in re.split(r"[，,]", keywords_line) if k.strip()]
        if not case_id:
            continue
        cases.append(
            ComplaintCase(
                case_id=case_id,
                user_id=user_id or "UNKNOWN",
                user_name=user_name or "UNKNOWN",
                channel=channel or "UNKNOWN",
                product=product or "UNKNOWN",
                severity=severity or "UNKNOWN",
                raw_text=b.strip(),
                keywords=keywords,
            )
        )
    return cases


def parse_agents(text: str) -> List[AgentProfile]:
    blocks = re.split(r"\n\s*\n(?=\[AGENT_ID:)", text.strip(), flags=re.M)
    agents: List[AgentProfile] = []
    for b in blocks:
        agent_id = _find_field(b, r"\[AGENT_ID:\s*(A-\d{2})\]")
        name = _find_field(b, r"姓名:\s*([^\n]+)")
        level = _find_field(b, r"级别:\s*([^\n（]+)")
        skills_line = _find_field(b, r"擅长领域:\s*([^\n]+)", default="")
        products_line = _find_field(b, r"负责产品线:\s*([^\n]+)", default="")
        skills = [s.strip() for s in re.split(r"[；;、,，]", skills_line) if s.strip()]
        products = [p.strip() for p in re.split(r"[、,，]", products_line) if p.strip()]
        if not agent_id:
            continue
        agents.append(
            AgentProfile(
                agent_id=agent_id,
                name=name or "UNKNOWN",
                level=level or "UNKNOWN",
                skills=skills,
                products=products,
                raw_text=b.strip(),
            )
        )
    return agents


def _find_field(text: str, pattern: str, default: str = "") -> str:
    m = re.search(pattern, text)
    return m.group(1).strip() if m else default


# ============================================================
# 图谱构建（Neo4j）
# ============================================================

class GraphStore:
    """
    负责 Neo4j 的建图与查询；若 Neo4j 不可用，提供最小内存实现用于演示流程。
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._driver = None
        self._inmem = {
            "case_to_agent": {},  # case_id -> (agent_id, agent_name, confidence)
            "agent_skills": {},   # agent_id -> skills/products
            "case_meta": {},      # case_id -> dict
        }

    def connect(self) -> None:
        if GraphDatabase is None:
            self._driver = None
            return
        try:
            self._driver = GraphDatabase.driver(
                self.cfg.neo4j_uri, auth=(self.cfg.neo4j_user, self.cfg.neo4j_password)
            )
        except Exception:
            self._driver = None

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    def build_graph(self, cases: List[ComplaintCase], agents: List[AgentProfile]) -> Dict[str, Any]:
        """
        结构化建图策略（客服场景）：
        - (:Case {id, severity, channel})-[:ABOUT_PRODUCT]->(:Product {name})
        - (:Case)-[:RAISED_BY]->(:User {id, name})
        - (:Case)-[:ASSIGNED_TO {confidence}]->(:Agent {id, name, level})
        - (:Agent)-[:HAS_SKILL]->(:Skill {name})
        - (:Agent)-[:SUPPORTS_PRODUCT]->(:Product)
        """

        # 先做“分配策略”（规则化，可解释）：根据产品线 / 关键词 / 严重度匹配 agent
        assignments = self._assign_agents(cases, agents)

        if self._driver is None:
            # 内存模式：保存 assignment + meta
            for c in cases:
                self._inmem["case_meta"][c.case_id] = {
                    "case_id": c.case_id,
                    "user_id": c.user_id,
                    "user_name": c.user_name,
                    "channel": c.channel,
                    "product": c.product,
                    "severity": c.severity,
                    "keywords": c.keywords,
                }
            for a in agents:
                self._inmem["agent_skills"][a.agent_id] = {
                    "name": a.name,
                    "level": a.level,
                    "skills": a.skills,
                    "products": a.products,
                }
            for case_id, (agent_id, agent_name, conf, why) in assignments.items():
                self._inmem["case_to_agent"][case_id] = {
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "confidence": conf,
                    "why": why,
                }
            return {"mode": "inmem", "assigned": len(assignments)}

        # Neo4j 模式
        with self._driver.session(database=self.cfg.neo4j_database) as session:
            session.run("MATCH (n) DETACH DELETE n")

            # 索引
            session.run("CREATE CONSTRAINT case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE")
            session.run("CREATE CONSTRAINT agent_id IF NOT EXISTS FOR (a:Agent) REQUIRE a.id IS UNIQUE")
            session.run("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
            session.run("CREATE CONSTRAINT product_name IF NOT EXISTS FOR (p:Product) REQUIRE p.name IS UNIQUE")
            session.run("CREATE CONSTRAINT skill_name IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE")

            # 批量写入：Agents / Skills / Product support
            for a in agents:
                session.run(
                    """
                    MERGE (ag:Agent {id:$id})
                    SET ag.name=$name, ag.level=$level
                    """,
                    id=a.agent_id, name=a.name, level=a.level
                )
                for sk in a.skills:
                    session.run(
                        """
                        MERGE (s:Skill {name:$skill})
                        MATCH (ag:Agent {id:$aid})
                        MERGE (ag)-[:HAS_SKILL]->(s)
                        """,
                        skill=sk, aid=a.agent_id
                    )
                for p in a.products:
                    session.run(
                        """
                        MERGE (p:Product {name:$pname})
                        MATCH (ag:Agent {id:$aid})
                        MERGE (ag)-[:SUPPORTS_PRODUCT]->(p)
                        """,
                        pname=p, aid=a.agent_id
                    )

            # 批量写入：Cases / Users / Products / Assignment
            for c in cases:
                session.run(
                    """
                    MERGE (cs:Case {id:$cid})
                    SET cs.severity=$sev, cs.channel=$ch
                    MERGE (u:User {id:$uid})
                    SET u.name=$uname
                    MERGE (p:Product {name:$pname})
                    MERGE (cs)-[:RAISED_BY]->(u)
                    MERGE (cs)-[:ABOUT_PRODUCT]->(p)
                    """,
                    cid=c.case_id, sev=c.severity, ch=c.channel,
                    uid=c.user_id, uname=c.user_name, pname=c.product
                )

                # Assignment
                if c.case_id in assignments:
                    agent_id, agent_name, conf, why = assignments[c.case_id]
                    session.run(
                        """
                        MATCH (cs:Case {id:$cid})
                        MATCH (ag:Agent {id:$aid})
                        MERGE (cs)-[r:ASSIGNED_TO]->(ag)
                        SET r.confidence=$conf, r.why=$why
                        """,
                        cid=c.case_id, aid=agent_id, conf=conf, why=why
                    )

            return {"mode": "neo4j", "assigned": len(assignments)}

    def query_case_owner(self, case_id: str) -> Dict[str, Any]:
        """
        多跳图谱推理示例：
        - Case -> ASSIGNED_TO -> Agent
        - 同时返回 why/confidence 作为 KG 置信度证据
        """
        if self._driver is None:
            item = self._inmem["case_to_agent"].get(case_id)
            if not item:
                return {"found": False, "case_id": case_id, "rows": []}
            return {
                "found": True,
                "case_id": case_id,
                "rows": [{
                    "agent_id": item["agent_id"],
                    "agent_name": item["agent_name"],
                    "kg_confidence": float(item["confidence"]),
                    "why": item["why"],
                }]
            }

        cypher = """
        MATCH (cs:Case {id:$cid})-[r:ASSIGNED_TO]->(ag:Agent)
        RETURN ag.id AS agent_id, ag.name AS agent_name, r.confidence AS kg_confidence, r.why AS why
        LIMIT 3
        """
        with self._driver.session(database=self.cfg.neo4j_database) as session:
            res = session.run(cypher, cid=case_id)
            rows = [dict(r) for r in res]
        return {"found": len(rows) > 0, "case_id": case_id, "rows": rows, "cypher": cypher.strip()}

    def _assign_agents(self, cases: List[ComplaintCase], agents: List[AgentProfile]) -> Dict[str, Tuple[str, str, float, str]]:
        """
        可解释的规则分配（工程化替代“黑盒”）：
        - 先按产品线匹配 agent
        - 再按关键词/严重度微调置信度
        - P0 优先分配具备“紧急/安全/上门”权限或更高等级
        """
        agent_by_product: Dict[str, List[AgentProfile]] = {}
        for a in agents:
            for p in a.products:
                agent_by_product.setdefault(p, []).append(a)

        def level_rank(level: str) -> int:
            m = re.search(r"L(\d)", level)
            return int(m.group(1)) if m else 0

        assignments: Dict[str, Tuple[str, str, float, str]] = {}
        for c in cases:
            candidates = agent_by_product.get(c.product, [])
            if not candidates:
                # 回退：找 skills 命中最多的
                candidates = agents[:]

            # scoring
            best = None
            best_score = -1.0
            best_why = ""
            for a in candidates:
                score = 0.40  # base
                why_parts = []

                # 产品线强匹配
                if c.product in a.products:
                    score += 0.35
                    why_parts.append(f"产品线匹配({c.product})")

                # 严重度
                if "P0" in c.severity:
                    score += 0.10
                    why_parts.append("P0优先")
                    score += 0.03 * level_rank(a.level)
                    why_parts.append(f"等级加成({a.level})")
                elif "P1" in c.severity:
                    score += 0.05

                # 关键词/技能命中
                hit = 0
                for kw in c.keywords:
                    for sk in a.skills:
                        if kw.lower() in sk.lower() or sk.lower() in kw.lower():
                            hit += 1
                            break
                if hit > 0:
                    score += min(0.20, 0.06 * hit)
                    why_parts.append(f"技能/关键词命中({hit})")

                # 特殊硬规则：安全/入侵 -> 网络专家；反锁/上门 -> 售后调度；计费/风控 -> 计费专家
                c_text = c.raw_text
                if ("安全" in c_text or "入侵" in c_text or "异常登录" in c_text) and ("网络" in "".join(a.skills) or c.product in a.products):
                    score += 0.10
                    why_parts.append("安全事件加权")
                if ("反锁" in c_text or "上门" in c_text or "机械" in c_text) and ("上门" in "".join(a.skills) or c.product in a.products):
                    score += 0.08
                    why_parts.append("上门/售后加权")
                if ("误扣费" in c_text or "订阅" in c_text or "RSK-" in c_text) and ("计费" in "".join(a.skills) or "风控" in "".join(a.skills)):
                    score += 0.10
                    why_parts.append("计费/风控加权")

                if score > best_score:
                    best_score = score
                    best = a
                    best_why = "；".join(why_parts) if why_parts else "规则分配"

            if best:
                assignments[c.case_id] = (best.agent_id, best.name, float(min(0.99, best_score)), best_why)

        return assignments


# ============================================================
# RAG：Milvus 优先，失败则本地向量索引 fallback
# ============================================================

class RAGStore:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._index = None
        self._mode = "none"
        self._mock_embedder = MockEmbedder(dim=256)
        self._docs_cache: List[Tuple[str, str]] = []  # (doc_id, text)

    def build_index(self, cases: List[ComplaintCase], agents: List[AgentProfile]) -> Dict[str, Any]:
        docs: List[Tuple[str, str]] = []
        for c in cases:
            docs.append((f"case::{c.case_id}", c.raw_text))
        for a in agents:
            docs.append((f"agent::{a.agent_id}", a.raw_text))
        self._docs_cache = docs

        # 尝试 Milvus + LlamaIndex
        if MILVUS_AVAILABLE and Document is not None and VectorStoreIndex is not None:
            try:
                connections.connect(alias="default", host=self.cfg.milvus_host, port=self.cfg.milvus_port)
                if self.cfg.milvus_overwrite and utility.has_collection(self.cfg.milvus_collection):
                    utility.drop_collection(self.cfg.milvus_collection)

                vector_store = MilvusVectorStore(
                    collection_name=self.cfg.milvus_collection,
                    uri=f"http://{self.cfg.milvus_host}:{self.cfg.milvus_port}",
                    dim=self.cfg.milvus_dim,
                    overwrite=self.cfg.milvus_overwrite,
                )
                storage_context = StorageContext.from_defaults(vector_store=vector_store)

                documents = []
                for doc_id, text in docs:
                    documents.append(Document(text=text, metadata={"doc_id": doc_id}))

                # 节点切分：更稳健的检索
                parser = SentenceSplitter(chunk_size=512, chunk_overlap=80)
                nodes = parser.get_nodes_from_documents(documents)

                self._index = VectorStoreIndex(nodes, storage_context=storage_context)
                self._mode = "milvus"
                return {"mode": "milvus", "docs": len(docs), "nodes": len(nodes)}
            except Exception:
                # fall back
                pass

        # 本地 fallback：不依赖外部服务（可运行，但不符合“必须 Milvus”的严格环境）
        if Document is None:
            self._mode = "simple_fallback"
            self._index = None
            return {"mode": "simple_fallback", "docs": len(docs), "note": "llama_index not installed"}

        try:
            os.makedirs(self.cfg.local_index_dir, exist_ok=True)
        except Exception:
            pass

        # 采用 LlamaIndex 默认向量存储（本地）
        documents = [Document(text=t, metadata={"doc_id": did}) for did, t in docs]
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=80)
        nodes = parser.get_nodes_from_documents(documents)
        self._index = VectorStoreIndex(nodes)
        self._mode = "local"
        return {"mode": "local", "docs": len(docs), "nodes": len(nodes)}

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        返回 RAG 检索结果：[{doc_id, text, rag_score}]
        - rag_score 统一归一化到 [0,1]（用于联合评分）
        """
        if self._index is None:
            # 极简 fallback：关键词匹配
            hits = []
            q = query.lower()
            for doc_id, text in self._docs_cache:
                score = 0.0
                for tok in re.split(r"\W+", q):
                    if tok and tok in text.lower():
                        score += 0.05
                if score > 0:
                    hits.append((doc_id, text, min(1.0, score)))
            hits.sort(key=lambda x: x[2], reverse=True)
            return [{"doc_id": h[0], "text": h[1], "rag_score": float(h[2])} for h in hits[:top_k]]

        qe = self._index.as_retriever(similarity_top_k=top_k)
        # retriever 返回 NodeWithScore
        nodes = qe.retrieve(query)
        results: List[Dict[str, Any]] = []
        for n in nodes:
            # score：不同后端范围不同，这里做一个稳健归一化（经验：cosine 近似 0~1）
            raw = float(getattr(n, "score", 0.0) or 0.0)
            norm = raw
            if raw < 0:
                norm = 0.0
            if raw > 1:
                # 可能是距离（越小越好）或其他量纲，做简单压缩
                norm = 1.0 / (1.0 + raw)
            meta = getattr(n.node, "metadata", {}) or {}
            results.append(
                {
                    "doc_id": meta.get("doc_id", "unknown"),
                    "text": n.node.get_content(),
                    "rag_score": float(max(0.0, min(1.0, norm))),
                }
            )
        return results

    @property
    def mode(self) -> str:
        return self._mode


# ============================================================
# 多跳问答：RAG -> KG -> Joint Scoring -> Guardrail -> LLM
# ============================================================

class MultiHopQA:
    def __init__(self, cfg: AppConfig, rag: RAGStore, graph: GraphStore, llm: Optional[Any] = None):
        self.cfg = cfg
        self.rag = rag
        self.graph = graph
        self.llm = llm or MockLLM()

    def answer(self, question: str) -> Dict[str, Any]:
        reasoning: List[str] = []
        start_ts = time.time()

        # Step 0: 实体抽取（case_id 优先；否则尝试 user_id/错误码）
        entity_prompt = (
            f"从以下问题中提取出客服案例编号（CASE_ID，如 C-2025-004）或用户ID（U1001）或错误码：'{question}'\n"
            "只返回名称，不要添加任何其他文字。"
        )
        entity = self.llm.complete(entity_prompt).text.strip()
        reasoning.append(f"步骤 0: 实体抽取 -> '{entity}'")

        # Step 1: RAG 检索（证据）
        rag_hits = self.rag.retrieve(query=question, top_k=self.cfg.rag_top_k)
        if rag_hits:
            reasoning.append(f"步骤 1: RAG 检索(top_k={self.cfg.rag_top_k}) 命中 {len(rag_hits)} 条，展示前 1 条摘要：")
            reasoning.append(f"   - doc_id={rag_hits[0]['doc_id']} rag_score={rag_hits[0]['rag_score']:.3f} text={rag_hits[0]['text'][:180]}...")
        else:
            reasoning.append("步骤 1: RAG 检索未命中（可能数据不足或向量库不可用），将进入降级路径。")

        # Step 2: KG 推理（多跳 Cypher）
        case_id = entity if re.match(r"^C-\d{4}-\d{3}$", entity) else self._infer_case_id_from_rag(rag_hits)
        kg_result = {"found": False, "rows": []}
        if case_id:
            kg_result = self.graph.query_case_owner(case_id=case_id)
            if kg_result.get("found"):
                reasoning.append(f"步骤 2: KG 查询 case_id='{case_id}' 命中 {len(kg_result['rows'])} 条。")
                if "cypher" in kg_result:
                    reasoning.append(f"   - Cypher: {kg_result['cypher']}")
                reasoning.append(f"   - KG Top1: {kg_result['rows'][0]}")
            else:
                reasoning.append(f"步骤 2: KG 查询 case_id='{case_id}' 未命中。")
        else:
            reasoning.append("步骤 2: 未能确定 case_id，KG 推理跳过。")

        # Step 3: 联合评分（RAG + KG）与一致性校验（防止错误传播）
        joint = self._joint_rank(rag_hits, kg_result)
        reasoning.append(
            "步骤 3: 联合评分（RAG 相似度 + KG 置信度 + 一致性校验）结果："
        )
        reasoning.append(f"   - joint_score={joint['joint_score']:.3f}, pass_threshold={joint['pass_threshold']}, consistency_hits={joint['consistency_hits']}")
        if not joint["pass_threshold"]:
            reasoning.append("   - Guardrail: 联合评分未达阈值，将输出保守回答并提示补充信息。")

        # Step 4: 生成最终回答（LLM 汇总；无 LLM 则模板化）
        rag_context = self._pack_rag_context(rag_hits)
        final_prompt = (
            "你是一个专业的客服运营/技术支持负责人。请根据以下信息回答用户问题，并保持可执行建议。\n"
            "--- 用户问题 ---\n"
            f"{question}\n\n"
            "--- 知识图谱查询结果 ---\n"
            f"{json.dumps(kg_result, ensure_ascii=False)}\n\n"
            "--- 相关文档信息 ---\n"
            f"{rag_context}\n\n"
            "--- 最终回答 ---\n"
        )

        if joint["pass_threshold"]:
            final_answer = self.llm.complete(final_prompt).text.strip()
        else:
            final_answer = self._conservative_answer(question, kg_result, rag_hits)

        elapsed = time.time() - start_ts
        reasoning.append(f"步骤 5: 完成，用时 {elapsed:.2f}s（RAG mode={self.rag.mode}）。")

        return {"final_answer": final_answer, "reasoning_path": reasoning}

    def _infer_case_id_from_rag(self, rag_hits: List[Dict[str, Any]]) -> Optional[str]:
        for h in rag_hits:
            m = re.search(r"(C-\d{4}-\d{3})", h.get("text", ""))
            if m:
                return m.group(1)
        return None

    def _pack_rag_context(self, rag_hits: List[Dict[str, Any]]) -> str:
        parts = []
        for h in rag_hits[: min(len(rag_hits), 3)]:
            parts.append(f"[{h['doc_id']}|score={h['rag_score']:.3f}]\n{h['text']}")
        return "\n\n".join(parts)

    def _joint_rank(self, rag_hits: List[Dict[str, Any]], kg_result: Dict[str, Any]) -> Dict[str, Any]:
        # KG 信度：来自 ASSIGNED_TO.confidence（0~1）；无则 0
        kg_conf = 0.0
        agent_name = None
        if kg_result.get("found") and kg_result.get("rows"):
            kg_conf = float(kg_result["rows"][0].get("kg_confidence") or 0.0)
            agent_name = kg_result["rows"][0].get("agent_name")

        # RAG 信度：取 Top1 score
        rag_conf = float(rag_hits[0]["rag_score"]) if rag_hits else 0.0

        # 一致性：RAG 文本中是否包含 case_id / agent_name / 产品线关键词等
        consistency_hits = 0
        joined_text = "\n".join([h["text"] for h in rag_hits])
        if kg_result.get("case_id") and kg_result["case_id"] in joined_text:
            consistency_hits += 1
        if agent_name and agent_name in joined_text:
            consistency_hits += 1

        # 联合评分：可调权重（工程常用：结构化证据优先）
        joint_score = 0.45 * rag_conf + 0.45 * kg_conf + 0.10 * min(1.0, consistency_hits / 2.0)

        pass_threshold = (joint_score >= self.cfg.joint_score_threshold) and (consistency_hits >= self.cfg.consistency_min_hits or kg_conf >= 0.85)
        return {
            "rag_conf": rag_conf,
            "kg_conf": kg_conf,
            "consistency_hits": consistency_hits,
            "joint_score": float(joint_score),
            "pass_threshold": bool(pass_threshold),
        }

    def _conservative_answer(self, question: str, kg_result: Dict[str, Any], rag_hits: List[Dict[str, Any]]) -> str:
        # 保守回答：明确“不确定”，并给出补充信息建议，避免错误传播
        hints = []
        if not rag_hits:
            hints.append("未从文档中检索到足够上下文（可能未建立向量索引或数据不包含该案例）。")
        if not kg_result.get("found"):
            hints.append("知识图谱中未查到该案例的指派/关系记录（可能 case_id 不存在或图谱未构建）。")

        return (
            f"问题：{question}\n\n"
            "当前无法在证据一致的前提下给出确定结论。\n"
            f"原因：{'；'.join(hints) if hints else '证据不足或存在冲突。'}\n\n"
            "建议你补充以下信息后再查询：\n"
            "1) 明确案例编号（例如 C-2025-004），或提供用户ID/订单号/错误码；\n"
            "2) 提供问题截图、日志片段或发生时间（便于检索命中）；\n"
            "3) 若系统环境允许，请先执行 --mode build 重新构建图谱与向量索引。\n"
        )


# ============================================================
# API（FastAPI）
# ============================================================

def create_app(qa: MultiHopQA) -> Any:
    if FastAPI is None:
        raise RuntimeError("fastapi 未安装，无法启动服务。请 pip install fastapi uvicorn")

    app = FastAPI(
        title="CS GraphRAG 多跳问答系统（客服场景）",
        description="融合 Milvus 文档检索 + Neo4j 图谱推理 + 可解释推理路径",
        version="1.0.0",
    )

    class QueryRequest(BaseModel):
        question: str = Field(..., example="C-2025-004 这个客诉由谁负责？为什么？")

    class QueryResponse(BaseModel):
        final_answer: str
        reasoning_path: List[str]

    @app.get("/")
    def root():
        return {"message": "OK. Use POST /api/query"}

    @app.post("/api/query", response_model=QueryResponse)
    def query(req: QueryRequest):
        if not req.question:
            raise HTTPException(status_code=400, detail="question 不能为空")
        return qa.answer(req.question)

    return app


# ============================================================
# 主流程
# ============================================================

def load_data(cfg: AppConfig) -> Tuple[List[ComplaintCase], List[AgentProfile]]:
    ensure_data_files(cfg)
    with open(cfg.complaints_path, "r", encoding="utf-8") as f:
        ctext = f.read()
    with open(cfg.agents_path, "r", encoding="utf-8") as f:
        atext = f.read()
    cases = parse_complaints(ctext)
    agents = parse_agents(atext)
    return cases, agents


def build_all(cfg: AppConfig) -> Dict[str, Any]:
    cases, agents = load_data(cfg)

    # Graph
    graph = GraphStore(cfg)
    graph.connect()
    graph_stat = graph.build_graph(cases, agents)

    # RAG
    rag = RAGStore(cfg)
    rag_stat = rag.build_index(cases, agents)

    graph.close()

    return {
        "graph": graph_stat,
        "rag": rag_stat,
        "cases": len(cases),
        "agents": len(agents),
    }


def serve(cfg: AppConfig) -> None:
    cases, agents = load_data(cfg)

    graph = GraphStore(cfg)
    graph.connect()
    graph.build_graph(cases, agents)  # 确保图谱存在（可按需改为只读）

    rag = RAGStore(cfg)
    rag.build_index(cases, agents)

    qa = MultiHopQA(cfg, rag, graph, llm=MockLLM())
    app = create_app(qa)

    uvicorn.run(app, host="0.0.0.0", port=8000)


def cli(cfg: AppConfig, question: str) -> None:
    cases, agents = load_data(cfg)

    graph = GraphStore(cfg)
    graph.connect()
    graph.build_graph(cases, agents)

    rag = RAGStore(cfg)
    rag.build_index(cases, agents)

    qa = MultiHopQA(cfg, rag, graph, llm=MockLLM())
    res = qa.answer(question)
    print(json.dumps(res, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build", "serve", "cli"], required=True)
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = AppConfig(verbose=not args.quiet)

    if args.mode == "build":
        stat = build_all(cfg)
        print(json.dumps(stat, ensure_ascii=False, indent=2))
    elif args.mode == "serve":
        serve(cfg)
    elif args.mode == "cli":
        if not args.question:
            raise SystemExit("--mode cli 必须提供 --question")
        cli(cfg, args.question)


if __name__ == "__main__":
    main()