# 运行方式：
# 1) 启动 MCP 服务器（终端 A）：
#    python main.py server --host 127.0.0.1 --port 8000
#
# 2) 启动 LangGraph 多代理写作客户端（终端 B）：
#    python main.py client --topic "写一篇介绍RAG理论与研究前沿的文章" --style "通俗但专业" --length 1400
#
# 3) 输出会在当前目录生成：
#    article_output_YYYYMMDD_HHMMSS.md
#
# 依赖：
#   pip install fastmcp duckduckgo-search langgraph langchain langchain-community langchain-mcp-adapters python-dotenv
#
#
# 说明：
# - 本文件同时包含：MCP 工具服务器 + LangGraph 多代理工作流 + 三级重试策略
# - 搜索工具使用 DuckDuckGo（duckduckgo-search）；网络不可用时会触发重试并降级

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Tuple

# -----------------------------
# Optional dependencies import
# -----------------------------
_FASTMCP_OK = True
_DDGS_OK = True
_LANGGRAPH_OK = True
_MCP_ADAPTER_OK = True
_TONGYI_OK = True

try:
    from fastmcp import FastMCP
except Exception:
    _FASTMCP_OK = False

try:
    from duckduckgo_search import DDGS
except Exception:
    _DDGS_OK = False

try:
    from langgraph.graph import StateGraph, END
except Exception:
    _LANGGRAPH_OK = False

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
except Exception:
    _MCP_ADAPTER_OK = False

try:
    # 通义千问（示例与助教样例一致）；若不可用则走 Mock
    from langchain_community.chat_models import ChatTongyi
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    _TONGYI_OK = False

try:
    from dotenv import load_dotenv
except Exception:
    # 不强依赖 dotenv
    load_dotenv = None


# -----------------------------
# Prompts
# -----------------------------
RESEARCH_PROMPT = """
你是一个AI研究员。你的任务是根据给定的主题，使用搜索工具收集相关信息，并输出一份结构化的研究资料（Markdown）。

输出必须包含：
1) 核心概念：关键术语与定义（用简洁但准确的表述）
2) 关键技术/机制：列出 5-8 个关键点，每个点 2-4 句说明
3) 代表性论文/系统：按年份列出 6-10 条（标题、作者/机构、贡献点）
4) 工程实践要点：数据、检索、重排、生成、评估、监控等至少 6 条建议
5) 风险与误区：至少 5 条（如幻觉、过拟合检索、评估偏差、隐私合规等）
6) 参考链接：给出来源 URL 列表（可从搜索结果中提取）

约束：
- 研究资料要“可复用、可引用、可下游写作”，避免空泛口号
- 对于不确定的结论要显式标注“可能/通常/在部分工作中”
"""

WRITING_PROMPT = """
你是一位专业的AI科技文章撰稿人。根据研究资料撰写文章初稿（Markdown）。

要求：
- 文章结构：引言 → 原理与范式 → 关键技术栈 → 评估与落地 → 前沿方向 → 结语
- 语言：面向工程读者，“通俗但专业”，避免学术腔堆砌
- 风格：{style}
- 长度：约 {length} 字
- 必须包含：至少一个对比表（例如：RAG vs Fine-tuning vs Tool-use），至少一段“落地 checklist”
"""

REVIEW_PROMPT = """
你是一位经验丰富的技术编辑与审稿人。请审查文章初稿（Markdown），输出“问题清单 + 修改建议”。

至少覆盖：
- 事实准确性与表述风险（是否存在过度断言）
- 结构与逻辑（是否有跳跃、冗余、缺关键段）
- 工程可落地性（评估指标、监控、数据治理是否具体）
- 语言与一致性（术语统一、段落衔接）
- 引用与来源（是否缺关键参考）

输出格式（必须）：
1) 总体评价（2-4 句）
2) 高优先级问题（P0）列表（不少于 5 条）
3) 中优先级问题（P1）列表（不少于 5 条）
4) 低优先级建议（P2）列表（不少于 3 条）
"""

POLISHING_PROMPT = """
你是一位顶级技术写作润色专家。请结合“文章初稿 + 审核建议”，生成最终终稿（Markdown）。

要求：
- 采纳所有合理的 P0/P1 建议
- 保持结构清晰、语言一致
- 对不确定结论加限定语或补引用
- 保留并优化：对比表、checklist、前沿方向
- 不要输出额外解释，只输出最终文章正文
"""

# 备用审核/润色代理（用于二级重试）
SENIOR_REVIEW_PROMPT = """
你是“高级技术审稿人”，标准比普通审核更严格。请对文章初稿进行更细致的审阅与风险控制，尤其关注：
- 是否存在事实性错误或过度营销式表述
- 概念边界是否清晰（RAG、Agent、Tool-use、Memory、Fine-tuning）
- 评估体系是否完整（离线、在线、人工、自动化、对抗测试）
- 安全与合规（隐私、版权、提示注入、数据外泄）

输出格式同普通审核，但 P0 必须 >= 7 条，且每条给出“建议改法”。
"""

SENIOR_POLISH_PROMPT = """
你是“高级技术写作总编”。请对终稿进行最后把关：
- 让文章读起来更像“工程团队内部技术白皮书 + 对外科普”融合体
- 进一步压缩冗余，提升信息密度
- 增强小标题的可扫描性（scannability）
- 补齐必要的 caveats（限定条件）

只输出终稿正文（Markdown）。
"""

PROMPTS: Dict[str, str] = {
    "research": RESEARCH_PROMPT,
    "write": WRITING_PROMPT,
    "review": REVIEW_PROMPT,
    "polish": POLISHING_PROMPT,
    "review_senior": SENIOR_REVIEW_PROMPT,
    "polish_senior": SENIOR_POLISH_PROMPT,
}


# -----------------------------
# State definition
# -----------------------------
class AgentState(TypedDict, total=False):
    topic: str
    style: str
    length: int

    research_report: str
    draft: str
    review_suggestions: str
    final_article: str

    # process observability
    log: List[str]                  # human-readable process log
    exception_log: List[str]        # structured exception/retry log

    # retry control
    retry_counts: Dict[str, int]    # per-agent retries (same agent)
    used_fallback: Dict[str, bool]  # whether fallback agent has been used

    # user clarification channel (for L3 retry)
    user_clarifications: Dict[str, str]


# -----------------------------
# Utilities: logging & retry
# -----------------------------
def _now_ts() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(state: AgentState, header: str, body_md: str) -> None:
    state.setdefault("log", [])
    state["log"].append(f"## {header}\n\n{body_md}\n")


def append_exception(state: AgentState, agent: str, level: str, msg: str, detail: Optional[str] = None) -> None:
    state.setdefault("exception_log", [])
    payload = {
        "time": _now_ts(),
        "agent": agent,
        "retry_level": level,   # L1 / L2 / L3
        "message": msg,
    }
    if detail:
        payload["detail"] = detail
    state["exception_log"].append(json.dumps(payload, ensure_ascii=False))


def should_retry_same_agent(state: AgentState, agent: str, max_times: int = 2) -> bool:
    state.setdefault("retry_counts", {})
    cnt = state["retry_counts"].get(agent, 0)
    return cnt < max_times


def mark_retry_same_agent(state: AgentState, agent: str) -> None:
    state.setdefault("retry_counts", {})
    state["retry_counts"][agent] = state["retry_counts"].get(agent, 0) + 1


def can_use_fallback(state: AgentState, agent: str) -> bool:
    state.setdefault("used_fallback", {})
    return not state["used_fallback"].get(agent, False)


def mark_used_fallback(state: AgentState, agent: str) -> None:
    state.setdefault("used_fallback", {})
    state["used_fallback"][agent] = True


def require_user_clarification(state: AgentState, agent: str, question: str) -> str:
    """
    L3：向用户请求补充信息。
    - 为了作业“可自动跑通”，这里支持两种模式：
      1) 交互式：stdin input
      2) 非交互：使用默认回答（并记录）
    """
    state.setdefault("user_clarifications", {})
    if sys.stdin is None or not sys.stdin.isatty():
        # non-interactive fallback
        answer = "默认：无需补充；按常规假设执行（面向工程读者，强调评估与安全，引用尽量给出链接）。"
        state["user_clarifications"][agent] = answer
        append_exception(state, agent, "L3", "Non-interactive mode: use default clarification", answer)
        return answer

    print("\n" + "=" * 70)
    print(f"⚠️ 需要用户补充信息（代理：{agent}）")
    print(question)
    print("=" * 70)
    answer = input("你的补充信息（可直接回车使用默认）： ").strip()
    if not answer:
        answer = "默认：无需补充；按常规假设执行（面向工程读者，强调评估与安全，引用尽量给出链接）。"
    state["user_clarifications"][agent] = answer
    append_exception(state, agent, "L3", "User clarification captured", answer)
    return answer


# -----------------------------
# LLM abstraction
# -----------------------------
class BaseLLM:
    async def ainvoke(self, system_prompt: str, user_content: str) -> str:
        raise NotImplementedError
    
class MockLLM(BaseLLM):
    pass


class TongyiLLM(BaseLLM):
    def __init__(self, model: str = "qwen-plus"):
        self.model = model
        self._llm = ChatTongyi(model=model)

    async def ainvoke(self, system_prompt: str, user_content: str) -> str:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
        resp = await self._llm.ainvoke(messages)
        return getattr(resp, "content", str(resp))


def build_llm() -> BaseLLM:
    # dotenv load
    if load_dotenv is not None:
        load_dotenv()

    # 若可用 Tongyi 且提供 key，则使用；否则 Mock
    key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASH_SCOPE_API_KEY")
    if _TONGYI_OK and key:
        return TongyiLLM(model=os.getenv("TONGYI_MODEL", "qwen-plus"))
    return MockLLM()


# -----------------------------
# MCP server (tools)
# -----------------------------
def create_mcp_server() -> "FastMCP":
    if not _FASTMCP_OK:
        raise RuntimeError("fastmcp not installed. Please pip install fastmcp")

    mcp = FastMCP("MCP Writer Tools (Single File)")

    @mcp.tool
    def get_prompt(agent_name: str) -> str:
        """根据代理名称获取对应系统提示词。"""
        print(f"MCP Server: 📄 get_prompt('{agent_name}')")
        return PROMPTS.get(agent_name, "Error: Prompt not found.")

    @mcp.tool
    def search(topic: str, max_results: int = 6) -> str:
        """DuckDuckGo 搜索并返回 JSON 文本（每项通常包含 title/href/body）。"""
        print(f"MCP Server: 🔍 search('{topic}', max_results={max_results})")
        if not _DDGS_OK:
            return json.dumps({"error": "duckduckgo-search not installed"}, ensure_ascii=False)

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(topic, max_results=max_results))
            return json.dumps(results, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    return mcp


def run_server(host: str, port: int) -> None:
    mcp = create_mcp_server()
    print(f"🚀 MCP Server is running at http://{host}:{port}/mcp")
    # streamable-http 更贴近“微服务工具服务器”
    mcp.run(transport="streamable-http", host=host, port=port)


# -----------------------------
# Agent nodes (LangGraph)
# -----------------------------
@dataclass
class AgentNodes:
    mcp_tools: Dict[str, Any]
    llm: BaseLLM

    async def _call_tool(self, name: str, **kwargs) -> str:
        if name not in self.mcp_tools:
            raise ValueError(f"Tool '{name}' not found on MCP server.")
        tool = self.mcp_tools[name]
        out = await tool.ainvoke(kwargs)
        # MCP adapter 可能返回 dict / str，这里统一成 str
        if isinstance(out, (dict, list)):
            return json.dumps(out, ensure_ascii=False)
        return str(out)

    async def research_node(self, state: AgentState) -> AgentState:
        agent = "researcher"
        print("--- 节点: 研究代理 (Research Agent) ---")

        prompt = await self._call_tool("get_prompt", agent_name="research")
        # 搜索：对 topic 做一次“学术+工程”扩展
        topic = state.get("topic", "")
        search_query = f"{topic} RAG retrieval augmented generation evaluation reranking prompt injection arxiv"
        raw_search = await self._call_tool("search", topic=search_query, max_results=8)

        user_content = f"主题：{topic}\n\n搜索结果(JSON)：\n{raw_search}\n"
        report = await self.llm.ainvoke(prompt, user_content)

        state["research_report"] = report
        append_log(state, "研究代理输出（Research Report）", report)
        print("✅ 研究资料生成完毕。")
        return state

    async def writing_node(self, state: AgentState) -> AgentState:
        agent = "writer"
        print("--- 节点: 撰写代理 (Writing Agent) ---")

        prompt_tpl = await self._call_tool("get_prompt", agent_name="write")
        prompt = prompt_tpl.format(style=state.get("style", "通俗但专业"), length=state.get("length", 1200))

        clar = state.get("user_clarifications", {}).get(agent)
        extra = f"\n\n用户补充信息：{clar}\n" if clar else ""
        user_content = f"{state.get('research_report','')}\n{extra}"

        draft = await self.llm.ainvoke(prompt, user_content)

        state["draft"] = draft
        append_log(state, "撰写代理输出（Draft）", draft)
        print("✅ 初稿完成。")
        return state

    async def review_node(self, state: AgentState) -> AgentState:
        agent = "reviewer"
        print("--- 节点: 审核代理 (Review Agent) ---")

        # 普通审核 prompt
        prompt = await self._call_tool("get_prompt", agent_name="review")

        clar = state.get("user_clarifications", {}).get(agent)
        extra = f"\n\n用户补充信息：{clar}\n" if clar else ""
        user_content = f"{state.get('draft','')}\n{extra}"

        suggestions = await self.llm.ainvoke(prompt, user_content)

        state["review_suggestions"] = suggestions
        append_log(state, "审核代理输出（Review Suggestions）", suggestions)
        print("✅ 审核完成。")
        return state

    async def polishing_node(self, state: AgentState) -> AgentState:
        agent = "polisher"
        print("--- 节点: 润色代理 (Polishing Agent) ---")

        prompt = await self._call_tool("get_prompt", agent_name="polish")

        clar = state.get("user_clarifications", {}).get(agent)
        extra = f"\n\n用户补充信息：{clar}\n" if clar else ""
        user_content = (
            f"文章初稿：\n\n{state.get('draft','')}\n\n"
            f"审核建议：\n\n{state.get('review_suggestions','')}\n"
            f"{extra}"
        )

        final_article = await self.llm.ainvoke(prompt, user_content)

        state["final_article"] = final_article
        append_log(state, "润色代理输出（Final Article）", final_article)
        print("✅ 终稿生成完成。")
        return state


# -----------------------------
# Retry wrapper for nodes
# -----------------------------
async def run_with_retry(
    state: AgentState,
    agent_name: str,
    primary_fn,
    fallback_fn=None,
    clarification_question: Optional[str] = None,
) -> AgentState:
    """
    三级重试策略（扩展项）：
      L1：同一代理重试（最多 2 次）
      L2：切换备用代理（一次）
      L3：向用户请求补充信息（一次） -> 再执行主代理（一次）
    """
    # L1
    while True:
        try:
            return await primary_fn(state)
        except Exception as e:
            detail = repr(e)
            if should_retry_same_agent(state, agent_name, max_times=2):
                mark_retry_same_agent(state, agent_name)
                append_exception(state, agent_name, "L1", "Retry same agent", detail)
                print(f"⚠️ {agent_name} 失败，L1 重试中... ({state['retry_counts'][agent_name]}/2)")
                continue
            break

    # L2
    if fallback_fn is not None and can_use_fallback(state, agent_name):
        try:
            mark_used_fallback(state, agent_name)
            append_exception(state, agent_name, "L2", "Switch to fallback agent", "fallback invoked")
            print(f"⚠️ {agent_name} 失败，切换到备用代理（L2）...")
            return await fallback_fn(state)
        except Exception as e:
            append_exception(state, agent_name, "L2", "Fallback failed", repr(e))

    # L3
    if clarification_question:
        _ = require_user_clarification(state, agent_name, clarification_question)
        try:
            append_exception(state, agent_name, "L3", "Re-run after clarification", "re-run primary")
            print(f"⚠️ {agent_name} 进入 L3：已获取补充信息，重新执行主代理...")
            return await primary_fn(state)
        except Exception as e:
            append_exception(state, agent_name, "L3", "Failed after clarification", repr(e))

    # 无法恢复：保留状态并抛出
    raise RuntimeError(f"Agent '{agent_name}' failed after retry strategy. See exception_log.")


# -----------------------------
# Graph construction
# -----------------------------
async def create_graph(mcp_session) -> Any:
    if not (_LANGGRAPH_OK and _MCP_ADAPTER_OK):
        raise RuntimeError("Missing langgraph or langchain-mcp-adapters dependencies.")

    mcp_tools_list = await load_mcp_tools(mcp_session)
    mcp_tools = {t.name: t for t in mcp_tools_list}

    llm = build_llm()
    nodes = AgentNodes(mcp_tools=mcp_tools, llm=llm)

    # fallback agents: use senior prompts by calling get_prompt with review_senior / polish_senior
    async def review_fallback(state: AgentState) -> AgentState:
        agent = "reviewer"
        print("--- 节点: 备用审核代理 (Senior Review) ---")
        prompt = await nodes._call_tool("get_prompt", agent_name="review_senior")
        user_content = state.get("draft", "")
        suggestions = await llm.ainvoke(prompt, user_content)
        state["review_suggestions"] = suggestions
        append_log(state, "备用审核代理输出（Senior Review Suggestions）", suggestions)
        return state

    async def polish_fallback(state: AgentState) -> AgentState:
        agent = "polisher"
        print("--- 节点: 备用润色代理 (Senior Polish) ---")
        prompt = await nodes._call_tool("get_prompt", agent_name="polish_senior")
        user_content = (
            f"文章初稿：\n\n{state.get('draft','')}\n\n"
            f"审核建议：\n\n{state.get('review_suggestions','')}\n"
        )
        final_article = await llm.ainvoke(prompt, user_content)
        state["final_article"] = final_article
        append_log(state, "备用润色代理输出（Senior Final Article）", final_article)
        return state

    workflow = StateGraph(AgentState)

    workflow.add_node(
        "researcher",
        lambda s: run_with_retry(
            s,
            "researcher",
            nodes.research_node,
            fallback_fn=None,
            clarification_question="请补充：文章读者是谁（工程/科研/产品）？是否需要聚焦某个子方向（评估/安全/系统架构）？",
        ),
    )
    workflow.add_node(
        "writer",
        lambda s: run_with_retry(
            s,
            "writer",
            nodes.writing_node,
            fallback_fn=None,
            clarification_question="请补充：文章风格与重点（更学术/更工程/更面向业务）？是否需要案例（医疗/金融/客服/代码检索）？",
        ),
    )
    workflow.add_node(
        "reviewer",
        lambda s: run_with_retry(
            s,
            "reviewer",
            nodes.review_node,
            fallback_fn=review_fallback,
            clarification_question="请补充：你更在意哪类问题（事实正确性/工程落地/写作表达/引用完整性）？",
        ),
    )
    workflow.add_node(
        "polisher",
        lambda s: run_with_retry(
            s,
            "polisher",
            nodes.polishing_node,
            fallback_fn=polish_fallback,
            clarification_question="请补充：是否需要更短/更长？是否需要加入小结、要点列表或更强的结语建议？",
        ),
    )

    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "reviewer")
    workflow.add_edge("reviewer", "polisher")
    workflow.add_edge("polisher", END)

    return workflow.compile()


# -----------------------------
# Client runner
# -----------------------------
def build_initial_state(topic: str, style: str, length: int) -> AgentState:
    return AgentState(
        topic=topic,
        style=style,
        length=length,
        log=[f"# 多代理协作写作流程记录\n\n**任务主题：** {topic}\n\n**风格：** {style}\n\n**目标长度：** {length}\n"],
        exception_log=[],
        retry_counts={},
        used_fallback={},
        user_clarifications={},
    )


def render_output_markdown(final_state: AgentState) -> str:
    topic = final_state.get("topic", "未命名主题")
    final_article = final_state.get("final_article", "未能生成最终文章。")
    process_log = "\n".join(final_state.get("log", []))

    # 异常处理日志（扩展项）
    ex_lines: List[str] = []
    for line in final_state.get("exception_log", []):
        try:
            obj = json.loads(line)
            ex_lines.append(
                f"- {obj.get('time')} | agent={obj.get('agent')} | level={obj.get('retry_level')} | {obj.get('message')}"
                + (f" | detail={obj.get('detail')}" if obj.get("detail") else "")
            )
        except Exception:
            ex_lines.append(f"- {line}")

    exception_md = "无。\n" if not ex_lines else "\n".join(ex_lines) + "\n"

    return (
        f"# 最终文章：{topic}\n\n"
        f"{final_article}\n\n"
        f"---\n\n"
        f"# 执行过程记录\n\n"
        f"{process_log}\n\n"
        f"---\n\n"
        f"# 异常处理日志\n\n"
        f"{exception_md}"
    )


async def run_client(
    server_url: str,
    topic: str,
    style: str,
    length: int,
) -> str:
    if not _MCP_ADAPTER_OK:
        raise RuntimeError("langchain-mcp-adapters not installed.")
    if not _LANGGRAPH_OK:
        raise RuntimeError("langgraph not installed.")

    client = MultiServerMCPClient(
        {
            "tools_server": {
                "url": server_url,
                "transport": "streamable_http",
            }
        }
    )

    async with client.session("tools_server") as mcp_session:
        print("✅ MCP 客户端已连接到工具服务器。")
        app = await create_graph(mcp_session)

        print("\n" + "=" * 70)
        print("🚀 LangGraph 多代理工作流启动")
        print("=" * 70 + "\n")

        state = build_initial_state(topic=topic, style=style, length=length)
        final_state = await app.ainvoke(state)

        md = render_output_markdown(final_state)
        return md


def write_output_file(md: str) -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"article_output_{ts}.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    return out


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="multi-agent-mcp", add_help=True)

    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("server", help="Run MCP tool server")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)

    cp = sub.add_parser("client", help="Run LangGraph multi-agent client")
    cp.add_argument("--server-url", default="http://127.0.0.1:8000/mcp")
    cp.add_argument("--topic", default="写一篇介绍RAG的理论与研究前沿的文章")
    cp.add_argument("--style", default="通俗但专业")
    cp.add_argument("--length", type=int, default=1400)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.cmd == "server":
        if not _FASTMCP_OK:
            print("ERROR: fastmcp not installed. pip install fastmcp")
            sys.exit(2)
        run_server(args.host, args.port)
        return

    if args.cmd == "client":
        try:
            md = asyncio.run(run_client(args.server_url, args.topic, args.style, args.length))
            out = write_output_file(md)
            print("\n" + "=" * 70)
            print("✅ 任务完成")
            print(f"🎉 输出文件：{out}")
            print("=" * 70 + "\n")
        except KeyboardInterrupt:
            print("\n程序已由用户中断。")
        except Exception as e:
            print("\n发生错误：", repr(e))
            print("\n排查建议：")
            print("1) 确保已启动 MCP Server：python main.py server")
            print("2) 确保依赖已安装：fastmcp duckduckgo-search langgraph langchain langchain-mcp-adapters")
            print("3) 若需真实 LLM：设置 DASHSCOPE_API_KEY；否则会使用 MockLLM（仍可运行）")
            sys.exit(1)
        return


if __name__ == "__main__":
    main()
