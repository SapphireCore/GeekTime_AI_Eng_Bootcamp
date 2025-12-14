 # main.py
"""
Week 4 Homework - Smart Customer Service (Single-file + minimal plugins)
- Multi-turn dialogue (order query / refund / invoice)
- Tool calling (deterministic "auto tool call" via LangGraph nodes)
- Hot update:
  1) Model hot update (switch LLM backend; supports mock)
  2) Plugin hot reload (invoice plugin v1/v2 via importlib)
  3) Ensure old sessions not impacted: session pins to a "service version"
- Production endpoints:
  /health
  /chat
  /hot-update
- Automated tests:
  python main.py --run-tests
- Run API:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain / LangGraph
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver


# -----------------------------
# 0) Utilities
# -----------------------------

def _now_local() -> datetime:
    """
    Homework requirement mentions: infer "yesterday" based on current time.
    In real service, use timezone-aware datetime; here keep local naive datetime.
    """
    return datetime.now()


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def _normalize_text(s: str) -> str:
    return (s or "").strip()


def _extract_order_id(text: str) -> Optional[str]:
    """
    Extract common order_id patterns (e.g., SN20240924001).
    """
    if not text:
        return None
    m = re.search(r"\bSN\d{8,}\b", text.upper())
    return m.group(0) if m else None


def _looks_like_relative_time(text: str) -> bool:
    return any(k in (text or "") for k in ["昨天", "前天", "今天", "上周", "本周", "上个月", "上月"])


def _detect_intent(text: str) -> Literal["order_query", "refund", "invoice", "chitchat"]:
    t = text or ""
    if any(k in t for k in ["查订单", "查询订单", "订单状态", "物流"]):
        return "order_query"
    if any(k in t for k in ["退款", "退货", "申请退款"]):
        return "refund"
    if any(k in t for k in ["发票", "开票", "开具发票", "invoice"]):
        return "invoice"
    return "chitchat"


# -----------------------------
# 1) Tools (baseline)
# -----------------------------

@tool
def query_order(order_id: str) -> dict:
    """
    Query order status & logistics by order_id (mock DB).
    """
    print(f"--- [Tool] query_order: order_id={order_id} ---")
    time.sleep(0.2)  # keep quick for homework/demo

    mock_db = {
        "SN20240924001": {"status": "已发货", "tracking_number": "SF123456789", "items": ["LangChain入门实战T恤"]},
        "SN20240925001": {"status": "已发货", "tracking_number": "SF987654321", "items": ["AI Agent开发者马克杯"]},
        "SN20240924002": {"status": "待支付", "tracking_number": None, "items": ["LangGraph高级教程贴纸"]},
        "SN20240924003": {"status": "已完成", "tracking_number": "JD987654321", "items": ["AI Agent开发者马克杯"]},
    }
    order_info = mock_db.get(order_id)
    if order_info:
        return {
            "success": True,
            "order_id": order_id,
            "status": order_info["status"],
            "tracking_number": order_info["tracking_number"],
            "details": f"订单中的商品: {', '.join(order_info['items'])}",
        }
    return {"success": False, "order_id": order_id, "error": "未找到该订单，请检查订单号是否正确。"}


@tool
def apply_refund(order_id: str, reason: str) -> dict:
    """
    Apply refund for an order. Mocked acceptance by basic validation.
    """
    print(f"--- [Tool] apply_refund: order_id={order_id}, reason={reason} ---")
    time.sleep(0.2)

    if order_id and order_id.upper().startswith("SN"):
        refund_id = f"REFUND_{uuid.uuid4().hex[:8].upper()}"
        return {"success": True, "order_id": order_id, "refund_id": refund_id, "message": "退款申请已提交，审核通过后将原路退回。"}
    return {"success": False, "order_id": order_id, "error": "无效的订单号，无法申请退款。"}


@tool
def get_date_for_relative_time(relative_time_str: str) -> str:
    """
    Convert relative time text to YYYY-MM-DD.
    Production: implement robust NLU; here implement the required minimum set.
    """
    print(f"--- [Tool] get_date_for_relative_time: text={relative_time_str} ---")
    today = _now_local().date()
    s = (relative_time_str or "").lower()

    if "昨天" in s:
        target = today - timedelta(days=1)
        return target.strftime("%Y-%m-%d")
    if "前天" in s:
        target = today - timedelta(days=2)
        return target.strftime("%Y-%m-%d")
    if "今天" in s:
        return today.strftime("%Y-%m-%d")

    # Minimal "上周X"
    if "上周" in s:
        weekday_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
        target_weekday = None
        for ch, idx in weekday_map.items():
            if ch in s:
                target_weekday = idx
                break
        if target_weekday is None:
            return "无法识别的星期信息。"

        today_wd = today.weekday()  # Mon=0
        # Move to last week's target weekday
        days_ago = (today_wd - target_weekday + 7) % 7 + 7
        target = today - timedelta(days=days_ago)
        return target.strftime("%Y-%m-%d")

    return "无法解析该相对时间，请使用更明确的描述。"


# -----------------------------
# 2) Plugin system (invoice hot-reload)
# -----------------------------

class PluginManager:
    """
    Minimal plugin loader for invoice generation.
    Supports:
      - load plugin module by name under tools/
      - importlib.reload for hot reload
    """

    def __init__(self, tools_package: str = "tools"):
        self.tools_package = tools_package
        self._loaded_modules: Dict[str, Any] = {}

    def ensure_tools_on_path(self):
        """
        Ensure current working directory is on sys.path so `tools.*` imports work.
        """
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)

    def load_invoice_tool(self, plugin_name: str) -> Any:
        """
        plugin_name: "invoice_plugin_v1" or "invoice_plugin_v2"
        Must export function `generate_invoice_tool` decorated by @tool.
        """
        self.ensure_tools_on_path()
        module_path = f"{self.tools_package}.{plugin_name}"
        try:
            if module_path in self._loaded_modules:
                mod = importlib.reload(self._loaded_modules[module_path])
            else:
                mod = importlib.import_module(module_path)
                self._loaded_modules[module_path] = mod

            if not hasattr(mod, "generate_invoice_tool"):
                raise AttributeError("Plugin must define generate_invoice_tool")
            return getattr(mod, "generate_invoice_tool")
        except Exception as e:
            raise RuntimeError(f"Failed to load invoice plugin {module_path}: {e}") from e


# -----------------------------
# 3) "Model" abstraction (real or mock)
# -----------------------------

class BaseLLM:
    """
    Minimal interface:
      - generate(messages) -> AIMessage
    """

    def generate(self, messages: Sequence[BaseMessage]) -> AIMessage:
        raise NotImplementedError


class MockLLM(BaseLLM):
    """
    Deterministic assistant response for chitchat / fallback.
    Tool calling is handled by LangGraph controller, not by LLM tool_calls.
    """

    def __init__(self, model_name: str = "mock-llm"):
        self.model_name = model_name

    def generate(self, messages: Sequence[BaseMessage]) -> AIMessage:
        last_user = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_user = m.content
                break
        # Keep concise; do not hallucinate tool results
        if _looks_like_relative_time(last_user):
            return AIMessage(content="我可以帮你把相对时间转换为具体日期，并继续协助查询订单/退款/开票。请告诉我你想做什么？")
        return AIMessage(content="我可以帮你：查订单、申请退款、开具发票。你想先做哪一项？")


# Optional: if you want to integrate a real model later, plug in here.
# For homework deliverable, mock is acceptable when API not available.
def create_llm(model_name: str) -> BaseLLM:
    return MockLLM(model_name=model_name)


# -----------------------------
# 4) Service versioning & session pinning
# -----------------------------

@dataclass(frozen=True)
class ServiceSnapshot:
    version: int
    model_name: str
    invoice_plugin: str
    tools: Tuple[Any, ...]  # langchain tools objects
    llm: BaseLLM


class ServiceManager:
    """
    Global service manager (current "active" service).
    Hot updates create a new version; sessions pin to a snapshot version.
    """

    def __init__(self, plugin_manager: PluginManager):
        self.plugin_manager = plugin_manager
        self._version = 1
        self._model_name = "mock-llm"
        self._invoice_plugin = "invoice_plugin_v1"
        self._snapshot_cache: Dict[int, ServiceSnapshot] = {}
        self._rebuild_current_snapshot()

    def _rebuild_current_snapshot(self):
        invoice_tool = self.plugin_manager.load_invoice_tool(self._invoice_plugin)

        tools = (
            query_order,
            apply_refund,
            get_date_for_relative_time,
            invoice_tool,
        )
        llm = create_llm(self._model_name)

        snapshot = ServiceSnapshot(
            version=self._version,
            model_name=self._model_name,
            invoice_plugin=self._invoice_plugin,
            tools=tools,
            llm=llm,
        )
        self._snapshot_cache[self._version] = snapshot

    def get_current_snapshot(self) -> ServiceSnapshot:
        return self._snapshot_cache[self._version]

    def get_snapshot(self, version: int) -> ServiceSnapshot:
        if version not in self._snapshot_cache:
            raise KeyError(f"Unknown service version: {version}")
        return self._snapshot_cache[version]

    def hot_update_model(self, new_model_name: str):
        self._version += 1
        self._model_name = new_model_name
        self._rebuild_current_snapshot()

    def hot_update_invoice_plugin(self, plugin_name: str):
        self._version += 1
        self._invoice_plugin = plugin_name
        self._rebuild_current_snapshot()

    def health(self) -> dict:
        cur = self.get_current_snapshot()
        return {
            "current_version": cur.version,
            "model": cur.model_name,
            "invoice_plugin": cur.invoice_plugin,
            "tools": [t.name for t in cur.tools],
            "known_versions": sorted(self._snapshot_cache.keys()),
        }


@dataclass
class SessionContext:
    thread_id: str
    service_version: int
    graph_app: Any  # compiled LangGraph app
    created_at: float


class SessionRegistry:
    """
    Stores session contexts. Session pins to service version at creation time.
    """

    def __init__(self):
        self._sessions: Dict[str, SessionContext] = {}

    def get_or_create(self, thread_id: str, service_version: int, graph_app_factory) -> SessionContext:
        if thread_id in self._sessions:
            return self._sessions[thread_id]
        app = graph_app_factory(service_version)
        ctx = SessionContext(thread_id=thread_id, service_version=service_version, graph_app=app, created_at=time.time())
        self._sessions[thread_id] = ctx
        return ctx

    def get(self, thread_id: str) -> Optional[SessionContext]:
        return self._sessions.get(thread_id)

    def stats(self) -> dict:
        return {
            "active_sessions": len(self._sessions),
            "sessions": [
                {"thread_id": k, "service_version": v.service_version, "age_sec": int(time.time() - v.created_at)}
                for k, v in self._sessions.items()
            ],
        }


# -----------------------------
# 5) LangGraph: multi-turn + tool calling controller
# -----------------------------

class AgentState(Dict[str, Any]):
    """
    State schema:
      - messages: List[BaseMessage]
      - slots: dict (intent/order_id/refund_reason/relative_time)
      - tool_result: dict|str
    """


class GraphFactory:
    """
    Builds a LangGraph app for a given service snapshot version.
    Uses MemorySaver to keep conversation state per thread_id.
    """

    def __init__(self, service_manager: ServiceManager):
        self.service_manager = service_manager
        self._app_cache: Dict[int, Any] = {}

    def get_app(self, service_version: int) -> Any:
        if service_version in self._app_cache:
            return self._app_cache[service_version]
        app = self._build(service_version)
        self._app_cache[service_version] = app
        return app

    def _build(self, service_version: int) -> Any:
        snapshot = self.service_manager.get_snapshot(service_version)

        workflow = StateGraph(AgentState)

        workflow.add_node("router", lambda s: self._router(s))
        workflow.add_node("ask_order_id", lambda s: self._ask_order_id(s))
        workflow.add_node("ask_refund_reason", lambda s: self._ask_refund_reason(s))
        workflow.add_node("ask_invoice_order_id", lambda s: self._ask_invoice_order_id(s))
        workflow.add_node("parse_slots", lambda s: self._parse_slots(s))
        workflow.add_node("tool_exec", lambda s: self._tool_exec(s, snapshot))
        workflow.add_node("respond", lambda s: self._respond(s, snapshot))

        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "parse_slots": "parse_slots",
                "ask_order_id": "ask_order_id",
                "ask_refund_reason": "ask_refund_reason",
                "ask_invoice_order_id": "ask_invoice_order_id",
                "respond": "respond",
            },
        )

        workflow.add_edge("ask_order_id", END)
        workflow.add_edge("ask_refund_reason", END)
        workflow.add_edge("ask_invoice_order_id", END)

        workflow.add_edge("parse_slots", "tool_exec")
        workflow.add_edge("tool_exec", "respond")
        workflow.add_edge("respond", END)

        checkpointer = MemorySaver()
        app = workflow.compile(checkpointer=checkpointer)
        print(f"✅ LangGraph app compiled for service_version={service_version}")
        return app

    @staticmethod
    def _ensure_slots(state: AgentState) -> AgentState:
        if "slots" not in state or not isinstance(state["slots"], dict):
            state["slots"] = {}
        return state

    def _router(self, state: AgentState) -> AgentState:
        state = self._ensure_slots(state)
        messages: List[BaseMessage] = state.get("messages", [])
        last_user = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_user = m.content
                break

        intent = _detect_intent(last_user)
        state["slots"]["intent"] = intent

        # Very lightweight multi-turn gatekeeping:
        if intent == "order_query":
            # If user didn't provide order id and isn't asking with relative time to infer date, ask for it.
            if _extract_order_id(last_user) is None:
                # If only said "我昨天下的单" without order id: allow parse_slots -> time tool -> respond guidance.
                if "下单" in last_user and _looks_like_relative_time(last_user):
                    return state
                state["slots"]["need"] = "order_id"
        elif intent == "refund":
            if _extract_order_id(last_user) is None:
                state["slots"]["need"] = "order_id"
            else:
                # refund needs a reason; if absent, ask.
                if not any(k in last_user for k in ["因为", "原因", "不想要", "买错", "重复", "质量", "破损"]):
                    state["slots"]["need"] = "refund_reason"
        elif intent == "invoice":
            if _extract_order_id(last_user) is None:
                state["slots"]["need"] = "invoice_order_id"

        return state

    def _route_decision(self, state: AgentState) -> Literal["parse_slots", "ask_order_id", "ask_refund_reason", "ask_invoice_order_id", "respond"]:
        state = self._ensure_slots(state)
        need = state["slots"].get("need")
        if need == "order_id":
            return "ask_order_id"
        if need == "refund_reason":
            return "ask_refund_reason"
        if need == "invoice_order_id":
            return "ask_invoice_order_id"

        # If user only mentions relative time without a clear task, respond via LLM fallback
        messages: List[BaseMessage] = state.get("messages", [])
        last_user = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_user = m.content
                break
        if _looks_like_relative_time(last_user) and _detect_intent(last_user) == "chitchat":
            return "respond"

        return "parse_slots"

    @staticmethod
    def _ask_order_id(state: AgentState) -> AgentState:
        msg = AIMessage(content="好的，请提供订单号（例如：SN20240924001）。")
        state.setdefault("messages", []).append(msg)
        return state

    @staticmethod
    def _ask_refund_reason(state: AgentState) -> AgentState:
        msg = AIMessage(content="好的，请补充退款原因（例如：买错/重复购买/质量问题/不想要了）。")
        state.setdefault("messages", []).append(msg)
        return state

    @staticmethod
    def _ask_invoice_order_id(state: AgentState) -> AgentState:
        msg = AIMessage(content="好的，请提供需要开具发票的订单号（例如：SN20240924001）。")
        state.setdefault("messages", []).append(msg)
        return state

    def _parse_slots(self, state: AgentState) -> AgentState:
        state = self._ensure_slots(state)
        messages: List[BaseMessage] = state.get("messages", [])
        last_user = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_user = m.content
                break
        last_user = _normalize_text(last_user)

        state["slots"]["order_id"] = state["slots"].get("order_id") or _extract_order_id(last_user)

        # Refund reason heuristic
        if state["slots"].get("intent") == "refund":
            if "因为" in last_user:
                reason = last_user.split("因为", 1)[1].strip()
            else:
                # fallback: take whole sentence if short
                reason = last_user if len(last_user) <= 50 else "用户未明确说明"
            state["slots"]["refund_reason"] = reason

        # Relative time: keep raw for tool conversion
        if _looks_like_relative_time(last_user):
            state["slots"]["relative_time_text"] = last_user

        return state

    def _tool_exec(self, state: AgentState, snapshot: ServiceSnapshot) -> AgentState:
        state = self._ensure_slots(state)
        intent = state["slots"].get("intent", "chitchat")
        order_id = state["slots"].get("order_id")
        refund_reason = state["slots"].get("refund_reason")
        rel_time_text = state["slots"].get("relative_time_text")

        tool_result: Any = None

        # Tool selection (auto)
        if rel_time_text:
            # Always provide date inference if user mentioned relative time
            tool_result = {"relative_date": get_date_for_relative_time.invoke({"relative_time_str": rel_time_text})}

        if intent == "order_query":
            if order_id:
                r = query_order.invoke({"order_id": order_id})
                tool_result = {"relative_time": tool_result, "order": r} if tool_result else {"order": r}
            else:
                # if no order id (e.g., "我昨天下的单"), keep only relative date
                tool_result = tool_result or {"info": "用户未提供订单号"}
        elif intent == "refund":
            if order_id and refund_reason:
                r = apply_refund.invoke({"order_id": order_id, "reason": refund_reason})
                tool_result = {"relative_time": tool_result, "refund": r} if tool_result else {"refund": r}
            else:
                tool_result = tool_result or {"error": "缺少订单号或退款原因"}
        elif intent == "invoice":
            if order_id:
                # invoice tool is plugin-based, selected from snapshot.tools by name
                inv_tool = next((t for t in snapshot.tools if t.name == "generate_invoice_tool"), None)
                if inv_tool is None:
                    tool_result = {"success": False, "error": "发票插件未加载"}
                else:
                    r = inv_tool.invoke({"order_id": order_id})
                    tool_result = {"invoice": r}
            else:
                tool_result = tool_result or {"error": "缺少订单号"}

        state["tool_result"] = tool_result
        return state

    def _respond(self, state: AgentState, snapshot: ServiceSnapshot) -> AgentState:
        state = self._ensure_slots(state)
        intent = state["slots"].get("intent", "chitchat")
        tool_result = state.get("tool_result")

        # If no tool_result, use LLM fallback (mock)
        if not tool_result:
            msg = snapshot.llm.generate(state.get("messages", []))
            state.setdefault("messages", []).append(msg)
            return state

        # Format response (professional, deterministic)
        parts: List[str] = []

        if isinstance(tool_result, dict) and tool_result.get("relative_date"):
            parts.append(f"我已将相对时间解析为日期：{tool_result['relative_date']}。")

        if intent == "order_query":
            if isinstance(tool_result, dict) and "order" in tool_result:
                order = tool_result["order"]
                if order.get("success"):
                    parts.append(f"订单号：{order['order_id']}")
                    parts.append(f"状态：{order['status']}")
                    parts.append(f"物流单号：{order['tracking_number'] or '暂无'}")
                    parts.append(f"明细：{order.get('details', '')}")
                else:
                    parts.append(f"查询失败：{order.get('error', '未知错误')}")
            else:
                parts.append("你提到下单时间，但尚未提供订单号。请提供订单号（例如：SN20240924001），我再帮你查询订单状态与物流。")

        elif intent == "refund":
            refund = tool_result.get("refund") if isinstance(tool_result, dict) else None
            if refund and refund.get("success"):
                parts.append(f"退款申请已提交。订单号：{refund['order_id']}，退款单号：{refund['refund_id']}。")
                parts.append(refund.get("message", ""))
            elif refund:
                parts.append(f"退款申请失败：{refund.get('error', '未知错误')}")
            else:
                parts.append("退款需要订单号与原因。请按提示补充信息。")

        elif intent == "invoice":
            invoice = tool_result.get("invoice") if isinstance(tool_result, dict) else None
            if invoice and invoice.get("success"):
                parts.append(f"发票已生成：{invoice.get('invoice_url', '')}")
                parts.append(invoice.get("message", ""))
            elif invoice:
                parts.append(f"发票生成失败：{invoice.get('error', '未知错误')}")
            else:
                parts.append("开具发票需要订单号。请提供订单号（例如：SN20240924001）。")

        else:
            # chitchat + relative time parsed
            parts.append("我可以继续帮你：查订单 / 申请退款 / 开具发票。请告诉我你的诉求。")

        msg = AIMessage(content="\n".join([p for p in parts if p]))
        state.setdefault("messages", []).append(msg)
        return state


# -----------------------------
# 6) FastAPI schema & endpoints
# -----------------------------

app = FastAPI(
    title="Smart Customer Service API",
    description="集成 LangGraph + 工具调用 + 模型/插件热更新 + 会话版本隔离 的智能客服API",
    version="1.0.0",
)

plugin_manager = PluginManager(tools_package="tools")
service_manager = ServiceManager(plugin_manager=plugin_manager)
graph_factory = GraphFactory(service_manager=service_manager)
session_registry = SessionRegistry()


class ChatRequest(BaseModel):
    user_id: str
    query: str


class HotUpdateRequest(BaseModel):
    type: str  # "model" or "plugin"
    name: str  # model_name or plugin_name


@app.get("/health", summary="健康检查")
async def health_check():
    return {
        "status": "healthy",
        "services": service_manager.health(),
        "sessions": session_registry.stats(),
    }


@app.post("/chat", summary="进行对话")
async def chat(req: ChatRequest):
    thread_id = _normalize_text(req.user_id)
    if not thread_id:
        raise HTTPException(status_code=400, detail="user_id 不能为空")

    user_text = _normalize_text(req.query)
    if not user_text:
        raise HTTPException(status_code=400, detail="query 不能为空")

    # Session pins to current service version on first use
    current_version = service_manager.get_current_snapshot().version
    session = session_registry.get_or_create(
        thread_id=thread_id,
        service_version=current_version,
        graph_app_factory=lambda v: graph_factory.get_app(v),
    )

    config = {"configurable": {"thread_id": thread_id}}

    # Append user message into graph
    inputs = {"messages": [HumanMessage(content=user_text)]}

    final_response = ""
    try:
        # Stream-like iteration; take the last AI message as final response
        for event in session.graph_app.stream(inputs, config=config, stream_mode="values"):
            msgs = event.get("messages") or []
            if msgs:
                last = msgs[-1]
                if isinstance(last, AIMessage):
                    final_response = last.content or final_response

        if not final_response:
            final_response = "抱歉，我暂时无法回答这个问题。"

        return {
            "user_id": thread_id,
            "service_version": session.service_version,
            "response": final_response,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"chat failed: {e}")


@app.post("/hot-update", summary="热更新模型或插件")
async def hot_update(req: HotUpdateRequest):
    t = _normalize_text(req.type).lower()
    name = _normalize_text(req.name)
    if t not in {"model", "plugin"}:
        raise HTTPException(status_code=400, detail="type 必须是 'model' 或 'plugin'")
    if not name:
        raise HTTPException(status_code=400, detail="name 不能为空")

    try:
        if t == "model":
            service_manager.hot_update_model(new_model_name=name)
        elif t == "plugin":
            # name examples: invoice_plugin_v1 / invoice_plugin_v2
            service_manager.hot_update_invoice_plugin(plugin_name=name)

        # Important: Do NOT mutate existing sessions' pinned service_version / graph.
        # New sessions will bind to new service version.

        cur = service_manager.get_current_snapshot()
        return {
            "status": "success",
            "message": f"{t} 热更新完成",
            "current_version": cur.version,
            "model": cur.model_name,
            "invoice_plugin": cur.invoice_plugin,
            "tools": [tt.name for tt in cur.tools],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"热更新失败: {e}")


# -----------------------------
# 7) Automated tests (scriptable)
# -----------------------------

def _local_test_assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def run_tests():
    """
    Required by homework:
      - Test "invoice plugin" correctness
      - Verify after hot update, old sessions are not impacted
    """
    print("=== Running Tests ===")

    # Create a simulated session A -> pins current version (v1) with invoice_plugin_v1
    user_a = "user_A"
    v_before = service_manager.get_current_snapshot().version
    sA = session_registry.get_or_create(user_a, v_before, lambda v: graph_factory.get_app(v))
    _local_test_assert(sA.service_version == v_before, "Session A should pin to initial service version")

    # Ask invoice with known order
    configA = {"configurable": {"thread_id": user_a}}
    outA = ""
    for ev in sA.graph_app.stream({"messages": [HumanMessage(content="我要开票，订单号 SN20240924001")]}, config=configA, stream_mode="values"):
        msgs = ev.get("messages") or []
        if msgs and isinstance(msgs[-1], AIMessage):
            outA = msgs[-1].content or outA

    print("[Test] Session A invoice output:", outA)
    _local_test_assert("发票已生成" in outA, "Invoice should be generated in session A")

    # Hot update invoice plugin to v2
    service_manager.hot_update_invoice_plugin("invoice_plugin_v2")
    v_after = service_manager.get_current_snapshot().version
    _local_test_assert(v_after > v_before, "Service version should increase after hot update")

    # Create session B -> pins new version (v2)
    user_b = "user_B"
    sB = session_registry.get_or_create(user_b, v_after, lambda v: graph_factory.get_app(v))
    _local_test_assert(sB.service_version == v_after, "Session B should pin to new service version")

    # B invoices same order, should reflect v2 behavior (different URL domain/path)
    configB = {"configurable": {"thread_id": user_b}}
    outB = ""
    for ev in sB.graph_app.stream({"messages": [HumanMessage(content="发票 SN20240924001")]}, config=configB, stream_mode="values"):
        msgs = ev.get("messages") or []
        if msgs and isinstance(msgs[-1], AIMessage):
            outB = msgs[-1].content or outB

    print("[Test] Session B invoice output:", outB)
    _local_test_assert("v2" in outB or "new-invoice" in outB or "invoices-v2" in outB, "Session B should use v2 plugin output markers")

    # Session A invoices again: must still be v1 output (old sessions unaffected)
    outA2 = ""
    for ev in sA.graph_app.stream({"messages": [HumanMessage(content="再开一次发票 SN20240924001")]}, config=configA, stream_mode="values"):
        msgs = ev.get("messages") or []
        if msgs and isinstance(msgs[-1], AIMessage):
            outA2 = msgs[-1].content or outA2

    print("[Test] Session A invoice output after hot update:", outA2)
    _local_test_assert("invoices-v1" in outA2 or "v1" in outA2, "Session A should remain on v1 plugin output markers")

    print("✅ All tests passed.")


# -----------------------------
# 8) Main entry
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tests", action="store_true", help="Run automated tests then exit")
    parser.add_argument("--serve", action="store_true", help="Run FastAPI via uvicorn (python main.py --serve)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.run_tests:
        run_tests()
        raise SystemExit(0)

    if args.serve:
        import uvicorn
        uvicorn.run("main:app", host=args.host, port=args.port, reload=False)
        raise SystemExit(0)

    # Default: print quick usage
    print("Usage:")
    print("  1) Run API:     uvicorn main:app --host 0.0.0.0 --port 8000")
    print("  2) Run tests:   python main.py --run-tests")
    print("  3) Serve via py python main.py --serve --host 0.0.0.0 --port 8000")
