# REPORT.md

## 1. 项目背景与目标

本项目实现一个小型多轮对话智能客服：
- 基础对话链：支持理解“昨天”等相对时间并推断为具体日期（YYYY-MM-DD）
- 多轮对话与工具调用：支持订单查询、退款申请、发票开具等流程，并由系统自动选择并调用工具
- 热更新与生产部署能力：支持模型热更新、插件热重载；提供健康检查接口；编写自动化测试验证“热更新后旧会话不受影响”

交付物：
- `main.py`：单文件主程序，包含 FastAPI 服务、LangGraph 编排、工具与热更新逻辑、自动化测试入口
- `tools/`：最小化插件目录（仅发票插件 v1/v2），用于演示“插件热重载/切换”

---

## 2. 总体架构设计说明

### 2.1 架构概览（逻辑分层）

- API 层（FastAPI）
  - `POST /chat`：会话对话入口（多轮）
  - `POST /hot-update`：热更新入口（model/plugin）
  - `GET /health`：健康检查（含当前服务版本、会话统计）

- 服务管理层（ServiceManager）
  - 维护“当前生效”的模型与工具集合
  - 每次热更新都创建新的 `service_version`
  - 提供 `ServiceSnapshot`：包含 model_name / tools / invoice_plugin / llm 等，供会话绑定

- 会话层（SessionRegistry）
  - 以 `thread_id(user_id)` 为 key
  - 会话创建时“绑定并固定”某个 `service_version`
  - 保障：热更新后，旧会话仍使用旧版本 snapshot（不受影响）

- 编排层（LangGraph）
  - 每个 `service_version` 对应一个编译好的 Graph（GraphFactory 缓存）
  - Graph 节点：router → (ask_xxx | parse_slots → tool_exec → respond)
  - MemorySaver 作为 checkpointer：按 thread_id 维护多轮状态

- 插件层（PluginManager）
  - 从 `tools/` 目录加载 `invoice_plugin_v1|v2`
  - 通过 `importlib.reload()` 支持热重载

### 2.2 关键机制：版本隔离（Hot Update 不影响旧会话）

- `ServiceManager` 维护递增版本号 `version`
- 每次热更新生成新的 `ServiceSnapshot(version=N)`
- `SessionRegistry` 中的 session 记录 `service_version`
- `/chat`：
  - 新会话：绑定“当前 version”
  - 旧会话：沿用原 `service_version`
- 因此：热更新只影响“新会话”（或你显式新开 thread_id），不会污染旧对话上下文

---

## 3. 核心模块与关键实现逻辑

### 3.1 相对时间解析（阶段一）

工具：`get_date_for_relative_time(relative_time_str) -> YYYY-MM-DD`
- 覆盖：今天、昨天、前天、上周X
- 通过 `_now_local()` 获取运行时当前时间；满足“结合当前时间推断昨天”的要求

在 Graph 的 `tool_exec` 中：
- 若用户话术包含相对时间关键词，则优先调用该工具，并把解析结果合并到最终回复中

### 3.2 多轮对话（阶段二）

Graph 关键节点：
- `router`：轻量意图识别（订单/退款/发票/闲聊），并决定是否需要追问 slot
- `ask_order_id / ask_refund_reason / ask_invoice_order_id`：当缺少必要字段时立即追问并结束本轮
- `parse_slots`：从用户文本提取结构化信息（order_id, refund_reason, relative_time_text）
- `tool_exec`：根据 intent 自动调用对应工具（query_order/apply_refund/invoice_plugin + time_tool）
- `respond`：把工具结果格式化成“可提交的客服回复”

多轮示例（订单查询）：
1) 用户：查订单
2) 系统：请提供订单号
3) 用户：SN20240924001
4) 系统：调用 query_order → 返回订单状态与物流信息

### 3.3 工具调用策略（Auto Tool Call）

本作业实现“自动工具调用”采用工程可控策略：
- 不依赖外部 LLM 的 tool_calls 能力（降低环境依赖与不确定性）
- 使用确定性规则选择工具：
  - 订单查询 → query_order
  - 退款 → apply_refund
  - 发票 → generate_invoice_tool（插件）
  - 相对时间 → get_date_for_relative_time
- 仍保留 LLM 抽象（MockLLM）作为闲聊/兜底回复，方便未来替换真实模型

---

## 4. 关键技术选型与权衡（Trade-offs）

### 4.1 为什么用 LangGraph 而不是纯链式 Prompt → LLM？

- 多轮对话是“状态机问题”，图编排更清晰：路由、追问、工具调用、回复格式化都可显式建模
- 便于插入“生产特性”：热更新后新会话版本切换、对不同版本 graph 进行缓存
- 更易测：节点行为确定，自动化测试可以稳定复现

### 4.2 为什么工具调用不完全依赖 LLM tool_calls？

- 作业允许在无法真实调用 API 时 mock；真实 LLM 的 tool_calls 在不同模型/SDK/Key 缺失时不稳定
- 采用 deterministic controller 可显著提高“可运行、可提交、可测”的交付质量
- 同时保留 LLM 抽象层，后续可把 `MockLLM` 替换为真实模型并改造成 LLM 驱动的工具调用

### 4.3 版本隔离的实现方式

- 方案 A：热更新后强制所有会话迁移（风险：破坏对话一致性）
- 方案 B：会话固定版本（本项目采用）
  - 优点：旧会话不受影响；符合题目“验证热更新后旧会话不受影响”
  - 缺点：同一服务会并存多个版本 graph，需管理缓存（本项目采用按版本缓存，规模可控）

---

## 5. LLM / RAG / Agent 使用方式与调试验证

### 5.1 LLM 使用方式

- 本项目提供 `BaseLLM` 与 `MockLLM`
- 对于：
  - 闲聊/不明确需求：MockLLM 返回引导式回复
  - 明确业务意图：由图编排执行工具并生成确定性回复

### 5.2 Prompt 设计思路（可扩展）

当前为简化依赖未引入复杂 Prompt；若替换真实模型建议：
- System prompt 明确工具使用规则、输出格式
- 对关键 slot（order_id/reason）采用结构化输出（JSON Schema）以提升鲁棒性

### 5.3 调试方法

- 运行时日志：
  - Tool 调用打印 `--- [Tool] ... ---`
  - 插件调用打印 `--- [Plugin:v1/v2] ... ---`
  - 可快速定位“路由—追问—工具执行”路径

- 健康检查：
  - `/health` 返回当前版本、工具列表、会话统计，便于部署监控

---

## 6. 运行方式说明（How to Run）

### 6.1 启动 API 服务

```bash
pip install fastapi uvicorn langchain langgraph
uvicorn main:app --host 0.0.0.0 --port 8000
```
### 6.2 调用示例

1. 健康检查

```bash
curl http://localhost:8000/health
```

2. 对话

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1","query":"查订单"}'
```

3. 热更新插件（切换到 v2）

```bash
curl -X POST http://localhost:8000/hot-update \
  -H "Content-Type: application/json" \
  -d '{"type":"plugin","name":"invoice_plugin_v2"}'
```

---

## 7. 示例运行结果 / 模拟输出（终端日志）

### 7.1 订单查询多轮

用户（u1）：查订单
系统：好的，请提供订单号（例如：SN20240924001）。

用户（u1）：SN20240924001
终端日志（节选）：

* --- [Tool] query_order: order_id=SN20240924001 ---

系统回复（示例）：

* 订单号：SN20240924001
* 状态：已发货
* 物流单号：SF123456789
* 明细：订单中的商品: LangChain入门实战T恤

### 7.2 相对时间推断

用户：我昨天下的单
终端日志：

* --- [Tool] get_date_for_relative_time: text=我昨天下的单 ---

系统回复（示例）：

* 我已将相对时间解析为日期：2025-12-12。
* 你提到下单时间，但尚未提供订单号。请提供订单号（例如：SN20240924001），我再帮你查询订单状态与物流。

（注：日期基于运行时系统时间自动推断。）

---

## 8. 自动化测试脚本（要求点覆盖）

直接运行：

```bash
python main.py --run-tests
```

测试点：

1. “发票开具”插件功能正确性（v1）
2. 热更新插件到 v2 后：

   * 新会话（新 thread_id）应使用 v2 输出
   * 旧会话（已创建 thread_id）仍使用 v1 输出，验证“不受影响”

---

## 9. 已知限制与可扩展方向

### 已知限制

* 当前工具调用由规则控制；未使用真实 LLM 的 tool_calls
* 相对时间解析覆盖有限（仅满足作业最小集合）

### 可扩展方向

* 替换 MockLLM 为真实模型，并采用结构化输出与 tool_calls
* 引入持久化存储（Redis/DB）替代 MemorySaver 以支持多实例部署
* 插件系统扩展为“插件注册表 + 签名校验 + 灰度发布”
* 增加可观测性：Prometheus 指标、结构化日志、trace id
