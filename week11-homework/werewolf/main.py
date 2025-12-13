
"""
狼人杀游戏示例 - 基于 LangGraph 的状态机实现

依赖：
  pip install "langchain>=0.3.0" "langchain-openai>=0.2.0" "langgraph>=0.2.0" "faiss-cpu"

运行前：
  export OPENAI_API_KEY=<your_api_key_here>
  python werewolf_langgraph_demo.py
"""

from __future__ import annotations
from typing import TypedDict, Dict, List, Literal, Optional
import random

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END


# -----------------------------
# 1. 全局 LLM & RAG 知识库
# -----------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)

# 一个很小的狼人杀规则 + 策略知识库示例
RAG_DOCS = [
    """
    狼人杀基础规则：共有狼人和好人两大阵营。
    夜晚：狼人统一选择一名玩家杀害；好人阵营中可能有预言家等功能牌。
    白天：主持人公布昨晚死亡玩家，所有存活玩家依次发言，然后进行投票处决一人。
    胜负条件：所有狼人出局，则好人阵营获胜；当狼人数量 >= 好人数量时，狼人阵营获胜。
    """,
    """
    狼人发言策略：
    - 注意隐藏身份，不要过度带节奏。
    - 尽量附和大多数人合理的怀疑方向，把票集中到好人身上。
    - 避免过早暴露与同伴的关系，减少“狼队友互踩过假”的痕迹。
    """,
    """
    村民发言与投票策略：
    - 关注说话内容是否前后自洽，有无逻辑漏洞。
    - 注意谁在悄悄跟风投票，而不是自己给出独立理由。
    - 记录每轮投票，寻找“总是投错”的玩家。
    """,
    """
    主持人（法官）职责：
    - 控制游戏流程和阶段切换。
    - 用简洁的语言描述局面：第几天，谁死亡，当前存活名单。
    - 不参与阵营，只负责叙述和执行规则。
    """,
]

emb = OpenAIEmbeddings()
vs = FAISS.from_documents([Document(page_content=t) for t in RAG_DOCS], emb)
retriever = vs.as_retriever(search_kwargs={"k": 3})


def rag_context(question: str) -> str:
    """简单的 RAG：根据当前问题检索规则/策略。"""
    docs = retriever.invoke(question)
    return "\n\n".join([d.page_content for d in docs])


# -----------------------------
# 2. LangGraph 状态定义
# -----------------------------

Role = Literal["wolf", "villager"]


class Player(TypedDict):
    name: str
    role: Role
    alive: bool
    # 长期记忆：该玩家记住的部分历史信息
    memory: List[str]


class GameState(TypedDict):
    day: int
    phase: Literal["night", "day_announce", "day_talk", "vote", "end"]
    players: Dict[str, Player]          # player_id -> Player
    history: List[str]                  # 全局可视化日志
    last_night_kill: Optional[str]      # 昨晚死亡玩家 id
    winner: Optional[str]               # "wolves" / "villagers" / None


def get_alive_players(state: GameState) -> List[str]:
    return [pid for pid, p in state["players"].items() if p["alive"]]


def get_alive_by_role(state: GameState, role: Role) -> List[str]:
    return [pid for pid, p in state["players"].items() if p["alive"] and p["role"] == role]


def short_public_view(state: GameState) -> str:
    alive = get_alive_players(state)
    dead = [pid for pid in state["players"] if pid not in alive]
    return (
        f"第 {state['day']} 天，当前存活玩家：{alive}；死亡玩家：{dead}；"
        f"上一晚死亡：{state['last_night_kill']}。"
    )


def call_player_model(
    state: GameState,
    pid: str,
    intent: Literal["night_action", "day_speech", "vote"],
    extra_info: str,
) -> str:
    """统一封装玩家 LLM 调用，带 RAG + 记忆。"""
    player = state["players"][pid]
    role = player["role"]
    memory_text = "\n".join(player["memory"][-6:])  # 取最近若干条
    public_view = short_public_view(state)

    # 构造检索问题
    rag_q = f"我在狼人杀中扮演{role}，当前阶段是{intent}，局面是：{public_view}。请给出简要策略建议。"
    context = rag_context(rag_q)

    sys_prompt = f"""
你现在是狼人杀游戏中的一名玩家 {pid}，你的真实身份是：{role}。
- 你必须严格保守自己的真实身份，只能通过发言“暗示”或“误导”。
- 你不能直接说出“我是狼人/我是好人阵营中的XX牌”这一类信息。
- 回答时使用简短的中文自然语言，不要剧透游戏规则。

【你的长期记忆】：
{memory_text or "（暂无）"}

【来自策略资料库的参考信息】：
{context}
    """

    if intent == "night_action":
        user_prompt = f"""
当前是夜晚阶段。你是{role}。
如果你是狼人，请在存活玩家中选择一个目标击杀，并给出一句理由。
如果你是村民，则只需说“我在睡觉”，不要输出别的信息。

局面信息：{public_view}
额外信息：{extra_info}

输出格式示例：
目标: P3
理由: 他白天发言很奇怪。
（如果你在睡觉就说：目标: NONE 理由: 我在睡觉。）
        """.strip()
    elif intent == "day_speech":
        user_prompt = f"""
当前是白天发言阶段。请你围绕昨晚死亡情况和投票情况，进行一段简短发言（2-4 句）。
- 狼人需要隐藏身份，可怀疑其他人。
- 村民需要尽量找出可疑对象。

局面信息：{public_view}
额外信息：{extra_info}

直接给出发言内容，不要再解释规则。
        """.strip()
    else:  # vote
        alive = get_alive_players(state)
        user_prompt = f"""
当前是投票阶段，请你在存活玩家 {alive} 中选择一人投票处决（不能投给已死亡玩家）。
给出一个你最想投票的玩家 ID，以及一句话理由。
输出格式：
投票: P3
理由: 他的发言前后矛盾。

局面信息：{public_view}
额外信息：{extra_info}
        """.strip()

    resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    text = resp.content.strip()
    # 把对局信息加入个人记忆
    player["memory"].append(f"[{intent}] {text}")
    return text


# -----------------------------
# 3. LangGraph 各阶段节点
# -----------------------------

def night_phase(state: GameState) -> GameState:
    """夜晚：狼人协商并选择击杀目标。"""
    alive_wolves = get_alive_by_role(state, "wolf")
    alive_players = get_alive_players(state)

    state["history"].append(f"=== 夜晚开始（第 {state['day']} 天）===")

    if not alive_wolves:
        # 没有狼，直接天亮
        state["last_night_kill"] = None
        state["phase"] = "day_announce"
        return state

    proposals = []
    for pid in alive_wolves:
        text = call_player_model(
            state,
            pid,
            "night_action",
            extra_info="请小心选择目标，不要选择自己或已死亡玩家。",
        )
        state["history"].append(f"【夜晚行动】{pid} 的想法：{text}")
        # 简单解析“目标: Px”
        target = None
        for token in text.split():
            token_clean = token.strip("：:，,。.")
            if token_clean in alive_players and token_clean not in alive_wolves:
                target = token_clean
                break
        if target:
            proposals.append(target)

    if not proposals:
        state["last_night_kill"] = None
    else:
        # 多数票简单决定，平票随机
        candidate = max(set(proposals), key=proposals.count)
        state["players"][candidate]["alive"] = False
        state["last_night_kill"] = candidate
        state["history"].append(f"【系统】夜晚狼人最终击杀了 {candidate}。")

    state["phase"] = "day_announce"
    return state


def day_announce(state: GameState) -> GameState:
    """主持人宣布昨晚死亡。"""
    view = short_public_view(state)
    sys_prompt = """
你是狼人杀游戏的主持人（法官）。
你的任务是用简短的中文描述昨晚发生的事情和当前局面，不要发表主观推理。
    """
    user_prompt = f"请面向所有玩家宣布：昨晚谁死亡，现在还有谁存活。信息：{view}"

    resp = llm.invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
    )
    ann = resp.content.strip()
    state["history"].append(f"【主持人】{ann}")
    state["phase"] = "day_talk"
    return state


def day_talk(state: GameState) -> GameState:
    """白天发言：所有存活玩家轮流发言。"""
    state["history"].append(f"=== 白天发言阶段（第 {state['day']} 天）===")
    alive = get_alive_players(state)

    for pid in alive:
        text = call_player_model(
            state,
            pid,
            "day_speech",
            extra_info="请控制在 2-4 句之内，尽量自然。",
        )
        state["history"].append(f"【发言】{pid}: {text}")

    state["phase"] = "vote"
    return state


def vote_phase(state: GameState) -> GameState:
    """投票阶段：所有存活玩家选择一人处决。"""
    state["history"].append(f"=== 投票阶段（第 {state['day']} 天）===")
    alive = get_alive_players(state)
    votes: Dict[str, int] = {pid: 0 for pid in alive}

    for pid in alive:
        text = call_player_model(
            state,
            pid,
            "vote",
            extra_info="请务必投一个存活玩家。",
        )
        state["history"].append(f"【投票声明】{pid}: {text}")
        chosen = None
        for token in text.split():
            token_clean = token.strip("：:，,。.")
            if token_clean in alive:
                chosen = token_clean
                break
        if chosen:
            votes[chosen] += 1

    # 统计票数
    if votes:
        max_votes = max(votes.values())
        candidates = [pid for pid, v in votes.items() if v == max_votes]
        executed = random.choice(candidates)
        state["players"][executed]["alive"] = False
        state["history"].append(
            f"【主持人】本轮投票最高票为 {executed}（{max_votes} 票），将被处决。"
        )
    else:
        executed = None
        state["history"].append("【主持人】本轮投票无效，没有人被处决。")

    # 检查胜负
    wolves_alive = len(get_alive_by_role(state, "wolf"))
    villagers_alive = len(get_alive_by_role(state, "villager"))

    if wolves_alive == 0:
        state["winner"] = "villagers"
        state["history"].append("【系统】所有狼人出局，好人阵营获胜！")
        state["phase"] = "end"
    elif wolves_alive >= villagers_alive:
        state["winner"] = "wolves"
        state["history"].append("【系统】狼人数量不小于好人，狼人阵营获胜！")
        state["phase"] = "end"
    else:
        state["day"] += 1
        state["phase"] = "night"

    return state


def phase_router(state: GameState) -> GameState:
    """路由节点：根据 phase 决定下一步走哪个子图节点。"""
    return state


# -----------------------------
# 4. 构建 LangGraph
# -----------------------------

def build_graph():
    g = StateGraph(GameState)

    g.add_node("router", phase_router)
    g.add_node("night", night_phase)
    g.add_node("day_announce", day_announce)
    g.add_node("day_talk", day_talk)
    g.add_node("vote", vote_phase)

    g.set_entry_point("router")

    # 根据 phase 决定走向
    def route_phase(state: GameState):
        return state["phase"]

    g.add_conditional_edges(
        "router",
        route_phase,
        {
            "night": "night",
            "day_announce": "day_announce",
            "day_talk": "day_talk",
            "vote": "vote",
            "end": END,
        },
    )

    # 每个阶段结束后回到 router
    g.add_edge("night", "router")
    g.add_edge("day_announce", "router")
    g.add_edge("day_talk", "router")
    g.add_edge("vote", "router")

    return g.compile()


# -----------------------------
# 5. 初始化游戏 & 运行示例
# -----------------------------

def init_game(seed: int = 42) -> GameState:
    random.seed(seed)
    # 固定 5 名玩家
    player_ids = ["P1", "P2", "P3", "P4", "P5"]
    roles: List[Role] = ["wolf", "wolf", "villager", "villager", "villager"]
    random.shuffle(roles)

    players: Dict[str, Player] = {}
    for pid, role in zip(player_ids, roles):
        players[pid] = {
            "name": pid,
            "role": role,
            "alive": True,
            "memory": [],
        }

    state: GameState = {
        "day": 1,
        "phase": "night",
        "players": players,
        "history": [],
        "last_night_kill": None,
        "winner": None,
    }
    return state


def main():
    app = build_graph()
    state = init_game()

    final_state = app.invoke(state)

    print("\n================ 游戏日志（节选） ================\n")
    for line in final_state["history"]:
        print(line)

    print("\n================ 游戏结果 ================")
    print("玩家身份：")
    for pid, p in final_state["players"].items():
        print(f"  {pid} -> {p['role']}，存活={p['alive']}")
    print("胜利阵营：", final_state["winner"])


if __name__ == "__main__":
    main()