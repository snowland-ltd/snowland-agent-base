# Copyright 2026 Snowland
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LangGraph 编排层: GraphAgent —— 继承 :class:`BaseAgent` 的图编排实现。

:class:`GraphAgent` **继承** :class:`~snowland_agent_base.agent.BaseAgent`, 复用其
感知 / 决策 / 执行 / 记忆四层及装配逻辑, 仅把主循环从手写 for 替换为 StateGraph,
从而获得:
    - 持久化 checkpoint / 断点续跑
    - 人机介入 (interrupt)
    - 流式输出 (stream_events v3)
    - 多 Agent / 子图编排

与基类的关系:
    - 复用: 层次装配、``_perceive`` / ``_recall`` / ``_memorize`` / ``_build_history`` / ``_render_observation``、
      ``add_rule`` / ``register_tool``、``_build_prompt_context`` / ``_build_system_prompt``
    - 覆盖: ``run()`` (改为走图, 并接受 thread_id)、图节点与路由
    - 新增: ``invoke`` / ``stream`` / ``stream_events`` / ``resume`` /
      ``get_state`` / ``update_state`` / ``add_subgraph``

依赖 (可选): ``pip install langgraph``

基本用法:
    agent = GraphAgent(llm=llm, embedding=embedding)
    agent.register_tool("search", func)
    result = agent.invoke("你好", thread_id="session-1")

需要人工审批时::
    st = agent.stream_events("删除订单 123", thread_id="t1")
    if st.interrupted:
        st = agent.resume(True, thread_id="t1")

注意 (LangGraph interrupt 规则):
    1. ``interrupt`` 通过抛异常实现, 节点内**不可**用裸 try/except 包裹,
       因此审批逻辑独立成 ``approval`` 节点, 且置于工具执行之前。
    2. 节点恢复时会**从节点开头重跑**, 故本模块所有 state 更新均采用
       "基于旧值重算"的覆盖式写法 (而非 add reducer), 保证幂等。
"""
from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Set, Union

# 直接引用模块而非包, 避免与 snowland_agent_base/__init__.py 形成循环导入
from snowland_agent_base.agent import BaseAgent
from snowland_agent_base.messages import (
    Message,
    assistant_message,
    ensure_messages,
    tool_message,
    user_message,
)

if TYPE_CHECKING:  # 仅用于类型注解
    from snowland_agent_base.skill import Skill, SkillRegistry

if TYPE_CHECKING:  # 以下仅用于类型注解, 运行时无需导入 (见 PEP 563)
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel

    from snowland_agent_base.decision import DecisionLayer
    from snowland_agent_base.execution import ToolExecutor
    from snowland_agent_base.memory import VectorMemory, MemoryBackend
    from snowland_agent_base.perception import PerceptionLayer, PerceptionBackend

logger = logging.getLogger(__name__)

try:  # pragma: no cover - 依赖缺失时给出明确提示
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    _LANGGRAPH_MISSING: Optional[str] = None
except ImportError as exc:  # pragma: no cover
    TypedDict = None  # type: ignore[assignment]
    END = START = StateGraph = Command = interrupt = None  # type: ignore[assignment]
    _LANGGRAPH_MISSING = str(exc)

    class AgentState(dict):  # type: ignore[no-redef]
        """langgraph 缺失时的占位, 使模块可被导入。"""

else:

    class AgentState(TypedDict, total=False):
        """图状态。

        所有字段均可选; 列表字段采用覆盖式更新以保证节点重跑幂等。
        """

        input: str
        perception: Optional[Dict[str, Any]]
        context: Dict[str, Any]
        recalled: List[str]
        #: 传给决策层的上下文: 记忆消息 + 工具观测消息
        history: List[Message]
        #: 工具观测消息, 与记忆分开保存以便组装输出
        observations: List[Message]
        #: 调用方传入的前置历史会话 (区别于 checkpointer 自动恢复的状态)
        input_history: List[Message]
        #: 已激活的 skill 名, 由 perceive 节点确定并贯穿整轮运行
        active_skills: List[str]
        action: Optional[Dict[str, Any]]
        observation: Optional[Dict[str, Any]]
        steps: List[Dict[str, Any]]
        step_count: int
        answer: Optional[str]
        #: 完整对话轨迹 (输出), 由 finalize 组装
        messages: List[Message]
        approval: Optional[Dict[str, Any]]


class GraphAgent(BaseAgent):
    """基于 LangGraph 的 Agent 实现, 继承并复用 :class:`BaseAgent` 的四层能力。

    图结构::

        START -> perceive -> decide -+-> finalize -> END
                                     |      ^
                                     +-> approval -+-> act -+ (步数未达上限则回 decide)
                                     |                      +-> finalize (达上限)
                                     +-> subgraph__<name> -> finalize
    """

    def __init__(
        self,
        llm: BaseChatModel,
        embedding: Optional[Embeddings] = None,
        perception: Optional[Union[PerceptionLayer, PerceptionBackend]] = None,
        decision: Optional[DecisionLayer] = None,
        executor: Optional[ToolExecutor] = None,
        memory: Optional[Union[VectorMemory, MemoryBackend]] = None,
        max_steps: int = 5,
        use_perception: bool = True,
        use_memory: bool = True,
        checkpointer: Optional[Any] = None,
        store: Optional[Any] = None,
        approval_tools: Optional[Sequence[str]] = None,
        require_approval: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        system_prompt: Optional[str] = None,
        skills: Optional[SkillRegistry] = None,
        auto_activate_skills: bool = True,
    ) -> None:
        """初始化 GraphAgent。

        Args:
            llm: LangChain ChatModel。
            embedding: Embeddings 实例, 开启长期记忆时必需。
            perception / decision / executor / memory: 可注入的层次实现, 缺省自动构建。
            max_steps: 最大工具调用轮次, 超限时收敛到 finalize。
            use_perception: 是否启用感知层。
            use_memory: 是否启用长期记忆。
            checkpointer: LangGraph checkpointer, 缺省 ``InMemorySaver()``。
                持久化到磁盘请用 ``SqliteSaver``。
            store: 可选的跨会话长期存储 (LangGraph Store)。
            approval_tools: 执行前需人工审批的工具名列表。
            require_approval: 自定义审批判定, 签名 ``(tool, tool_input) -> bool``,
                优先级高于 ``approval_tools``。
            system_prompt: 角色/人设提示词, 同 :class:`BaseAgent`。
            skills: skill 注册表, 同 :class:`BaseAgent`。
            auto_activate_skills: 是否自动激活相关 skill, 同 :class:`BaseAgent`。
        """
        if _LANGGRAPH_MISSING:
            raise ImportError(
                f"GraphAgent 需要 langgraph, 请执行: pip install langgraph ({_LANGGRAPH_MISSING})"
            )

        # 复用 BaseAgent 的层次装配 (感知/决策/执行/记忆), 不重复实现
        super().__init__(
            llm=llm,
            embedding=embedding,
            perception=perception,
            decision=decision,
            executor=executor,
            memory=memory,
            max_steps=max_steps,
            use_perception=use_perception,
            use_memory=use_memory,
            system_prompt=system_prompt,
            skills=skills,
            auto_activate_skills=auto_activate_skills,
        )

        # 以下为 GraphAgent 专有能力
        self._approval_tools: Set[str] = set(approval_tools or ())
        self._require_approval = require_approval
        self._subgraphs: Dict[str, Any] = {}
        self._checkpointer = checkpointer
        self._store = store
        self._graph: Optional[Any] = None

    # ------------------------------------------------------------------ #
    # 扩展点
    # ------------------------------------------------------------------ #
    # 注: add_rule / register_tool / _build_prompt_context / _build_system_prompt
    #     均继承自 BaseAgent, 此处不再重复实现。

    def add_subgraph(self, name: str, subgraph: Any) -> None:
        """注册一个已编译的子图, 作为可被决策层调用的「工具」。

        决策层若输出 ``tool == name``, 则路由进该子图, 子图结束后收敛到 finalize。
        需先调用 :meth:`build` 才会生效。
        """
        self._subgraphs[name] = subgraph
        self._graph = None

    # ------------------------------------------------------------------ #
    # 图构建
    # ------------------------------------------------------------------ #
    def build(self) -> Any:
        """构建并编译 StateGraph。"""
        if self._graph is not None:
            return self._graph

        if self._checkpointer is None:
            from langgraph.checkpoint.memory import InMemorySaver

            self._checkpointer = InMemorySaver()

        builder = StateGraph(AgentState)
        builder.add_node("perceive", self._node_perceive)
        builder.add_node("decide", self._node_decide)
        builder.add_node("approval", self._node_approval)
        builder.add_node("act", self._node_act)
        builder.add_node("finalize", self._node_finalize)

        builder.add_edge(START, "perceive")
        builder.add_edge("perceive", "decide")

        routes: List[str] = ["finalize", "approval", "act"]
        for name, subgraph in self._subgraphs.items():
            # 注意: ':' 是 LangGraph 节点名保留字符, 不可用
            node_name = self._subgraph_node_name(name)
            builder.add_node(node_name, self._make_subgraph_node(name, subgraph))
            builder.add_edge(node_name, "finalize")
            routes.append(node_name)

        builder.add_conditional_edges("decide", self._route_decide, routes)
        builder.add_edge("approval", "act")
        builder.add_conditional_edges("act", self._route_act, ["decide", "finalize"])
        builder.add_edge("finalize", END)

        self._graph = builder.compile(checkpointer=self._checkpointer, store=self._store)
        return self._graph

    # ------------------------------------------------------------------ #
    # 节点
    # ------------------------------------------------------------------ #
    def _node_perceive(self, state: AgentState) -> Dict[str, Any]:
        """感知 + 记忆检索节点。"""
        user_input = state.get("input", "")
        perception_result = self._perceive(user_input)
        context = self._build_prompt_context(user_input, perception_result)
        recalled = self._recall(user_input)
        memory_messages = self._build_history(recalled)
        # 顺序: 检索到的记忆 -> 调用方传入的前置历史 (旧 -> 新)
        prior_history = ensure_messages(state.get("input_history"))
        active_skills = self._activate_skills(user_input)
        return {
            "active_skills": [s.name for s in active_skills],
            "perception": perception_result,
            "context": context,
            "recalled": recalled,
            "history": memory_messages + prior_history,
            "input_history": prior_history,
            "observations": [],
            "steps": [],
            "step_count": 0,
            "action": None,
            "observation": None,
            "answer": None,
            "messages": [],
            "approval": None,
        }

    def _node_decide(self, state: AgentState) -> Dict[str, Any]:
        """决策节点: 规则命中或 LLM 链式思考, 产出 action。"""
        action = self.decision.decide(
            user_input=state.get("input", ""),
            tools=self._tool_names(),
            history=state.get("history") or [],
            context=state.get("context") or {},
            system_prompt=self._build_system_prompt(
                state.get("input", ""), state.get("perception")
            ),
            skills=self._resolve_active_skills(state),
        )
        return {"action": action.to_dict()}

    def _resolve_active_skills(self, state: "AgentState") -> List[Skill]:
        """按 state 中记录的 skill 名解析出 skill 对象 (保证节点重跑幂等)。"""
        names = state.get("active_skills") or []
        resolved: List[Skill] = []
        for name in names:
            skill_obj = self.skills.get(name)
            if skill_obj is not None:
                resolved.append(skill_obj)
        return resolved

    def _node_approval(self, state: AgentState) -> Any:
        """人工审批节点。

        注意: 此处**不可**用裸 try/except 包裹 interrupt, 否则中断信号会被吞掉。
        """
        action = state.get("action") or {}
        tool_input = action.get("tool_input") or {}

        decision = interrupt(
            {
                "question": "执行该工具需要人工批准",
                "tool": action.get("tool"),
                "tool_input": tool_input,
            }
        )

        if isinstance(decision, dict):
            approved = bool(decision.get("approved", False))
            final_input = decision.get("tool_input") or tool_input
        else:
            approved = bool(decision)
            final_input = tool_input

        if approved:
            new_action = dict(action)
            new_action["tool_input"] = final_input
            return Command(goto="act", update={"action": new_action, "approval": {"approved": True}})

        rejected_answer = f"已取消执行工具 {action.get('tool')}。"
        return Command(
            goto="finalize",
            update={
                "approval": {"approved": False, "tool": action.get("tool")},
                "action": {"thought": "用户拒绝执行工具", "final_answer": rejected_answer},
            },
        )

    def _node_act(self, state: AgentState) -> Dict[str, Any]:
        """执行节点: 调用工具并记录观测。"""
        action = state.get("action") or {}
        tool = action.get("tool") or ""
        observation = self.executor.execute(tool, **(action.get("tool_input") or {}))

        steps = list(state.get("steps") or [])
        step_count = int(state.get("step_count") or 0) + 1
        steps.append({"step": step_count, "action": action, "observation": observation})

        obs_message = tool_message(
            self._render_observation(observation), name=tool, cls=self._message_cls
        )
        observations = list(state.get("observations") or []) + [obs_message]
        history = list(state.get("history") or []) + [obs_message]
        return {
            "observation": observation,
            "steps": steps,
            "step_count": step_count,
            "observations": observations,
            "history": history,
        }

    def _node_finalize(self, state: AgentState) -> Dict[str, Any]:
        """收尾节点: 确定最终答复并写入长期记忆。

        优先级: 节点已写入的 answer (如子图结果) > 决策层 final_answer > 默认文案。
        直接取 action.final_answer 会覆盖掉子图写入的结果。
        """
        answer = state.get("answer")
        if answer is None:
            action = state.get("action") or {}
            answer = action.get("final_answer")
        if answer is None:
            answer = f"已在 {self.max_steps} 步内完成可用操作, 未得出最终结论。"
        self._memorize(state.get("input", ""), answer)

        # 组装完整对话轨迹: 记忆 -> 前置历史 -> 用户输入 -> 工具观测 -> 助手答复
        memory_messages = self._build_history(state.get("recalled") or [])
        prior_history = ensure_messages(state.get("input_history"))
        observations = list(state.get("observations") or [])
        messages: List[Message] = (
            memory_messages
            + prior_history
            + [user_message(state.get("input", ""), cls=self._message_cls)]
            + observations
        )
        if answer:
            messages.append(assistant_message(answer, cls=self._message_cls))
        return {"answer": answer, "messages": messages}

    @staticmethod
    def _subgraph_node_name(name: str) -> str:
        """子图在父图中的节点名。

        ``:`` 是 LangGraph 节点名保留字符, 故用 ``__`` 分隔。
        """
        return f"subgraph__{name}"

    def _make_subgraph_node(self, name: str, subgraph: Any) -> Callable[[AgentState], Dict[str, Any]]:
        """把已编译子图包装为父图节点 (state 不共享, 以 input 传递)。"""

        def _node(state: AgentState) -> Dict[str, Any]:
            result = subgraph.invoke({"input": state.get("input", "")})
            answer = result.get("answer") if isinstance(result, dict) else None
            steps = list(state.get("steps") or [])
            step_count = int(state.get("step_count") or 0) + 1
            steps.append(
                {
                    "step": step_count,
                    "action": {"tool": name, "tool_input": {}},
                    "observation": {"success": True, "name": name, "result": answer, "error": None},
                }
            )
            return {"answer": answer, "steps": steps, "step_count": step_count}

        return _node

    # ------------------------------------------------------------------ #
    # 路由
    # ------------------------------------------------------------------ #
    def _route_decide(self, state: AgentState) -> str:
        """决策后路由: 终答 / 子图 / 需审批 / 直接执行。"""
        action = state.get("action") or {}
        if action.get("final_answer") is not None:
            return "finalize"

        tool = action.get("tool")
        if tool in self._subgraphs:
            return self._subgraph_node_name(tool)
        if self._needs_approval(tool or "", action.get("tool_input") or {}):
            return "approval"
        return "act"

    def _route_act(self, state: AgentState) -> str:
        """执行后路由: 未达步数上限则继续决策, 否则收敛。"""
        if int(state.get("step_count") or 0) >= self.max_steps:
            return "finalize"
        return "decide"

    def _needs_approval(self, tool: str, tool_input: Dict[str, Any]) -> bool:
        """判定工具是否需要人工审批。"""
        if self._require_approval is not None:
            return bool(self._require_approval(tool, tool_input))
        return tool in self._approval_tools

    def _tool_names(self) -> List[str]:
        """决策层可见的工具名 (含已注册子图)。"""
        return self.executor.list_tools() + list(self._subgraphs)

    # ------------------------------------------------------------------ #
    # 运行接口
    # ------------------------------------------------------------------ #
    def _config(self, thread_id: Optional[str]) -> Dict[str, Any]:
        return {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}

    @staticmethod
    def _payload(
        user_input: Any, history: Union[str, Sequence[Message], None] = None
    ) -> Dict[str, Any]:
        """构造图的输入 state。

        Args:
            user_input: 用户输入, 或直接传入 state 字典。
            history: 前置历史会话 (消息列表, 兼容字符串)。

        Returns:
            图输入 dict。
        """
        payload = dict(user_input) if isinstance(user_input, dict) else {"input": user_input}
        if history is not None:
            payload["input_history"] = ensure_messages(history)
        return payload

    def run(
        self,
        user_input: str,
        thread_id: Optional[str] = None,
        history: Union[str, Sequence[Message], None] = None,
    ) -> Dict[str, Any]:
        """保持与 :class:`BaseAgent` 一致的入口, 便于多态替换。

        Args:
            user_input: 用户输入。
            thread_id: 会话线程 ID; 不传则每次新建线程 (无断点续跑)。
            history: 前置历史会话。

        Returns:
            同 :meth:`invoke`。
        """
        return self.invoke(user_input, thread_id=thread_id, history=history)

    def invoke(
        self,
        user_input: Any,
        thread_id: Optional[str] = None,
        history: Union[str, Sequence[Message], None] = None,
    ) -> Dict[str, Any]:
        """同步运行, 返回与 :class:`BaseAgent` 一致的结构化结果。

        历史会话有两种提供方式:
            1. ``history`` 参数: 显式传入, 适合自行管理会话存储的场景;
            2. ``thread_id``: 由 checkpointer 自动恢复, 适合交给 LangGraph 托管。

        两者可同时使用。

        Args:
            user_input: 用户输入。
            thread_id: 会话线程 ID (断点续跑)。
            history: 前置历史会话, 为 :data:`Message` 列表 (兼容字符串)。

        Returns:
            ``{"input", "perception", "steps", "answer", "messages", "success",
            "thread_id", "interrupted", "interrupts"}``
        """
        graph = self.build()
        resolved_thread_id = thread_id or uuid.uuid4().hex
        state = graph.invoke(
            self._payload(user_input, history), config=self._config(resolved_thread_id)
        )
        interrupts = state.pop("__interrupt__", None) if isinstance(state, dict) else None
        return self._format(
            state,
            interrupted=bool(interrupts),
            interrupts=interrupts,
            thread_id=resolved_thread_id,
        )

    def stream_events(
        self,
        user_input: Any = None,
        thread_id: Optional[str] = None,
        version: str = "v3",
        history: Union[str, Sequence[Message], None] = None,
    ) -> Any:
        """事件流式运行, 支持 messages / values / interrupts 三类投影。

        推荐用于人机介入场景::

            st = agent.stream_events("删除订单 123", thread_id="t1")
            _ = st.output            # 驱动流执行
            if st.interrupted:
                st = agent.resume(True, thread_id="t1")

        Args:
            user_input: 用户输入。
            thread_id: 会话线程 ID。
            version: 流式协议版本。
            history: 前置历史会话。
        """
        graph = self.build()
        return graph.stream_events(
            self._payload(user_input, history), config=self._config(thread_id), version=version
        )

    def stream(
        self,
        user_input: Any,
        thread_id: Optional[str] = None,
        stream_mode: str = "values",
        history: Union[str, Sequence[Message], None] = None,
    ) -> Any:
        """按 state 快照 / 增量流式输出。

        Args:
            user_input: 用户输入。
            thread_id: 会话线程 ID。
            stream_mode: ``values`` (完整快照) / ``updates`` (增量) / ``messages`` (token 流)。
            history: 前置历史会话。
        """
        graph = self.build()
        return graph.stream(
            self._payload(user_input, history),
            config=self._config(thread_id),
            stream_mode=stream_mode,
        )

    def resume(self, decision: Any, thread_id: str) -> Any:
        """在 interrupt 处恢复执行。

        Args:
            decision: 传给 ``interrupt()`` 的返回值, 可为 bool 或
                ``{"approved": bool, "tool_input": {...}}`` (审批时改写入参)。
            thread_id: 必须与中断时的 thread_id 一致。
        """
        graph = self.build()
        return graph.stream_events(
            Command(resume=decision), config=self._config(thread_id), version="v3"
        )

    def get_state(self, thread_id: str) -> Any:
        """读取指定线程的当前状态快照。"""
        return self.build().get_state(self._config(thread_id))

    def update_state(self, thread_id: str, values: Dict[str, Any]) -> Any:
        """手动修改状态 (时间旅行 / 人工纠错后继续)。"""
        return self.build().update_state(self._config(thread_id), values)

    # 注: _perceive / _recall / _memorize / _build_history / _render_observation 均继承自 BaseAgent。

    @staticmethod
    def _format(
        state: Dict[str, Any],
        interrupted: bool,
        interrupts: Any,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """统一结果结构, 与 BaseAgent 对齐。"""
        return {
            "input": state.get("input", ""),
            "perception": state.get("perception"),
            "steps": state.get("steps") or [],
            "answer": state.get("answer"),
            "messages": state.get("messages") or [],
            "skills": list(state.get("active_skills") or []),
            "success": True,
            "thread_id": thread_id,
            "interrupted": interrupted,
            "interrupts": interrupts,
        }
