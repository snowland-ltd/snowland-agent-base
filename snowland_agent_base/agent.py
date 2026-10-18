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

"""AI Agent 基类: 串联 感知 -> 决策 -> 执行 -> 记忆 的完整循环。

子类只需继承 :class:`BaseAgent`, 通过 ``executor.register(...)`` 注册业务工具、
按需覆盖 :meth:`BaseAgent._build_prompt_context` 即可, 无需改动主循环。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:  # 仅用于类型注解, 运行时无需导入 (见 PEP 563)
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel

from snowland_agent_base.decision import AgentAction, DecisionLayer, Rule
from snowland_agent_base.execution import ToolExecutor
from snowland_agent_base.memory import VectorMemory, MemoryBackend
from snowland_agent_base.messages import (
    Message,
    assistant_message,
    system_message,
    tool_message,
    user_message,
)
from snowland_agent_base.model import LazyChatModel, RetryOnEmptyChatModel
from snowland_agent_base.model.registry import get_message_class
from snowland_agent_base.perception import PerceptionLayer, PerceptionBackend
from snowland_agent_base.skill import Skill, SkillRegistry

logger = logging.getLogger(__name__)


class BaseAgent:
    """AI Agent 基类。

    运行循环 (ReAct 风格):
        1. 感知层解析输入 -> 实体 / 意图
        2. 记忆层语义检索 -> 相关长期记忆
        3. 决策层产出动作 -> 规则兜底 或 LLM 链式思考
        4. 执行层调用工具 -> 结果回流, 继续下一轮 (受 ``max_steps`` 限制)
        5. 结束时把交互写入长期记忆
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
        system_prompt: Optional[str] = None,
        skills: Optional[SkillRegistry] = None,
        auto_activate_skills: bool = True,
        backend: Optional[str] = None,
        enable_retry: bool = True,
        retry_max_retries: int = 2,
    ) -> None:
        """初始化 Agent。

        Args:
            llm: LangChain ChatModel (参见 :func:`snowland_agent_base.model.build_llm`)。
            embedding: Embeddings 实例, 开启记忆时必需。
            perception: 感知层, 缺省自动构建。
            decision: 决策层, 缺省基于 ``llm`` 构建 (此时 ``system_prompt`` 生效)。
            executor: 执行层, 缺省自动构建。
            memory: 记忆层; 缺省且提供 ``embedding`` 时自动构建。
            max_steps: 单次运行最大推理步数, 防止死循环。
            use_perception: 是否启用感知层 (关闭可省去 spaCy 依赖)。
            use_memory: 是否启用长期记忆。
            system_prompt: 角色/人设提示词, 如 ``"你是一个电商客服助手"``。
                仅在 ``decision`` 未显式传入时生效;
                需动态生成时可覆盖 :meth:`_build_system_prompt`。
            skills: skill 注册表 (见 :mod:`snowland_agent_base.skill`)。
            auto_activate_skills: 是否按输入自动激活相关 skill (词面粗筛)。
                关闭后仅注入 skill 目录, 由 LLM 自行判断是否使用。
            enable_retry: 是否给 LLM 包一层「空 content 自愈重试」代理 (默认开)。
                所有 Agent 默认获得与业务层一致的空响应自愈能力; 正常模型零额外开销,
                推理模型撞满额度返回空 content 时自动重试。传入的 llm 若是代理本身则
                跳过以避免重复包裹; 若传入 :class:`LazyChatModel`, 则在外层再包一层
                重试, 同时具备「懒加载 + 重试」。
            retry_max_retries: 空 content 重试次数 (不含首次), 默认 2。
        """
        # 统一空响应自愈重试: 让所有 Agent 默认具备与业务层一致的「空 content 自动重试」
        # 能力。传入的 llm 若是 RetryOnEmptyChatModel 本身则跳过, 避免重复包裹; 若传入
        # LazyChatModel, 则在外层再包一层重试, 同时获得「懒加载 + 重试」。
        if enable_retry and not isinstance(llm, RetryOnEmptyChatModel):
            llm = RetryOnEmptyChatModel(inner=llm, max_retries=retry_max_retries)
        self.llm = llm
        self.max_steps = max_steps
        self.use_perception = use_perception
        self.use_memory = use_memory and embedding is not None
        self._system_prompt = system_prompt
        # 后端名称 (如 "anthropic"): 仅用于选择配套的消息风格类,
        # 构造消息时统一用 self._message_cls, 决策层不再感知具体后端。
        self._backend = backend
        self._message_cls = get_message_class(backend)

        self.skills = skills or SkillRegistry()
        self.auto_activate_skills = auto_activate_skills

        if perception is None:
            self.perception = PerceptionLayer() if use_perception else None
        elif isinstance(perception, PerceptionLayer):
            self.perception = perception
        else:
            # 传入的是具体后端实例, 用门面包一层以统一接口
            self.perception = PerceptionLayer(backend=perception)
        if decision is not None:
            if system_prompt is not None:
                logger.warning(
                    "同时传入 decision 与 system_prompt, system_prompt 将被忽略; "
                    "请改为在 DecisionLayer(system_prompt=...) 上设置"
                )
            if skills is not None:
                logger.warning(
                    "同时传入 decision 与 skills, skills 将被忽略; "
                    "请改为在 DecisionLayer(skills=...) 上设置"
                )
            self.decision = decision
        else:
            self.decision = DecisionLayer(
                llm=llm, system_prompt=system_prompt, skills=self.skills
            )
        self.executor = executor or ToolExecutor()
        # 构造时传入的注册表, 其 skill 携带的工具也需注册进执行器
        for existing in self.skills.list_skills():
            for tool_name, func in existing.tools.items():
                self.register_tool(tool_name, func)

        if memory is None:
            self.memory = (
                VectorMemory(embedding=embedding)
                if (self.use_memory and embedding is not None)
                else None
            )
        elif isinstance(memory, VectorMemory):
            self.memory = memory
        else:
            # 传入的是具体后端实例 (如 ChromaMemoryBackend), 用门面包一层以统一接口
            self.memory = VectorMemory(backend=memory)

    # ------------------------------------------------------------------ #
    # 扩展点
    # ------------------------------------------------------------------ #
    def add_rule(self, rule: Rule) -> None:
        """注册一条决策规则 (兜底用)。"""
        self.decision.add_rule(rule)

    def register_tool(self, name: str, func: callable) -> None:
        """注册一个可执行工具。"""
        self.executor.register(name, func)

    # ------------------------------------------------------------------ #
    # Skill
    # ------------------------------------------------------------------ #
    def add_skill(self, skill_obj: Skill, overwrite: bool = False) -> None:
        """注册一个 skill。

        Args:
            skill_obj: :class:`~snowland_agent_base.skill.Skill` 实例。
            overwrite: 同名时是否覆盖。
        """
        self.skills.register(skill_obj, overwrite=overwrite)
        # skill 携带的工具注册进执行器, 使其可被决策层调用
        for tool_name, func in skill_obj.tools.items():
            self.register_tool(tool_name, func)

    def load_skills(self, directory: Any, overwrite: bool = False) -> List[Skill]:
        """从目录扫描加载 skill (自动发现含 SKILL.md 的子目录)。

        Args:
            directory: skills 根目录或单个 skill 目录。
            overwrite: 同名时是否覆盖。

        Returns:
            本次加载的 skill 列表。
        """
        loaded = self.skills.load_directory(directory, overwrite=overwrite)
        for skill_obj in loaded:
            for tool_name, func in skill_obj.tools.items():
                self.register_tool(tool_name, func)
        return loaded

    def _activate_skills(self, user_input: str) -> List[Skill]:
        """按输入粗筛出待激活的 skill (渐进式披露第二级)。

        子类可覆盖以实现基于向量检索 / LLM 判定的激活策略。

        Args:
            user_input: 用户输入。

        Returns:
            待激活的 skill 列表。
        """
        if not self.auto_activate_skills or not self.skills:
            return []
        return self.skills.match(user_input)

    def _build_prompt_context(
        self, user_input: str, perception_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """构建决策上下文, 子类可覆盖以注入业务字段。"""
        return perception_result or {"text": user_input, "entities": [], "intent": "unknown"}

    def _build_system_prompt(
        self, user_input: str, perception_result: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """构建系统提示词, 子类可覆盖以实现按请求动态生成的人设。

        常用于注入用户画像、当前时间、业务状态等。返回 ``None`` 表示沿用构造时
        传入的 ``system_prompt`` (或默认人设)。

        Args:
            user_input: 用户输入。
            perception_result: 感知层结果, 可能为空。

        Returns:
            系统提示词, 或 ``None``。
        """
        return None

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    def run(self, user_input: str) -> Dict[str, Any]:
        """运行一次完整推理, 返回结构化结果。

        Args:
            user_input: 用户输入。

        Returns:
            ``{"input", "perception", "steps", "answer", "messages", "success"}``

            - ``steps``: 每轮的 action / observation 明细。
            - ``answer``: 最终答复文本。
            - ``messages``: 完整对话轨迹, 为 :data:`Message` 列表,
              顺序为「记忆 -> 用户输入 -> 工具观测 -> 助手答复」。
        """
        perception_result = self._perceive(user_input)
        context = self._build_prompt_context(user_input, perception_result)
        recalled = self._recall(user_input)

        system_prompt = self._build_system_prompt(user_input, perception_result)
        active_skills = self._activate_skills(user_input)
        memory_messages = self._build_history(recalled)
        history: List[Message] = list(memory_messages)  # 传给决策层的上下文
        observations: List[Message] = []  # 工具观测, 用于组装输出
        steps: List[Dict[str, Any]] = []
        answer: Optional[str] = None

        for step in range(1, self.max_steps + 1):
            action = self.decision.decide(
                user_input=user_input,
                tools=self.executor.list_tools(),
                history=history,
                context=context,
                system_prompt=system_prompt,
                skills=active_skills,
            )
            logger.debug("step %s -> %s", step, action.to_dict())

            if action.is_final:
                answer = action.final_answer
                steps.append({"step": step, "action": action.to_dict(), "observation": None})
                break

            observation = self.executor.execute(action.tool or "", **action.tool_input)
            steps.append(
                {"step": step, "action": action.to_dict(), "observation": observation}
            )
            obs_message = tool_message(
                self._render_observation(observation),
                name=action.tool or "",
                cls=self._message_cls,
            )
            observations.append(obs_message)
            history.append(obs_message)

        if answer is None:
            answer = f"已在 {self.max_steps} 步内完成可用操作, 未得出最终结论。"

        self._memorize(user_input, answer)

        messages: List[Message] = (
            memory_messages
            + [user_message(user_input, cls=self._message_cls)]
            + observations
        )
        if answer:
            messages.append(assistant_message(answer, cls=self._message_cls))

        return {
            "input": user_input,
            "perception": perception_result,
            "steps": steps,
            "answer": answer,
            "messages": messages,
            "skills": [s.name for s in active_skills],
            "success": True,
        }

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #
    def _perceive(self, user_input: str) -> Optional[Dict[str, Any]]:
        if self.perception is None:
            return None
        try:
            return self.perception.perceive(user_input)
        except Exception as exc:  # noqa: BLE001 - 感知失败不应阻断主流程
            logger.warning("感知层失败, 已降级: %s", exc)
            return None

    def _recall(self, user_input: str) -> List[str]:
        if self.memory is None:
            return []
        try:
            return self.memory.search(user_input)
        except Exception as exc:  # noqa: BLE001
            logger.warning("记忆检索失败: %s", exc)
            return []

    def _memorize(self, user_input: str, answer: str) -> None:
        if self.memory is None:
            return
        try:
            self.memory.add([f"用户: {user_input}\n助手: {answer}"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("记忆写入失败: %s", exc)

    def _build_history(self, recalled: List[str]) -> List[Message]:
        """把检索到的长期记忆组装为历史消息列表。

        Args:
            recalled: 语义检索得到的记忆文本列表。

        Returns:
            :data:`Message` 列表; 无记忆时为空列表。
        """
        if not recalled:
            return []
        body = "\n".join(f"- {r}" for r in recalled)
        return [system_message(f"检索到的相关记忆:\n{body}", cls=self._message_cls)]

    @staticmethod
    def _render_observation(observation: Dict[str, Any]) -> str:
        """把工具执行结果渲染为观测文本。"""
        if observation.get("success"):
            return f"调用成功: {observation.get('result')}"
        return f"调用失败: {observation.get('error')}"
