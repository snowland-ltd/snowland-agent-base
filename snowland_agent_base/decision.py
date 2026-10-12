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

"""决策层: 基于 LLM 的链式思考 (CoT) 拆解复杂任务, 搭配规则引擎兜底, 保证输出稳定性。

流程:
    1. 先过规则引擎 (``Rule`` 列表) —— 命中则直接返回确定性动作, 不调用 LLM。
    2. 未命中则交给 LLM 链式思考, 产出结构化 ``AgentAction``。
    3. LLM 输出解析失败或异常时, 回退到 ``fallback_action`` 兜底。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from snowland_agent_base.messages import Message, ensure_messages, to_langchain_messages
from snowland_agent_base.skill import Skill, SkillRegistry

logger = logging.getLogger(__name__)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """从文本中抽取第一个配平的 JSON 对象。

    先尝试整段直接 ``json.loads``; 失败则从首个 ``{`` 起做括号配平扫描
    (正确处理字符串内的引号与转义), 取首个完整对象。抽不到返回 ``None``。
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


@dataclass
class AgentAction:
    """决策层产出的结构化动作。"""

    thought: str = ""
    tool: Optional[str] = None
    tool_input: Dict[str, Any] = field(default_factory=dict)
    final_answer: Optional[str] = None

    @property
    def is_final(self) -> bool:
        """是否直接给出最终答复 (无需调用工具)。"""
        return self.final_answer is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought": self.thought,
            "tool": self.tool,
            "tool_input": self.tool_input,
            "final_answer": self.final_answer,
        }


@dataclass
class Rule:
    """规则引擎规则: ``predicate`` 命中时返回固定 ``action``。"""

    name: str
    predicate: Callable[[str, Dict[str, Any]], bool]
    action: AgentAction


COT_INSTRUCTIONS = """请按链式思考 (Chain-of-Thought) 拆解任务。

你必须只输出一个 JSON 对象, 不要输出任何额外文本, 格式如下:

当需要使用工具时:
{"thought": "你的推理过程", "tool": "工具名", "tool_input": {"参数名": "参数值"}, "final_answer": null}

当可以直接回答时:
{"thought": "你的推理过程", "tool": null, "tool_input": {}, "final_answer": "最终答复"}
"""

#: 未显式指定 system_prompt 时的默认角色设定
DEFAULT_SYSTEM_PROMPT = "你是一个 AI Agent 的决策中枢。"


class DecisionLayer:
    """决策: 规则兜底 + LLM 链式思考。"""

    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: Optional[str] = None,
        rules: Optional[List[Rule]] = None,
        fallback_answer: str = "抱歉, 我暂时无法处理这个请求。",
        max_parse_retries: int = 1,
        skills: Optional[SkillRegistry] = None,
    ) -> None:
        """初始化决策层。

        Args:
            llm: LangChain ChatModel。
            system_prompt: 角色/人设提示词, 置于 CoT 指令之前;
                缺省使用 :data:`DEFAULT_SYSTEM_PROMPT`。
            rules: 规则引擎规则列表, 按序匹配。
            fallback_answer: 解析/调用失败时的兜底答复。
            max_parse_retries: JSON 解析失败重问次数。
            skills: skill 注册表; 提供后其目录 (name + description) 会常驻
                系统提示词, 供模型判断相关性。
        """
        self._llm = llm
        self._system_prompt = system_prompt
        self._rules: List[Rule] = list(rules or [])
        self._fallback_answer = fallback_answer
        self._max_parse_retries = max_parse_retries
        self._skills = skills

    def build_prompt(
        self,
        tools: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        skills: Optional[Sequence[Skill]] = None,
    ) -> str:
        """组装系统提示词: 人设 + CoT 指令 + 工具列表 + Skill 指令。

        采用拼接而非 ``str.format()``, 避免 JSON 示例中的花括号被误解析。

        历史会话不再塞进系统提示词, 而是以独立消息参与多轮对话
        (见 :meth:`build_messages`), 更符合 LLM 的对话范式。

        Skill 遵循渐进式披露: 此处只注入**已激活** skill 的完整指令;
        未激活的仅以 name + description 出现在目录里
        (见 :meth:`SkillRegistry.render_catalog`)。

        Args:
            tools: 可用工具名列表。
            system_prompt: 覆盖实例级人设 (用于按请求动态生成)。
            skills: 已激活的 skill 列表。

        Returns:
            组装后的系统提示词。
        """
        persona = system_prompt if system_prompt is not None else self._system_prompt
        if not persona:
            persona = DEFAULT_SYSTEM_PROMPT

        tools_block = "\n".join(f"- {t}" for t in (tools or [])) or "(无可用工具)"
        blocks = [
            persona,
            COT_INSTRUCTIONS,
            f"可用工具列表:\n{tools_block}",
        ]

        # 第一级披露: 全部 skill 的 name + description, 供模型判断相关性
        if self._skills is not None:
            catalog = self._skills.render_catalog()
            if catalog:
                blocks.append(catalog)

        # 第二级披露: 已激活 skill 的完整指令
        if skills:
            rendered = "\n\n".join(s.render_instructions() for s in skills)
            blocks.append(f"已激活 Skill:\n\n{rendered}")
        return "\n\n".join(blocks)

    def build_messages(
        self,
        user_input: str,
        tools: Optional[List[str]] = None,
        history: Union[str, Sequence[Message], None] = None,
        system_prompt: Optional[str] = None,
        skills: Optional[Sequence[Skill]] = None,
    ) -> List[Any]:
        """组装完整消息序列: 系统提示词 + 历史消息 + 当前输入。

        Args:
            user_input: 当前用户输入。
            tools: 可用工具名列表。
            history: 历史会话 (消息列表, 兼容字符串)。
            system_prompt: 按请求动态指定人设。
            skills: 已激活的 skill 列表。

        Returns:
            LangChain 消息对象列表。
        """
        prompt = self.build_prompt(
            tools=tools, system_prompt=system_prompt, skills=skills
        )
        messages: List[Any] = [SystemMessage(content=prompt)]
        messages.extend(to_langchain_messages(ensure_messages(history)))
        messages.append(HumanMessage(content=user_input))
        return messages

    def add_rule(self, rule: Rule) -> None:
        """追加一条规则。"""
        self._rules.append(rule)

    def match_rule(self, user_input: str, context: Dict[str, Any]) -> Optional[AgentAction]:
        """按序匹配规则, 返回首个命中的动作。"""
        for rule in self._rules:
            try:
                if rule.predicate(user_input, context):
                    logger.debug("规则命中: %s", rule.name)
                    return rule.action
            except Exception as exc:  # noqa: BLE001 - 规则自身异常不应中断
                logger.warning("规则 '%s' 执行异常: %s", rule.name, exc)
        return None

    def decide(
        self,
        user_input: str,
        tools: Optional[List[str]] = None,
        history: Union[str, Sequence[Message], None] = None,
        context: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        skills: Optional[Sequence[Skill]] = None,
    ) -> AgentAction:
        """产出下一步动作。

        Args:
            user_input: 用户输入。
            tools: 可用工具名列表。
            history: 历史会话 (消息列表, 兼容字符串)。
            context: 感知层产出的上下文 (意图、实体等)。
            system_prompt: 按请求动态指定人设, 覆盖实例级配置。
            skills: 已激活的 skill 列表, 其指令会注入系统提示词。

        Returns:
            :class:`AgentAction`
        """
        context = context or {}
        matched = self.match_rule(user_input, context)
        if matched is not None:
            return matched

        messages = self.build_messages(
            user_input=user_input,
            tools=tools,
            history=history,
            system_prompt=system_prompt,
            skills=skills,
        )

        last_raw = ""
        for attempt in range(self._max_parse_retries + 1):
            try:
                response = self._llm.invoke(messages)
                last_raw = self._extract_text(response)
                return self._parse_action(last_raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM 决策第 %s 次失败: %s", attempt + 1, exc)

        logger.error("决策失败, 走兜底。原始输出: %s", last_raw)
        return AgentAction(thought="决策失败, 使用兜底答复", final_answer=self._fallback_answer)

    @staticmethod
    def _extract_text(response: Any) -> str:
        """从 LLM 响应中取出可供决策解析的文本。

        多数模型把答案放在 ``content``; 但推理模型(如 orcarouter/free)会把最终
        JSON 动作塞进 ``reasoning_content`` 而让 ``content`` 为空, 直接解析会失败。
        这里优先用 ``content``, 为空时回退到 ``reasoning_content``, 保证决策层
        能拿到动作 JSON, 而不是一路走兜底。
        """
        content = getattr(response, "content", "") or ""
        if isinstance(content, str) and content.strip():
            return content
        reasoning = ""
        extra = getattr(response, "additional_kwargs", None) or {}
        if isinstance(extra, dict):
            reasoning = extra.get("reasoning_content", "") or ""
        if not reasoning and hasattr(response, "reasoning_content"):
            reasoning = getattr(response, "reasoning_content", "") or ""
        return reasoning if isinstance(reasoning, str) else ""

    @staticmethod
    def _parse_action(raw: str) -> AgentAction:
        """从 LLM 输出中解析 JSON 动作, 兼容 ```json 代码块包裹与 JSON 内嵌在文本中。

        推理模型(如 orcarouter/free)常把最终 JSON 答案混在叙述/推理里返回, 直接
        ``json.loads`` 会失败; 这里在直接解析失败时, 退而抽取文本中第一个配平的 JSON
        对象, 提升对这类模型的兼容性。
        """
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

        data = _extract_json_object(text)
        if data is None:
            raise ValueError("无法从 LLM 输出中解析出 JSON 动作")
        return AgentAction(
            thought=data.get("thought", ""),
            tool=data.get("tool"),
            tool_input=data.get("tool_input") or {},
            final_answer=data.get("final_answer"),
        )
