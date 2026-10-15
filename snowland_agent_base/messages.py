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

"""消息模型: 历史会话与 Agent 输出的统一结构。

设计: ``Message`` 为抽象基类, 持有 ``role`` / ``content`` / ``tool_call_id`` /
``tool_calls`` 等通用字段; 具体厂商格式 (OpenAI / Anthropic) 由子类
:class:`OpenAIMessage` / :class:`AnthropicMessage` 在 :meth:`Message.to_langchain_messages`
中实现, 二者提供**相同的方法**, 调用方只依赖抽象接口, 不感知底层协议差异, 从而通用。

此前历史会话以拼接的字符串传递, 存在两个问题:
    1. 无法区分角色 (用户输入 / 助手回复 / 工具观测), 模型难以利用对话结构;
    2. 输出只有一段文本, 拿不到结构化轨迹, 不便于持久化和前端渲染。

故统一改用 :data:`Message` 列表:
    - **历史会话** (``history``) 为 ``List[Message]``;
    - **输出** 除 ``answer`` 文本外, 还提供完整的 ``messages`` 列表。

:class:`Message` 同时提供 :meth:`Message.to_dict` / :meth:`Message.from_dict`,
便于需要 JSON 序列化的场景 (如持久化 checkpointer)。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union


#: 角色常量
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"


@dataclass
class Message(ABC):
    """消息统一抽象: 持有通用字段, 子类负责转换为具体厂商格式。

    子类只需实现 :meth:`to_langchain_messages`; 二者方法签名一致,
    因此决策层 / Agent 层无需感知当前是 OpenAI 还是 Anthropic 后端。

    采用 ``@dataclass`` 是为了让 LangGraph 的 checkpointer (msgpack 序列化) 能
    自动 (反)序列化消息对象: ``langgraph`` 对 dataclass 有原生编解码分支, 会按
    字段存 ``(__module__, __name__, {字段})`` 并在读取时 ``cls(**字段)`` 重建。
    若不如此, 自定义 ``Message`` 无法被 msgpack 序列化, 会导致图状态持久化时报
    ``TypeError: Type is not msgpack serializable``。
    """

    #: 角色 (system / user / assistant / tool)
    role: str
    #: 文本内容
    content: str
    #: 可选: 工具名 / 发言人名
    name: Optional[str] = None
    #: 可选: 工具调用 id (配合 tool_calls 使用)
    tool_call_id: Optional[str] = None
    #: 可选: 工具调用列表 (Anthropic 原生 tool use)
    tool_calls: Optional[List[Dict[str, Any]]] = None

    @abstractmethod
    def to_langchain_messages(self) -> List[Any]:
        """转换为 LangChain 消息对象列表 (厂商格式由子类决定)。"""

    # ------------------------------------------------------------------ #
    # 通用辅助: 序列化 / 比较 (便于持久化与测试)
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """导出为普通 dict, 用于 JSON 序列化。"""
        data: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            data["name"] = self.name
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            data["tool_calls"] = self.tool_calls
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从 dict 还原为消息对象 (缺省还原为 :class:`OpenAIMessage`)。

        子类可覆盖以保留自身厂商语义。
        """
        return OpenAIMessage(
            role=data.get("role", ROLE_USER),
            content=str(data.get("content", "")),
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
            tool_calls=data.get("tool_calls"),
        )


class OpenAIMessage(Message):
    """OpenAI 风格消息。

    与 OpenAI 兼容端点 (``/v1/chat/completions``) 对应的转换:
        - ``system`` -> ``SystemMessage``
        - ``user`` -> ``HumanMessage``
        - ``assistant`` -> ``AIMessage``
        - ``tool`` -> 作为系统上下文注入 ``SystemMessage`` (观测文本化; 该写法已弃用,
          仅作回退; 真正的原生 tool use 由 :class:`AnthropicMessage` 承担)

    未启用 ``bind_tools`` 时, ``tool_call_id`` / ``tool_calls`` 字段被忽略,
    以保证纯文本 CoT 决策流行为不变。
    """

    def to_langchain_messages(self) -> List[Any]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        role = self.role
        content = str(self.content or "")
        if not content and role not in (ROLE_TOOL, ROLE_ASSISTANT):
            return []
        if role == ROLE_USER:
            return [HumanMessage(content=content)]
        if role == ROLE_ASSISTANT:
            return [AIMessage(content=content)]
        if role == ROLE_TOOL:
            # 弃用回退: 旧写法把工具观测当系统上下文注入
            text = f"[工具 {self.name}] {content}" if self.name else content
            return [SystemMessage(content=text)]
        # system
        return [SystemMessage(content=content)]


class AnthropicMessage(Message):
    """Anthropic 原生 Messages API 风格消息。

    与 ``langchain_anthropic.ChatAnthropic`` 对应的转换 (``system`` 由库自动提至顶层,
    ``content`` 由库自动包成内容块数组), 这里只需产出正确的 LangChain 消息:
        - ``tool`` + ``tool_call_id`` -> ``ToolMessage``
        - ``assistant`` + ``tool_calls`` -> 带 ``tool_calls`` 的 ``AIMessage``

    仅当消息确实携带上述字段才走原生路径, 否则回退到旧写法, 行为不变。
    """

    def to_langchain_messages(self) -> List[Any]:
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        role = self.role
        content = str(self.content or "")
        if not content and role not in (ROLE_TOOL, ROLE_ASSISTANT):
            return []
        if role == ROLE_USER:
            return [HumanMessage(content=content)]
        if role == ROLE_ASSISTANT:
            if self.tool_calls:
                lc_calls = [
                    {
                        "id": tc.get("id", ""),
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {}) or {},
                        "type": "tool_call",
                    }
                    for tc in self.tool_calls
                ]
                return [AIMessage(content=content, tool_calls=lc_calls)]
            return [AIMessage(content=content)]
        if role == ROLE_TOOL:
            if self.tool_call_id:
                return [
                    ToolMessage(
                        content=content,
                        tool_call_id=self.tool_call_id,
                        name=self.name or None,
                    )
                ]
            # 回退: 无 tool_call_id 时当作系统上下文注入
            text = f"[工具 {self.name}] {content}" if self.name else content
            return [SystemMessage(content=text)]
        # system
        return [SystemMessage(content=content)]


# ---------------------------------------------------------------------- #
# 构造器: 默认产出 OpenAIMessage, 可通过 ``cls`` 指定厂商风格
# ---------------------------------------------------------------------- #
def system_message(content: str, *, cls: type[Message] = OpenAIMessage) -> Message:
    """构造一条系统消息。"""
    return cls(ROLE_SYSTEM, content)


def user_message(content: str, *, cls: type[Message] = OpenAIMessage) -> Message:
    """构造一条用户消息。"""
    return cls(ROLE_USER, content)


def assistant_message(content: str, *, cls: type[Message] = OpenAIMessage) -> Message:
    """构造一条助手消息。"""
    return cls(ROLE_ASSISTANT, content)


def tool_message(
    content: str,
    name: str = "",
    tool_call_id: Optional[str] = None,
    *,
    cls: type[Message] = OpenAIMessage,
) -> Message:
    """构造一条工具观测消息。

    Args:
        content: 工具返回内容的文本化结果。
        name: 工具名。
        tool_call_id: 可选; 对应 assistant 发起的 tool_call 的 id。
            提供后, 在 :class:`AnthropicMessage` 下会转换为原生 ``ToolMessage``。
        cls: 消息风格类 (``OpenAIMessage`` / ``AnthropicMessage``), 默认 OpenAI 风格。
    """
    return cls(ROLE_TOOL, content, name=name or None, tool_call_id=tool_call_id)


def tool_result_message(
    content: str,
    tool_call_id: str,
    name: str = "",
    *,
    cls: type[Message] = OpenAIMessage,
) -> Message:
    """构造一条工具观测消息 (Anthropic 原生 tool use 专用, 必须携带 ``tool_call_id``)。

    等价于 ``tool_message(content, name=name, tool_call_id=tool_call_id)``,
    单独命名以明确语义。
    """
    return tool_message(content, name=name, tool_call_id=tool_call_id, cls=cls)


def assistant_tool_call_message(
    tool_calls: Sequence[Dict[str, Any]],
    content: str = "",
    *,
    cls: type[Message] = OpenAIMessage,
) -> Message:
    """构造一条携带 ``tool_calls`` 的助手消息 (Anthropic 原生 tool use 专用)。

    ``tool_calls`` 中每项形如 ``{"id": str, "name": str, "args": dict}``,
    转换层会补齐 ``type="tool_call"`` 以符合 LangChain 规范。

    Args:
        tool_calls: 模型发起的工具调用列表。
        content: 助手文本 (可与工具调用并存, 通常为空)。
        cls: 消息风格类, 默认 OpenAI 风格。
    """
    return cls(ROLE_ASSISTANT, content, tool_calls=list(tool_calls))


def ensure_messages(history: Union[str, Sequence[Message], None]) -> List[Message]:
    """把历史参数统一为 :data:`Message` 列表。

    兼容旧的字符串用法: 传入 ``str`` 时包装为单条系统消息, 空串则为空列表。
    列表中的元素若为 dict 则按 :class:`OpenAIMessage` 还原, 若为 ``Message``
    则原样保留; 始终返回副本, 避免调用方列表被外部修改污染。

    Args:
        history: 消息列表、字符串或 ``None``。

    Returns:
        :data:`Message` 列表。
    """
    if history is None:
        return []
    if isinstance(history, str):
        return [OpenAIMessage(ROLE_SYSTEM, history)] if history else []
    result: List[Message] = []
    for item in history:
        if isinstance(item, Message):
            result.append(item)
        elif isinstance(item, dict):
            result.append(OpenAIMessage.from_dict(item))
        else:
            # 兜底: 非预期类型当作用户文本
            result.append(OpenAIMessage(ROLE_USER, str(item)))
    return result


def render_messages(messages: Sequence[Message], include_role: bool = True) -> str:
    """把消息列表渲染为文本, 用于日志或需要纯文本摘要的场景。

    Args:
        messages: 消息列表。
        include_role: 是否带 ``role:`` 前缀。

    Returns:
        渲染后的文本 (空列表返回空串)。
    """
    lines: List[str] = []
    for msg in messages or []:
        content = str(msg.content or "")
        if not content:
            continue
        if include_role:
            prefix = msg.role
            name = msg.name
            if name:
                prefix = f"{prefix}({name})"
            lines.append(f"{prefix}: {content}")
        else:
            lines.append(content)
    return "\n".join(lines)


def to_langchain_messages(messages: Sequence[Message]) -> List[Any]:
    """把 :data:`Message` 列表转为 LangChain 消息对象, 作为多轮对话上下文传入。

    转换逻辑由每条消息自身的 :meth:`Message.to_langchain_messages` 决定
    (OpenAI / Anthropic 风格已在子类内实现), 本函数仅做扁平聚合,
    不再依赖 ``backend`` 字符串分派。

    Args:
        messages: 消息列表。

    Returns:
        LangChain 消息对象列表。
    """
    converted: List[Any] = []
    for msg in messages or []:
        converted.extend(msg.to_langchain_messages())
    return converted
