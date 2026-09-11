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

此前历史会话以拼接的字符串传递, 存在两个问题:
    1. 无法区分角色 (用户输入 / 助手回复 / 工具观测), 模型难以利用对话结构;
    2. 输出只有一段文本, 拿不到结构化轨迹, 不便于持久化和前端渲染。

故统一改用 :data:`Message` 列表:
    - **历史会话** (``history``) 为 ``List[Message]``;
    - **输出** 除 ``answer`` 文本外, 还提供完整的 ``messages`` 列表。

:data:`Message` 为 ``TypedDict`` (即普通 dict), 便于 LangGraph state 序列化。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

try:  # pragma: no cover - TypedDict 在 3.8+ 的 typing 中可用
    from typing import TypedDict
except ImportError:  # pragma: no cover
    from typing_extensions import TypedDict


#: 角色常量
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"


class Message(TypedDict, total=False):
    """一条对话消息。

    Attributes:
        role: 角色, 取值 :data:`ROLE_SYSTEM` / ``ROLE_USER`` / ``ROLE_ASSISTANT`` /
            :data:`ROLE_TOOL`。
        content: 消息文本。
        name: 可选; role 为 ``tool`` 时为工具名。
    """

    role: str
    content: str
    name: str


def system_message(content: str) -> Message:
    """构造一条系统消息。"""
    return {"role": ROLE_SYSTEM, "content": content}


def user_message(content: str) -> Message:
    """构造一条用户消息。"""
    return {"role": ROLE_USER, "content": content}


def assistant_message(content: str) -> Message:
    """构造一条助手消息。"""
    return {"role": ROLE_ASSISTANT, "content": content}


def tool_message(content: str, name: str = "") -> Message:
    """构造一条工具观测消息。

    Args:
        content: 工具返回内容的文本化结果。
        name: 工具名。
    """
    message: Message = {"role": ROLE_TOOL, "content": content}
    if name:
        message["name"] = name
    return message


def ensure_messages(history: Union[str, Sequence[Message], None]) -> List[Message]:
    """把历史参数统一为 :data:`Message` 列表。

    兼容旧的字符串用法: 传入 ``str`` 时包装为单条系统消息, 空串则为空列表。

    Args:
        history: 消息列表、字符串或 ``None``。

    Returns:
        :data:`Message` 列表。
    """
    if history is None:
        return []
    if isinstance(history, str):
        return [system_message(history)] if history else []
    return [dict(m) for m in history]  # type: ignore[arg-type]


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
        content = str(msg.get("content", ""))
        if not content:
            continue
        if include_role:
            prefix = msg.get("role", ROLE_USER)
            name = msg.get("name")
            if name:
                prefix = f"{prefix}({name})"
            lines.append(f"{prefix}: {content}")
        else:
            lines.append(content)
    return "\n".join(lines)


def to_langchain_messages(messages: Sequence[Message]) -> List[Any]:
    """把 :data:`Message` 列表转为 LangChain 消息对象, 作为多轮对话上下文传入。

    角色映射:
        - ``system`` -> ``SystemMessage``
        - ``user`` -> ``HumanMessage``
        - ``assistant`` -> ``AIMessage``
        - ``tool`` -> ``SystemMessage`` (观测作为上下文注入; 未使用 bind_tools,
          故没有 tool_call_id, 不适合用 ToolMessage)

    Args:
        messages: 消息列表。

    Returns:
        LangChain 消息对象列表。
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    converted: List[Any] = []
    for msg in messages or []:
        role = msg.get("role", ROLE_USER)
        content = str(msg.get("content", ""))
        if not content:
            continue
        if role == ROLE_USER:
            converted.append(HumanMessage(content=content))
        elif role == ROLE_ASSISTANT:
            converted.append(AIMessage(content=content))
        else:  # system / tool 均作为系统侧上下文
            name = msg.get("name")
            text = f"[工具 {name}] {content}" if role == ROLE_TOOL and name else content
            converted.append(SystemMessage(content=text))
    return converted
