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
        tool_call_id: 可选; role 为 ``tool`` 时对应 assistant 发起的 tool_call 的 id
            (Anthropic 原生 tool use 需要, 用于把观测结果与调用配对)。
        tool_calls: 可选; role 为 ``assistant`` 时模型发起的工具调用列表,
            每项形如 ``{"id": str, "name": str, "args": dict}``。
    """

    role: str
    content: str
    name: str
    tool_call_id: str
    tool_calls: List[Dict[str, Any]]


def system_message(content: str) -> Message:
    """构造一条系统消息。"""
    return {"role": ROLE_SYSTEM, "content": content}


def user_message(content: str) -> Message:
    """构造一条用户消息。"""
    return {"role": ROLE_USER, "content": content}


def assistant_message(content: str) -> Message:
    """构造一条助手消息。"""
    return {"role": ROLE_ASSISTANT, "content": content}


def tool_message(
    content: str, name: str = "", tool_call_id: Optional[str] = None
) -> Message:
    """构造一条工具观测消息。

    Args:
        content: 工具返回内容的文本化结果。
        name: 工具名。
        tool_call_id: 可选; 对应 assistant 发起的 tool_call 的 id。
            提供后, 在 ``backend="anthropic"`` 下会转换为原生 ``ToolMessage``。
    """
    message: Message = {"role": ROLE_TOOL, "content": content}
    if name:
        message["name"] = name
    if tool_call_id:
        message["tool_call_id"] = tool_call_id
    return message


def tool_result_message(content: str, tool_call_id: str, name: str = "") -> Message:
    """构造一条工具观测消息 (Anthropic 原生 tool use 专用, 必须携带 ``tool_call_id``)。

    等价于 ``tool_message(content, name=name, tool_call_id=tool_call_id)``,
    单独命名以明确语义。
    """
    return tool_message(content, name=name, tool_call_id=tool_call_id)


def assistant_tool_call_message(
    tool_calls: Sequence[Dict[str, Any]], content: str = ""
) -> Message:
    """构造一条携带 ``tool_calls`` 的助手消息 (Anthropic 原生 tool use 专用)。

    ``tool_calls`` 中每项形如 ``{"id": str, "name": str, "args": dict}``,
    转换层会补齐 ``type="tool_call"`` 以符合 LangChain 规范。

    Args:
        tool_calls: 模型发起的工具调用列表。
        content: 助手文本 (可与工具调用并存, 通常为空)。
    """
    return {"role": ROLE_ASSISTANT, "content": content, "tool_calls": list(tool_calls)}


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


def _supports_native_tool_use(backend: Optional[str]) -> bool:
    """判断后端是否走 Anthropic 原生 Messages API (需配套 ``ToolMessage``)。"""
    return backend == "anthropic"


def to_langchain_messages(
    messages: Sequence[Message], backend: Optional[str] = None
) -> List[Any]:
    """把 :data:`Message` 列表转为 LangChain 消息对象, 作为多轮对话上下文传入。

    角色映射 (默认, 兼容旧行为):
        - ``system`` -> ``SystemMessage``
        - ``user`` -> ``HumanMessage``
        - ``assistant`` -> ``AIMessage``
        - ``tool`` -> ``SystemMessage`` (观测作为上下文注入; 该写法已弃用, 仅作回退)

    当 ``backend="anthropic"`` (原生 Messages API) 且消息自带 ``tool_call_id`` /
    ``tool_calls`` 时, 升级为真正的原生 tool use:
        - ``tool`` + ``tool_call_id`` -> ``ToolMessage``
        - ``assistant`` + ``tool_calls`` -> 带 ``tool_calls`` 的 ``AIMessage``

    注意: 仅当消息确实携带上述字段才走原生路径, 否则仍回退到旧写法,
    以保证未启用原生 tool use 时行为不变。

    Args:
        messages: 消息列表。
        backend: 后端名称; ``"anthropic"`` 启用原生 tool use 转换。

    Returns:
        LangChain 消息对象列表。
    """
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    native_tool = _supports_native_tool_use(backend)
    converted: List[Any] = []
    for msg in messages or []:
        role = msg.get("role", ROLE_USER)
        content = str(msg.get("content", ""))
        if not content and role not in (ROLE_TOOL, ROLE_ASSISTANT):
            continue
        if role == ROLE_USER:
            converted.append(HumanMessage(content=content))
        elif role == ROLE_ASSISTANT:
            tool_calls = msg.get("tool_calls")
            if native_tool and tool_calls:
                lc_calls = [
                    {
                        "id": tc.get("id", ""),
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {}) or {},
                        "type": "tool_call",
                    }
                    for tc in tool_calls
                ]
                converted.append(AIMessage(content=content, tool_calls=lc_calls))
            else:
                converted.append(AIMessage(content=content))
        elif role == ROLE_TOOL:
            tool_call_id = msg.get("tool_call_id")
            name = msg.get("name")
            if native_tool and tool_call_id:
                converted.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=tool_call_id,
                        name=name or None,
                    )
                )
            else:
                # 弃用回退: 旧写法把工具观测当系统上下文注入
                text = f"[工具 {name}] {content}" if name else content
                converted.append(SystemMessage(content=text))
        else:  # system
            converted.append(SystemMessage(content=content))
    return converted
