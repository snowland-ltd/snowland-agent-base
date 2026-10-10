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

"""LLM 后端抽象接口。

把「构建 ChatModel」抽象为统一后端接口 :class:`LLMBackend`, 不同厂商可热插拔。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

from snowland_agent_base.messages import OpenAIMessage


class LLMBackend(ABC):
    """大模型后端统一接口: 产出 LangChain ``BaseChatModel``。

    该基类是「backend 派生出两种接口」的抽象根: 不同厂商后端 (OpenAI 兼容 /
    Anthropic 原生) 继承自它, 并实现**相同的方法** (``build`` 与 ``message_class``),
    调用方只依赖抽象, 不感知具体厂商, 从而通用。

    - ``build``: 构建 ChatModel 实例 (各后端实现不同);
    - ``message_class``: 该后端配套的消息风格类 (``OpenAIMessage`` /
      ``AnthropicMessage``), 子类覆写以绑定自身协议; 默认 ``OpenAIMessage``
      (多数厂商走 OpenAI 兼容协议)。
    """

    #: 该后端配套的消息风格类; 子类按自身协议覆写。
    message_class = OpenAIMessage

    @abstractmethod
    def build(self) -> BaseChatModel:
        """构建并返回一个 ``BaseChatModel`` 实例。"""
