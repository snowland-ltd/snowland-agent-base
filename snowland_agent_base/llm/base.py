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


class LLMBackend(ABC):
    """大模型后端统一接口: 产出 LangChain ``BaseChatModel``。"""

    @abstractmethod
    def build(self) -> BaseChatModel:
        """构建并返回一个 ``BaseChatModel`` 实例。"""
