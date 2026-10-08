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

"""可插拔大模型接入层。

把「构建 ChatModel」抽象为统一后端接口 :class:`LLMBackend`, 不同厂商可热插拔:

- :class:`DashScopeLLMBackend`: 基于阿里云百炼 (DashScope) 服务 (默认)。

:func:`build_llm` 是向后兼容的门面, 通过 ``backend`` 选择后端::

    build_llm()                                   # 默认百炼
    build_llm(backend="dashscope", model="qwen-max")
    build_llm(backend=MyLLMBackend(...))          # 直接注入实例

新增后端只需继承 :class:`LLMBackend` 并 ``register_llm_backend`` 注册,
即可被名称字符串解析, 实现真正的插拔式扩展。
"""
from __future__ import annotations

from snowland_agent_base.llm.base import LLMBackend
from snowland_agent_base.llm.backends import DashScopeLLMBackend
from snowland_agent_base.llm.factory import build_llm
from snowland_agent_base.llm.registry import (
    create_llm_backend,
    get_llm_backend,
    list_llm_backends,
    register_llm_backend,
)


__all__ = [
    "LLMBackend",
    "DashScopeLLMBackend",
    "build_llm",
    "register_llm_backend",
    "get_llm_backend",
    "list_llm_backends",
    "create_llm_backend",
]
