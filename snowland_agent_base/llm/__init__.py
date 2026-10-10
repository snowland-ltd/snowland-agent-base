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

- :class:`DashScopeLLMBackend`: 基于阿里云百炼 (DashScope) / 通义千问 (Qwen) 服务 (默认)。
- :class:`OrcaRouterLLMBackend`: 基于 OrcaRouter (OpenAI 兼容路由服务)。
- :class:`ZhipuLLMBackend`: 基于智谱 AI (BigModel / GLM) OpenAI 兼容接口。
- :class:`DeepSeekLLMBackend`: 基于 DeepSeek OpenAI 兼容接口。
- :class:`AnthropicLLMBackend`: 基于 Anthropic 原生 Messages API (langchain_anthropic)。
- :class:`AnthropicOpenAILLMBackend`: 基于 Anthropic OpenAI 兼容接口
  (langchain_openai, 复用 OpenAI 兼容基类)。

:func:`build_llm` 是向后兼容的门面, 通过 ``backend`` 选择后端::

    build_llm()                                              # 默认百炼
    build_llm(backend="dashscope", model="qwen-max")
    build_llm(backend="qwen", model="qwen-max")              # 通义千问 (别名)
    build_llm(backend="orcarouter", base_url="http://orcarouter:8000/v1", model="orca-router")
    build_llm(backend="zhipu", api_key="...")                # 智谱 GLM
    build_llm(backend="deepseek", api_key="...")             # DeepSeek
    build_llm(backend=MyLLMBackend(...))                     # 直接注入实例

新增后端只需继承 :class:`LLMBackend` (OpenAI 兼容厂商继承
:class:`OpenAICompatLLMBackend` 即可) 并 ``register_llm_backend`` 注册,
即可被名称字符串解析, 实现真正的插拔式扩展。
"""
from __future__ import annotations

from snowland_agent_base.llm.base import LLMBackend
from snowland_agent_base.llm.backends import (
    DashScopeLLMBackend,
    DeepSeekLLMBackend,
    OrcaRouterLLMBackend,
    ZhipuLLMBackend,
    AnthropicLLMBackend,
    AnthropicOpenAILLMBackend,
)
from snowland_agent_base.llm.factory import build_llm
from snowland_agent_base.llm.registry import (
    create_llm_backend,
    get_llm_backend,
    get_message_class,
    list_llm_backends,
    register_llm_backend,
)


__all__ = [
    "LLMBackend",
    "DashScopeLLMBackend",
    "OrcaRouterLLMBackend",
    "ZhipuLLMBackend",
    "DeepSeekLLMBackend",
    "AnthropicLLMBackend",
    "AnthropicOpenAILLMBackend",
    "build_llm",
    "register_llm_backend",
    "get_llm_backend",
    "list_llm_backends",
    "create_llm_backend",
    "get_message_class",
]
