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

"""内置 LLM 后端实现集合。

每个「大脑」(厂商后端) 单独成文件, 通过 :mod:`snowland_agent_base.llm.registry`
注册后即可被名称字符串解析。
"""
from __future__ import annotations

from snowland_agent_base.llm.backends.dashscope import DashScopeLLMBackend
from snowland_agent_base.llm.backends.deepseek import DeepSeekLLMBackend
from snowland_agent_base.llm.backends.orcarouter import (
    ORCAROUTER_REF,
    OrcaRouterLLMBackend,
    build_orcarouter_auth_url,
    exchange_orcarouter_code,
)
from snowland_agent_base.llm.backends.zhipu import ZhipuLLMBackend
from snowland_agent_base.llm.backends.anthropic import AnthropicLLMBackend
from snowland_agent_base.llm.backends.anthropic_openai import (
    AnthropicOpenAILLMBackend,
)


__all__ = [
    "DashScopeLLMBackend",
    "OrcaRouterLLMBackend",
    "ZhipuLLMBackend",
    "DeepSeekLLMBackend",
    "AnthropicLLMBackend",
    "AnthropicOpenAILLMBackend",
    "ORCAROUTER_REF",
    "build_orcarouter_auth_url",
    "exchange_orcarouter_code",
]
