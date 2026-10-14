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

"""内置模型后端实现集合。

每个「大脑」(厂商后端) 单独成文件, 通过
:mod:`snowland_agent_base.model.registry` 注册后即可被名称字符串解析。

具体模型实现类 (``*Model``) 同样在此导出, 方便调用方按需引用 / 继承 / 测试。
"""
from __future__ import annotations

from snowland_agent_base.model.backends.dashscope import (
    DashScopeEmbeddingBackend,
    DashScopeImageBackend,
    DashScopeImageModel,
    DashScopeLLMBackend,
    DashScopeRerankBackend,
    DashScopeRerankModel,
    DashScopeTTSBackend,
    DashScopeTTSModel,
    DashScopeVideoBackend,
    DashScopeVideoModel,
)
from snowland_agent_base.model.backends.deepseek import DeepSeekLLMBackend
from snowland_agent_base.model.backends.orcarouter import (
    ORCAROUTER_REF,
    OrcaRouterLLMBackend,
    build_orcarouter_auth_url,
    exchange_orcarouter_code,
)
from snowland_agent_base.model.backends.zhipu import (
    ZhipuLLMBackend,
    ZhipuVideoBackend,
    ZhipuVideoModel,
)
from snowland_agent_base.model.backends.anthropic import AnthropicLLMBackend
from snowland_agent_base.model.backends.anthropic_openai import (
    AnthropicOpenAILLMBackend,
)
from snowland_agent_base.model.backends.openai_compat import (
    OpenAICompatEmbeddingBackend,
    OpenAICompatImageBackend,
    OpenAICompatImageModel,
    OpenAICompatLLMBackend,
    OpenAICompatRerankBackend,
    OpenAICompatRerankModel,
    OpenAICompatTTSBackend,
    OpenAICompatTTSModel,
    OpenAICompatVideoBackend,
    OpenAICompatVideoModel,
)


__all__ = [
    "DashScopeLLMBackend",
    "DashScopeEmbeddingBackend",
    "DashScopeImageBackend",
    "DashScopeImageModel",
    "DashScopeVideoBackend",
    "DashScopeVideoModel",
    "DashScopeTTSBackend",
    "DashScopeTTSModel",
    "DashScopeRerankBackend",
    "DashScopeRerankModel",
    "OrcaRouterLLMBackend",
    "ZhipuLLMBackend",
    "ZhipuVideoBackend",
    "ZhipuVideoModel",
    "DeepSeekLLMBackend",
    "AnthropicLLMBackend",
    "AnthropicOpenAILLMBackend",
    "OpenAICompatLLMBackend",
    "OpenAICompatEmbeddingBackend",
    "OpenAICompatImageBackend",
    "OpenAICompatImageModel",
    "OpenAICompatVideoBackend",
    "OpenAICompatVideoModel",
    "OpenAICompatTTSBackend",
    "OpenAICompatTTSModel",
    "OpenAICompatRerankBackend",
    "OpenAICompatRerankModel",
    "ORCAROUTER_REF",
    "build_orcarouter_auth_url",
    "exchange_orcarouter_code",
]
