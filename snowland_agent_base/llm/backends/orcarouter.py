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

"""OrcaRouter LLM 后端实现。

基于 OrcaRouter (OpenAI 兼容路由服务) 的 ChatModel 后端, 复用
:class:`OpenAICompatLLMBackend` 基类。
"""
from __future__ import annotations

from snowland_agent_base.llm.backends.openai_compat import (
    OpenAICompatLLMBackend,
)


class OrcaRouterLLMBackend(OpenAICompatLLMBackend):
    """基于 OrcaRouter (OpenAI 兼容路由服务) 的 ChatModel 后端。"""

    default_base_url = ""
    api_key_env = "ORCAROUTER_API_KEY"
    base_url_env = "ORCAROUTER_BASE_URL"
    default_model = "orca-router"
