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

"""智谱 AI (BigModel / GLM) LLM 后端实现。

基于智谱 AI 的 OpenAI 兼容接口 (``/v1/chat/completions``) 封装。
官方文档: https://open.bigmodel.cn/dev/api/normal-session/chat
"""
from __future__ import annotations

from snowland_agent_base.llm.backends.openai_compat import (
    OpenAICompatLLMBackend,
)


class ZhipuLLMBackend(OpenAICompatLLMBackend):
    """基于智谱 AI (BigModel / GLM) OpenAI 兼容接口的 ChatModel 后端。"""

    default_base_url = "https://open.bigmodel.cn/api/paas/v4/"
    api_key_env = "ZHIPU_API_KEY"
    base_url_env = "ZHIPU_BASE_URL"
    default_model = "glm-4"
