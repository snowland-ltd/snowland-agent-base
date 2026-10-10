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

"""OpenAI 兼容 LLM 后端基类。

OrcaRouter / 智谱 / DeepSeek 等多数厂商提供 OpenAI 兼容的
``/v1/chat/completions`` 接口, 统一用 ``langchain_openai.ChatOpenAI`` 指向其
``base_url`` 封装。子类仅需覆盖类属性 ``default_base_url`` / ``api_key_env`` /
``base_url_env`` / ``default_model`` 即可成为一个可插拔后端。
"""
from __future__ import annotations

import os
from typing import Dict, Optional

from langchain_core.language_models import BaseChatModel

from snowland_agent_base._compat import ensure_pydantic_v1_shim
from snowland_agent_base.llm.base import LLMBackend
from snowland_agent_base.messages import OpenAIMessage


class OpenAICompatLLMBackend(LLMBackend):
    """OpenAI 兼容 ChatModel 后端基类。"""

    #: 该后端使用的消息风格 (OpenAI 兼容协议 -> OpenAIMessage)。
    message_class = OpenAIMessage

    #: 厂商 OpenAI 兼容接口地址 (缺省回退 base_url_env)。
    default_base_url: str = ""
    #: API Key 环境变量名。
    api_key_env: str = "OPENAI_API_KEY"
    #: base_url 环境变量名 (覆盖 default_base_url)。
    base_url_env: str = "OPENAI_BASE_URL"
    #: 默认模型名。
    default_model: str = "gpt-4o"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.8,
        **kwargs,
    ) -> None:
        """初始化 OpenAI 兼容后端。

        Args:
            model: 模型名; 缺省取 ``default_model``。
            api_key: API Key; 缺省从 ``api_key_env`` 读取。
            base_url: 服务地址; 缺省从 ``default_base_url`` / ``base_url_env`` 读取。
            temperature: 采样温度 (0~1)。
            max_tokens: 单次回复最大 token 数。
            top_p: 核采样概率。
            kwargs: 透传给 LangChain ``ChatOpenAI`` 的其他参数。
        """
        resolved_base_url = (
            base_url
            or os.getenv(self.base_url_env)
            or self.default_base_url
        )
        self._params: Dict[str, object] = {
            "model": model or self.default_model,
            "api_key": api_key or os.getenv(self.api_key_env),
            "base_url": resolved_base_url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            **kwargs,
        }

    def build(self) -> BaseChatModel:
        if not self._params.get("base_url"):
            raise ValueError(
                f"{type(self).__name__} base_url 未提供: 请传入 base_url "
                f"或设置环境变量 {self.base_url_env}"
            )
        # 共享环境 langchain-core 1.x 已移除 pydantic_v1;
        # 用 pydantic.v1 注入 shim, 避免 langchain_openai 运行期导入失败。
        ensure_pydantic_v1_shim()
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**self._params)
