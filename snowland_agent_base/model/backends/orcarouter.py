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

基于 OrcaRouter (OpenAI 兼容路由网关) 的 ChatModel 后端, 复用
:class:`OpenAICompatLLMBackend` 基类。接入方式见 OrcaRouter 官方文档:
base_url 指向 ``https://api.orcarouter.ai/v1``, 鉴权用标准 OpenAI 头部
``Authorization: Bearer <ORCA_KEY>``, 任意请求/响应均为 OpenAI JSON 形状。

本模块同时提供 OrcaRouter 合作伙伴接入所需的工具:

- :data:`ORCAROUTER_REF`: 本仓库的推广码 (归因用);
- :func:`build_orcarouter_auth_url` / :func:`exchange_orcarouter_code`:
  OAuth (PKCE) 流程, 用于把你的应用端用户引导到 OrcaRouter 注册/登录,
  新注册用户自动归因到 :data:`ORCAROUTER_REF`。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import urllib.request
from typing import Any, Dict, Optional
from urllib.parse import urlencode

from snowland_agent_base.model.backends.openai_compat import (
    OpenAICompatEmbeddingBackend,
    OpenAICompatImageBackend,
    OpenAICompatLLMBackend,
    OpenAICompatRerankBackend,
    OpenAICompatTTSBackend,
    OpenAICompatVideoBackend,
)

#: 合作伙伴推广码 (归因用). 见 OrcaRouter 合作伙伴仪表板.
ORCAROUTER_REF = "ref_808b7e19b052dee31192"
#: OAuth 授权页地址.
ORCAROUTER_AUTH_URL = "https://www.orcarouter.ai/auth"
#: OAuth 换 key 地址 (PKCE 第二步).
ORCAROUTER_TOKEN_URL = "https://api.orcarouter.ai/api/v1/auth/keys"

#: 优先读取的 API Key 环境变量 (对齐 OrcaRouter provider 配置 ``env_key``).
ORCA_KEY_ENV = "ORCA_KEY"
#: 兼容旧名.
ORCAROUTER_API_KEY_ENV = "ORCAROUTER_API_KEY"


class OrcaRouterLLMBackend(OpenAICompatLLMBackend):
    """基于 OrcaRouter (OpenAI 兼容路由网关) 的 ChatModel 后端。

    典型模型名:

    - ``orcarouter/auto``: 智能路由, 自动选最便宜的在线模型 (默认);
    - ``orcarouter/fusion-mini``: 命名路由器 (fusion 系列);
    - ``orcarouter/free``: 免费推理模型 (答案可能放在 ``reasoning_content``,
      已由 :class:`~snowland_agent_base.model.backends.openai_compat.ReasoningChatOpenAI`
      兼容处理)。

    可通过 ``extra_body={"models": [...]}`` (透传 kwargs) 设置 fallback 链。

    API Key 读取优先级: 构造参数 ``api_key`` > 环境变量 ``ORCA_KEY`` >
    环境变量 ``ORCAROUTER_API_KEY``。
    """

    default_base_url = "https://api.orcarouter.ai/v1"
    api_key_env = ORCA_KEY_ENV
    base_url_env = "ORCAROUTER_BASE_URL"
    default_model = "orcarouter/auto"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.8,
        **kwargs: Any,
    ) -> None:
        resolved_key = (
            api_key
            or os.getenv(ORCA_KEY_ENV)
            or os.getenv(ORCAROUTER_API_KEY_ENV)
        )
        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )


class _OrcaRouterBackendMixin:
    """OrcaRouter OpenAI 兼容后端的共享配置。

    统一复用 OrcaRouter 网关地址 ``https://api.orcarouter.ai/v1`` 与 API Key
    解析 (``ORCA_KEY`` 优先, 回退 ``ORCAROUTER_API_KEY``)。
    """

    #: OrcaRouter OpenAI 兼容接口地址。
    default_base_url = "https://api.orcarouter.ai/v1"
    #: API Key 环境变量 (对齐 OrcaRouter provider 配置 ``env_key``)。
    api_key_env = ORCA_KEY_ENV
    #: base_url 环境变量。
    base_url_env = "ORCAROUTER_BASE_URL"

    def _orca_key(self, api_key: Optional[str]) -> Optional[str]:
        """解析 OrcaRouter API Key: 构造参数 > ``ORCA_KEY`` > ``ORCAROUTER_API_KEY``。"""
        return api_key or os.getenv(ORCA_KEY_ENV) or os.getenv(ORCAROUTER_API_KEY_ENV)


class OrcaRouterEmbeddingBackend(_OrcaRouterBackendMixin, OpenAICompatEmbeddingBackend):
    """OrcaRouter 向量后端 (OpenAI 兼容 ``/embeddings``)。"""

    default_model = "text-embedding-3-small"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=self._orca_key(api_key),
            base_url=base_url,
            **kwargs,
        )


class OrcaRouterImageBackend(_OrcaRouterBackendMixin, OpenAICompatImageBackend):
    """OrcaRouter 图像生成后端 (OpenAI 兼容 ``/images/generations``)。"""

    default_model = "dall-e-3"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        n: int = 1,
        size: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=self._orca_key(api_key),
            base_url=base_url,
            n=n,
            size=size,
            **kwargs,
        )


class OrcaRouterVideoBackend(_OrcaRouterBackendMixin, OpenAICompatVideoBackend):
    """OrcaRouter 视频生成后端 (OpenAI 兼容 ``/videos/generations``)。"""

    default_model = "video"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=self._orca_key(api_key),
            base_url=base_url,
            **kwargs,
        )


class OrcaRouterTTSBackend(_OrcaRouterBackendMixin, OpenAICompatTTSBackend):
    """OrcaRouter 语音合成后端 (OpenAI 兼容 ``/audio/speech``)。"""

    default_model = "tts-1"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        voice: str = "alloy",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=self._orca_key(api_key),
            base_url=base_url,
            voice=voice,
            **kwargs,
        )


class OrcaRouterRerankBackend(_OrcaRouterBackendMixin, OpenAICompatRerankBackend):
    """OrcaRouter 语义重排后端 (OpenAI 兼容 ``/rerank``)。"""

    default_model = "rerank"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        top_n: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=self._orca_key(api_key),
            base_url=base_url,
            top_n=top_n,
            **kwargs,
        )


def build_orcarouter_auth_url(
    *,
    app_name: str,
    port: int,
    code_verifier: str,
    state: Optional[str] = None,
    callback_path: str = "/cb",
    ref: str = ORCAROUTER_REF,
) -> "tuple[str, str]":
    """构造 OrcaRouter OAuth (PKCE) 授权 URL。

    对应合作伙伴文档「连接你的应用」第 1 步。调用方需在本地起一个回环
    (loopback) 监听器 (端口即 ``port``, RFC 8252 §7.3 仅允许端口变动),
    捕获重定向后用 :func:`exchange_orcarouter_code` 换 key。

    Args:
        app_name: 你的应用名 (显示在授权页)。
        port: 回环监听端口。
        code_verifier: 你自行生成的随机串, 需保存以在换 key 时配对。
        state: 防 CSRF 的随机值; 缺省自动生成。回调返回的 state 必须与之相等。
        callback_path: 回环回调路径, 默认 ``/cb``。
        ref: 推广码, 默认本仓库的 :data:`ORCAROUTER_REF`。

    Returns:
        ``(auth_url, state)`` —— state 需用于比对回调, 防止 CSRF。
    """
    code_challenge = (
        base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode("utf-8")).digest()
        )
        .rstrip(b"=")
        .decode("ascii")
    )
    state = state or secrets.token_urlsafe(16)
    callback_url = f"http://127.0.0.1:{port}{callback_path}"
    params = {
        "callback_url": callback_url,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "app_name": app_name,
        "ref": ref,
    }
    return f"{ORCAROUTER_AUTH_URL}?{urlencode(params)}", state


def exchange_orcarouter_code(*, code: str, code_verifier: str) -> Dict[str, Any]:
    """用授权码 + code_verifier 换取 OrcaRouter API Key (PKCE 第二步)。

    对应合作伙伴文档「连接你的应用」第 2 步。**调用前必须先校验回调返回的
    ``state`` 与 :func:`build_orcarouter_auth_url` 生成的一致** (那是你的 CSRF
    防护), 校验通过再调用本函数。

    Args:
        code: 重定向回调里携带的 ``code``。
        code_verifier: 你之前生成并哈希的随机串 (原始值, 非 challenge)。

    Returns:
        解析后的 JSON 响应 (含 API Key)。
    """
    payload = json.dumps(
        {"code": code, "code_verifier": code_verifier}
    ).encode("utf-8")
    request = urllib.request.Request(
        ORCAROUTER_TOKEN_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))
