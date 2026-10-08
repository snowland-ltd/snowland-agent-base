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

"""向后兼容门面: 构建 ChatModel 实例。"""
from __future__ import annotations

from typing import Optional

from langchain_core.language_models import BaseChatModel

from snowland_agent_base.llm.registry import create_llm_backend


def build_llm(
    backend: str = "dashscope",
    model: str = "qwen-plus",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.8,
    enable_thinking: bool = False,
    **kwargs,
) -> BaseChatModel:
    """构建一个 ChatModel 实例 (门面)。

    Args:
        backend: 后端名称或已注册别名, 默认 ``"dashscope"`` (亦可用 ``"bailian"``)。
        model: 百炼平台模型名, 如 ``qwen-plus`` / ``qwen-max`` / ``qwen-turbo``。
        api_key: DashScope API Key; 缺省时从环境变量 ``DASHSCOPE_API_KEY`` 读取。
        temperature: 采样温度 (0~1)。
        max_tokens: 单次回复最大 token 数。
        top_p: 核采样概率。
        enable_thinking: 是否开启 qwq 系列模型的思考模式。
        kwargs: 透传给具体后端构造器的其他参数。

    Returns:
        LangChain ``BaseChatModel`` 实例。
    """
    return create_llm_backend(
        backend,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        enable_thinking=enable_thinking,
        **kwargs,
    ).build()
