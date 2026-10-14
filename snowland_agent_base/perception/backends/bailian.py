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

"""阿里云百炼 (DashScope) 感知后端实现。

基于大模型的实体识别与意图分类, 适合 spaCy 未覆盖的语言 / 领域。
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from langchain_core.language_models import BaseChatModel

from snowland_agent_base.perception.base import PerceptionBackend


_BAILIAN_PERCEIVE_TEMPLATE = """你是一个文本感知(NLU)助手。请分析用户输入, 完成实体识别与意图分类。

只输出一个 JSON 对象, 不要包含任何额外说明, 也不要使用 Markdown 代码块标记, 格式如下:
{{
  "entities": [{{"text": "实体文本", "label": "实体类型"}}],
  "intent": "意图类别"
}}

意图类别从以下候选中选择(若都不匹配则填 "unknown"):
{intent_candidates}

用户输入:
{text}"""

_BAILIAN_INTENT_TEMPLATE = """请判断以下用户输入的意图类别。只回复类别名称本身, 不要输出 JSON, 不要输出任何解释。

候选类别:
{intent_candidates}

用户输入:
{text}"""


class BailianPerceptionBackend(PerceptionBackend):
    """基于阿里云百炼 (DashScope) 大模型的感知后端。

    通过 LLM 完成实体识别与意图分类, 适合 spaCy 未覆盖的语言 / 领域。
    ``llm`` 缺省时惰性调用 :func:`snowland_agent_base.model.build_llm` 构建
    (需设置 ``DASHSCOPE_API_KEY`` 环境变量或传入 ``api_key``)。
    """

    DEFAULT_INTENT_CANDIDATES: List[str] = [
        "greeting",
        "question",
        "search",
        "execute",
        "farewell",
        "unknown",
    ]

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        model: Optional[str] = None,
        intent_candidates: Optional[List[str]] = None,
    ) -> None:
        """初始化百炼感知后端。

        Args:
            llm: 已构建的 LangChain ChatModel; 缺省惰性构建 DashScope 模型。
            model: 百炼模型名 (如 ``qwen-plus`` / ``qwen-max``), 仅 ``llm`` 缺省时生效。
            intent_candidates: 意图候选类别, 约束模型输出范围。
        """
        self._llm = llm
        self._model = model
        self._intent_candidates = list(
            intent_candidates or self.DEFAULT_INTENT_CANDIDATES
        )

    @property
    def llm(self) -> BaseChatModel:
        """惰性构建 DashScope ChatModel。"""
        if self._llm is None:
            from snowland_agent_base.model import build_llm

            self._llm = (
                build_llm(model=self._model) if self._model else build_llm()
            )
        return self._llm

    @property
    def intent_candidates(self) -> List[str]:
        return self._intent_candidates

    # -- 内部工具 --------------------------------------------------------- #
    def _ask(self, prompt: str) -> str:
        from langchain_core.messages import HumanMessage

        resp = self.llm.invoke([HumanMessage(content=prompt)])
        return getattr(resp, "content", "") or ""

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, object]]:
        """从模型回复中提取首个 JSON 对象, 容错 fenced / 多余说明文本。"""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:  # pragma: no cover - 模型异常输出
            return None

    def _normalize_intent(self, raw: str, default: str = "unknown") -> str:
        """把模型输出的意图文本归一化到候选类别。"""
        first = (raw or "").strip().splitlines()[0] if (raw or "").strip() else ""
        first = first.strip("`\"'{} ").strip()
        candidates_lower = {c.lower(): c for c in self._intent_candidates}
        if first.lower() in candidates_lower:
            return candidates_lower[first.lower()]
        for cand in self._intent_candidates:
            if cand.lower() in first.lower():
                return cand
        return default

    # -- 接口实现 --------------------------------------------------------- #
    def perceive(self, text: str) -> Dict[str, object]:
        """调用大模型一次性得到实体与意图。"""
        prompt = _BAILIAN_PERCEIVE_TEMPLATE.format(
            text=text, intent_candidates=", ".join(self._intent_candidates)
        )
        data = self._extract_json(self._ask(prompt)) or {}
        entities = [
            {"text": e.get("text", ""), "label": e.get("label", "")}
            for e in data.get("entities", [])
            if isinstance(e, dict)
        ]
        intent = self._normalize_intent(
            str(data.get("intent", "")), default="unknown"
        )
        return {"text": text, "entities": entities, "intent": intent}

    def classify_intent(self, text: str, default: str = "unknown") -> str:
        """调用大模型做意图分类。"""
        prompt = _BAILIAN_INTENT_TEMPLATE.format(
            text=text, intent_candidates=", ".join(self._intent_candidates)
        )
        return self._normalize_intent(self._ask(prompt), default=default)

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """调用大模型做实体识别。"""
        return self.perceive(text)["entities"]
