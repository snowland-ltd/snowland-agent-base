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

"""spaCy 感知后端实现。

基于 spaCy 的实体识别 + 规则关键词意图分类 (默认, 轻量)。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from snowland_agent_base.perception.base import PerceptionBackend


class SpacyPerceptionBackend(PerceptionBackend):
    """输入感知: spaCy 实体识别 + 规则关键词意图分类。

    意图分类采用「规则关键词匹配 + 可扩展打分」的轻量实现。
    如需更高精度, 可改用 :class:`BailianPerceptionBackend` 或自写后端。
    """

    #: 默认意图关键词表, 可通过 ``intent_keywords`` 覆盖或扩展
    DEFAULT_INTENT_KEYWORDS: Dict[str, List[str]] = {
        "greeting": ["你好", "您好", "hi", "hello", "在吗"],
        "question": ["什么", "为什么", "如何", "怎么", "?", "？"],
        "search": ["搜索", "查询", "查找", "search"],
        "execute": ["执行", "运行", "调用", "发送", "创建", "删除"],
        "farewell": ["再见", "拜拜", "bye", "goodbye"],
    }

    def __init__(
        self,
        model: str = "zh_core_web_sm",
        intent_keywords: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """初始化感知层。

        Args:
            model: spaCy 模型名。
            intent_keywords: 自定义意图关键词表, 与默认表合并 (同名覆盖)。
        """
        self._model_name = model
        self._nlp = None
        keywords = dict(self.DEFAULT_INTENT_KEYWORDS)
        if intent_keywords:
            keywords.update(intent_keywords)
        self._intent_keywords = keywords

    @property
    def nlp(self):
        """惰性加载 spaCy 模型, 避免启动时开销。"""
        if self._nlp is None:
            import spacy

            try:
                self._nlp = spacy.load(self._model_name)
            except OSError as exc:  # pragma: no cover - 依赖模型下载
                raise RuntimeError(
                    f"spaCy 模型 '{self._model_name}' 未安装, "
                    f"请执行: python -m spacy download {self._model_name}"
                ) from exc
        return self._nlp

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """实体识别, 返回 ``[{"text": ..., "label": ...}, ...]``。"""
        doc = self.nlp(text)
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    def classify_intent(self, text: str, default: str = "unknown") -> str:
        """意图分类: 关键词命中数打分, 取得分最高意图。"""
        lowered = text.lower()
        scores: Dict[str, int] = {}
        for intent, keywords in self._intent_keywords.items():
            score = sum(1 for kw in keywords if kw.lower() in lowered)
            if score:
                scores[intent] = score
        if not scores:
            return default
        return max(scores, key=scores.get)

    def perceive(self, text: str) -> Dict[str, object]:
        """统一入口: 一次性产出实体与意图。"""
        return {
            "text": text,
            "entities": self.extract_entities(text),
            "intent": self.classify_intent(text),
        }
