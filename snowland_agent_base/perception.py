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

"""可插拔感知层。

把「感知」抽象为统一后端接口 :class:`PerceptionBackend`, 不同实现可热插拔:

- :class:`SpacyPerceptionBackend`: 基于 spaCy 的实体识别 + 规则意图分类 (默认, 轻量)。
- :class:`BailianPerceptionBackend`: 基于阿里云百炼 (DashScope) 大模型完成 NER 与意图分类。

:class:`PerceptionLayer` 是向后兼容的门面, 通过 ``backend`` 选择具体后端::

    # 名称式 (经注册表解析, 可插拔)
    PerceptionLayer()                              # 默认 spaCy
    PerceptionLayer(backend="bailian")             # 百炼大模型
    PerceptionLayer(backend="bailian", model="qwen-max")

    # 实例式 (直接注入, 便于测试 / 自定义)
    PerceptionLayer(backend=BailianPerceptionBackend(llm=my_llm))

新增后端只需继承 :class:`PerceptionBackend` 并 ``register_perception_backend`` 注册,
即可被名称字符串解析, 实现真正的插拔式扩展。
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 抽象后端
# --------------------------------------------------------------------------- #
class PerceptionBackend(ABC):
    """感知后端统一接口。

    所有具体后端 (spaCy / 百炼 / 自定义) 都必须实现这三个方法,
    ``perceive`` 的返回结构约定为 ``{"text", "entities", "intent"}``。
    """

    @abstractmethod
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """实体识别, 返回 ``[{"text": ..., "label": ...}, ...]``。"""

    @abstractmethod
    def classify_intent(self, text: str, default: str = "unknown") -> str:
        """意图分类, 返回意图类别字符串。"""

    @abstractmethod
    def perceive(self, text: str) -> Dict[str, object]:
        """统一入口: 一次性产出实体与意图。"""


# --------------------------------------------------------------------------- #
# 后端 1: spaCy (实体) + 规则 (意图)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 后端 2: 阿里云百炼 (DashScope) 大模型
# --------------------------------------------------------------------------- #
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
    ``llm`` 缺省时惰性调用 :func:`snowland_agent_base.llm.build_llm` 构建
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
        self._intent_candidates = list(intent_candidates or self.DEFAULT_INTENT_CANDIDATES)

    @property
    def llm(self) -> BaseChatModel:
        """惰性构建 DashScope ChatModel。"""
        if self._llm is None:
            from snowland_agent_base.llm import build_llm

            self._llm = build_llm(model=self._model) if self._model else build_llm()
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
        intent = self._normalize_intent(str(data.get("intent", "")), default="unknown")
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


# --------------------------------------------------------------------------- #
# 注册表 + 工厂 (插拔核心)
# --------------------------------------------------------------------------- #
_PERCEPTION_BACKENDS: Dict[str, type] = {}


def register_perception_backend(name: str, cls: type) -> None:
    """注册一个感知后端, 使其可通过名称字符串被工厂解析。

    Args:
        name: 后端名称, 如 ``"spacy"`` / ``"bailian"``。
        cls: 继承自 :class:`PerceptionBackend` 的后端类。
    """
    if not (isinstance(cls, type) and issubclass(cls, PerceptionBackend)):
        raise TypeError(f"感知后端 {cls!r} 必须继承 PerceptionBackend")
    _PERCEPTION_BACKENDS[name] = cls


def get_perception_backend(name: str) -> type:
    """按名称取回已注册的感知后端类。"""
    if name not in _PERCEPTION_BACKENDS:
        raise KeyError(
            f"未知感知后端: {name!r}; 可用: {list_perception_backends()}"
        )
    return _PERCEPTION_BACKENDS[name]


def list_perception_backends() -> List[str]:
    """列出所有已注册的感知后端名称。"""
    return list(_PERCEPTION_BACKENDS)


def create_perception_backend(name: str = "spacy", **kwargs) -> PerceptionBackend:
    """按名称构建一个感知后端实例。"""
    return get_perception_backend(name)(**kwargs)


# 注册内置后端
register_perception_backend("spacy", SpacyPerceptionBackend)
register_perception_backend("bailian", BailianPerceptionBackend)


# --------------------------------------------------------------------------- #
# 门面 (向后兼容)
# --------------------------------------------------------------------------- #
class PerceptionLayer:
    """可插拔感知层门面。

    对外暴露与原 :class:`PerceptionLayer` 完全一致的 ``perceive`` /
    ``extract_entities`` / ``classify_intent`` 接口, 内部委派给具体后端。

    ``backend`` 支持两种形式:
        - 字符串名称 (经注册表解析, 可插拔): ``"spacy"`` / ``"bailian"``。
        - 后端实例 (直接注入, 便于测试 / 自定义): ``SpacyPerceptionBackend()``。
    缺省 (不传) 使用 spaCy 后端, 与原行为兼容。
    """

    def __init__(self, backend: Optional[str] = "spacy", **kwargs) -> None:
        if isinstance(backend, PerceptionBackend):
            self._backend = backend
            self._backend_name = type(backend).__name__
        elif isinstance(backend, str):
            self._backend = create_perception_backend(backend, **kwargs)
            self._backend_name = backend
        elif backend is None:
            self._backend = None  # type: ignore[assignment]
            self._backend_name = "none"
        else:
            raise TypeError("backend 需为 PerceptionBackend 实例或已注册的名称字符串")

    @property
    def backend(self) -> Optional[PerceptionBackend]:
        """当前实际使用的感知后端实例。"""
        return self._backend

    @property
    def backend_name(self) -> str:
        """当前后端的名称 (字符串名或类名)。"""
        return self._backend_name

    def perceive(self, text: str) -> Dict[str, object]:
        if self._backend is None:
            return {"text": text, "entities": [], "intent": "unknown"}
        return self._backend.perceive(text)

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        if self._backend is None:
            return []
        return self._backend.extract_entities(text)

    def classify_intent(self, text: str, default: str = "unknown") -> str:
        if self._backend is None:
            return default
        return self._backend.classify_intent(text, default)


__all__ = [
    "PerceptionBackend",
    "SpacyPerceptionBackend",
    "BailianPerceptionBackend",
    "PerceptionLayer",
    "register_perception_backend",
    "get_perception_backend",
    "list_perception_backends",
    "create_perception_backend",
]
