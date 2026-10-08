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

"""SkillRegistry 门面 (向后兼容)。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from snowland_agent_base.skill.backends import FileSkillBackend
from snowland_agent_base.skill.base import SkillBackend, SkillMatcher
from snowland_agent_base.skill.models import Skill, SkillError
from snowland_agent_base.skill.registry import create_skill_backend, create_skill_matcher


logger = logging.getLogger(__name__)


class SkillRegistry:
    """可插拔 skill 注册表门面: 注册、发现、召回、加载。

    对外暴露与原 :class:`SkillRegistry` 完全一致的接口, 内部聚合:
        - 一个内存注册表 (``register`` / ``get`` / ``load_directory`` 的落点);
        - 一个可插拔的**召回策略** :class:`SkillMatcher` (默认 ``idf``);
        - 可选**来源后端** :class:`SkillBackend` (构造时传入则自动发现其 skill)。

    Args:
        matcher: 召回策略, 名称字符串 (如 ``"idf"``) 或直接传入
            :class:`SkillMatcher` 实例。
        backend: 可选来源后端, 名称字符串或直接传入 :class:`SkillBackend`
            实例; 提供时其 ``discover()`` 结果会在构造时自动注册。
    """

    def __init__(
        self,
        matcher: Optional[Any] = "idf",
        backend: Optional[Any] = None,
    ) -> None:
        self._skills: Dict[str, Skill] = {}

        if isinstance(matcher, SkillMatcher):
            self._matcher = matcher
            self._matcher_name = type(matcher).__name__
        elif isinstance(matcher, str):
            self._matcher = create_skill_matcher(matcher)
            self._matcher_name = matcher
        elif matcher is None:
            self._matcher = create_skill_matcher("idf")
            self._matcher_name = "idf"
        else:
            raise TypeError("matcher 需为 SkillMatcher 实例或已注册的名称字符串")

        if isinstance(backend, SkillBackend):
            self._backend_name = type(backend).__name__
            for skill_obj in backend.discover():
                self.register(skill_obj)
        elif isinstance(backend, str):
            self._backend_name = backend
            for skill_obj in create_skill_backend(backend).discover():
                self.register(skill_obj)
        elif backend is None:
            self._backend_name = "memory"
        else:
            raise TypeError("backend 需为 SkillBackend 实例或已注册的名称字符串")

    @property
    def matcher_name(self) -> str:
        """当前召回策略名称 (字符串名或类名)。"""
        return self._matcher_name

    @property
    def backend_name(self) -> str:
        """当前来源后端名称 (字符串名或类名)。"""
        return self._backend_name

    # ------------------------------------------------------------------ #
    # 注册 / 注销
    # ------------------------------------------------------------------ #
    def register(self, skill_obj: Skill, overwrite: bool = False) -> None:
        """注册一个 skill。

        Args:
            skill_obj: skill 实例。
            overwrite: 同名时是否覆盖; 否则抛 :class:`SkillError`。
        """
        if not overwrite and skill_obj.name in self._skills:
            raise SkillError(f"skill 已存在: {skill_obj.name}")
        self._skills[skill_obj.name] = skill_obj

    def unregister(self, name: str) -> None:
        """注销 skill (不存在时静默)。"""
        self._skills.pop(name, None)

    def get(self, name: str) -> Optional[Skill]:
        """按名取 skill。"""
        return self._skills.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def list_skills(self) -> List[Skill]:
        """返回全部 skill。"""
        return list(self._skills.values())

    @property
    def names(self) -> List[str]:
        """全部 skill 名称。"""
        return list(self._skills)

    # ------------------------------------------------------------------ #
    # 目录扫描 (委托 FileSkillBackend)
    # ------------------------------------------------------------------ #
    def load_directory(self, root: Any, overwrite: bool = False) -> List[Skill]:
        """扫描目录, 加载其中所有含 SKILL.md 的子目录。

        ``root`` 既可以是 skills 根目录 (扫描其下一级子目录),
        也可以直接是单个 skill 目录。

        Args:
            root: 目录路径 (str 或 Path)。
            overwrite: 同名时是否覆盖。

        Returns:
            本次加载的 skill 列表。
        """
        loaded: List[Skill] = []
        for skill_obj in FileSkillBackend(Path(root)).discover():
            if not overwrite and skill_obj.name in self._skills:
                logger.warning("skill 已存在, 跳过: %s", skill_obj.name)
                continue
            self.register(skill_obj, overwrite=overwrite)
            loaded.append(skill_obj)
        return loaded

    # ------------------------------------------------------------------ #
    # 渐进式披露
    # ------------------------------------------------------------------ #
    def render_catalog(self) -> str:
        """第一级: 渲染所有 skill 的元数据 (~100 tokens/个)。

        供模型判断「用不用某个 skill」, 内容仅含 name + description。
        """
        if not self._skills:
            return ""
        body = "\n".join(s.render_metadata() for s in self._skills.values())
        return f"可用 Skill 列表:\n{body}"

    def match(self, text: str, limit: Optional[int] = None) -> List[Skill]:
        """召回与输入相关的 skill (委托当前召回策略)。

        这是**召回**阶段: 用可插拔的匹配策略缩小候选集; 最终「激活哪个」
        由 LLM 依据 :meth:`render_catalog` 决定。

        Args:
            text: 用户输入。
            limit: 最多返回个数。

        Returns:
            候选 skill 列表, 按相关度降序。
        """
        return self._matcher.match(self.list_skills(), text, limit)

    def collect_tools(self, skills: Sequence[Skill]) -> Dict[str, Callable[..., Any]]:
        """汇总给定 skill 携带的工具。"""
        tools: Dict[str, Callable[..., Any]] = {}
        for skill_obj in skills:
            tools.update(skill_obj.tools)
        return tools

    def all_tool_names(self) -> Set[str]:
        """全部 skill 提供的工具名。"""
        names: Set[str] = set()
        for skill_obj in self._skills.values():
            names.update(skill_obj.tools)
        return names
