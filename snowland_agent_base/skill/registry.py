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

"""Skill 来源后端 / 召回策略 注册表与工厂 (插拔核心)。"""
from __future__ import annotations

from typing import Dict, Type

from snowland_agent_base.skill.backends import FileSkillBackend, MemorySkillBackend
from snowland_agent_base.skill.base import SkillBackend, SkillMatcher
from snowland_agent_base.skill.matchers import IdfSkillMatcher


_SKILL_BACKENDS: Dict[str, Type[SkillBackend]] = {}
_SKILL_MATCHERS: Dict[str, Type[SkillMatcher]] = {}


# --------------------------------------------------------------------------- #
# 来源后端注册表
# --------------------------------------------------------------------------- #
def register_skill_backend(name: str, cls: Type[SkillBackend]) -> None:
    """注册一个 skill 来源后端, 使其可通过名称字符串被工厂解析。"""
    if not (isinstance(cls, type) and issubclass(cls, SkillBackend)):
        raise TypeError(f"skill 后端 {cls!r} 必须继承 SkillBackend")
    _SKILL_BACKENDS[name] = cls


def get_skill_backend(name: str) -> Type[SkillBackend]:
    """按名称取回已注册的 skill 来源后端类。"""
    if name not in _SKILL_BACKENDS:
        raise KeyError(f"未知 skill 后端: {name!r}; 可用: {list_skill_backends()}")
    return _SKILL_BACKENDS[name]


def list_skill_backends() -> list:
    """列出所有已注册的 skill 来源后端名称。"""
    return list(_SKILL_BACKENDS)


def create_skill_backend(name: str = "file", **kwargs) -> SkillBackend:
    """按名称构建一个 skill 来源后端实例。"""
    return get_skill_backend(name)(**kwargs)


# --------------------------------------------------------------------------- #
# 召回策略注册表
# --------------------------------------------------------------------------- #
def register_skill_matcher(name: str, cls: Type[SkillMatcher]) -> None:
    """注册一个召回策略, 使其可通过名称字符串被工厂解析。"""
    if not (isinstance(cls, type) and issubclass(cls, SkillMatcher)):
        raise TypeError(f"召回策略 {cls!r} 必须继承 SkillMatcher")
    _SKILL_MATCHERS[name] = cls


def get_skill_matcher(name: str) -> Type[SkillMatcher]:
    """按名称取回已注册的召回策略类。"""
    if name not in _SKILL_MATCHERS:
        raise KeyError(f"未知召回策略: {name!r}; 可用: {list_skill_matchers()}")
    return _SKILL_MATCHERS[name]


def list_skill_matchers() -> list:
    """列出所有已注册的召回策略名称。"""
    return list(_SKILL_MATCHERS)


def create_skill_matcher(name: str = "idf", **kwargs) -> SkillMatcher:
    """按名称构建一个召回策略实例。"""
    return get_skill_matcher(name)(**kwargs)


# 注册内置后端与策略
register_skill_backend("file", FileSkillBackend)
register_skill_backend("memory", MemorySkillBackend)
register_skill_matcher("idf", IdfSkillMatcher)
