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

"""Skill 能力层: 让 Agent 按需加载领域知识与 SOP。

设计遵循 `Agent Skills 规范 <https://agentskills.io/specification>`_,
采用**渐进式披露 (progressive disclosure)**, 分三级加载:

    1. **元数据** (启动时): 仅 name + description, 供模型判断相关性 (~100 tokens)
    2. **指令** (激活时): 完整 SKILL.md 正文, 注入系统提示词
    3. **资源** (按需): scripts/ references/ assets/ 下的文件, 需要时再读

同时支持两种 skill 形态:
    - **指令型**: 一段领域指令 (SOP / 领域知识), 激活后注入 system prompt;
    - **工具包型**: 携带自己的工具集, 激活后其工具对决策层可见。

本模块为**可插拔**设计, 两个维度均可热插拔:

- **skill 来源后端** :class:`SkillBackend`: 决定 skill 从哪来、如何被发现。
  内置 :class:`FileSkillBackend` (扫描 ``SKILL.md`` 目录) 与
  :class:`MemorySkillBackend` (代码注册的内存集合)。新增来源只需继承
  :class:`SkillBackend` 并 ``register_skill_backend`` 注册。
- **召回策略** :class:`SkillMatcher`: 决定如何按输入召回相关 skill。
  内置 :class:`IdfSkillMatcher` (IDF 加权词面匹配)。新增策略只需继承
  :class:`SkillMatcher` 并 ``register_skill_matcher`` 注册。

:class:`SkillRegistry` 是向后兼容的门面, 内部聚合「内存注册表 + 可选来源后端 +
召回策略」, 对外暴露与原版一致的 ``register`` / ``load_directory`` / ``match`` 等接口::

    SkillRegistry()                              # 默认 idf 召回 + 内存注册表
    SkillRegistry(matcher="idf")                 # 显式指定召回策略
    SkillRegistry(matcher=EmbeddingSkillMatcher())  # 直接注入自定义召回
    SkillRegistry(backend=FileSkillBackend(root))   # 构造即自动发现目录 skill
"""
from __future__ import annotations

from snowland_agent_base.skill.base import SkillBackend, SkillMatcher
from snowland_agent_base.skill.backends import FileSkillBackend, MemorySkillBackend
from snowland_agent_base.skill.facade import SkillRegistry
from snowland_agent_base.skill.matchers import IdfSkillMatcher
from snowland_agent_base.skill.models import (
    SKILL_FILENAME,
    Skill,
    SkillError,
    skill,
    validate_skill_name,
)
from snowland_agent_base.skill.parsing import parse_skill_file, parse_simple_yaml
from snowland_agent_base.skill.registry import (
    create_skill_backend,
    create_skill_matcher,
    get_skill_backend,
    get_skill_matcher,
    list_skill_backends,
    list_skill_matchers,
    register_skill_backend,
    register_skill_matcher,
)


__all__ = [
    "Skill",
    "SkillError",
    "SkillBackend",
    "FileSkillBackend",
    "MemorySkillBackend",
    "SkillMatcher",
    "IdfSkillMatcher",
    "SkillRegistry",
    "register_skill_backend",
    "get_skill_backend",
    "list_skill_backends",
    "create_skill_backend",
    "register_skill_matcher",
    "get_skill_matcher",
    "list_skill_matchers",
    "create_skill_matcher",
    "skill",
    "parse_skill_file",
    "parse_simple_yaml",
    "validate_skill_name",
    "SKILL_FILENAME",
]
