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

"""内存型 skill 来源后端。"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from snowland_agent_base.skill.base import SkillBackend
from snowland_agent_base.skill.models import Skill, SkillError


class MemorySkillBackend(SkillBackend):
    """内存型 skill 来源: 代码注册的集合, 支持 register/unregister。"""

    def __init__(self, initial: Optional[Sequence[Skill]] = None) -> None:
        self._skills: Dict[str, Skill] = {}
        for skill_obj in initial or []:
            self.register(skill_obj)

    def register(self, skill_obj: Skill, overwrite: bool = False) -> None:
        if not overwrite and skill_obj.name in self._skills:
            raise SkillError(f"skill 已存在: {skill_obj.name}")
        self._skills[skill_obj.name] = skill_obj

    def unregister(self, name: str) -> None:
        self._skills.pop(name, None)

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def discover(self) -> Sequence[Skill]:
        return list(self._skills.values())
