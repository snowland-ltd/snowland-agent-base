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

"""Skill 可插拔维度的抽象接口。

两个可插拔维度:

- :class:`SkillBackend`: skill 来源后端, 决定 skill 从哪来、如何被发现;
- :class:`SkillMatcher`: 召回策略, 决定如何按输入召回相关 skill。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from snowland_agent_base.skill.models import Skill


class SkillBackend(ABC):
    """skill 来源后端统一接口: 决定 skill 从哪来、如何被发现。

    子类实现 :meth:`discover` 返回一组 :class:`Skill`。
    """

    @abstractmethod
    def discover(self) -> List[Skill]:
        """发现并返回本来源提供的一组 skill。"""


class SkillMatcher(ABC):
    """skill 召回策略统一接口: 按输入召回相关 skill。"""

    @abstractmethod
    def match(
        self, skills: Sequence[Skill], text: str, limit: Optional[int] = None
    ) -> List[Skill]:
        """从候选 skill 中召回与 ``text`` 相关的, 按相关度降序返回。"""
