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

"""基于目录扫描的 skill 来源后端。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from snowland_agent_base.skill.base import SkillBackend
from snowland_agent_base.skill.models import SKILL_FILENAME, Skill, SkillError
from snowland_agent_base.skill.parsing import parse_skill_file


logger = logging.getLogger(__name__)


class FileSkillBackend(SkillBackend):
    """基于目录扫描的 skill 来源: 自动发现 ``SKILL.md`` 文件。

    ``root`` 既可以是 skills 根目录 (扫描其下一级子目录),
    也可以直接是单个 skill 目录。
    """

    def __init__(self, root: Any) -> None:  # noqa: ANN401 - 接受 str/Path
        self.root = Path(root)

    def discover(self) -> List[Skill]:
        if not self.root.is_dir():
            raise SkillError(f"目录不存在: {self.root}")

        candidates = (
            [self.root]
            if (self.root / SKILL_FILENAME).is_file()
            else [d for d in sorted(self.root.iterdir()) if d.is_dir()]
        )

        found: List[Skill] = []
        for directory in candidates:
            skill_file = directory / SKILL_FILENAME
            if not skill_file.is_file():
                continue
            try:
                found.append(parse_skill_file(skill_file))
            except SkillError as exc:
                logger.warning("跳过 skill 目录 %s: %s", directory, exc)
        return found
