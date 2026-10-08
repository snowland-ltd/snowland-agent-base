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

"""Skill 数据模型、命名约束与便捷构造器。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


#: SKILL.md 文件名
SKILL_FILENAME = "SKILL.md"

#: name 字段约束: 小写字母/数字/连字符, 不以连字符开头或结尾, 无连续连字符
_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

_MAX_NAME_LEN = 64
_MAX_DESC_LEN = 1024


class SkillError(ValueError):
    """skill 定义或加载错误。"""


@dataclass
class Skill:
    """一个 skill。

    Attributes:
        name: 唯一标识, 需符合命名约束 (小写字母/数字/连字符)。
        description: 描述「做什么 + 何时用」, 供模型判断相关性, 必填。
        instructions: SKILL.md 正文, 激活后注入系统提示词。
        tools: 该 skill 携带的工具名 -> 可调用对象; 激活后可用。
        allowed_tools: ``allowed-tools`` 字段声明的工具名 (仅声明, 不提供实现)。
        path: skill 目录路径, 用于解析 references/ scripts/ 等相对路径。
        metadata: 额外的元数据键值对。
        license: 许可声明。
        compatibility: 环境要求说明。
    """

    name: str
    description: str
    instructions: str = ""
    tools: Dict[str, Callable[..., Any]] = field(default_factory=dict)
    allowed_tools: List[str] = field(default_factory=list)
    path: Optional[Path] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    license: Optional[str] = None
    compatibility: Optional[str] = None

    def __post_init__(self) -> None:
        validate_skill_name(self.name)
        if not self.description or not self.description.strip():
            raise SkillError(f"skill '{self.name}' 的 description 不可为空")

    @property
    def tool_names(self) -> List[str]:
        """该 skill 提供的工具名。"""
        return list(self.tools)

    def render_metadata(self) -> str:
        """渲染第一级: 元数据 (~100 tokens), 供模型判断相关性。"""
        return f"- {self.name}: {self.description}"

    def render_instructions(self) -> str:
        """渲染第二级: 完整指令, 激活后注入系统提示词。"""
        blocks = [f"## Skill: {self.name}", self.instructions.strip()]
        if self.tool_names:
            blocks.append(f"该 skill 提供工具: {', '.join(self.tool_names)}")
        if not self.path:
            return "\n\n".join(b for b in blocks if b)

        # 第三级: 告诉模型有哪些资源可用 (不加载内容, 按需再读)
        resources = self.list_resources()
        if resources:
            rel = "\n".join(f"  - {r}" for r in resources)
            blocks.append(f"可用资源 (需要时读取):\n{rel}")
        return "\n\n".join(b for b in blocks if b)

    def list_resources(self) -> List[str]:
        """列出 skill 目录下的资源文件 (相对路径, 仅一级)。"""
        if not self.path or not self.path.is_dir():
            return []
        resources: List[str] = []
        for sub in ("scripts", "references", "assets"):
            directory = self.path / sub
            if not directory.is_dir():
                continue
            for file_path in sorted(directory.iterdir()):
                if file_path.is_file():
                    resources.append(f"{sub}/{file_path.name}")
        return resources

    def read_resource(self, relative_path: str) -> str:
        """读取 skill 资源文件内容。

        Args:
            relative_path: 相对 skill 根目录的路径, 如 ``references/REFERENCE.md``。

        Returns:
            文件内容。

        Raises:
            SkillError: 路径不存在或越界。
        """
        if not self.path:
            raise SkillError(f"skill '{self.name}' 无目录路径, 无法读取资源")
        target = (self.path / relative_path).resolve()
        root = self.path.resolve()
        if not str(target).startswith(str(root)):
            raise SkillError(f"资源路径越界: {relative_path}")
        if not target.is_file():
            raise SkillError(f"资源不存在: {relative_path}")
        return target.read_text(encoding="utf-8")


def validate_skill_name(name: str) -> None:
    """校验 skill 名称, 不符合规范则抛 :class:`SkillError`。"""
    if not name:
        raise SkillError("skill name 不可为空")
    if len(name) > _MAX_NAME_LEN:
        raise SkillError(f"skill name 超过 {_MAX_NAME_LEN} 字符: {name}")
    if not _NAME_RE.match(name):
        raise SkillError(
            f"skill name 只允许小写字母/数字/连字符, 且不能以连字符开头或结尾、"
            f"不能含连续连字符: {name}"
        )


def skill(
    name: str,
    description: str,
    instructions: str = "",
    tools: Optional[Dict[str, Callable[..., Any]]] = None,
    **kwargs: Any,
) -> Skill:
    """便捷构造函数。

    Args:
        name: skill 名称。
        description: 描述「做什么 + 何时用」。
        instructions: 指令正文。
        tools: 该 skill 携带的工具。
        kwargs: 透传给 :class:`Skill` 的其他字段。

    Returns:
        :class:`Skill` 实例。
    """
    if len(description) > _MAX_DESC_LEN:
        raise SkillError(f"skill '{name}' 的 description 超过 {_MAX_DESC_LEN} 字符")
    return Skill(
        name=name,
        description=description,
        instructions=instructions,
        tools=dict(tools or {}),
        **kwargs,
    )
