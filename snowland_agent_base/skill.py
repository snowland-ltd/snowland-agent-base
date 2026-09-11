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

加载方式:
    - 代码注册: :func:`skill` 装饰器或 :meth:`SkillRegistry.register`
    - 目录扫描: :meth:`SkillRegistry.load_directory`, 自动发现 ``SKILL.md``

目录结构::

    skills/
    └── pdf-processing/
        ├── SKILL.md          # 必需: YAML frontmatter + Markdown 指令
        ├── scripts/          # 可选
        ├── references/       # 可选
        └── assets/           # 可选

SKILL.md 示例::

    ---
    name: pdf-processing
    description: 提取 PDF 文本与表格. 当用户提到 PDF、表单、文档抽取时使用.
    allowed-tools: extract_pdf merge_pdf
    ---

    ## 步骤
    1. 调用 extract_pdf 提取文本
    ...
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)

#: SKILL.md 文件名
SKILL_FILENAME = "SKILL.md"

#: name 字段约束: 小写字母/数字/连字符, 不以连字符开头或结尾, 无连续连字符
_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

#: frontmatter 分隔符
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)

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


_ASCII_WORD_RE = re.compile(r"[a-z0-9]{2,}")
#: 含中文的连续片段 (中英文混排, 如 "a股" 需整体保留)
_CJK_CHUNK_RE = re.compile(r"[a-z0-9\u4e00-\u9fff]+")


def extract_terms(text: str) -> Set[str]:
    """从文本中提取用于召回的 term 集合。

    中文无空格分词, 采用 2-gram; 英文/数字按词提取。中英混排片段
    (如 ``A股``) 会整体参与 2-gram, 避免把 ``A`` 与 ``股`` 拆散。

    Args:
        text: 输入文本。

    Returns:
        term 集合 (已小写)。
    """
    if not text:
        return set()
    lowered = text.lower()
    terms: Set[str] = set(_ASCII_WORD_RE.findall(lowered))

    for chunk in _CJK_CHUNK_RE.findall(lowered):
        # 纯 ASCII 片段已由上面覆盖, 此处只处理含中文的
        if not any("\u4e00" <= ch <= "\u9fff" for ch in chunk):
            continue
        if len(chunk) == 1:
            terms.add(chunk)
            continue
        terms.update(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return terms


def _idf(term: str, df: Dict[str, int], total: int) -> float:
    """计算逆文档频率权重; 未出现过的词给最高权重。"""
    freq = df.get(term, 0)
    return math.log((total + 1) / (freq + 1)) + 1.0


def _first_present(data: Dict[str, Any], *keys: str) -> Any:
    """按顺序返回首个存在且非空的字段值。

    真实 skill 仓库中字段命名并不统一 (如 ``description`` 缺失但有
    ``description_zh`` / ``description_en``), 此处做容错回退。
    """
    for key in keys:
        value = data.get(key)
        if value not in (None, "", {}, []):
            return value
    return None


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


class SkillRegistry:
    """skill 注册表: 注册、发现、匹配、加载。

    召回采用 **IDF 加权**: 在大规模 skill 仓库 (数百个) 中, 朴素词频打分会把
    "AI"、"数据" 这类高频词的无关 skill 排到前面 (如查询「写邮件」却命中含
    "AI" 的资讯 skill)。故按词的文档频率降权, 并给 name 命中更高权重。
    """

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}
        self._index: Optional[Dict[str, Any]] = None

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
        self._index = None

    def unregister(self, name: str) -> None:
        """注销 skill (不存在时静默)。"""
        self._skills.pop(name, None)
        self._index = None

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
    # 目录扫描
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
        root_path = Path(root)
        if not root_path.is_dir():
            raise SkillError(f"目录不存在: {root_path}")

        candidates = [root_path] if (root_path / SKILL_FILENAME).is_file() else [
            d for d in sorted(root_path.iterdir()) if d.is_dir()
        ]

        loaded: List[Skill] = []
        for directory in candidates:
            skill_file = directory / SKILL_FILENAME
            if not skill_file.is_file():
                continue
            try:
                skill_obj = parse_skill_file(skill_file)
            except SkillError as exc:
                logger.warning("跳过 skill 目录 %s: %s", directory, exc)
                continue
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
        """召回与输入相关的 skill (离线、零成本)。

        这是**召回**阶段: 用 IDF 加权的词面匹配缩小候选集; 最终「激活哪个」
        由 LLM 依据 :meth:`render_catalog` 决定。

        Args:
            text: 用户输入。
            limit: 最多返回个数。

        Returns:
            候选 skill 列表, 按相关度降序。
        """
        if not text or not self._skills:
            return []

        index = self._ensure_index()
        df = index["df"]
        total = len(self._skills)
        query_terms = extract_terms(text)
        if not query_terms:
            return []

        scored: List[tuple] = []
        for skill_obj in self._skills.values():
            terms = index["terms"].get(skill_obj.name)
            if not terms:
                continue
            name_terms, desc_terms = terms
            score = 0.0
            for term in query_terms:
                # IDF: 越稀有的词权重越高
                weight = _idf(term, df, total)
                if term in name_terms:
                    score += weight * 2.0  # name 命中权重更高
                elif term in desc_terms:
                    score += weight
            if score > 0:
                scored.append((score, skill_obj.name, skill_obj))

        scored.sort(key=lambda item: (-item[0], item[1]))
        result = [s for _, _, s in scored]
        return result[:limit] if limit else result

    def _ensure_index(self) -> Dict[str, Any]:
        """惰性构建倒排索引 (文档频率 + 每个 skill 的 term 集合)。"""
        if self._index is not None:
            return self._index

        df: Dict[str, int] = {}
        terms: Dict[str, tuple] = {}
        for name, skill_obj in self._skills.items():
            name_terms = extract_terms(skill_obj.name)
            desc_terms = extract_terms(skill_obj.description)
            terms[name] = (name_terms, desc_terms)
            for term in name_terms | desc_terms:
                df[term] = df.get(term, 0) + 1

        self._index = {"df": df, "terms": terms}
        return self._index

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


# ---------------------------------------------------------------------- #
# SKILL.md 解析
# ---------------------------------------------------------------------- #
def parse_skill_file(path: Any) -> Skill:
    """解析 SKILL.md 文件。

    Args:
        path: SKILL.md 路径。

    Returns:
        :class:`Skill` 实例。

    Raises:
        SkillError: 文件缺失或 frontmatter 不合法。
    """
    path = Path(path)
    if not path.is_file():
        raise SkillError(f"文件不存在: {path}")

    raw = path.read_text(encoding="utf-8")
    frontmatter, body = split_frontmatter(raw)

    name = str(_first_present(frontmatter, "name", "display_name", "displayName", "slug") or "")
    name = name.strip()
    description = str(
        _first_present(
            frontmatter,
            "description",
            "description_zh",
            "description_en",
            "display_name",
            "display_name_zh",
        )
        or ""
    ).strip()
    if not name:
        raise SkillError(f"{path}: frontmatter 缺少 name")
    if not description:
        raise SkillError(f"{path}: frontmatter 缺少 description (或 description_zh/en)")
    if len(description) > _MAX_DESC_LEN:
        raise SkillError(f"{path}: description 超过 {_MAX_DESC_LEN} 字符")

    allowed = frontmatter.get("allowed-tools") or ""
    allowed_tools = str(allowed).split() if isinstance(allowed, str) else list(allowed)

    metadata = frontmatter.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    skill_obj = Skill(
        name=name,
        description=description,
        instructions=body.strip(),
        allowed_tools=list(allowed_tools),
        path=path.parent,
        metadata={str(k): str(v) for k, v in metadata.items()},
        license=frontmatter.get("license"),
        compatibility=frontmatter.get("compatibility"),
    )

    # 规范: name 必须与父目录名一致
    if path.parent.name and path.parent.name != name:
        logger.warning(
            "skill name '%s' 与目录名 '%s' 不一致 (规范要求一致)", name, path.parent.name
        )
    return skill_obj


def split_frontmatter(raw: str) -> tuple:
    """拆分 YAML frontmatter 与正文。

    Returns:
        ``(frontmatter_dict, body)``; 无 frontmatter 时返回 ``({}, raw)``。
    """
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    yaml_text = match.group(1)
    body = raw[match.end() :]
    return parse_simple_yaml(yaml_text), body


def parse_simple_yaml(text: str) -> Dict[str, Any]:
    """解析 SKILL.md 用到的 frontmatter 子集。

    仅依赖标准库: 支持 ``key: value``、嵌套一层的 ``key:\\n  k: v``、
    以及 ``#`` 注释。不追求完整 YAML 兼容。

    Args:
        text: frontmatter 文本。

    Returns:
        解析结果字典。
    """
    result: Dict[str, Any] = {}
    current_key: Optional[str] = None

    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        indented = line[:1].isspace()
        stripped = line.strip()

        if indented and current_key:
            if ":" not in stripped:
                continue
            key, _, value = stripped.partition(":")
            nested = result.setdefault(current_key, {})
            if isinstance(nested, dict):
                nested[key.strip()] = _coerce(value.strip())
            continue

        if ":" not in stripped:
            continue

        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        current_key = key
        if value == "":
            # 可能是嵌套映射, 先占位
            result[key] = {}
        else:
            result[key] = _coerce(value)

    # 清理空的嵌套占位
    return {k: (v if v != {} else "") for k, v in result.items()}


def _coerce(value: str) -> Any:
    """去除引号并做基础类型推断。"""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    return value
