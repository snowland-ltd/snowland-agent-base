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

"""SKILL.md 解析与词面召回辅助。"""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, Set

from snowland_agent_base.skill.models import Skill, SkillError, _MAX_DESC_LEN


logger = logging.getLogger(__name__)

#: frontmatter 分隔符
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)

#: 含中文的连续片段 (中英文混排, 如 "a股" 需整体保留)
_CJK_CHUNK_RE = re.compile(r"[a-z0-9\u4e00-\u9fff]+")
_ASCII_WORD_RE = re.compile(r"[a-z0-9]{2,}")


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
    current_key: Any = None

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
