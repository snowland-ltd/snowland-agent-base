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

"""Skill 能力层测试: 渐进式披露、代码注册、目录扫描、工具包型 skill。"""
import tempfile
import unittest
from pathlib import Path

from snowland_agent_base.skill import (
    Skill,
    SkillError,
    SkillRegistry,
    parse_skill_file,
    parse_simple_yaml,
    skill,
    validate_skill_name,
)


class TestValidateName(unittest.TestCase):
    def test_valid(self):
        for name in ("pdf", "pdf-processing", "data2-analysis", "a"):
            validate_skill_name(name)

    def test_empty(self):
        with self.assertRaises(SkillError):
            validate_skill_name("")

    def test_uppercase(self):
        with self.assertRaises(SkillError):
            validate_skill_name("PDF")

    def test_leading_hyphen(self):
        with self.assertRaises(SkillError):
            validate_skill_name("-pdf")

    def test_trailing_hyphen(self):
        with self.assertRaises(SkillError):
            validate_skill_name("pdf-")

    def test_consecutive_hyphens(self):
        with self.assertRaises(SkillError):
            validate_skill_name("pdf--x")

    def test_too_long(self):
        with self.assertRaises(SkillError):
            validate_skill_name("a" * 65)


class TestSkillModel(unittest.TestCase):
    def test_description_required(self):
        with self.assertRaises(SkillError):
            Skill(name="x", description="")

    def test_constructor_helper(self):
        s = skill("demo", "做某些事", "步骤一")
        self.assertEqual(s.name, "demo")
        self.assertEqual(s.instructions, "步骤一")

    def test_description_too_long(self):
        with self.assertRaises(SkillError):
            skill("demo", "x" * 1025)

    def test_render_metadata(self):
        """第一级: 仅 name + description。"""
        s = skill("pdf", "处理 PDF 文档")
        self.assertEqual(s.render_metadata(), "- pdf: 处理 PDF 文档")
        self.assertNotIn("步骤", s.render_metadata())

    def test_render_instructions(self):
        s = skill("pdf", "处理 PDF", "## 步骤\n1. 提取")
        text = s.render_instructions()
        self.assertIn("## Skill: pdf", text)
        self.assertIn("1. 提取", text)

    def test_render_includes_tool_names(self):
        s = skill("pdf", "处理 PDF", "正文", tools={"extract": lambda: 1})
        self.assertIn("extract", s.render_instructions())

    def test_tool_names(self):
        s = skill("x", "y", tools={"a": lambda: 1, "b": lambda: 2})
        self.assertEqual(sorted(s.tool_names), ["a", "b"])


class TestRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = SkillRegistry()

    def test_register_and_get(self):
        s = skill("pdf", "处理 PDF")
        self.registry.register(s)
        self.assertIs(self.registry.get("pdf"), s)
        self.assertIn("pdf", self.registry)
        self.assertEqual(len(self.registry), 1)

    def test_duplicate_rejected(self):
        self.registry.register(skill("pdf", "a"))
        with self.assertRaises(SkillError):
            self.registry.register(skill("pdf", "b"))

    def test_duplicate_overwrite(self):
        self.registry.register(skill("pdf", "a"))
        self.registry.register(skill("pdf", "b"), overwrite=True)
        self.assertEqual(self.registry.get("pdf").description, "b")

    def test_unregister(self):
        self.registry.register(skill("pdf", "a"))
        self.registry.unregister("pdf")
        self.assertIsNone(self.registry.get("pdf"))

    def test_render_catalog_empty(self):
        self.assertEqual(self.registry.render_catalog(), "")

    def test_render_catalog(self):
        self.registry.register(skill("pdf", "处理 PDF"))
        self.registry.register(skill("sql", "写 SQL"))
        catalog = self.registry.render_catalog()
        self.assertIn("- pdf: 处理 PDF", catalog)
        self.assertIn("- sql: 写 SQL", catalog)

    def test_collect_tools(self):
        a = skill("a", "x", tools={"t1": lambda: 1})
        b = skill("b", "y", tools={"t2": lambda: 2})
        tools = self.registry.collect_tools([a, b])
        self.assertEqual(sorted(tools), ["t1", "t2"])

    def test_all_tool_names(self):
        self.registry.register(skill("a", "x", tools={"t1": lambda: 1}))
        self.assertEqual(self.registry.all_tool_names(), {"t1"})


class TestMatch(unittest.TestCase):
    """离线召回: 中英文词面匹配。"""

    def setUp(self) -> None:
        self.registry = SkillRegistry()
        self.registry.register(
            skill("pdf-processing", "提取 PDF 文本与表格, 处理 PDF 文档时使用")
        )
        self.registry.register(
            skill("sql-helper", "编写与优化 SQL 查询语句, 涉及数据库时使用")
        )

    def test_match_english(self):
        matched = self.registry.match("I have a PDF file")
        self.assertEqual([s.name for s in matched], ["pdf-processing"])

    def test_match_chinese(self):
        matched = self.registry.match("帮我提取PDF文本")
        self.assertEqual([s.name for s in matched], ["pdf-processing"])

    def test_match_sql(self):
        matched = self.registry.match("写个 SQL 查询")
        self.assertEqual([s.name for s in matched], ["sql-helper"])

    def test_no_match(self):
        self.assertEqual(self.registry.match("今天天气不错"), [])

    def test_empty_query(self):
        self.assertEqual(self.registry.match(""), [])

    def test_limit(self):
        self.registry.register(skill("pdf-other", "PDF 的另一个用途"))
        matched = self.registry.match("PDF", limit=1)
        self.assertEqual(len(matched), 1)


class TestParseYaml(unittest.TestCase):
    def test_simple(self):
        data = parse_simple_yaml("name: pdf\ndescription: 处理 PDF")
        self.assertEqual(data["name"], "pdf")
        self.assertEqual(data["description"], "处理 PDF")

    def test_quoted(self):
        self.assertEqual(parse_simple_yaml('name: "pdf"')["name"], "pdf")

    def test_nested(self):
        data = parse_simple_yaml('metadata:\n  author: me\n  version: "1.0"')
        self.assertEqual(data["metadata"]["author"], "me")

    def test_comment_ignored(self):
        data = parse_simple_yaml("# comment\nname: pdf")
        self.assertEqual(data["name"], "pdf")

    def test_bool(self):
        self.assertIs(parse_simple_yaml("flag: true")["flag"], True)


class TestSkillFile(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, rel: str, content: str) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_parse_basic(self):
        path = self._write(
            "pdf-processing/SKILL.md",
            "---\nname: pdf-processing\ndescription: 处理 PDF\n---\n\n## 步骤\n1. 提取\n",
        )
        s = parse_skill_file(path)
        self.assertEqual(s.name, "pdf-processing")
        self.assertEqual(s.description, "处理 PDF")
        self.assertIn("1. 提取", s.instructions)
        self.assertEqual(s.path, path.parent)

    def test_parse_metadata_and_allowed_tools(self):
        path = self._write(
            "demo/SKILL.md",
            "---\nname: demo\ndescription: 演示\nallowed-tools: a b\n"
            "metadata:\n  author: me\n---\n正文\n",
        )
        s = parse_skill_file(path)
        self.assertEqual(s.allowed_tools, ["a", "b"])
        self.assertEqual(s.metadata["author"], "me")
        self.assertEqual(s.instructions, "正文")

    def test_missing_name(self):
        path = self._write("demo/SKILL.md", "---\ndescription: x\n---\n正文")
        with self.assertRaises(SkillError):
            parse_skill_file(path)

    def test_missing_description(self):
        path = self._write("demo/SKILL.md", "---\nname: demo\n---\n正文")
        with self.assertRaises(SkillError):
            parse_skill_file(path)

    def test_no_frontmatter(self):
        """无 frontmatter -> 缺 name, 应报错。"""
        path = self._write("demo/SKILL.md", "纯正文")
        with self.assertRaises(SkillError):
            parse_skill_file(path)

    def test_file_not_exists(self):
        with self.assertRaises(SkillError):
            parse_skill_file(self.root / "nope" / "SKILL.md")

    def test_resources_listed(self):
        self._write("demo/SKILL.md", "---\nname: demo\ndescription: 演示\n---\n正文")
        self._write("demo/references/REF.md", "详细参考")
        self._write("demo/scripts/run.py", "print(1)")
        s = parse_skill_file(self.root / "demo" / "SKILL.md")
        self.assertIn("references/REF.md", s.list_resources())
        self.assertIn("scripts/run.py", s.list_resources())

    def test_read_resource(self):
        self._write("demo/SKILL.md", "---\nname: demo\ndescription: 演示\n---\n正文")
        self._write("demo/references/REF.md", "详细内容")
        s = parse_skill_file(self.root / "demo" / "SKILL.md")
        self.assertEqual(s.read_resource("references/REF.md"), "详细内容")

    def test_read_resource_outside(self):
        self._write("demo/SKILL.md", "---\nname: demo\ndescription: 演示\n---\n正文")
        s = parse_skill_file(self.root / "demo" / "SKILL.md")
        with self.assertRaises(SkillError):
            s.read_resource("../../secret.txt")

    def test_read_resource_missing(self):
        self._write("demo/SKILL.md", "---\nname: demo\ndescription: 演示\n---\n正文")
        s = parse_skill_file(self.root / "demo" / "SKILL.md")
        with self.assertRaises(SkillError):
            s.read_resource("references/nope.md")


class TestLoadDirectory(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.registry = SkillRegistry()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, rel: str, content: str) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_load_multiple_skills(self):
        self._write("a/SKILL.md", "---\nname: a\ndescription: 技能 A\n---\nA 正文")
        self._write("b/SKILL.md", "---\nname: b\ndescription: 技能 B\n---\nB 正文")
        loaded = self.registry.load_directory(self.root)
        self.assertEqual(sorted(s.name for s in loaded), ["a", "b"])
        self.assertEqual(len(self.registry), 2)

    def test_skip_dir_without_skill_file(self):
        self._write("a/SKILL.md", "---\nname: a\ndescription: 技能 A\n---\nA")
        self._write("no-skill/readme.txt", "无 SKILL.md")
        loaded = self.registry.load_directory(self.root)
        self.assertEqual([s.name for s in loaded], ["a"])

    def test_skip_invalid_skill(self):
        """单个 skill 解析失败不应中断整体加载。"""
        self._write("good/SKILL.md", "---\nname: good\ndescription: 好的\n---\n正文")
        self._write("bad/SKILL.md", "---\nname: BAD\ndescription: 坏名\n---\n正文")
        loaded = self.registry.load_directory(self.root)
        self.assertEqual([s.name for s in loaded], ["good"])

    def test_load_single_skill_dir(self):
        """直接传入单个 skill 目录也应可用。"""
        self._write("a/SKILL.md", "---\nname: a\ndescription: 技能 A\n---\nA")
        loaded = self.registry.load_directory(self.root / "a")
        self.assertEqual([s.name for s in loaded], ["a"])

    def test_missing_dir(self):
        with self.assertRaises(SkillError):
            self.registry.load_directory(self.root / "nope")


if __name__ == "__main__":
    unittest.main()
