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

"""可插拔 Skill 层单元测试: 来源后端 / 召回策略 / 注册表 / 门面注入。"""
import tempfile
import textwrap
import unittest
from pathlib import Path

from snowland_agent_base.skill import (
    FileSkillBackend,
    IdfSkillMatcher,
    MemorySkillBackend,
    Skill,
    SkillBackend,
    SkillError,
    SkillMatcher,
    SkillRegistry,
    create_skill_backend,
    create_skill_matcher,
    list_skill_backends,
    list_skill_matchers,
    register_skill_backend,
    register_skill_matcher,
)


def _write_skill(root: Path, name: str, description: str, body: str = "instructions") -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    text = textwrap.dedent(
        f"""\
        ---
        name: {name}
        description: {description}
        ---

        {body}
        """
    )
    (directory / "SKILL.md").write_text(text, encoding="utf-8")
    return directory


class TestSkillBackendRegistry(unittest.TestCase):
    def test_builtin_backends_registered(self):
        self.assertIn("file", list_skill_backends())
        self.assertIn("memory", list_skill_backends())

    def test_builtin_matchers_registered(self):
        self.assertIn("idf", list_skill_matchers())

    def test_unknown_backend_raises(self):
        with self.assertRaises(KeyError):
            create_skill_backend("nope")

    def test_unknown_matcher_raises(self):
        with self.assertRaises(KeyError):
            create_skill_matcher("nope")

    def test_custom_backend_registration(self):
        class RemoteSkillBackend(SkillBackend):
            def __init__(self, url=None):
                self._url = url

            def discover(self):
                return [Skill(name="remote-a", description="remote skill")]

        register_skill_backend("remote", RemoteSkillBackend)
        self.assertIn("remote", list_skill_backends())
        backend = create_skill_backend("remote", url="http://x")
        self.assertIsInstance(backend, RemoteSkillBackend)
        self.assertEqual(len(backend.discover()), 1)

    def test_custom_matcher_registration(self):
        class AlwaysFirstMatcher(SkillMatcher):
            def match(self, skills, text, limit=None):
                result = list(skills)[:1]
                return result[:limit] if limit else result

        register_skill_matcher("first", AlwaysFirstMatcher)
        self.assertIn("first", list_skill_matchers())
        self.assertIsInstance(create_skill_matcher("first"), AlwaysFirstMatcher)


class TestFileSkillBackend(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        _write_skill(self.tmp, "email-writer", "写邮件相关")
        _write_skill(self.tmp, "code-review", "代码审查相关")

    def test_discover(self):
        skills = FileSkillBackend(self.tmp).discover()
        names = {s.name for s in skills}
        self.assertEqual(names, {"email-writer", "code-review"})

    def test_single_skill_directory(self):
        single = self.tmp / "email-writer"
        skills = FileSkillBackend(single).discover()
        self.assertEqual([s.name for s in skills], ["email-writer"])

    def test_missing_dir_raises(self):
        with self.assertRaises(SkillError):
            FileSkillBackend(self.tmp / "nope").discover()


class TestMemorySkillBackend(unittest.TestCase):
    def test_register_and_discover(self):
        backend = MemorySkillBackend()
        backend.register(Skill(name="a", description="d-a"))
        backend.register(Skill(name="b", description="d-b"))
        self.assertEqual({s.name for s in backend.discover()}, {"a", "b"})

    def test_duplicate_raises(self):
        backend = MemorySkillBackend()
        backend.register(Skill(name="a", description="d-a"))
        with self.assertRaises(SkillError):
            backend.register(Skill(name="a", description="d-a2"))

    def test_overwrite(self):
        backend = MemorySkillBackend()
        backend.register(Skill(name="a", description="d-a"))
        backend.register(Skill(name="a", description="d-a2"), overwrite=True)
        self.assertEqual(backend.get("a").description, "d-a2")


class TestIdfMatcher(unittest.TestCase):
    def setUp(self):
        self.skills = [
            Skill(name="email-writer", description="写邮件 发送邮件"),
            Skill(name="code-review", description="代码审查 评审"),
        ]

    def test_match_orders_by_relevance(self):
        result = IdfSkillMatcher().match(self.skills, "如何写一封邮件")
        self.assertEqual(result[0].name, "email-writer")

    def test_match_limit(self):
        result = IdfSkillMatcher().match(self.skills, "邮件 代码", limit=1)
        self.assertEqual(len(result), 1)

    def test_no_match_returns_empty(self):
        self.assertEqual(IdfSkillMatcher().match(self.skills, ""), [])


class TestFacadeInjection(unittest.TestCase):
    def test_default_matcher_is_idf(self):
        reg = SkillRegistry()
        self.assertEqual(reg.matcher_name, "idf")
        self.assertEqual(reg.backend_name, "memory")

    def test_inject_matcher_instance(self):
        reg = SkillRegistry(matcher=IdfSkillMatcher())
        self.assertEqual(reg.matcher_name, "IdfSkillMatcher")

    def test_inject_file_backend_discovers(self):
        tmp = Path(tempfile.mkdtemp())
        _write_skill(tmp, "pdf-processing", "处理 PDF 文档")
        reg = SkillRegistry(backend=FileSkillBackend(tmp))
        self.assertIn("pdf-processing", reg.names)
        self.assertEqual(reg.backend_name, "FileSkillBackend")

    def test_inject_memory_backend_with_initial(self):
        backend = MemorySkillBackend([Skill(name="x", description="dx")])
        reg = SkillRegistry(backend=backend)
        self.assertIn("x", reg.names)

    def test_match_delegates_to_matcher(self):
        tmp = Path(tempfile.mkdtemp())
        _write_skill(tmp, "email-writer", "写邮件 发送邮件")
        reg = SkillRegistry(backend=FileSkillBackend(tmp))
        result = reg.match("写一封邮件给客户")
        self.assertEqual(result[0].name, "email-writer")

    def test_render_catalog_uses_registry(self):
        reg = SkillRegistry()
        reg.register(Skill(name="a", description="desc-a"))
        catalog = reg.render_catalog()
        self.assertIn("desc-a", catalog)


if __name__ == "__main__":
    unittest.main()
