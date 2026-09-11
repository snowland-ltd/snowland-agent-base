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

"""Skill 与 Agent 集成测试: 渐进式披露的实际效果与工具包型 skill。"""
import tempfile
import unittest
from pathlib import Path

from snowland_agent_base.agent import BaseAgent
from snowland_agent_base.skill import SkillRegistry, skill

from tests.test_agent import FakeChatModel


class TestSkillInjection(unittest.TestCase):
    """验证 skill 按要求分两级注入 prompt。"""

    def setUp(self) -> None:
        self.llm = FakeChatModel()
        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']

    def _system(self) -> str:
        return self.llm.last_messages[0].content

    def test_catalog_always_present(self):
        """第一级: 无论是否激活, 目录 (name + description) 都应出现。"""
        registry = SkillRegistry()
        registry.register(skill("pdf-processing", "处理 PDF 文档"))
        agent = BaseAgent(llm=self.llm, use_perception=False, use_memory=False, skills=registry)

        agent.run("今天天气如何")
        system = self._system()
        self.assertIn("- pdf-processing: 处理 PDF 文档", system)
        self.assertIn("可用 Skill 列表", system)

    def test_instructions_only_when_activated(self):
        """第二级: 完整指令仅在激活时注入。"""
        registry = SkillRegistry()
        registry.register(
            skill("pdf-processing", "处理 PDF 文档", "## 步骤\n1. 调用提取工具")
        )
        agent = BaseAgent(llm=self.llm, use_perception=False, use_memory=False, skills=registry)

        # 不相关输入: 不应注入正文
        agent.run("今天天气如何")
        self.assertNotIn("1. 调用提取工具", self._system())

        # 相关输入: 应注入正文
        agent.run("帮我处理这个 PDF")
        system = self._system()
        self.assertIn("1. 调用提取工具", system)
        self.assertIn("已激活 Skill", system)

    def test_activated_skill_in_output(self):
        registry = SkillRegistry()
        registry.register(skill("pdf-processing", "处理 PDF 文档"))
        agent = BaseAgent(llm=self.llm, use_perception=False, use_memory=False, skills=registry)

        result = agent.run("处理 PDF")
        self.assertEqual(result["skills"], ["pdf-processing"])

    def test_no_activation_when_disabled(self):
        """关闭自动激活后, 仅注入目录。"""
        registry = SkillRegistry()
        registry.register(skill("pdf-processing", "处理 PDF 文档", "秘密指令"))
        agent = BaseAgent(
            llm=self.llm,
            use_perception=False,
            use_memory=False,
            skills=registry,
            auto_activate_skills=False,
        )
        agent.run("处理 PDF")
        system = self._system()
        self.assertIn("- pdf-processing", system)
        self.assertNotIn("秘密指令", system)

    def test_no_skills_no_section(self):
        agent = BaseAgent(llm=self.llm, use_perception=False, use_memory=False)
        agent.run("你好")
        self.assertNotIn("Skill", self._system())


class TestSkillTools(unittest.TestCase):
    """工具包型 skill: 携带的工具应注册进执行器。"""

    def setUp(self) -> None:
        self.llm = FakeChatModel()

    def test_skill_tools_registered(self):
        registry = SkillRegistry()
        registry.register(
            skill("calc", "做计算", "步骤", tools={"add": lambda a, b: a + b})
        )
        agent = BaseAgent(llm=self.llm, use_perception=False, use_memory=False, skills=registry)
        agent.add_skill(skill("x", "y", tools={"noop": lambda: 1}))

        self.assertIn("add", agent.executor.list_tools())
        self.assertIn("noop", agent.executor.list_tools())

    def test_skill_tool_executable(self):
        self.llm.responses = [
            '{"thought": "计算", "tool": "add", "tool_input": {"a": 1, "b": 2}, "final_answer": null}'
        ]
        registry = SkillRegistry()
        registry.register(skill("calc", "做计算", "步骤", tools={"add": lambda a, b: a + b}))
        agent = BaseAgent(
            llm=self.llm,
            use_perception=False,
            use_memory=False,
            skills=registry,
            max_steps=1,
        )
        result = agent.run("1+2")
        self.assertEqual(result["steps"][0]["observation"]["result"], 3)

    def test_tool_listed_in_prompt(self):
        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']
        registry = SkillRegistry()
        registry.register(skill("calc", "做计算", "步骤", tools={"add": lambda a, b: a + b}))
        agent = BaseAgent(llm=self.llm, use_perception=False, use_memory=False, skills=registry)
        agent.run("算一下")
        self.assertIn("- add", self.llm.last_messages[0].content)


class TestLoadSkillsFromDir(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = FakeChatModel()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_agent_load_directory(self):
        d = self.root / "pdf-processing"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            "---\nname: pdf-processing\ndescription: 处理 PDF 文档\n---\n\n## 步骤\n1. 提取\n",
            encoding="utf-8",
        )
        agent = BaseAgent(llm=self.llm, use_perception=False, use_memory=False)
        loaded = agent.load_skills(self.root)

        self.assertEqual([s.name for s in loaded], ["pdf-processing"])
        self.assertIn("pdf-processing", agent.skills.names)

        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']
        result = agent.run("处理 PDF")
        self.assertEqual(result["skills"], ["pdf-processing"])
        self.assertIn("1. 提取", self.llm.last_messages[0].content)


if __name__ == "__main__":
    unittest.main()
