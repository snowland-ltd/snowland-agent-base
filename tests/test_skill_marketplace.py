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

"""真实技能市场集成测试: 输入一条指令, 能从 skill 仓库中找到对应的 skill。

用途: 验证本框架对**大规模真实 skill 仓库**的兼容性与召回能力。

数据源为本机的技能市场目录 (含 290+ 个遵循 Agent Skills 规范的 skill):
    C:\\Users\\PC\\.codebuddy\\skills-marketplace\\skills

若该目录不存在 (如 CI 环境), 全部用例自动跳过, 不影响其余测试。
可设置环境变量 SNOWLAND_SKILL_MARKET 指向其他目录。
"""
import os
import unittest
from pathlib import Path

from snowland_agent_base.skill import SkillRegistry

try:
    from snowland_agent_base.agent import BaseAgent

    from tests.test_agent import FakeChatModel

    AGENT_AVAILABLE = True
except ImportError:  # pragma: no cover
    AGENT_AVAILABLE = False

#: 技能市场根目录
DEFAULT_MARKET = Path(r"C:\Users\PC\.codebuddy\skills-marketplace\skills")
MARKET = Path(os.environ.get("SNOWLAND_SKILL_MARKET", DEFAULT_MARKET))

MARKET_EXISTS = MARKET.is_dir()


@unittest.skipUnless(MARKET_EXISTS, f"技能市场目录不存在: {MARKET}")
class TestMarketplaceLoading(unittest.TestCase):
    """能否完整加载真实 skill 仓库。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = SkillRegistry()
        cls.loaded = cls.registry.load_directory(MARKET)

    def test_loads_bulk_skills(self):
        """应能加载绝大多数 skill (仓库约 290+ 个)。"""
        self.assertGreater(len(self.loaded), 200, "加载数量过少, 兼容性可能有问题")

    def test_all_have_name_and_description(self):
        for s in self.loaded:
            self.assertTrue(s.name, "skill 缺少 name")
            self.assertTrue(s.description, f"skill {s.name} 缺少 description")

    def test_catalog_rendered(self):
        catalog = self.registry.render_catalog()
        self.assertIn("可用 Skill 列表", catalog)
        self.assertGreater(len(catalog), 1000)


@unittest.skipUnless(MARKET_EXISTS, f"技能市场目录不存在: {MARKET}")
class TestFindSkillByInstruction(unittest.TestCase):
    """核心用例: 给定一条自然语言指令, 召回对应的 skill。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = SkillRegistry()
        cls.registry.load_directory(MARKET)

    def _top_names(self, instruction: str, limit: int = 10) -> list:
        return [s.name for s in self.registry.match(instruction, limit=limit)]

    def test_find_pdf_ocr_skill(self):
        """「提取 PDF 里的文字」应召回 OCR / 文档处理类 skill。"""
        names = self._top_names("帮我处理这个 PDF 文件, 提取里面的文字")
        self.assertTrue(
            any("ocr" in n or "pdf" in n for n in names),
            f"未召回 OCR/PDF 类 skill, 实际: {names}",
        )

    def test_find_stock_skill(self):
        """「查询 A 股行情」应召回股票数据类 skill。"""
        names = self._top_names("查询 A 股股票行情数据")
        expected = {"a-stock-data", "westockdata", "futuapi", "lingxi-financialsearch-skill"}
        self.assertTrue(
            expected & set(names),
            f"未召回股票类 skill, 实际: {names}",
        )

    def test_find_weather_skill(self):
        """「查天气」应召回天气类 skill。"""
        names = self._top_names("帮我查一下北京今天的天气")
        self.assertIn(
            "weather-open-meteo", names, f"未召回天气 skill, 实际: {names}"
        )

    def test_find_translation_skill(self):
        """「翻译论文」应召回翻译/学术类 skill。"""
        names = self._top_names("帮我把这篇英文学术论文翻译成中文", limit=15)
        expected = {"academic-translation", "arxiv-reader", "academic-tutor"}
        self.assertTrue(
            expected & set(names),
            f"未召回翻译/学术类 skill, 实际: {names}",
        )

    def test_irrelevant_query_returns_nothing(self):
        """完全无关的乱码输入不应召回 (避免误激活)。"""
        names = self._top_names("zzqqxwv")
        self.assertEqual(names, [], f"无关输入不应召回, 实际: {names}")


@unittest.skipUnless(MARKET_EXISTS and AGENT_AVAILABLE, "需要技能市场目录与 Agent 模块")
class TestAgentFindsSkillEndToEnd(unittest.TestCase):
    """端到端: Agent 运行时自动激活匹配的 skill。"""

    def setUp(self) -> None:
        self.llm = FakeChatModel()
        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']

    def _make_agent(self) -> "BaseAgent":
        return BaseAgent(llm=self.llm, use_perception=False, use_memory=False)

    def test_agent_activates_stock_skill(self):
        agent = self._make_agent()
        agent.load_skills(MARKET)
        result = agent.run("查询 A 股股票行情数据")

        self.assertTrue(result["skills"], "未激活任何 skill")
        # 激活的 skill 应包含股票类
        self.assertTrue(
            any(
                kw in " ".join(result["skills"])
                for kw in ("stock", "futu", "westock", "financial")
            ),
            f"激活的 skill 与股票无关: {result['skills']}",
        )

    def test_agent_activates_weather_skill(self):
        agent = self._make_agent()
        agent.load_skills(MARKET)
        result = agent.run("帮我查一下北京今天的天气")

        self.assertIn(
            "weather-open-meteo", result["skills"], f"实际激活: {result['skills']}"
        )

    def test_activated_instructions_injected(self):
        """激活的 skill 完整指令应注入系统提示词 (渐进式披露第二级)。"""
        agent = self._make_agent()
        agent.load_skills(MARKET)
        agent.run("帮我查一下北京今天的天气")

        system = self.llm.last_messages[0].content
        self.assertIn("已激活 Skill", system)
        self.assertIn("weather-open-meteo", system)


if __name__ == "__main__":
    unittest.main()
