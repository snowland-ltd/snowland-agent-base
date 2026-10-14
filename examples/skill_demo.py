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

"""Skill 能力演示: 代码注册、目录扫描、渐进式披露、工具包型 skill。

运行前准备:
    pip install -r requirements.txt
    set DASHSCOPE_API_KEY=sk-xxxxxx

目录结构 (本示例自带):
    examples/skills/
    ├── pdf-processing/
    │   ├── SKILL.md            # frontmatter + 指令
    │   └── references/TABLES.md
    └── code-review/
        └── SKILL.md
"""
from pathlib import Path

from langchain_dashscope import DashScopeEmbeddings

from snowland_agent_base.agent import BaseAgent
from snowland_agent_base.model import build_llm
from snowland_agent_base.skill import SkillRegistry, skill

SKILLS_DIR = Path(__file__).parent / "skills"


def make_agent() -> BaseAgent:
    llm = build_llm(model="qwen-plus")
    embedding = DashScopeEmbeddings(model="text-embedding-v2")
    return BaseAgent(llm=llm, embedding=embedding)


def demo_code_registration(agent: BaseAgent) -> None:
    """方式一: 代码注册 (工具包型 skill, 携带自己的工具)。"""
    print("\n=== 1. 代码注册 skill ===")

    def get_order_status(order_id: str) -> str:
        return f"订单 {order_id} 已发货"

    def refund(order_id: str, amount: float) -> str:
        return f"已为订单 {order_id} 退款 {amount} 元"

    agent.add_skill(
        skill(
            name="order-support",
            description="查询订单状态、办理退款。当用户提到订单、物流、退款时使用。",
            instructions=(
                "## 流程\n"
                "1. 先确认订单号; 缺少时向用户询问, 不要猜测。\n"
                "2. 查询状态用 get_order_status。\n"
                "3. 退款需用户明确金额, 调用 refund。\n"
            ),
            tools={"get_order_status": get_order_status, "refund": refund},
        )
    )
    print("已注册 skill:", agent.skills.names)
    print("skill 携带的工具已注册:", agent.executor.list_tools())


def demo_directory_scan(agent: BaseAgent) -> None:
    """方式二: 目录扫描, 自动发现含 SKILL.md 的子目录。"""
    print("\n=== 2. 目录扫描加载 skill ===")
    if not SKILLS_DIR.is_dir():
        print(f"目录不存在, 跳过: {SKILLS_DIR}")
        return

    loaded = agent.load_skills(SKILLS_DIR)
    print(f"从 {SKILLS_DIR} 加载了 {len(loaded)} 个 skill:")
    for s in loaded:
        print(f"  - {s.name}: {s.description}")
        if s.list_resources():
            print(f"      资源: {s.list_resources()}")


def demo_progressive_disclosure(agent: BaseAgent) -> None:
    """渐进式披露: 目录常驻, 指令按需加载。"""
    print("\n=== 3. 渐进式披露 ===")
    print("第一级 - 目录 (常驻 system prompt, 约 100 tokens/skill):")
    print(agent.skills.render_catalog())

    # 第二级: 命中后才加载完整指令
    matched = agent.skills.match("帮我提取这个 PDF 的表格")
    print("\n第二级 - 按输入召回的 skill:", [s.name for s in matched])
    for s in matched:
        print(f"\n--- {s.name} 的完整指令 ---")
        print(s.render_instructions())

    # 第三级: 按需读取资源
    for s in matched:
        for resource in s.list_resources():
            print(f"\n--- {s.name}/{resource} (按需读取) ---")
            print(s.read_resource(resource)[:200])


def demo_run(agent: BaseAgent) -> None:
    """实际运行: 观察 skill 是否被激活。"""
    print("\n=== 4. 运行 Agent ===")
    for question in ["帮我提取 PDF 里的表格", "今天天气怎么样"]:
        result = agent.run(question)
        print(f"\n问题: {question}")
        print(f"激活的 skill: {result['skills']}")
        print(f"答复: {result['answer']}")


def main() -> None:
    agent = make_agent()
    demo_code_registration(agent)
    demo_directory_scan(agent)
    demo_progressive_disclosure(agent)
    demo_run(agent)


if __name__ == "__main__":
    main()
