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

"""快速上手示例: 构建一个带工具、规则兜底和长期记忆的 Agent。

运行前准备:
    pip install -e .
    python -m spacy download zh_core_web_sm
    set DASHSCOPE_API_KEY=sk-xxxxxx   (Linux: export DASHSCOPE_API_KEY=sk-xxxxxx)

注意:
- 若不希望联网调用大模型, 可在 tests/ 中用 FakeChatModel 替代 (参见 tests/test_agent.py)。
- 若未安装 spaCy 模型, 可使用 ``use_perception=False`` 跳过感知层。
"""
from langchain_dashscope import DashScopeEmbeddings

from snowland_agent_base.agent import BaseAgent
from snowland_agent_base.decision import AgentAction, Rule
from snowland_agent_base.execution import build_http_client, request_json
from snowland_agent_base.llm import build_llm


def make_weather_tool():
    """用 snowland-http 封装外部 API 工具 (超时在请求级, 重试由执行器统一负责)。"""
    client = build_http_client(base_url="https://api.example.com")

    def get_weather(city: str) -> str:
        data = request_json(client, "get", "/weather", timeout=10.0, params={"city": city})
        return data.get("weather", "未知")

    return get_weather


def main() -> None:
    # 1. 大模型: 阿里云百炼 (DashScope)
    llm = build_llm(model="qwen-plus", temperature=0.7)

    # 2. 记忆存储: ChromaDB 向量知识库
    embedding = DashScopeEmbeddings(model="text-embedding-v2")

    # 3. 组装 Agent (system_prompt 定义角色/人设)
    agent = BaseAgent(
        llm=llm,
        embedding=embedding,
        max_steps=5,
        system_prompt=(
            "你是一个专业的电商客服助手, 语气友好简洁。"
            "回答基于商品信息, 不确定时主动说明。"
        ),
    )

    # 4. 注册执行层工具
    agent.register_tool("get_weather", make_weather_tool())
    agent.register_tool("calculator", lambda expression: eval(expression))  # noqa: S307 - 示例

    # 5. 决策层规则兜底: 命中则不调用 LLM
    agent.add_rule(
        Rule(
            name="greeting",
            predicate=lambda text, ctx: ctx.get("intent") == "greeting",
            action=AgentAction(thought="问候语, 规则直出", final_answer="你好, 有什么可以帮您?"),
        )
    )

    # 6. 运行
    result = agent.run("北京今天天气怎么样?")
    print("感知:", result["perception"])
    print("答复:", result["answer"])

    # 输出同时提供完整的 messages 列表 (user / tool / assistant)
    print("对话轨迹:")
    for msg in result["messages"]:
        print(f"  [{msg['role']}] {msg['content']}")

    for step in result["steps"]:
        print(f"  步骤{step['step']}: {step['action'].get('tool')} -> {step['observation']}")


if __name__ == "__main__":
    main()
