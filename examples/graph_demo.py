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

"""GraphAgent 能力演示: 持久化 / 人机介入 / 流式输出 / 子图编排。

运行前准备:
    pip install -r requirements.txt
    pip install langgraph langgraph-checkpoint-sqlite
    python -m spacy download zh_core_web_sm
    set DASHSCOPE_API_KEY=sk-xxxxxx
"""
# langchain-dashscope 0.1.8 仍 import 已被移除的 langchain_core.pydantic_v1;
# 在导入它之前注入 pydantic.v1 兼容 shim (不改动任何已安装包)。
from snowland_agent_base._compat import ensure_pydantic_v1_shim

ensure_pydantic_v1_shim()
from langchain_dashscope import DashScopeEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver

from snowland_agent_base.graph import GraphAgent
from snowland_agent_base.llm import build_llm


def make_agent(thread_id: str) -> GraphAgent:
    """构建一个带持久化 checkpointer 的 GraphAgent。"""
    import sqlite3

    llm = build_llm(model="qwen-plus")
    embedding = DashScopeEmbeddings(model="text-embedding-v2")

    # SQLite checkpointer: 进程重启后仍可恢复 (生产可用 PostgresSaver)
    conn = sqlite3.connect("agent_state.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return GraphAgent(
        llm=llm,
        embedding=embedding,
        checkpointer=checkpointer,
        max_steps=5,
        # 这些工具执行前需人工审批
        approval_tools=["delete_order", "refund"],
    )


def demo_persistence(agent: GraphAgent) -> None:
    """持久化: 同一 thread_id 跨调用保留状态, 可断点续跑。"""
    print("\n=== 1. 持久化 / 断点续跑 ===")
    agent.invoke("我叫小明", thread_id="user-1")
    result = agent.invoke("我叫什么名字?", thread_id="user-1")
    print("答复:", result["answer"])

    # 随时查看该线程的状态快照
    state = agent.get_state("user-1")
    print("已保存步数:", len(state.values.get("steps") or []))


def demo_human_in_the_loop(agent: GraphAgent) -> None:
    """人机介入: 危险工具执行前暂停等待审批。"""
    print("\n=== 2. 人机介入 (interrupt) ===")
    agent.register_tool("delete_order", lambda order_id: f"订单 {order_id} 已删除")

    stream = agent.stream_events("请删除订单 12345", thread_id="user-2")
    _ = stream.output  # 驱动流执行到中断点

    if stream.interrupted:
        payload = stream.interrupts[0].value
        print("待审批:", payload)

        # 模拟人工决策: 批准(可同时改写入参)
        resumed = agent.resume({"approved": True}, thread_id="user-2")
        print("结果:", resumed.output["answer"])
    else:
        print("未触发审批, 直接完成:", stream.output.get("answer"))


def demo_history(agent: GraphAgent) -> None:
    """显式传入历史会话列表 (自行管理会话存储时使用)。

    若不传 history, 也可用 thread_id 让 checkpointer 自动恢复。
    """
    print("\n=== 5. 传入历史会话 ===")
    history = [
        {"role": "user", "content": "我叫小明"},
        {"role": "assistant", "content": "你好, 小明!"},
    ]
    result = agent.invoke("我叫什么名字?", thread_id="user-5", history=history)
    print("答复:", result["answer"])

    print("完整轨迹 (含历史):")
    for msg in result["messages"]:
        print(f"  [{msg['role']}] {msg['content']}")


def demo_streaming(agent: GraphAgent) -> None:
    """流式输出: 按节点增量产出。"""
    print("\n=== 3. 流式输出 ===")
    for chunk in agent.stream("介绍一下你自己", thread_id="user-3", stream_mode="updates"):
        for node_name, update in chunk.items():
            print(f"  [{node_name}] ->", update)


def demo_subgraph(parent: GraphAgent, llm) -> None:
    """子图编排: 把专用 Agent 注册为父 Agent 的「工具」。"""
    print("\n=== 4. 子图编排 (多 Agent) ===")

    # 子 Agent: 专职翻译
    translator = GraphAgent(llm=llm, use_perception=False, use_memory=False)
    translator.register_tool("translate", lambda text: f"[译文] {text}")

    # 父 Agent 挂载子图; 决策层若输出 tool=="translator" 则路由进子图
    parent.add_subgraph("translator", translator.build())
    result = parent.invoke("帮我把这段话翻译成英文", thread_id="user-4")
    print("答复:", result["answer"])


def main() -> None:
    llm = build_llm(model="qwen-plus")
    agent = make_agent(thread_id="demo")

    demo_persistence(agent)
    demo_human_in_the_loop(agent)
    demo_streaming(agent)
    demo_subgraph(agent, llm)
    demo_history(agent)


if __name__ == "__main__":
    main()
