# snowland-agent-base

> English version: [README_EN.md](README.md)

一个可复用的 **AI Agent 基类**框架，按「感知 → 决策 → 执行 → 记忆」分层设计，基于 LangChain 生态搭建。

## 技术选型

| 层次 | 方案 | 说明 |
| --- | --- | --- |
| 核心框架 | LangChain | 内置记忆、工具调用等封装，减少重复开发 |
| 大模型接入 | 阿里云百炼 (DashScope) | 云端服务，无需本地部署与模型运维 |
| 记忆存储 | ChromaDB | 向量知识库，长期记忆语义检索，本地持久化 |
| 感知层 | spaCy | 实体识别、意图分类，无需从零开发 NLP |
| 决策层 | LLM 链式思考 + 规则引擎 | CoT 拆解复杂任务，规则兜底保证稳定性 |
| 执行层 | snowland-http | 统一超时、重试，快速对接外部 API |

## 架构

```
用户输入
   │
   ▼
┌──────────────┐   实体/意图
│  感知层       │──────────────┐
│ PerceptionLayer│             │
└──────────────┘             ▼
                     ┌──────────────┐
┌──────────────┐     │  决策层       │  规则命中 → 直接输出
│  记忆层       │────▶│ DecisionLayer│  否则 → LLM 链式思考
│ VectorMemory │ 检索 └──────┬───────┘
└──────────────┘            │ 动作
        ▲                   ▼
        │            ┌──────────────┐
        └────────────│  执行层       │ 超时/重试/异常收敛
          写回记忆    │ ToolExecutor │
                     └──────────────┘
```

`BaseAgent.run()` 将上述流程串成 ReAct 风格循环，受 `max_steps` 限制防止死循环。

## 安装

本包以**已发布的发行版（wheel）**形式安装使用，从 PyPI 导入，**不支持 editable（可编辑）安装**。

```bash
# 方式一：从 PyPI 安装（推荐，导入来源即 PyPI）
pip install snowland-agent-base

# 可选依赖（按需安装）
pip install "snowland-agent-base[graph]"   # GraphAgent 需要的 langgraph 相关依赖
pip install "snowland-http[httpx]"         # 执行层推荐的 HTTP 后端

# 方式二：本地构建 wheel 后常规安装（发布前自测用，非 editable）
python -m build --wheel        # 生成 dist/*.whl
pip install dist/*.whl         # 常规安装，从 site-packages 导入，不带 -e
```

`snowland-http` 需搭配一个 HTTP 后端使用，推荐 httpx：

下载 spaCy 中文模型（感知层 spaCy 后端需要）：

```bash
python -m spacy download zh_core_web_sm
```

配置百炼 API Key：

```bash
# Windows
set DASHSCOPE_API_KEY=sk-xxxxxx
# Linux / macOS
export DASHSCOPE_API_KEY=sk-xxxxxx
```

## 快速使用

```python
from langchain_dashscope import DashScopeEmbeddings

from snowland_agent_base.agent import BaseAgent
from snowland_agent_base.decision import AgentAction, Rule
from snowland_agent_base.model import build_llm

llm = build_llm(model="qwen-plus")
agent = BaseAgent(
    llm=llm,
    embedding=DashScopeEmbeddings(model="text-embedding-v2"),
    system_prompt="你是一个专业的电商客服助手, 语气友好简洁。",
)

# 注册工具
agent.register_tool("calculator", lambda expression: eval(expression))

# 规则兜底（命中则不调用大模型）
agent.add_rule(Rule(
    name="greeting",
    predicate=lambda text, ctx: ctx.get("intent") == "greeting",
    action=AgentAction(final_answer="你好，有什么可以帮您？"),
))

result = agent.run("1 + 2 等于几？")
print(result["answer"])
```

完整示例见 `examples/quickstart.py`。

## 模型后端（可插拔）

模型层按**能力类型**可插拔。一个后端由两条正交维度定位：

- **厂商**（vendor）：`dashscope` / `orcarouter` / `zhipu` / `deepseek` / `anthropic` / `anthropic-openai` / `openai-compat`
- **能力**（capability）：`ModelType`：推理 / 向量 / 图像 / 视频 / 语音 / 重排

所有后端继承自统一抽象根 `ModelBackend`，并派生出六种同形态接口（`LLMBackend` / `EmbeddingBackend` / `ImageBackend` / `VideoBackend` / `TTSBackend` / `RerankBackend`），调用方只依赖抽象能力，不感知具体厂商。

### 门面函数

```python
from snowland_agent_base.model import (
    build_llm, build_embedding, build_image, build_video, build_tts, build_rerank,
)

chat   = build_llm(model="qwen-plus")                           # BaseChatModel
emb    = build_embedding(backend="orcarouter")                  # Embeddings
img    = build_image(backend="orcarouter").generate("一只猫")     # ImageResult
video  = build_video(backend="dashscope").generate("海浪")         # VideoResult
audio  = build_tts(backend="dashscope", voice="alex").synthesize("你好")  # AudioResult
reranker = build_rerank(backend="dashscope")                    # RerankModel
ranked   = reranker.rerank(query="...", documents=[...])        # RerankResult
```

### AutoBackend —— 一个对象，调全部能力

`AutoBackend` 持有一个默认厂商，把各能力作为方法暴露，无需每个调用都重复写 backend：

```python
from snowland_agent_base.model import AutoBackend

backend = AutoBackend(vendor="dashscope")
chat  = backend.reasoning(model="qwen-plus")
emb   = backend.embedding()
img   = backend.image(model="wanx-v1").generate("一只猫")
video = backend.video(model="wanx-v1").generate("海浪")
audio = backend.tts(model="cosyvoice-v1").synthesize("你好")
rank  = backend.rerank(query="...", documents=[...])   # 直接返回 RerankResult
```

### 支持的厂商

| 厂商 | 环境变量（key / base_url） | 能力 |
| --- | --- | --- |
| `dashscope`（默认） | `DASHSCOPE_API_KEY` | **推理、向量、图像、视频、语音、重排**（全 6 类） |
| `orcarouter` | `ORCA_KEY`（或 `ORCAROUTER_API_KEY`）/ `ORCAROUTER_BASE_URL` | **推理、向量、图像、视频、语音、重排**（全 6 类） |
| `zhipu` | `ZHIPU_API_KEY` / `ZHIPU_BASE_URL` | 推理、向量、图像、视频、语音、重排（全 6 类） |
| `deepseek` | `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` | 推理、向量、重排 |
| `anthropic` | `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` | 推理（原生 Messages API） |
| `anthropic-openai` | `OPENAI_API_KEY` | 推理（Anthropic 走 OpenAI 兼容） |
| `openai-compat` | `OPENAI_API_KEY` / `OPENAI_BASE_URL` | 推理、向量、图像、视频、语音、重排（通用兜底） |

OrcaRouter 是一个 OpenAI 兼容的路由网关，可在 OpenAI、Anthropic、Gemini、DeepSeek、Qwen 等模型间路由，支持智能路由与 fallback 链：

```python
from snowland_agent_base.model import build_llm

llm = build_llm(backend="orcarouter", model="orcarouter/auto")
```

通过 `ORCA_KEY`（或 `ORCAROUTER_API_KEY`）环境变量设置 API Key。参考 provider 配置见 `examples/providers.toml`。

如果你觉得本项目有用，可以通过我们的推广链接注册 OrcaRouter —— 新账户会自动归因到我们，对你没有任何额外成本：

https://www.orcarouter.ai/ref/ref_808b7e19b052dee31192

### 自定义后端

继承对应能力接口并注册即可新增厂商：

```python
from snowland_agent_base.model import register_llm_backend, ModelType
from snowland_agent_base.model.backends.openai_compat import OpenAICompatLLMBackend

class MyLLMBackend(OpenAICompatLLMBackend):
    default_base_url = "https://my.api/v1"
    api_key_env = "MY_API_KEY"

register_llm_backend("myvendor", MyLLMBackend)
register_llm_backend("myvendor", MyOtherBackend, ModelType.EMBEDDING)
```

### ChatModel 代理（懒加载 / 空响应重试）

代理模块提供两个 `BaseChatModel` 装饰器，包裹已有的 ChatModel（如 `build_llm` 返回的对象），对调用方完全透明——可在任何需要 ChatModel 的地方直接替换使用：

- `LazyChatModel(builder=...)`：延迟到底层 LLM 的首次真正调用时才构造。适合不想让每个 Agent 在构造期就拉起 ChatModel 的场景。构造过程采用双重检查 + 加锁，可安全地在并行任务间共享同一实例。
- `RetryOnEmptyChatModel(inner=..., max_retries=2)`：当底层模型返回空 `content`（推理模型把额度花在 `reasoning_content` 上、撞满后一个 token 都不剩）时自动重试。正常模型永不触发重试，零额外开销。

```python
from snowland_agent_base.model import build_llm, LazyChatModel, RetryOnEmptyChatModel

# 懒加载 + 重试组合：仅在首次使用时构建 LLM，并对空响应自愈
llm = RetryOnEmptyChatModel(
    inner=LazyChatModel(builder=lambda: build_llm(model="qwen-plus")),
)
agent = BaseAgent(llm=llm, ...)
```

`BaseAgent` 默认就开启了这一能力：它会用 `RetryOnEmptyChatModel` 自动包裹 LLM（`enable_retry=True`、`retry_max_retries=2`）。若你传入的是 `LazyChatModel`，Agent 会在外层再包一层重试，同时获得「懒加载 + 重试」。

## 消息模型

历史会话与 Agent 输出均使用 **消息列表**（`List[Message]`），而非拼接的字符串 —— 这样能保留角色结构，便于持久化、多轮对话和前端渲染。

```python
from snowland_agent_base import user_message, assistant_message

result = agent.run("北京天气怎么样？")

result["answer"]     # "北京今天晴，25℃"  —— 最终答复文本
result["messages"]   # 完整对话轨迹：
# [
#   {"role": "user",      "content": "北京天气怎么样？"},
#   {"role": "tool",      "content": "调用成功: 晴，25℃", "name": "get_weather"},
#   {"role": "assistant", "content": "北京今天晴，25℃"},
# ]
```

`Message` 结构为 `{"role": ..., "content": ...}`，`role` 取值 `system` / `user` / `assistant` / `tool`（`tool` 消息带 `name` 字段标识工具名）。

历史会以**独立消息**参与多轮对话传给模型，而不是塞进系统提示词，更符合 LLM 的对话范式。

`GraphAgent` 还支持把上一轮的 `messages` 直接作为下一轮的历史传入，实现自行托管的多轮对话：

```python
history = []
for question in ["我叫小明", "我叫什么名字？"]:
    result = agent.invoke(question, thread_id="multi", history=history)
    history = result["messages"]   # 累积完整轨迹，带入下一轮
```

若不传 `history`，也可用 `thread_id` 让 checkpointer 自动恢复状态；两者可同时使用。

## Skill

Skill 用于沉淀领域知识、SOP 和复杂工作流，遵循 [Agent Skills 规范](https://agentskills.io/specification)，采用**渐进式披露**分三级加载，避免一次性占满上下文：

| 级别 | 加载时机 | 内容 |
| --- | --- | --- |
| 1. 元数据 | 启动时 | 仅 `name` + `description`（~100 tokens/skill），供模型判断相关性 |
| 2. 指令 | 激活时 | 完整 `SKILL.md` 正文，注入系统提示词 |
| 3. 资源 | 按需 | `scripts/` `references/` `assets/` 下的文件，需要时才读 |

支持两种形态：**指令型**（SOP/领域知识）与**工具包型**（携带自己的工具集，激活后工具可用）。

### 代码注册

```python
from snowland_agent_base import skill

agent.add_skill(skill(
    name="order-support",
    description="查询订单状态、办理退款。当用户提到订单、物流、退款时使用。",
    instructions="## 流程\n1. 先确认订单号...",
    tools={"get_order_status": get_order_status, "refund": refund},  # 工具包型
))
```

### 目录扫描

```
skills/
└── pdf-processing/
    ├── SKILL.md              # 必需
    ├── scripts/              # 可选
    ├── references/           # 可选
    └── assets/               # 可选
```

`SKILL.md` 格式（`name` / `description` 必填）：

```markdown
---
name: pdf-processing
description: 提取 PDF 文本与表格。当用户提到 PDF、表单、文档抽取时使用。
metadata:
  author: snowland
---

## 标准流程
1. 确认用户意图...
```

```python
agent.load_skills("./skills")   # 自动发现含 SKILL.md 的子目录
```

### 激活与观测

默认按输入做词面粗筛自动激活；最终「用不用」由 LLM 依据目录判断。

```python
result = agent.run("帮我提取 PDF 的表格")
result["skills"]   # ['pdf-processing']  —— 本轮激活的 skill
```

关闭自动激活（仅注入目录，完全交给 LLM 判断）：

```python
agent = BaseAgent(llm=llm, skills=registry, auto_activate_skills=False)
```

完整示例见 `examples/skill_demo.py` 与 `examples/skills/`。

## 两种 Agent 实现

| | `BaseAgent`（默认） | `GraphAgent`（需 langgraph） |
| --- | --- | --- |
| 编排方式 | 手写 ReAct 循环 | LangGraph `StateGraph` |
| 持久化 / 断点续跑 | 不支持 | checkpointer（Memory / SQLite / Postgres） |
| 人机介入 | 不支持 | `interrupt` 原生支持 |
| 流式输出 | 阻塞返回 | `stream_events` v3（messages / values / interrupts） |
| 多 Agent | 不支持 | 子图 / supervisor 模式 |
| 依赖 | 仅基础依赖 | 额外 `langgraph` |

两者**接口一致、可并存**，四层（感知/决策/执行/记忆）完全复用。简单场景用 `BaseAgent`，需要持久化、审批、流式或多 Agent 时用 `GraphAgent`。

### GraphAgent 用法

```bash
pip install langgraph langgraph-checkpoint-sqlite
```

```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from snowland_agent_base.graph import GraphAgent

agent = GraphAgent(
    llm=llm,
    embedding=embedding,
    checkpointer=SqliteSaver(sqlite3.connect("agent.db", check_same_thread=False)),
    approval_tools=["delete_order", "refund"],  # 这些工具执行前需人工审批
)
agent.register_tool("delete_order", delete_order)

# 普通调用（同一 thread_id 可断点续跑）
result = agent.invoke("你好", thread_id="session-1")

# 显式传入历史会话（自行管理会话存储时用；亦可与 thread_id 并存）
result = agent.invoke(
    "我叫什么名字？",
    thread_id="session-1",
    history=[
        {"role": "user", "content": "我叫小明"},
        {"role": "assistant", "content": "你好，小明！"},
    ],
)

# 人机介入：危险工具执行前暂停
stream = agent.stream_events("删除订单 12345", thread_id="session-1")
_ = stream.output                      # 驱动执行到中断点
if stream.interrupted:
    print(stream.interrupts[0].value)  # 待审批内容
    # 批准（亦可传 {"approved": True, "tool_input": {...}} 改写入参）
    stream = agent.resume(True, thread_id="session-1")
    print(stream.output["answer"])

# 流式输出
for chunk in agent.stream("介绍一下你自己", thread_id="s2", stream_mode="updates"):
    print(chunk)

# 子图编排：把专用 Agent 注册为「工具」
agent.add_subgraph("translator", translator_agent.build())
```

完整示例见 `examples/graph_demo.py`。

**实现要点**：`interrupt` 靠抛异常实现，因此审批逻辑独立成 `approval` 节点且置于工具执行之前，避免被 `ToolExecutor` 的 `except Exception` 吞掉；节点重跑时从开头执行，故所有 state 更新均为「基于旧值重算」的覆盖式写法以保证幂等。

## 扩展点

继承 `BaseAgent` 即可定制业务 Agent，无需改动主循环：

- **加工具**：`agent.register_tool(name, func)`，由执行层统一处理超时与重试。
- **加规则**：`agent.add_rule(Rule(...))`，用于稳定输出的确定性兜底。
- **注入业务上下文**：覆盖 `_build_prompt_context()`，把用户画像、会话状态等传给决策层。
- **定义角色人设**：构造时传 `system_prompt="..."`；需按请求动态生成时覆盖 `_build_system_prompt()`：

  ```python
  class MyAgent(BaseAgent):
      def _build_system_prompt(self, user_input, perception_result):
          return f"当前时间 {now()}，请用中文回答。用户问题：{user_input}"
  ```

  > 注意：若同时传入 `decision=` 和 `system_prompt=`，后者会被忽略（决策层已自带人设），此时应在 `DecisionLayer(system_prompt=...)` 上设置。

单项能力也可独立替换（依赖注入）：`perception` / `decision` / `executor` / `memory` 均可在构造时传入自定义实现。

## 降级策略

各层故障均不阻断主流程，自动降级：

- 感知层（spaCy 模型缺失）→ 返回 `None`，决策走默认意图。
- 记忆层（Chroma 不可用）→ 检索/写入忽略，退化为无记忆模式。
- 决策层（LLM 异常或返回非法 JSON）→ 重试后回退 `fallback_answer`。
- 执行层（工具抛错/超时）→ 指数退避重试，最终收敛为 `success=False` 的观测结果。

## 并发执行

对于「无共享状态任务」扇出，使用 `snowland_agent_base.parallel` 中的线程池工具：

- `map_parallel(fn, items, max_workers=None, catch_errors=True)`：把 `fn` 并发映射到 `items`，返回与 `items` **顺序一致**的结果列表。`max_workers <= 1` 或只有一个元素时自动退化为串行。
- `run_parallel(tasks, max_workers=None, catch_errors=True)`：并发执行一批**命名任务**（`{name: callable}`），返回 `{name: result}`。

默认每个任务的异常都会被捕获并转换为 `{"success": False, "error": str(e), ...}`，单个任务失败不会中断整体；传 `catch_errors=False` 则首个异常原样抛出。

```python
from snowland_agent_base import map_parallel, run_parallel

# 批量同构任务，顺序保持一致
summaries = map_parallel(summarize, documents)

# 命名并行步骤
out = run_parallel({
    "frontend": lambda: build_ui(...),
    "backend":  lambda: build_api(...),
})
ui, api = out["frontend"], out["backend"]
```

当多个任务共用同一个 LLM 时，用 `LazyChatModel` 包裹，可让底层 ChatModel 只构建一次并在多线程间安全复用。

## 测试

测试基于 Python 标准库 `unittest`，使用 `FakeChatModel` / `FakeEmbeddings` 替代外部依赖，**无需联网**：

```bash
python -m unittest discover -s tests -t .
```
