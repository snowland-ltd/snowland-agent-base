# snowland-agent-base

> 中文版: [README.md](README_zh.md)

A reusable **AI Agent base class** framework, designed with a layered
"Perception → Decision → Execution → Memory" architecture and built on the
LangChain ecosystem.

## Tech Stack

| Layer | Solution | Description |
| --- | --- | --- |
| Core framework | LangChain | Built-in memory, tool-calling wrappers, reducing duplicate development |
| LLM integration | Alibaba Cloud Bailian (DashScope) | Cloud service, no local deployment or model O&M |
| Memory storage | ChromaDB | Vector knowledge base, long-term semantic retrieval, local persistence |
| Perception layer | spaCy | Entity recognition, intent classification, no need to build NLP from scratch |
| Decision layer | LLM chain-of-thought + rule engine | CoT decomposes complex tasks; rules guarantee stability as fallback |
| Execution layer | snowland-http | Unified timeout/retry, fast integration with external APIs |

## Architecture

```
User input
   │
   ▼
┌──────────────┐   entity/intent
│ Perception   │──────────────┐
│ Layer        │             │
└──────────────┘             ▼
                     ┌──────────────┐
┌──────────────┐     │  Decision    │  rule hit → output directly
│  Memory      │────▶│  Layer       │  otherwise → LLM chain-of-thought
│  Layer       │retrieve└──────┬─────┘
└──────────────┘            │ action
        ▲                   ▼
        │            ┌──────────────┐
        └────────────│  Execution   │  timeout/retry/exception handling
           write back│  Layer       │
              memory │  ToolExecutor │
                     └──────────────┘
```

`BaseAgent.run()` chains the above into a ReAct-style loop, bounded by `max_steps`
to prevent infinite loops.

## Installation

This package is installed and used as a **published distribution (wheel)**, imported
from PyPI. **Editable (`-e`) installation is not supported.**

```bash
# Method 1: Install from PyPI (recommended; import source is PyPI)
pip install snowland-agent-base

# Optional dependencies (install as needed)
pip install "snowland-agent-base[graph]"   # langgraph deps required by GraphAgent
pip install "snowland-http[httpx]"         # recommended HTTP backend for the execution layer

# Method 2: Build the wheel locally then install normally (pre-release self-test, not editable)
python -m build --wheel        # generates dist/*.whl
pip install dist/*.whl         # normal install, imported from site-packages, without -e
```

`snowland-http` requires an HTTP backend; httpx is recommended.

Download the spaCy Chinese model (required by the spaCy backend in the perception layer):

```bash
python -m spacy download zh_core_web_sm
```

Configure the Bailian API Key:

```bash
# Windows
set DASHSCOPE_API_KEY=sk-xxxxxx
# Linux / macOS
export DASHSCOPE_API_KEY=sk-xxxxxx
```

## Quick Start

```python
from langchain_dashscope import DashScopeEmbeddings

from snowland_agent_base.agent import BaseAgent
from snowland_agent_base.decision import AgentAction, Rule
from snowland_agent_base.model import build_llm

llm = build_llm(model="qwen-plus")
agent = BaseAgent(
    llm=llm,
    embedding=DashScopeEmbeddings(model="text-embedding-v2"),
    system_prompt="You are a professional e-commerce customer service assistant, friendly and concise.",
)

# Register a tool
agent.register_tool("calculator", lambda expression: eval(expression))

# Rule-based fallback (no LLM call when matched)
agent.add_rule(Rule(
    name="greeting",
    predicate=lambda text, ctx: ctx.get("intent") == "greeting",
    action=AgentAction(final_answer="Hello, how can I help you?"),
))

result = agent.run("What is 1 + 2?")
print(result["answer"])
```

Full example: `examples/quickstart.py`.

## Model Backend (Pluggable)

The model layer is **pluggable by capability type**. A backend is addressed by two
orthogonal dimensions:

- **Vendor** (`dashscope` / `orcarouter` / `zhipu` / `deepseek` / `anthropic` / `anthropic-openai` / `openai-compat`)
- **Capability** (`ModelType`: reasoning / embedding / image / video / tts / rerank)

Every backend derives from a unified root `ModelBackend` and branches into six
same-shaped interfaces (`LLMBackend` / `EmbeddingBackend` / `ImageBackend` /
`VideoBackend` / `TTSBackend` / `RerankBackend`), so callers depend only on the
abstract capability, never on the concrete vendor.

### Facade functions

```python
from snowland_agent_base.model import (
    build_llm, build_embedding, build_image, build_video, build_tts, build_rerank,
)

chat   = build_llm(model="qwen-plus")                       # BaseChatModel
emb    = build_embedding(backend="orcarouter")              # Embeddings
img    = build_image(backend="orcarouter").generate("a cat")      # ImageResult
video  = build_video(backend="dashscope").generate("waves")       # VideoResult
audio  = build_tts(backend="dashscope", voice="alex").synthesize("hello")  # AudioResult
reranker = build_rerank(backend="dashscope")                # RerankModel
ranked   = reranker.rerank(query="...", documents=[...])    # RerankResult
```

### AutoBackend — one object, all capabilities

`AutoBackend` holds a default vendor and exposes every capability as a method, so you
never repeat the backend name:

```python
from snowland_agent_base.model import AutoBackend

backend = AutoBackend(vendor="dashscope")
chat  = backend.reasoning(model="qwen-plus")
emb   = backend.embedding()
img   = backend.image(model="wanx-v1").generate("a cat")
video = backend.video(model="wanx-v1").generate("waves")
audio = backend.tts(model="cosyvoice-v1").synthesize("你好")
rank  = backend.rerank(query="...", documents=[...])   # directly returns RerankResult
```

### Supported vendors

| Vendor | Env vars (key / base_url) | Capabilities |
| --- | --- | --- |
| `dashscope` (default) | `DASHSCOPE_API_KEY` | **reasoning, embedding, image, video, tts, rerank** (all 6) |
| `orcarouter` | `ORCA_KEY` (or `ORCAROUTER_API_KEY`) / `ORCAROUTER_BASE_URL` | **reasoning, embedding, image, video, tts, rerank** (all 6) |
| `zhipu` | `ZHIPU_API_KEY` / `ZHIPU_BASE_URL` | reasoning, embedding, image, video, tts, rerank (all 6) |
| `deepseek` | `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` | reasoning, embedding, rerank |
| `anthropic` | `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` | reasoning (native Messages API) |
| `anthropic-openai` | `OPENAI_API_KEY` | reasoning (Anthropic via OpenAI-compatible) |
| `openai-compat` | `OPENAI_API_KEY` / `OPENAI_BASE_URL` | reasoning, embedding, image, video, tts, rerank (generic fallback) |

OrcaRouter is an OpenAI-compatible routing gateway that fans out to OpenAI, Anthropic,
Gemini, DeepSeek, Qwen and more, with smart routing and fallback chains:

```python
from snowland_agent_base.model import build_llm

llm = build_llm(backend="orcarouter", model="orcarouter/auto")
```

Set the API key via the `ORCA_KEY` (or `ORCAROUTER_API_KEY`) environment variable.
A reference provider config lives at `examples/providers.toml`.

If you find this project useful, sign up for OrcaRouter through our referral link —
new accounts are attributed to us at no extra cost to you:

https://www.orcarouter.ai/ref/ref_808b7e19b052dee31192

### Custom backend

Add a vendor by subclassing the matching capability interface and registering it:

```python
from snowland_agent_base.model import register_llm_backend, ModelType
from snowland_agent_base.model.backends.openai_compat import OpenAICompatLLMBackend

class MyLLMBackend(OpenAICompatLLMBackend):
    default_base_url = "https://my.api/v1"
    api_key_env = "MY_API_KEY"

register_llm_backend("myvendor", MyLLMBackend)
register_llm_backend("myvendor", MyOtherBackend, ModelType.EMBEDDING)
```

### ChatModel proxies (lazy load / empty-response retry)

The proxy module provides two `BaseChatModel` decorators that wrap an existing
ChatModel (such as the one returned by `build_llm`) and transparently add
behavior — they can be used as drop-in replacements anywhere a ChatModel is
expected:

- `LazyChatModel(builder=...)` — defers construction of the underlying LLM until
  the first real call. Handy when you do not want every Agent to spin up a
  ChatModel at construction time. Construction is double-checked and lock-guarded,
  so a single instance is safe to share across parallel tasks.
- `RetryOnEmptyChatModel(inner=..., max_retries=2)` — when the underlying model
  returns an empty `content` (a reasoning model that spent its budget on
  `reasoning_content`), it retries automatically. Normal models never trigger a
  retry, so there is zero overhead for them.

```python
from snowland_agent_base.model import build_llm, LazyChatModel, RetryOnEmptyChatModel

# Lazy + retry combined: build the LLM only on first use, and self-heal empty responses
llm = RetryOnEmptyChatModel(
    inner=LazyChatModel(builder=lambda: build_llm(model="qwen-plus")),
)
agent = BaseAgent(llm=llm, ...)
```

`BaseAgent` enables this automatically: it wraps the LLM with
`RetryOnEmptyChatModel` by default (`enable_retry=True`, `retry_max_retries=2`).
If you pass a `LazyChatModel`, the agent adds the retry layer on top, giving you
both lazy loading and self-healing for free.

## Message Model

Both conversation history and Agent output use a **message list** (`List[Message]`)
rather than concatenated strings — this preserves role structure, facilitating
persistence, multi-turn dialogue, and frontend rendering.

```python
from snowland_agent_base import user_message, assistant_message

result = agent.run("What's the weather like in Beijing?")

result["answer"]     # "Beijing is sunny today, 25°C"  —— final answer text
result["messages"]   # full conversation trace:
# [
#   {"role": "user",      "content": "What's the weather like in Beijing?"},
#   {"role": "tool",      "content": "Call succeeded: sunny, 25°C", "name": "get_weather"},
#   {"role": "assistant", "content": "Beijing is sunny today, 25°C"},
# ]
```

The `Message` structure is `{"role": ..., "content": ...}`, with `role` being one of
`system` / `user` / `assistant` / `tool` (a `tool` message carries a `name` field
identifying the tool).

History participates in multi-turn dialogue as **independent messages** passed to the
model, rather than being stuffed into the system prompt, which better fits the LLM
conversation paradigm.

`GraphAgent` also supports passing the previous round's `messages` directly as the next
round's history, enabling self-managed multi-turn dialogue:

```python
history = []
for question in ["My name is Xiao Ming", "What is my name?"]:
    result = agent.invoke(question, thread_id="multi", history=history)
    history = result["messages"]   # accumulate the full trace, bring into the next round
```

If `history` is not passed, you can also use `thread_id` to let the checkpointer
automatically restore state; both can be used together.

## Skill

Skills are used to distill domain knowledge, SOPs, and complex workflows, following the
[Agent Skills specification](https://agentskills.io/specification). They use
**progressive disclosure** with three levels of loading to avoid filling the context all
at once:

| Level | Loaded when | Content |
| --- | --- | --- |
| 1. Metadata | At startup | Only `name` + `description` (~100 tokens/skill), for the model to judge relevance |
| 2. Instructions | On activation | Full `SKILL.md` body, injected into the system prompt |
| 3. Resources | On demand | Files under `scripts/` `references/` `assets/`, read only when needed |

Two forms are supported: **instruction-type** (SOP/domain knowledge) and
**toolkit-type** (carries its own toolset, tools become available after activation).

### Register in code

```python
from snowland_agent_base import skill

agent.add_skill(skill(
    name="order-support",
    description="Query order status, process refunds. Use when the user mentions orders, shipping, or refunds.",
    instructions="## Flow\n1. Confirm the order number first...",
    tools={"get_order_status": get_order_status, "refund": refund},  # toolkit-type
))
```

### Directory scanning

```
skills/
└── pdf-processing/
    ├── SKILL.md              # required
    ├── scripts/              # optional
    ├── references/           # optional
    └── assets/               # optional
```

`SKILL.md` format (`name` / `description` required):

```markdown
---
name: pdf-processing
description: Extract text and tables from PDFs. Use when the user mentions PDF, forms, or document extraction.
metadata:
  author: snowland
---

## Standard Flow
1. Confirm the user's intent...
```

```python
agent.load_skills("./skills")   # auto-discover subdirectories containing SKILL.md
```

### Activation and observation

By default, keyword-based coarse filtering on the input auto-activates skills; the final
"whether to use" is decided by the LLM based on the catalog.

```python
result = agent.run("Extract the tables from this PDF")
result["skills"]   # ['pdf-processing']  —— skills activated this round
```

Disable auto-activation (inject the catalog only, leave it entirely to the LLM):

```python
agent = BaseAgent(llm=llm, skills=registry, auto_activate_skills=False)
```

Full examples: `examples/skill_demo.py` and `examples/skills/`.

## Two Agent Implementations

| | `BaseAgent` (default) | `GraphAgent` (requires langgraph) |
| --- | --- | --- |
| Orchestration | Hand-written ReAct loop | LangGraph `StateGraph` |
| Persistence / resumability | Not supported | checkpointer (Memory / SQLite / Postgres) |
| Human-in-the-loop | Not supported | `interrupt` native support |
| Streaming | Blocking return | `stream_events` v3 (messages / values / interrupts) |
| Multi-Agent | Not supported | subgraph / supervisor pattern |
| Dependencies | base dependencies only | additional `langgraph` |

The two have **consistent interfaces and can coexist**, with the four layers
(perception/decision/execution/memory) fully reused. Use `BaseAgent` for simple
scenarios, and `GraphAgent` when you need persistence, approval, streaming, or
multi-agent orchestration.

### GraphAgent usage

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
    approval_tools=["delete_order", "refund"],  # these tools pause for approval before execution
)
agent.register_tool("delete_order", delete_order)

# Normal invocation (same thread_id can resume from a breakpoint)
result = agent.invoke("Hello", thread_id="session-1")

# Explicitly pass conversation history (for self-managed session storage; can also coexist with thread_id)
result = agent.invoke(
    "What is my name?",
    thread_id="session-1",
    history=[
        {"role": "user", "content": "My name is Xiao Ming"},
        {"role": "assistant", "content": "Hello, Xiao Ming!"},
    ],
)

# Human-in-the-loop: pause before executing dangerous tools
stream = agent.stream_events("Delete order 12345", thread_id="session-1")
_ = stream.output                      # drive execution up to the interrupt point
if stream.interrupted:
    print(stream.interrupts[0].value)  # content pending approval
    # Approve (or pass {"approved": True, "tool_input": {...}} to rewrite the input)
    stream = agent.resume(True, thread_id="session-1")
    print(stream.output["answer"])

# Streaming output
for chunk in agent.stream("Introduce yourself", thread_id="s2", stream_mode="updates"):
    print(chunk)

# Subgraph orchestration: register a dedicated Agent as a "tool"
agent.add_subgraph("translator", translator_agent.build())
```

Full example: `examples/graph_demo.py`.

**Implementation notes**: `interrupt` is implemented by raising an exception, so the
approval logic is isolated into an `approval` node placed before tool execution to avoid
being swallowed by `ToolExecutor`'s `except Exception`; nodes re-run from the beginning,
so all state updates use "recompute based on old value" overwrite-style writes to ensure
idempotency.

## Extension Points

Inherit `BaseAgent` to customize a business Agent without modifying the main loop:

- **Add tools**: `agent.register_tool(name, func)`, with unified timeout and retry handled by the execution layer.
- **Add rules**: `agent.add_rule(Rule(...))`, for deterministic fallback of stable outputs.
- **Inject business context**: override `_build_prompt_context()` to pass user profiles, session state, etc. to the decision layer.
- **Define role persona**: pass `system_prompt="..."` at construction; override `_build_system_prompt()` when it needs to be generated dynamically per request:

  ```python
  class MyAgent(BaseAgent):
      def _build_system_prompt(self, user_input, perception_result):
          return f"Current time {now()}, please answer in Chinese. User question: {user_input}"
  ```

  > Note: If both `decision=` and `system_prompt=` are passed, the latter is ignored (the decision layer already has its own persona); in that case set it on `DecisionLayer(system_prompt=...)`.

Individual capabilities can also be replaced independently (dependency injection):
`perception` / `decision` / `executor` / `memory` can all be passed as custom
implementations at construction.

## Degradation Strategy

Failures at any layer do not block the main flow and degrade automatically:

- Perception layer (spaCy model missing) → returns `None`, decision uses default intent.
- Memory layer (Chroma unavailable) → retrieval/writing ignored, degrades to memory-less mode.
- Decision layer (LLM exception or invalid JSON) → retry then fall back to `fallback_answer`.
- Execution layer (tool throws/times out) → exponential backoff retry, eventually converging to an observation with `success=False`.

## Parallel Execution

For fanning out independent, shared-nothing tasks, use the thread-pool helpers in
`snowland_agent_base.parallel`:

- `map_parallel(fn, items, max_workers=None, catch_errors=True)` — apply `fn` to
  each item concurrently and return a list **in the same order** as `items`.
  Auto-degrades to serial when `max_workers <= 1` or there is only a single item.
- `run_parallel(tasks, max_workers=None, catch_errors=True)` — run a dict of
  **named** tasks (`{name: callable}`) concurrently and return `{name: result}`.

By default each task's exception is caught and converted to
`{"success": False, "error": str(e), ...}` so one failure does not abort the whole
batch; pass `catch_errors=False` to raise on the first error.

```python
from snowland_agent_base import map_parallel, run_parallel

# batch homogeneous tasks, order preserved
summaries = map_parallel(summarize, documents)

# named parallel steps
out = run_parallel({
    "frontend": lambda: build_ui(...),
    "backend":  lambda: build_api(...),
})
ui, api = out["frontend"], out["backend"]
```

When tasks share an LLM, wrap it with `LazyChatModel` so the underlying ChatModel
is built once and safely reused across threads.

## Testing

Tests are based on Python's standard library `unittest`, using `FakeChatModel` /
`FakeEmbeddings` to replace external dependencies, **no network required**:

```bash
python -m unittest discover -s tests -t .
```
