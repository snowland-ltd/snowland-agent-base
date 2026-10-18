# snowland-agent-base API Reference (v0.2.1 English)

> Version: **v0.2.1**.
> This document describes the stable public API of `snowland-agent-base` using the format
> **Function / Parameters / Returns**. The top-level package `snowland_agent_base` exports the
> main entry points; all `build_*` facade functions (including `build_llm`) are exported from
> `snowland_agent_base.model`, and `build_llm` is additionally re-exported from the top level.

```python
import snowland_agent_base as sab
# main entry facades from the top level:
from snowland_agent_base import (
    BaseAgent,
    GraphAgent,
    build_llm,
    VectorMemory,
    PerceptionLayer,
    SkillRegistry,
    Message,
    OpenAIMessage,
    AnthropicMessage,
    map_parallel,
    run_parallel,
)

# v0.2: multi-capability facades live under .model:
from snowland_agent_base.model import (
    build_embedding,
    build_image,
    build_video,
    build_tts,
    build_rerank,
    build_model,
    AutoBackend,
    ModelType,
    LazyChatModel,
    RetryOnEmptyChatModel,
)
```

Concrete backend implementations (`DashScopeLLMBackend`, `OpenAICompatLLMBackend`,
`OrcaRouterImageBackend`, `ZhipuVideoBackend`, `DeepSeekRerankBackend`, `ChromaMemoryBackend`,
`SpacyPerceptionBackend`, `FileSkillBackend`, `IdfSkillMatcher`, ...) and the registration /
factory functions (`register_llm_backend`, `get_message_class`, ...) should be imported from
their submodules:

```python
from snowland_agent_base.model import (
    DashScopeLLMBackend,
    OrcaRouterLLMBackend,
    ZhipuLLMBackend,
    DeepSeekLLMBackend,
    AnthropicLLMBackend,
    OpenAICompatLLMBackend,
    register_llm_backend,
    get_llm_backend,
)
from snowland_agent_base.memory import ChromaMemoryBackend
from snowland_agent_base.skill import FileSkillBackend, IdfSkillMatcher
```

> Note: Chapters 1–5 and 7–8 are unchanged from v0.1 (the core framework is stable); **Chapter 6
> "Model Integration" was substantially extended in v0.2** — it evolved from a single reasoning
> backend into a pluggable model layer classified by *capability type*.

---

## 1. Agent

### 1.1 `BaseAgent` — hand-written ReAct-loop Agent

**Function**: Wires the four layers (perception → decision → execution → memory) and runs a
complete inference, returning a structured result.

**Constructor parameters** (`BaseAgent(...)`):

| Param | Type | Default | Description |
| --- | --- | --- | --- |
| `llm` | `BaseChatModel` | required | LangChain ChatModel (use `build_llm()`) |
| `embedding` | `Embeddings` | `None` | Required when long-term memory is enabled |
| `perception` | `PerceptionLayer` / `PerceptionBackend` | `None` | Perception layer; auto-built unless `use_perception=False` |
| `decision` | `DecisionLayer` | `None` | Decision layer; auto-built from `llm` (then `system_prompt` applies) |
| `executor` | `ToolExecutor` | `None` | Execution layer; auto-built |
| `memory` | `VectorMemory` / `MemoryBackend` | `None` | Memory layer; auto-built if `embedding` given |
| `max_steps` | `int` | `5` | Max reasoning steps per run (prevents infinite loops) |
| `use_perception` | `bool` | `True` | Enable perception (off to drop spaCy dependency) |
| `use_memory` | `bool` | `True` | Enable long-term memory |
| `system_prompt` | `str` | `None` | Persona/system prompt; only used when `decision` is not passed |
| `skills` | `SkillRegistry` | `None` | Skill registry |
| `auto_activate_skills` | `bool` | `True` | Auto-activate relevant skills by keyword match |

**Main methods**:

#### `run(user_input: str) -> Dict[str, Any]`
- **Function**: Run a complete inference pass.
- **Param**: `user_input` — the user input text.
- **Returns**: dict with keys `input`, `perception`, `steps` (per-step action/observation),
  `answer` (final answer text), `messages` (`Message` list: memory → user input → tool
  observations → assistant answer), `skills` (activated skill names), `success` (`bool`).

#### `register_tool(name: str, func: callable) -> None`
- **Function**: Register an executable tool.
- **Params**: `name` — tool name; `func` — callable.

#### `add_rule(rule: Rule) -> None`
- **Function**: Register a deterministic fallback rule.
- **Param**: `rule` — a `Rule` instance.

#### `add_skill(skill_obj: Skill, overwrite: bool = False) -> None`
- **Function**: Register a skill and register its tools into the executor.
- **Params**: `skill_obj` — `Skill`; `overwrite` — overwrite on name collision.

#### `load_skills(directory: Any, overwrite: bool = False) -> List[Skill]`
- **Function**: Scan a directory and load skills (auto-discovers subdirs containing `SKILL.md`).
- **Params**: `directory` — skills root or a single skill dir; `overwrite` — overwrite on collision.
- **Returns**: `List[Skill]` loaded this call.

#### Extension points (override in subclass)
- `_build_prompt_context(user_input, perception_result) -> Dict` — inject business context.
- `_build_system_prompt(user_input, perception_result) -> Optional[str]` — generate persona
  dynamically per request; return `None` to use the constructor `system_prompt`.

---

### 1.2 `GraphAgent(BaseAgent)` — LangGraph-orchestrated Agent (optional `langgraph` dep)

**Function**: Built on `BaseAgent`'s four layers, uses a `StateGraph` to add persistence /
resumability, human-in-the-loop, streaming, and multi-agent / subgraph orchestration.

**Constructor parameters**: all `BaseAgent` parameters plus:

| Param | Type | Default | Description |
| --- | --- | --- | --- |
| `checkpointer` | `Any` | `InMemorySaver()` | LangGraph checkpointer; use `SqliteSaver` for persistence |
| `store` | `Any` | `None` | Cross-session long-term store |
| `approval_tools` | `Sequence[str]` | `None` | Tool names requiring human approval before execution |
| `require_approval` | `Callable[[str, dict], bool]` | `None` | Custom approval predicate, takes precedence over `approval_tools` |

**Main methods**:

#### `run(user_input, thread_id: str = None, history = None) -> Dict[str, Any]`
- Same as `invoke`; returns additionally `thread_id`, `interrupted` (`bool`), `interrupts`.
- Use for drop-in replacement of `BaseAgent`.

#### `invoke(user_input, thread_id: str = None, history = None) -> Dict[str, Any]`
- **Function**: Run one conversation turn synchronously.
- **Params**: `history` — prior history (`Message` list or str); `thread_id` — session thread
  id for resumability. Both can be combined.
- **Returns**: structured result (same as `run`, plus `thread_id`/`interrupted`/`interrupts`).

#### `stream_events(user_input = None, thread_id = None, version = "v3", history = None)`
- **Function**: Event-streaming run (`messages`/`values`/`interrupts`), for human-in-the-loop.
- **Returns**: LangGraph stream object with `.output`, `.interrupted`, `.interrupts`.

#### `stream(user_input, thread_id = None, stream_mode = "values", history = None)`
- **Function**: Stream state snapshots / deltas.
- **Param**: `stream_mode` — `"values"` (full snapshot) / `"updates"` (delta) / `"messages"` (token stream).

#### `resume(decision, thread_id: str)`
- **Function**: Resume execution at the `interrupt` point.
- **Param**: `decision` — `bool` or `{"approved": bool, "tool_input": {...}}` (rewrite input at approval); `thread_id` — must match the interrupted thread.

#### `get_state(thread_id: str)` / `update_state(thread_id: str, values: dict)`
- **Function**: Read thread state snapshot / manually modify state (time-travel).

#### `add_subgraph(name: str, subgraph) -> None`
- **Function**: Register a compiled subgraph as a "tool" callable by the decision layer (takes effect after `build()`).

#### `build() -> Any`
- **Function**: Build and compile the `StateGraph` (with checkpointer/store).

---

## 2. Decision Layer

### 2.1 `DecisionLayer` — rule fallback + LLM chain-of-thought

**Constructor parameters** (`DecisionLayer(...)`):

| Param | Type | Default | Description |
| --- | --- | --- | --- |
| `llm` | `BaseChatModel` | required | LangChain ChatModel |
| `system_prompt` | `str` | `None` | Persona prompt; default if omitted |
| `rules` | `List[Rule]` | `None` | Rule list, matched in order |
| `fallback_answer` | `str` | `"抱歉, 我暂时无法处理这个请求。"` | Fallback answer on parse/call failure |
| `max_parse_retries` | `int` | `1` | Retries when JSON parsing fails |
| `skills` | `SkillRegistry` | `None` | Skill registry; its catalog is pinned into the system prompt |

**Main methods**:

#### `decide(user_input, tools=None, history=None, context=None, system_prompt=None, skills=None) -> AgentAction`
- **Function**: Produce the next action (rules first, else LLM CoT, else fallback).
- **Params**: `tools` — available tool names; `history` — conversation history; `context` —
  perception context (intent/entities); `system_prompt` — dynamic persona; `skills` — activated skills.
- **Returns**: `AgentAction`.

#### `add_rule(rule: Rule) -> None` / `match_rule(user_input, context) -> Optional[AgentAction]`
- Append a rule / return the first matched action.

#### `build_prompt(tools=None, system_prompt=None, skills=None) -> str`
- **Function**: Assemble system prompt (persona + CoT instructions + tool list + skill catalog/instructions).
- **Returns**: prompt text.

#### `build_messages(user_input, tools=None, history=None, system_prompt=None, skills=None) -> List`
- **Function**: Assemble full LangChain message sequence (system + history + input).
- **Returns**: list of LangChain message objects.

### 2.2 `AgentAction` — decision action (dataclass)
**Fields**: `thought: str`, `tool: Optional[str]`, `tool_input: Dict`, `final_answer: Optional[str]`.
- **Property** `is_final: bool` — whether a final answer is given (no tool call needed).
- **Method** `to_dict() -> Dict`.

### 2.3 `Rule` — rule (dataclass)
**Fields**: `name: str`, `predicate: Callable[[str, dict], bool]`, `action: AgentAction`.

---

## 3. Execution Layer

### 3.1 `ToolExecutor` — tool executor

**Constructor parameters** (`ToolExecutor(...)`):

| Param | Type | Default | Description |
| --- | --- | --- | --- |
| `timeout` | `float` | `30.0` | Timeout (seconds) per call |
| `max_retries` | `int` | `3` | Max retries |
| `retry_interval` | `float` | `1.0` | Base retry interval (exponential backoff) |

**Main methods**:

#### `register(name: str, func: Callable) -> None`
- **Function**: Register a tool (raises `TypeError` if not callable).

#### `unregister(name: str) -> None`
- **Function**: Unregister a tool (silent if absent).

#### `list_tools() -> list`
- **Returns**: registered tool names.

#### `execute(name: str, **kwargs) -> Dict[str, Any]`
- **Function**: Execute a tool with retry and exception aggregation (never breaks the main flow).
- **Params**: `name` — tool name; `kwargs` — forwarded to the tool.
- **Returns**: `{"success": bool, "name": str, "result": Any, "error": str|None}`.

### 3.2 `build_http_client(base_url, backend="auto", max_workers=10, max_concurrency=10, rate_limit=None)`
- **Function**: Build a `snowland-http` client (timeout/retry handled by `ToolExecutor`).
- **Params**: `backend` — `auto`/`httpx`/`requests`/`aiohttp`/`stdlib`; `rate_limit` — optional rate-limit config.
- **Returns**: `snowland-http` `HttpClient` instance (with `base_url` attribute).

### 3.3 `request_json(client, method, path, timeout=30.0, **kwargs) -> Any`
- **Function**: HTTP request and parse JSON (joins `base_url` and timeout automatically).
- **Params**: `method` — HTTP method; `path` — endpoint path; `kwargs` — forwarded `params`/`json`/`headers`.
- **Returns**: deserialized JSON.

---

## 4. Memory Layer

### 4.1 `VectorMemory` — pluggable memory facade

**Constructor parameters** (`VectorMemory(...)`):

| Param | Type | Default | Description |
| --- | --- | --- | --- |
| `backend` | `str` / `MemoryBackend` | `"chroma"` | Backend name (`"chroma"`) or instance |
| `embedding` | `Embeddings` | `None` | Embeddings instance (required) |
| `**kwargs` | — | — | Forwarded to the backend |

**Properties**: `backend` (current `MemoryBackend`), `backend_name` (`str`).

**Main methods**:

#### `add(texts: List[str], metadatas: List[dict] = None) -> None`
- **Function**: Store memory fragments.

#### `search(query: str, top_k: int = 4) -> List[str]`
- **Function**: Semantic retrieval of most relevant memories.
- **Returns**: list of memory texts.

#### `as_retriever(top_k: int = 4)`
- **Returns**: LangChain Retriever.

#### `clear() -> None`
- **Function**: Clear memory.

### 4.2 `MemoryBackend` (abstract base)
- `add(texts, metadatas=None) -> None`
- `search(query, top_k=4) -> List[str]`
- `as_retriever(top_k=4)`
- `clear() -> None` (optional)

Subclass, then `register_memory_backend` to be resolved by the name `"chroma"`.

---

## 5. Perception Layer

### 5.1 `PerceptionLayer` — pluggable perception facade

**Constructor parameters** (`PerceptionLayer(backend="spacy", **kwargs)`):
- `backend`: `str` (`"spacy"` / `"bailian"`) or a `PerceptionBackend` instance; default spaCy.

**Properties**: `backend`, `backend_name`.

**Main methods**:

#### `perceive(text: str) -> Dict[str, object]`
- **Returns**: `{"text": str, "entities": list, "intent": str}`; when backend is `None` returns
  `{"text", "entities": [], "intent": "unknown"}`.

#### `extract_entities(text: str) -> List[Dict[str, str]]`
- **Returns**: `[{"text": ..., "label": ...}, ...]`.

#### `classify_intent(text: str, default: str = "unknown") -> str`
- **Returns**: intent category string.

### 5.2 `PerceptionBackend` (abstract base)
- `extract_entities(text) -> List[Dict[str, str]]`
- `classify_intent(text, default="unknown") -> str`
- `perceive(text) -> Dict[str, object]`

Subclass, then `register_perception_backend` to be resolved by name.

---

## 6. Model Integration  ⭐ major v0.2 extension

A model backend is located by two **orthogonal dimensions**:

- **Vendor backend**: `dashscope` / `orcarouter` / `zhipu` / `deepseek` / `anthropic` / `anthropic-openai` / `openai-compat`;
- **Capability type `ModelType`**: `reasoning` (→ `BaseChatModel`), `embedding` (→ `Embeddings`),
  `image` (→ `ImageModel`), `video` (→ `VideoModel`), `tts` (→ `TTSModel`), `rerank` (→ `RerankModel`).

Every backend derives from a unified abstract root `ModelBackend` and branches into **six
same-shaped interfaces** `LLMBackend` / `EmbeddingBackend` / `ImageBackend` / `VideoBackend` /
`TTSBackend` / `RerankBackend`. Callers depend only on the abstraction and `ModelType`, never on
the concrete vendor — making it universal.

### 6.1 Facade functions (recommended entry points)

| Function | Key params | Returns | Description |
| --- | --- | --- | --- |
| `build_llm(backend="dashscope", model="qwen-plus", api_key=None, temperature=0.7, max_tokens=2048, top_p=0.8, enable_thinking=False, **kwargs)` | same as v0.1 | `BaseChatModel` | Reasoning / chat |
| `build_embedding(backend="dashscope", model="text-embedding-v2", **kwargs)` | `model` defaults per backend | `Embeddings` | Vector / embedding |
| `build_image(backend="dashscope", model="wanx-v1", **kwargs)` | — | `ImageModel` | Image generation |
| `build_video(backend="dashscope", model="wanx-v1", **kwargs)` | — | `VideoModel` | Video generation |
| `build_tts(backend="dashscope", model="cosyvoice-v1", **kwargs)` | e.g. `voice` | `TTSModel` | Text-to-speech |
| `build_rerank(backend="dashscope", model="gte-rerank", top_n=5, **kwargs)` | `top_n` default 5 | `RerankModel` | Semantic rerank |
| `build_model(model_type, backend="dashscope", model=None, **kwargs)` | `model_type`: `ModelType`/str | capability object | Generic entry (capability type + vendor) |

> All `build_*` are imported from `snowland_agent_base.model`; `build_llm` is additionally re-exported at the top level.

### 6.2 Capability models and result dataclasses

| Capability | Model interface | Method | Result dataclass (fields) |
| --- | --- | --- | --- |
| Image | `ImageModel` | `generate(prompt, **kwargs) -> ImageResult` | `ImageResult`: `urls: List[str]`, `raw` |
| Video | `VideoModel` | `generate(prompt, **kwargs) -> VideoResult` | `VideoResult`: `urls: List[str]`, `task_id: Optional[str]`, `raw` |
| Speech | `TTSModel` | `synthesize(text, **kwargs) -> AudioResult` | `AudioResult`: `audio: Any`, `format: Optional[str]`, `raw` |
| Rerank | `RerankModel` | `rerank(query, documents, top_n=None, **kwargs) -> RerankResult` | `RerankResult`: `results: List[RerankItem]`, `raw`; `RerankItem`: `index: int`, `document: Any`, `relevance_score: float` |

### 6.3 `AutoBackend` — unified auto backend

`AutoBackend(vendor="dashscope", **common)` holds a default vendor and delegates each capability
method to the matching `build_*` facade, automatically attaching that vendor so callers never
repeat the backend name:

| Method | Params | Returns |
| --- | --- | --- |
| `reasoning(model=None, **kwargs)` | — | `BaseChatModel` |
| `embedding(model=None, **kwargs)` | — | `Embeddings` |
| `image(model=None, **kwargs)` | — | `ImageModel` |
| `video(model=None, **kwargs)` | — | `VideoModel` |
| `tts(model=None, **kwargs)` | — | `TTSModel` |
| `rerank(query=None, documents=None, model=None, top_n=5, **kwargs)` | if both `query` and `documents` given | `RerankResult` directly; else `RerankModel` |
| `build(model_type, model=None, **kwargs)` | generic build by capability type | capability object |

`build_auto_backend(vendor="dashscope", **common) -> AutoBackend` is a convenience facade.

```python
from snowland_agent_base.model import AutoBackend

backend = AutoBackend(vendor="dashscope")
chat  = backend.reasoning(model="qwen-plus")
emb   = backend.embedding()
img   = backend.image(model="wanx-v1").generate("a cat")
video = backend.video(model="wanx-v1").generate("waves")
audio = backend.tts(model="cosyvoice-v1").synthesize("hello")
rank  = backend.rerank(query="...", documents=[...])   # returns RerankResult directly
```

### 6.4 Abstract base classes

`ModelBackend` (abstract root): class attrs `model_type: ModelType` (default `REASONING`),
`message_class` (meaningful only for reasoning backends, default `None`), abstract method
`build() -> Any`.

Six capability interfaces (each declares `model_type` and implements `build()` returning the
matching type):

- `LLMBackend` (REASONING) → `BaseChatModel`, `message_class = OpenAIMessage`
- `EmbeddingBackend` (EMBEDDING) → `Embeddings`
- `ImageBackend` (IMAGE) → `ImageModel`
- `VideoBackend` (VIDEO) → `VideoModel`
- `TTSBackend` (TTS) → `TTSModel`
- `RerankBackend` (RERANK) → `RerankModel`

### 6.5 Built-in vendors and capability matrix

| Vendor (aliases) | Env vars (key / base_url) | Registered capabilities |
| --- | --- | --- |
| `dashscope` (default; aliases `bailian`/`qwen`/`qianwen`) | `DASHSCOPE_API_KEY` | **reasoning, embedding, image, video, tts, rerank (all 6)** |
| `orcarouter` | `ORCA_KEY` (or `ORCAROUTER_API_KEY`) / `ORCAROUTER_BASE_URL` | **reasoning, embedding, image, video, tts, rerank (all 6)** |
| `zhipu` | `ZHIPU_API_KEY` / `ZHIPU_BASE_URL` | **reasoning, embedding, image, video, tts, rerank (all 6)** |
| `deepseek` | `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` | reasoning, embedding, rerank |
| `anthropic` | `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` | reasoning (native Messages API) |
| `anthropic-openai` (alias `anthropic-compatible`) | `OPENAI_API_KEY` | reasoning (Anthropic via OpenAI-compatible) |
| `openai-compat` | `OPENAI_API_KEY` / `OPENAI_BASE_URL` | **reasoning, embedding, image, video, tts, rerank (all 6, generic fallback)** |

`OpenAICompat*Backend` is the shared base for OpenAI-compatible vendors; a subclass becomes a
pluggable backend by overriding the class attributes `default_base_url` / `api_key_env` /
`base_url_env` / `default_model`.

### 6.6 Registry and factory

| Function | Params | Returns | Description |
| --- | --- | --- | --- |
| `register_llm_backend(name, cls, model_type=ModelType.REASONING)` | `str`, `type[ModelBackend]`, `ModelType` | `None` | Register a backend (`cls` must subclass a `ModelBackend` capability interface) |
| `get_llm_backend(name, model_type=REASONING) -> type[ModelBackend]` | `str`, `ModelType` | backend class | Resolve a registered backend class by name + capability type |
| `list_llm_backends(model_type=None) -> list` | `ModelType?` | name list | List registered backends (optionally filtered by capability type) |
| `create_model(model_type=REASONING, name="dashscope", **kwargs) -> ModelBackend` | `ModelType`, `str`, `**kwargs` | backend instance | Build an instance by capability type + name |
| `create_llm_backend(name="dashscope", **kwargs) -> ModelBackend` | `str`, `**kwargs` | backend instance | Build a reasoning (REASONING) backend (back-compat facade) |
| `get_message_class(backend) -> type[Message]` | `str?` | message class | Resolve the bound message-style class (`OpenAIMessage`/`AnthropicMessage`); falls back to `OpenAIMessage` for `None` / unknown names |

Subclass the matching capability base and `register_llm_backend` to be resolved by
`build_llm(backend=...)` / `build_embedding(backend=...)` / etc.

### 6.7 Custom backend

```python
from snowland_agent_base.model import register_llm_backend, ModelType
from snowland_agent_base.model.backends.openai_compat import OpenAICompatLLMBackend

class MyLLMBackend(OpenAICompatLLMBackend):
    default_base_url = "https://my.api/v1"
    api_key_env = "MY_API_KEY"

register_llm_backend("myvendor", MyLLMBackend)                          # reasoning
register_llm_backend("myvendor", MyOtherBackend, ModelType.EMBEDDING)   # embedding
```

### 6.8 ChatModel proxies (lazy load / empty-response retry)

Two `BaseChatModel` decorators wrap an existing ChatModel (e.g. the one returned by
`build_llm`) and transparently add behavior; they are drop-in replacements anywhere a
ChatModel is expected.

| Class | Params | Returns | Description |
| --- | --- | --- | --- |
| `LazyChatModel(builder=None, **kwargs)` | `builder: Callable[[], BaseChatModel]` | `BaseChatModel` | Defers construction of the underlying LLM until the first real call (double-checked + lock-guarded, safe to share across parallel tasks) |
| `RetryOnEmptyChatModel(inner=None, max_retries=2, **kwargs)` | `inner: BaseChatModel`, `max_retries: int` | `BaseChatModel` | Retries automatically when the underlying model returns an empty `content` (reasoning model that burned its budget on `reasoning_content`); normal models are never retried (zero overhead) |

Both implement the full sync/async call chain (`invoke` / `ainvoke` / `bind_tools` /
`stream` / `_generate` / `_agenerate` / `_stream` / `_astream`) and delegate to the
wrapped model, so they can be passed wherever a `BaseChatModel` is expected:

```python
from snowland_agent_base.model import build_llm, LazyChatModel, RetryOnEmptyChatModel

# Lazy + retry combined: build the LLM only on first use, and self-heal empty responses
llm = RetryOnEmptyChatModel(
    inner=LazyChatModel(builder=lambda: build_llm(model="qwen-plus")),
)
```

`BaseAgent` enables this automatically: it wraps the LLM with `RetryOnEmptyChatModel`
by default (`enable_retry=True`, `retry_max_retries=2`). If you pass a `LazyChatModel`,
the agent adds the retry layer on top, giving you both lazy loading and self-healing.

---

## 7. Skill Capability Layer

### 7.1 `Skill` — skill data model (dataclass)
**Fields**: `name: str` (unique id, lowercase letters/digits/hyphens), `description: str`
(required, helps the model judge relevance), `instructions: str=""` (SKILL.md body),
`tools: Dict[str, Callable]={}`, `allowed_tools: List[str]=[]`, `path: Optional[Path]=None`,
`metadata: Dict={}`, `license: Optional[str]=None`, `compatibility: Optional[str]=None`.

**Property**: `tool_names: List[str]` — provided tool names.

**Methods**:
- `render_metadata() -> str` — level-1 disclosure (name + description, ~100 tokens).
- `render_instructions() -> str` — level-2 disclosure (full instructions, injected on activation).
- `list_resources() -> List[str]` — list `scripts/references/assets/` resources.
- `read_resource(relative_path: str) -> str` — read a resource file's content.

### 7.2 `skill(name, description, instructions="", tools=None, **kwargs) -> Skill`
- **Function**: Convenience constructor for a `Skill`.
- **Params**: `name`/`description`/`instructions` as model fields; `tools` — tool dict.
- **Returns**: `Skill` instance.

### 7.3 `SkillRegistry` — skill registry facade

**Constructor parameters** (`SkillRegistry(matcher="idf", backend=None)`):
- `matcher`: `str` (`"idf"`) or a `SkillMatcher` instance.
- `backend`: `str` or a `SkillBackend` instance; when given, its skills are auto-discovered at construction.

**Main methods**:

| Method | Params | Returns | Description |
| --- | --- | --- | --- |
| `register(skill_obj, overwrite=False)` | `Skill`, `bool` | `None` | Register a skill |
| `unregister(name)` | `str` | `None` | Unregister a skill |
| `get(name) -> Optional[Skill]` | `str` | `Skill` | Get by name |
| `list_skills() -> List[Skill]` | — | `Skill` list | All skills |
| `load_directory(root, overwrite=False) -> List[Skill]` | `Any`, `bool` | `Skill` list | Scan & load |
| `render_catalog() -> str` | — | `str` | Render all skill metadata |
| `match(text, limit=None) -> List[Skill]` | `str`, `int` | `Skill` list | Recall relevant skills |
| `collect_tools(skills) -> Dict` | `Sequence[Skill]` | `Dict` | Aggregate skill tools |
| `all_tool_names() -> Set[str]` | — | `set` | All tool names |

**Properties**: `matcher_name`, `backend_name`, `names`, plus `__len__`, `__contains__`.

### 7.4 `SkillBackend` (abstract base)
- `discover() -> List[Skill]`: discover and return the skills this source provides.
- Built-ins: `FileSkillBackend` (directory scan), `MemorySkillBackend` (in-memory set).

### 7.5 `SkillMatcher` (abstract base)
- `match(skills, text, limit=None) -> List[Skill]`: recall relevant skills, descending by relevance.
- Built-in: `IdfSkillMatcher` (IDF-weighted lexical match).

### 7.6 `parse_skill_file(path: Any) -> Skill`
- **Function**: Parse a `SKILL.md` file (supports YAML frontmatter).
- **Param**: `path` — path to SKILL.md.
- **Returns**: `Skill` instance; raises `SkillError` on missing file or invalid frontmatter.

---

## 8. Message Model

`Message` is an **abstract base class** holding the common fields `role` / `content` / `name` /
`tool_call_id` / `tool_calls`; the concrete vendor format is decided by its subclasses, which
expose the **same method** `to_langchain_messages()`. Callers depend only on the `Message`
abstraction and need not know the underlying protocol — making it universal:

- `OpenAIMessage`: OpenAI-compatible (`system→SystemMessage`, `user→HumanMessage`,
  `assistant→AIMessage`, `tool→` system-context fallback).
- `AnthropicMessage`: Anthropic native Messages API (`tool` + `tool_call_id→ToolMessage`,
  `assistant` + `tool_calls→` `AIMessage` with `tool_calls`).

The Agent selects the style class via `get_message_class(backend)` and stays polymorphic
thereafter, without threading a `backend` string up the call stack.

`Message` provides `to_dict()` / `from_dict()` (JSON serialization), field-level `__eq__`,
and the `role` / `content` / `name` / `tool_call_id` / `tool_calls` attributes.

| Constructor | Params | Returns | Description |
| --- | --- | --- | --- |
| `system_message(content, *, cls=OpenAIMessage)` | `str`, `type[Message]` | `Message` | System message |
| `user_message(content, *, cls=OpenAIMessage)` | `str`, `type[Message]` | `Message` | User message |
| `assistant_message(content, *, cls=OpenAIMessage)` | `str`, `type[Message]` | `Message` | Assistant message |
| `tool_message(content, name="", tool_call_id=None, *, cls=OpenAIMessage)` | `str`, `str`, `str?`, `type[Message]` | `Message` | Tool observation |
| `tool_result_message(content, tool_call_id, name="", *, cls=OpenAIMessage)` | `str`, `str`, `str`, `type[Message]` | `Message` | Anthropic-native tool use only; requires `tool_call_id` |
| `assistant_tool_call_message(tool_calls, content="", *, cls=OpenAIMessage)` | `Sequence[Dict]`, `str`, `type[Message]` | `Message` | Assistant message carrying `tool_calls` |
| `ensure_messages(history)` | `str`/`List[Message]`/`None` | `List[Message]` | Normalize history to message list (str wrapped as a system message; dict restored as `OpenAIMessage`) |
| `render_messages(messages, include_role=True)` | `Sequence[Message]`, `bool` | `str` | Render to text (logs/summaries) |
| `to_langchain_messages(messages)` | `Sequence[Message]` | `List` | Convert to LangChain message objects (vendor format decided by each message's own `to_langchain_messages()`, no `backend` string dispatching) |

`SkillError` is the exception for skill definition/loading errors (subclass of `ValueError`).

---

## 9. Parallel Execution

For fanning out independent, shared-nothing tasks, use the thread-pool helpers in
`snowland_agent_base.parallel`:

| Function | Params | Returns | Description |
| --- | --- | --- | --- |
| `map_parallel(fn, items, max_workers=None, *, catch_errors=True)` | `fn: Callable[[T], R]`, `items: Sequence[T]`, `max_workers: int?`, `catch_errors: bool` | `List[R]` | Apply `fn` to each item concurrently, returning a list **in the same order** as `items`; auto-degrades to serial when `max_workers <= 1` or a single item |
| `run_parallel(tasks, max_workers=None, *, catch_errors=True)` | `tasks: Dict[str, Callable[[], Any]]`, `max_workers: int?`, `catch_errors: bool` | `Dict[str, Any]` | Run named tasks concurrently, returning `{name: result}` |

By default each task's exception is caught and converted to
`{"success": False, "error": str(e), "item" | "name": ...}` so one failure does not
abort the whole batch; pass `catch_errors=False` to raise on the first error.

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

When tasks share an LLM, wrap it with `LazyChatModel` (see §6.8) so the underlying
ChatModel is built once and safely reused across threads.
