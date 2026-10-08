# snowland-agent-base API Reference (English)

> This document describes the stable public API of `snowland-agent-base` using the format
> **Function / Parameters / Returns**. All classes and functions can be imported from the
> top-level `snowland_agent_base` package.

```python
import snowland_agent_base as sab
# or import by name:
from snowland_agent_base import BaseAgent, GraphAgent, build_llm, VectorMemory, PerceptionLayer, SkillRegistry
```

Concrete backend implementations (`DashScopeLLMBackend`, `ChromaMemoryBackend`,
`SpacyPerceptionBackend`, `FileSkillBackend`, `IdfSkillMatcher`, ...) and the registration /
factory functions should be imported from their submodules:

```python
from snowland_agent_base.llm import DashScopeLLMBackend, register_llm_backend
from snowland_agent_base.memory import ChromaMemoryBackend
from snowland_agent_base.skill import FileSkillBackend, IdfSkillMatcher
```

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

## 6. LLM Integration

### 6.1 `build_llm(...)` — ChatModel builder facade

**Parameters**:

| Param | Type | Default | Description |
| --- | --- | --- | --- |
| `backend` | `str` | `"dashscope"` | Backend name, alias `"bailian"` also accepted |
| `model` | `str` | `"qwen-plus"` | Model name (`qwen-plus`/`qwen-max`/`qwen-turbo`) |
| `api_key` | `str` | `None` | Falls back to env var `DASHSCOPE_API_KEY` |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `max_tokens` | `int` | `2048` | Max tokens |
| `top_p` | `float` | `0.8` | Nucleus sampling probability |
| `enable_thinking` | `bool` | `False` | Enable qwq-series thinking mode |
| `**kwargs` | — | — | Forwarded to the backend constructor |

**Returns**: LangChain `BaseChatModel` instance.

### 6.2 `DashScopeLLMBackend(LLMBackend)` — Bailian (DashScope) backend

**Constructor parameters**: `model`, `api_key`, `temperature=0.7`, `max_tokens=2048`,
`top_p=0.8`, `enable_thinking=False`, `**kwargs` (same as `build_llm`).

**Method**: `build() -> BaseChatModel` — builds and returns `ChatDashScope` (auto-injects the
pydantic v1 compatibility shim internally).

### 6.3 `LLMBackend` (abstract base)
- `build() -> BaseChatModel`: build a ChatModel.

Subclass, then `register_llm_backend` to be resolved by `build_llm(backend=...)`.

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

`Message` is a `TypedDict` (`{"role", "content", "name?"}`); `role` ∈
`system` / `user` / `assistant` / `tool`.

| Function | Params | Returns | Description |
| --- | --- | --- | --- |
| `system_message(content)` | `str` | `Message` | System message |
| `user_message(content)` | `str` | `Message` | User message |
| `assistant_message(content)` | `str` | `Message` | Assistant message |
| `tool_message(content, name="")` | `str`, `str` | `Message` | Tool observation (`tool` role with `name`) |
| `ensure_messages(history)` | `str`/`List[Message]`/`None` | `List[Message]` | Normalize history to message list (str wrapped as a system message) |
| `render_messages(messages, include_role=True)` | `Sequence[Message]`, `bool` | `str` | Render to text (logs/summaries) |
| `to_langchain_messages(messages)` | `Sequence[Message]` | `List` | Convert to LangChain message objects (multi-turn context) |

`SkillError` is the exception for skill definition/loading errors (subclass of `ValueError`).
