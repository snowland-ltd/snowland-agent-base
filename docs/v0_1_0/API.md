# snowland-agent-base API 文档（中文）

> 本文档按「功能 / 入参 / 出参」描述 `snowland-agent-base` 的稳定公共 API。
> 所有类与函数均可从 `snowland_agent_base` 顶层导入。

```python
import snowland_agent_base as sab
# 或按模块导入:
from snowland_agent_base import BaseAgent, GraphAgent, build_llm, VectorMemory, PerceptionLayer, SkillRegistry
```

具体后端实现类（`DashScopeLLMBackend`、`ChromaMemoryBackend`、`SpacyPerceptionBackend`、`FileSkillBackend`、`IdfSkillMatcher` 等）及注册/工厂函数请从对应子模块导入：

```python
from snowland_agent_base.llm import DashScopeLLMBackend, register_llm_backend
from snowland_agent_base.memory import ChromaMemoryBackend
from snowland_agent_base.skill import FileSkillBackend, IdfSkillMatcher
```

---

## 1. 代理 (Agent)

### 1.1 `BaseAgent` — 手写 ReAct 循环代理

**功能**：串联「感知 → 决策 → 执行 → 记忆」四层，运行一次完整推理，返回结构化结果。

**构造入参** (`BaseAgent(...)`)：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `llm` | `BaseChatModel` | 必填 | LangChain ChatModel（建议用 `build_llm()` 构建） |
| `embedding` | `Embeddings` | `None` | 开启长期记忆时必需 |
| `perception` | `PerceptionLayer` / `PerceptionBackend` | `None` | 感知层，缺省自动构建（受 `use_perception` 控制） |
| `decision` | `DecisionLayer` | `None` | 决策层，缺省基于 `llm` 构建（此时 `system_prompt` 生效） |
| `executor` | `ToolExecutor` | `None` | 执行层，缺省自动构建 |
| `memory` | `VectorMemory` / `MemoryBackend` | `None` | 记忆层，缺省且提供 `embedding` 时自动构建 |
| `max_steps` | `int` | `5` | 单次运行最大推理步数，防死循环 |
| `use_perception` | `bool` | `True` | 是否启用感知层（关闭可省 spaCy 依赖） |
| `use_memory` | `bool` | `True` | 是否启用长期记忆 |
| `system_prompt` | `str` | `None` | 角色/人设提示词；仅当 `decision` 未传时生效 |
| `skills` | `SkillRegistry` | `None` | skill 注册表 |
| `auto_activate_skills` | `bool` | `True` | 是否按输入自动激活相关 skill |

**主要方法**：

#### `run(user_input: str) -> Dict[str, Any]`
- **功能**：运行一次完整推理。
- **入参**：`user_input` — 用户输入文本。
- **出参**：字典，键含 `input`（输入）、`perception`（感知结果）、`steps`（每轮 action/observation 明细）、`answer`（最终答复文本）、`messages`（`Message` 列表：记忆→用户输入→工具观测→助手答复）、`skills`（本轮激活的 skill 名列表）、`success`（`bool`）。

#### `register_tool(name: str, func: callable) -> None`
- **功能**：注册可执行工具。
- **入参**：`name` — 工具名；`func` — 可调用对象。
- **出参**：无。

#### `add_rule(rule: Rule) -> None`
- **功能**：注册一条决策兜底规则。
- **入参**：`rule` — `Rule` 实例。
- **出参**：无。

#### `add_skill(skill_obj: Skill, overwrite: bool = False) -> None`
- **功能**：注册一个 skill，并将其携带的工具注册进执行器。
- **入参**：`skill_obj` — `Skill` 实例；`overwrite` — 同名是否覆盖。
- **出参**：无。

#### `load_skills(directory: Any, overwrite: bool = False) -> List[Skill]`
- **功能**：从目录扫描加载 skill（自动发现含 `SKILL.md` 的子目录）。
- **入参**：`directory` — skills 根目录或单个 skill 目录；`overwrite` — 同名是否覆盖。
- **出参**：`List[Skill]` — 本次加载的 skill 列表。

#### 扩展点（子类覆盖）
- `_build_prompt_context(user_input, perception_result) -> Dict`：注入业务上下文。
- `_build_system_prompt(user_input, perception_result) -> Optional[str]`：按请求动态生成人设，返回 `None` 沿用构造时 `system_prompt`。

---

### 1.2 `GraphAgent(BaseAgent)` — LangGraph 编排代理（可选依赖 `langgraph`）

**功能**：在 `BaseAgent` 四层能力基础上，用 `StateGraph` 实现持久化断点续跑、人机介入、流式输出、多 Agent/子图编排。

**构造入参**：在 `BaseAgent` 全部参数之外，额外支持：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `checkpointer` | `Any` | `InMemorySaver()` | LangGraph checkpointer，持久化用 `SqliteSaver` |
| `store` | `Any` | `None` | 跨会话长期存储 |
| `approval_tools` | `Sequence[str]` | `None` | 执行前需人工审批的工具名列表 |
| `require_approval` | `Callable[[str, dict], bool]` | `None` | 自定义审批判定，优先级高于 `approval_tools` |

**主要方法**：

#### `run(user_input, thread_id: str = None, history = None) -> Dict[str, Any]`
- 同 `invoke`，便于与 `BaseAgent` 多态替换。
- **出参** 额外含 `thread_id`、`interrupted`（`bool`）、`interrupts`。

#### `invoke(user_input, thread_id: str = None, history = None) -> Dict[str, Any]`
- **功能**：同步运行一轮对话。
- **入参**：`history` — 前置历史（`Message` 列表或字符串）；`thread_id` — 会话线程 ID（断点续跑）。二者可并存。
- **出参**：结构化结果（同 `run`，含 `thread_id`/`interrupted`/`interrupts`）。

#### `stream_events(user_input = None, thread_id = None, version = "v3", history = None)`
- **功能**：事件流式运行，支持 `messages`/`values`/`interrupts` 三类投影，适用于人机介入。
- **出参**：LangGraph 流式对象，含 `.output`、`.interrupted`、`.interrupts`。

#### `stream(user_input, thread_id = None, stream_mode = "values", history = None)`
- **功能**：按 state 快照/增量流式输出。
- **入参**：`stream_mode` — `"values"`（完整快照）/ `"updates"`（增量）/ `"messages"`（token 流）。

#### `resume(decision, thread_id: str)`
- **功能**：在 `interrupt` 处恢复执行。
- **入参**：`decision` — `bool` 或 `{"approved": bool, "tool_input": {...}}`（审批时可改写入参）；`thread_id` — 须与中断时一致。

#### `get_state(thread_id: str)` / `update_state(thread_id: str, values: dict)`
- **功能**：读取线程状态快照 / 手动修改状态（时间旅行）。

#### `add_subgraph(name: str, subgraph) -> None`
- **功能**：注册已编译子图，作为可被决策层调用的「工具」（需先 `build()` 生效）。

#### `build() -> Any`
- **功能**：构建并编译 `StateGraph`（带 checkpointer/store）。

---

## 2. 决策层 (Decision)

### 2.1 `DecisionLayer` — 规则兜底 + LLM 链式思考

**构造入参** (`DecisionLayer(...)`)：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `llm` | `BaseChatModel` | 必填 | LangChain ChatModel |
| `system_prompt` | `str` | `None` | 角色/人设提示词，缺省用默认 |
| `rules` | `List[Rule]` | `None` | 规则引擎规则列表，按序匹配 |
| `fallback_answer` | `str` | `"抱歉, 我暂时无法处理这个请求。"` | 解析/调用失败时的兜底答复 |
| `max_parse_retries` | `int` | `1` | JSON 解析失败重问次数 |
| `skills` | `SkillRegistry` | `None` | skill 注册表，其目录常驻系统提示词 |

**主要方法**：

#### `decide(user_input, tools=None, history=None, context=None, system_prompt=None, skills=None) -> AgentAction`
- **功能**：产出下一步动作（先过规则，未命中再 LLM 链式思考，失败回退兜底）。
- **入参**：`tools` — 可用工具名列表；`history` — 历史会话；`context` — 感知层上下文（意图/实体）；`system_prompt` — 动态人设；`skills` — 已激活 skill 列表。
- **出参**：`AgentAction`。

#### `add_rule(rule: Rule) -> None` / `match_rule(user_input, context) -> Optional[AgentAction]`
- 追加规则 / 按序匹配首个命中动作。

#### `build_prompt(tools=None, system_prompt=None, skills=None) -> str`
- **功能**：组装系统提示词（人设 + CoT 指令 + 工具列表 + skill 目录/指令）。
- **出参**：提示词文本。

#### `build_messages(user_input, tools=None, history=None, system_prompt=None, skills=None) -> List`
- **功能**：组装完整 LangChain 消息序列（系统提示词 + 历史 + 输入）。
- **出参**：LangChain 消息对象列表。

### 2.2 `AgentAction` — 决策动作（dataclass）

**字段**：`thought: str`（推理过程）、`tool: Optional[str]`（工具名）、`tool_input: Dict`（工具入参）、`final_answer: Optional[str]`（最终答复）。

- **属性** `is_final: bool` — 是否直接给出最终答复（无需调用工具）。
- **方法** `to_dict() -> Dict` — 转为字典。

### 2.3 `Rule` — 规则（dataclass）

**字段**：`name: str`（规则名）、`predicate: Callable[[str, dict], bool]`（命中判定）、`action: AgentAction`（命中时返回的固定动作）。

---

## 3. 执行层 (Execution)

### 3.1 `ToolExecutor` — 工具执行器

**构造入参** (`ToolExecutor(...)`)：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `timeout` | `float` | `30.0` | 单次调用超时秒数 |
| `max_retries` | `int` | `3` | 最大重试次数 |
| `retry_interval` | `float` | `1.0` | 重试间隔基数（指数退避） |

**主要方法**：

#### `register(name: str, func: Callable) -> None`
- **功能**：注册工具（不可调用对象会抛 `TypeError`）。

#### `unregister(name: str) -> None`
- **功能**：注销工具（不存在时静默）。

#### `list_tools() -> list`
- **出参**：已注册工具名列表。

#### `execute(name: str, **kwargs) -> Dict[str, Any]`
- **功能**：执行工具，带重试与异常收敛（失败不阻断主流程）。
- **入参**：`name` — 工具名；`kwargs` — 透传参数。
- **出参**：`{"success": bool, "name": str, "result": Any, "error": str|None}`。

### 3.2 `build_http_client(base_url, backend="auto", max_workers=10, max_concurrency=10, rate_limit=None)`
- **功能**：构建 `snowland-http` 客户端（统一超时/重试由 `ToolExecutor` 负责）。
- **入参**：`backend` — `auto`/`httpx`/`requests`/`aiohttp`/`stdlib`；`rate_limit` — 可选限流配置。
- **出参**：`snowland-http` 的 `HttpClient` 实例（含 `base_url` 属性）。

### 3.3 `request_json(client, method, path, timeout=30.0, **kwargs) -> Any`
- **功能**：发起 HTTP 请求并返回 JSON（自动拼接 `base_url` 与超时）。
- **入参**：`method` — HTTP 方法；`path` — 接口路径；`kwargs` — 透传 `params`/`json`/`headers` 等。
- **出参**：反序列化后的 JSON 数据。

---

## 4. 记忆层 (Memory)

### 4.1 `VectorMemory` — 可插拔记忆门面

**构造入参** (`VectorMemory(...)`)：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `backend` | `str` / `MemoryBackend` | `"chroma"` | 后端名（`"chroma"`）或后端实例 |
| `embedding` | `Embeddings` | `None` | Embeddings 实例（必需） |
| `**kwargs` | — | — | 透传给具体后端 |

**属性**：`backend`（`MemoryBackend`，当前后端实例）、`backend_name`（`str`）。

**主要方法**：

#### `add(texts: List[str], metadatas: List[dict] = None) -> None`
- **功能**：写入记忆片段。

#### `search(query: str, top_k: int = 4) -> List[str]`
- **功能**：语义检索最相关的历史记忆。
- **出参**：记忆文本列表。

#### `as_retriever(top_k: int = 4)`
- **出参**：LangChain Retriever。

#### `clear() -> None`
- **功能**：清空记忆。

### 4.2 `MemoryBackend`（抽象基类）
- `add(texts, metadatas=None) -> None`
- `search(query, top_k=4) -> List[str]`
- `as_retriever(top_k=4)`
- `clear() -> None`（可选实现）

继承并 `register_memory_backend` 注册后即可被 `"chroma"` 名称解析。

---

## 5. 感知层 (Perception)

### 5.1 `PerceptionLayer` — 可插拔感知门面

**构造入参** (`PerceptionLayer(backend="spacy", **kwargs)`)：
- `backend`：`str`（`"spacy"` / `"bailian"`）或 `PerceptionBackend` 实例；缺省 spaCy。

**属性**：`backend`、`backend_name`。

**主要方法**：

#### `perceive(text: str) -> Dict[str, object]`
- **出参**：`{"text": str, "entities": list, "intent": str}`；后端为 `None` 时返回 `{"text", "entities": [], "intent": "unknown"}`。

#### `extract_entities(text: str) -> List[Dict[str, str]]`
- **出参**：`[{"text": ..., "label": ...}, ...]`。

#### `classify_intent(text: str, default: str = "unknown") -> str`
- **出参**：意图类别字符串。

### 5.2 `PerceptionBackend`（抽象基类）
- `extract_entities(text) -> List[Dict[str, str]]`
- `classify_intent(text, default="unknown") -> str`
- `perceive(text) -> Dict[str, object]`

继承并 `register_perception_backend` 注册后即可被名称解析。

---

## 6. LLM 接入层 (LLM)

### 6.1 `build_llm(...)` — ChatModel 构建门面

**入参**：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `backend` | `str` | `"dashscope"` | 后端名，亦可用别名 `"bailian"` |
| `model` | `str` | `"qwen-plus"` | 模型名（`qwen-plus`/`qwen-max`/`qwen-turbo`） |
| `api_key` | `str` | `None` | 缺省读环境变量 `DASHSCOPE_API_KEY` |
| `temperature` | `float` | `0.7` | 采样温度 |
| `max_tokens` | `int` | `2048` | 最大 token 数 |
| `top_p` | `float` | `0.8` | 核采样概率 |
| `enable_thinking` | `bool` | `False` | 是否开启 qwq 系列思考模式 |
| `**kwargs` | — | — | 透传后端构造器 |

**出参**：LangChain `BaseChatModel` 实例。

### 6.2 `DashScopeLLMBackend(LLMBackend)` — 百炼后端

**构造入参**：`model`、`api_key`、`temperature=0.7`、`max_tokens=2048`、`top_p=0.8`、`enable_thinking=False`、`**kwargs`（同 `build_llm`）。

**方法**：`build() -> BaseChatModel` — 构建并返回 `ChatDashScope`（内部自动注入 pydantic v1 兼容垫片）。

### 6.3 `LLMBackend`（抽象基类）
- `build() -> BaseChatModel`：构建 ChatModel。

继承并 `register_llm_backend` 注册后即可被 `build_llm(backend=...)` 解析。

---

## 7. Skill 能力层 (Skill)

### 7.1 `Skill` — skill 数据模型（dataclass）

**字段**：`name: str`（唯一标识，小写字母/数字/连字符）、`description: str`（必填，供模型判断相关性）、`instructions: str=""`（SKILL.md 正文）、`tools: Dict[str, Callable]={}`、`allowed_tools: List[str]=[]`、`path: Optional[Path]=None`、`metadata: Dict={}`、`license: Optional[str]=None`、`compatibility: Optional[str]=None`。

**属性**：`tool_names: List[str]` — 提供的工具名。

**方法**：
- `render_metadata() -> str` — 第一级披露（name + description，~100 tokens）。
- `render_instructions() -> str` — 第二级披露（完整指令，激活后注入 system prompt）。
- `list_resources() -> List[str]` — 列出 `scripts/references/assets/` 资源。
- `read_resource(relative_path: str) -> str` — 读取资源文件内容。

### 7.2 `skill(name, description, instructions="", tools=None, **kwargs) -> Skill`
- **功能**：便捷构造一个 `Skill`。
- **入参**：`name`/`description`/`instructions` 同模型字段；`tools` — 工具字典。
- **出参**：`Skill` 实例。

### 7.3 `SkillRegistry` — skill 注册表门面

**构造入参** (`SkillRegistry(matcher="idf", backend=None)`)：
- `matcher`：`str`（`"idf"`）或 `SkillMatcher` 实例。
- `backend`：`str` 或 `SkillBackend` 实例；提供时构造即自动发现其 skill。

**主要方法**：

| 方法 | 入参 | 出参 | 说明 |
| --- | --- | --- | --- |
| `register(skill_obj, overwrite=False)` | `Skill`, `bool` | `None` | 注册 skill |
| `unregister(name)` | `str` | `None` | 注销 skill |
| `get(name) -> Optional[Skill]` | `str` | `Skill` | 按名取 skill |
| `list_skills() -> List[Skill]` | — | `Skill` 列表 | 全部 skill |
| `load_directory(root, overwrite=False) -> List[Skill]` | `Any`, `bool` | `Skill` 列表 | 扫描目录加载 |
| `render_catalog() -> str` | — | `str` | 渲染全部 skill 元数据 |
| `match(text, limit=None) -> List[Skill]` | `str`, `int` | `Skill` 列表 | 按输入召回相关 skill |
| `collect_tools(skills) -> Dict` | `Sequence[Skill]` | `Dict` | 汇总 skill 携带工具 |
| `all_tool_names() -> Set[str]` | — | `set` | 全部工具名 |

**属性**：`matcher_name`、`backend_name`、`names`、`__len__`、`__contains__`。

### 7.4 `SkillBackend`（抽象基类）
- `discover() -> List[Skill]`：发现并返回来源提供的一组 skill。
- 内置实现：`FileSkillBackend`（目录扫描）、`MemorySkillBackend`（内存集合）。

### 7.5 `SkillMatcher`（抽象基类）
- `match(skills, text, limit=None) -> List[Skill]`：按输入召回相关 skill，按相关度降序。
- 内置实现：`IdfSkillMatcher`（IDF 加权词面匹配）。

### 7.6 `parse_skill_file(path: Any) -> Skill`
- **功能**：解析 `SKILL.md` 文件（支持 YAML frontmatter）。
- **入参**：`path` — SKILL.md 路径。
- **出参**：`Skill` 实例；文件缺失或 frontmatter 不合法抛 `SkillError`。

---

## 8. 消息模型 (Messages)

`Message` 为 `TypedDict`（`{"role", "content", "name?"}`），`role` 取值 `system`/`user`/`assistant`/`tool`。

| 函数 | 入参 | 出参 | 说明 |
| --- | --- | --- | --- |
| `system_message(content)` | `str` | `Message` | 系统消息 |
| `user_message(content)` | `str` | `Message` | 用户消息 |
| `assistant_message(content)` | `str` | `Message` | 助手消息 |
| `tool_message(content, name="")` | `str`, `str` | `Message` | 工具观测消息（`tool` 角色带 `name`） |
| `ensure_messages(history)` | `str`/`List[Message]`/`None` | `List[Message]` | 统一历史为消息列表（字符串包为单条系统消息） |
| `render_messages(messages, include_role=True)` | `Sequence[Message]`, `bool` | `str` | 渲染为文本（日志/摘要） |
| `to_langchain_messages(messages)` | `Sequence[Message]` | `List` | 转为 LangChain 消息对象（多轮上下文） |

`SkillError` 为 skill 定义/加载错误异常（继承自 `ValueError`）。
