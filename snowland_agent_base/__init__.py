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

"""snowland-agent-base: A reusable AI Agent base framework.

技术栈映射:
- 核心框架: LangChain
- 大模型接入: 可插拔后端 (阿里云百炼 DashScope 等)
- 记忆存储: 可插拔后端 (ChromaDB 向量知识库等)
- 感知层: 可插拔后端 (spaCy / 阿里百炼 DashScope 等)
- 决策层: LLM 链式思考 + 规则引擎兜底
- 执行层: snowland-http 封装外部 API (超时 / 重试)
- 能力层: 可插拔 skill (来源后端 + 召回策略, 内置 file/memory 与 IDF 召回)

本模块只暴露**稳定的公共 API**(主入口门面 + 供扩展的抽象基类 + 消息模型)。
具体后端实现与注册/工厂函数请从其子模块导入, 例如::

    from snowland_agent_base.llm import DashScopeLLMBackend, register_llm_backend
    from snowland_agent_base.memory import ChromaMemoryBackend
    from snowland_agent_base.skill import FileSkillBackend, IdfSkillMatcher
"""


# ---- 主入口 / 代理 ----
from snowland_agent_base.agent import BaseAgent

# ---- 大模型接入层 (门面 + 抽象基类) ----
from snowland_agent_base.llm import LLMBackend, build_llm

# ---- 记忆层 (门面 + 抽象基类) ----
from snowland_agent_base.memory import MemoryBackend, VectorMemory

# ---- 感知层 (门面 + 抽象基类) ----
from snowland_agent_base.perception import PerceptionBackend, PerceptionLayer

# ---- 决策 / 执行层 ----
from snowland_agent_base.decision import DecisionLayer
from snowland_agent_base.execution import ToolExecutor

# ---- 消息模型与构造器 ----
from snowland_agent_base.messages import (
    Message,
    OpenAIMessage,
    AnthropicMessage,
    assistant_message,
    system_message,
    tool_message,
    user_message,
)

# ---- 能力层 skill (门面 + 模型 + 抽象基类 + 装饰器/工具) ----
from snowland_agent_base.skill import (
    Skill,
    SkillBackend,
    SkillError,
    SkillMatcher,
    SkillRegistry,
    parse_skill_file,
    skill,
)

__all__ = [
    "BaseAgent",
    # llm
    "build_llm",
    "LLMBackend",
    # memory
    "VectorMemory",
    "MemoryBackend",
    # perception
    "PerceptionLayer",
    "PerceptionBackend",
    # decision / execution
    "DecisionLayer",
    "ToolExecutor",
    # messages
    "Message",
    "OpenAIMessage",
    "AnthropicMessage",
    "system_message",
    "user_message",
    "assistant_message",
    "tool_message",
    # skill
    "Skill",
    "SkillError",
    "SkillBackend",
    "SkillMatcher",
    "SkillRegistry",
    "skill",
    "parse_skill_file",
]

# GraphAgent 依赖可选的 langgraph, 未安装时不阻断本包导入
try:  # pragma: no cover
    from snowland_agent_base.graph import GraphAgent

    __all__.append("GraphAgent")
except ImportError:  # pragma: no cover
    GraphAgent = None  # type: ignore[assignment]


__version__ = "0.1.0beta2"
