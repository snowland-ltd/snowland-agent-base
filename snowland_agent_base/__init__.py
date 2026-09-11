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
"""


from snowland_agent_base.agent import BaseAgent
from snowland_agent_base.memory import (
    ChromaMemoryBackend,
    MemoryBackend,
    VectorMemory,
    create_memory_backend,
    get_memory_backend,
    list_memory_backends,
    register_memory_backend,
)
from snowland_agent_base.perception import (
    BailianPerceptionBackend,
    PerceptionBackend,
    PerceptionLayer,
    create_perception_backend,
    get_perception_backend,
    list_perception_backends,
    register_perception_backend,
    SpacyPerceptionBackend,
)
from snowland_agent_base.decision import DecisionLayer
from snowland_agent_base.execution import ToolExecutor
from snowland_agent_base.llm import (
    DashScopeLLMBackend,
    LLMBackend,
    build_llm,
    create_llm_backend,
    get_llm_backend,
    list_llm_backends,
    register_llm_backend,
)
from snowland_agent_base.messages import (
    Message,
    assistant_message,
    system_message,
    tool_message,
    user_message,
)

from snowland_agent_base.skill import (
    Skill,
    SkillError,
    SkillRegistry,
    parse_skill_file,
    skill,
)

__all__ = [
    "BaseAgent",
    "VectorMemory",
    "MemoryBackend",
    "ChromaMemoryBackend",
    "register_memory_backend",
    "get_memory_backend",
    "list_memory_backends",
    "create_memory_backend",
    "PerceptionLayer",
    "PerceptionBackend",
    "SpacyPerceptionBackend",
    "BailianPerceptionBackend",
    "register_perception_backend",
    "get_perception_backend",
    "list_perception_backends",
    "create_perception_backend",
    "DecisionLayer",
    "ToolExecutor",
    "build_llm",
    "LLMBackend",
    "DashScopeLLMBackend",
    "register_llm_backend",
    "get_llm_backend",
    "list_llm_backends",
    "create_llm_backend",
    "Skill",
    "SkillRegistry",
    "SkillError",
    "skill",
    "parse_skill_file",
    "Message",
    "system_message",
    "user_message",
    "assistant_message",
    "tool_message",
]

# GraphAgent 依赖可选的 langgraph, 未安装时不阻断本包导入
try:  # pragma: no cover
    from snowland_agent_base.graph import GraphAgent

    __all__.append("GraphAgent")
except ImportError:  # pragma: no cover
    GraphAgent = None  # type: ignore[assignment]


__version__ = "0.1.0alpha1"
