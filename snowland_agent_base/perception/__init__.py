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

"""可插拔感知层。

把「感知」抽象为统一后端接口 :class:`PerceptionBackend`, 不同实现可热插拔:

- :class:`SpacyPerceptionBackend`: 基于 spaCy 的实体识别 + 规则意图分类 (默认, 轻量)。
- :class:`BailianPerceptionBackend`: 基于阿里云百炼 (DashScope) 大模型完成 NER 与意图分类。

:class:`PerceptionLayer` 是向后兼容的门面, 通过 ``backend`` 选择具体后端::

    # 名称式 (经注册表解析, 可插拔)
    PerceptionLayer()                              # 默认 spaCy
    PerceptionLayer(backend="bailian")             # 百炼大模型
    PerceptionLayer(backend="bailian", model="qwen-max")

    # 实例式 (直接注入, 便于测试 / 自定义)
    PerceptionLayer(backend=BailianPerceptionBackend(llm=my_llm))

新增后端只需继承 :class:`PerceptionBackend` 并 ``register_perception_backend`` 注册,
即可被名称字符串解析, 实现真正的插拔式扩展。
"""
from __future__ import annotations

from snowland_agent_base.perception.backends import (
    BailianPerceptionBackend,
    SpacyPerceptionBackend,
)
from snowland_agent_base.perception.base import PerceptionBackend
from snowland_agent_base.perception.facade import PerceptionLayer
from snowland_agent_base.perception.registry import (
    create_perception_backend,
    get_perception_backend,
    list_perception_backends,
    register_perception_backend,
)


__all__ = [
    "PerceptionBackend",
    "SpacyPerceptionBackend",
    "BailianPerceptionBackend",
    "PerceptionLayer",
    "register_perception_backend",
    "get_perception_backend",
    "list_perception_backends",
    "create_perception_backend",
]
