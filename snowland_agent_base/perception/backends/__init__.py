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

"""内置感知后端实现集合。

每个感知后端单独成文件, 通过 :mod:`snowland_agent_base.perception.registry`
注册后即可被名称字符串解析。
"""
from __future__ import annotations

from snowland_agent_base.perception.backends.bailian import BailianPerceptionBackend
from snowland_agent_base.perception.backends.spacy import SpacyPerceptionBackend


__all__ = ["SpacyPerceptionBackend", "BailianPerceptionBackend"]
