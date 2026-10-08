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

from __future__ import annotations

import sys


def ensure_pydantic_v1_shim() -> None:
    """确保 ``langchain_core.pydantic_v1`` 存在 (指向 ``pydantic.v1``)。"""
    shim_name = "langchain_core.pydantic_v1"
    if shim_name in sys.modules:
        return

    import langchain_core  # noqa: WPS433 - 运行期按需

    if hasattr(langchain_core, "pydantic_v1"):
        # 环境自带 (langchain-core < 0.3), 无需 shim
        sys.modules.setdefault(shim_name, langchain_core.pydantic_v1)
        return

    import pydantic  # noqa: WPS433 - 运行期按需

    if not hasattr(pydantic, "v1"):
        raise RuntimeError(
            "无法为 langchain_dashscope 提供 pydantic_v1 shim: "
            "当前 pydantic 版本不含 pydantic.v1 兼容命名空间"
        )

    langchain_core.pydantic_v1 = pydantic.v1  # type: ignore[attr-defined]
    sys.modules[shim_name] = pydantic.v1
