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

"""并发执行工具: 把「无共享状态的任务扇出」抽象为框架级能力。

业务侧两类并行都能用同一套设施覆盖:

- ``map_parallel``: 批量同构任务 (如对 N 款游戏各跑一遍生成), 返回与输入**顺序一致**的
  结果列表 (内部用 ``ThreadPoolExecutor`` + ``as_completed`` 落位);
- ``run_parallel``: 若干**命名**任务 (如前端∥后台并行编码), 返回 ``{name: result}``,
  方便按名取用。

二者都具备:

- ``max_workers<=1`` 或只有一个任务时**自动退化为串行**, 语义与结果顺序保持一致;
- 默认 ``catch_errors=True``: 单个任务异常被收敛为
  ``{"success": False, "error": str(e), ...}`` 而非打断整体 (符合「工具/阶段异常不中断
  主流程」的工程约定); 设 ``False`` 则首个异常原样抛出。

线程安全: ``concurrent.futures`` 的线程池本身是线程安全的; 若任务内部使用的 LLM 是
:class:`snowland_agent_base.model.LazyChatModel`, 其懒加载已加锁, 可在并行任务中安全复用。
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def map_parallel(
    fn: Callable[[Any], T],
    items: Sequence[Any],
    max_workers: Optional[int] = None,
    *,
    catch_errors: bool = True,
) -> List[T]:
    """把 ``fn`` 并发映射到 ``items``, 返回与 ``items`` 顺序一致的结果列表。

    适合「批量同构任务」扇出 (如批量游戏生成)。``max_workers<=1`` 或单元素时退化为串行。
    默认捕获异常: 失败项记为 ``{"success": False, "error": str(e), "item": ...}``;
    ``catch_errors=False`` 时首个异常原样抛出。
    """
    n = len(items)
    if n == 0:
        return []
    if (max_workers is not None and max_workers <= 1) or n == 1:
        return _map_serial(fn, items, catch_errors)
    out: List[Any] = [None] * n
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, it): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            out[i] = _resolve(fut, catch_errors, item=items[i])
    return out  # type: ignore[return-value]


def run_parallel(
    tasks: Dict[str, Callable[[], Any]],
    max_workers: Optional[int] = None,
    *,
    catch_errors: bool = True,
) -> Dict[str, Any]:
    """并发执行一批**命名任务**, 返回 ``{name: result}``。

    适合「多个无共享状态、可并行的步骤/智能体」扇出 (如前端∥后台并行编码)。每个任务以
    ``{name: callable}`` 给出, 返回同 key 的 ``{name: result}``——方便按名取用
    (对比 ``map_parallel`` 的顺序列表)。``max_workers<=1`` 或单任务时退化为串行。
    默认捕获异常: 失败任务记为 ``{"success": False, "error": str(e), "name": name}``
    而非中断整体; ``catch_errors=False`` 时首个异常原样抛出。
    """
    if not tasks:
        return {}
    if (max_workers is not None and max_workers <= 1) or len(tasks) == 1:
        out: Dict[str, Any] = {}
        for name, fn in tasks.items():
            if catch_errors:
                try:
                    out[name] = fn()
                except Exception as e:  # pragma: no cover - 容错路径
                    logger.exception("run_parallel 任务[%s]失败: %s", name, e)
                    out[name] = {"success": False, "error": str(e), "name": name}
            else:
                out[name] = fn()
        return out
    out: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            out[name] = _resolve(fut, catch_errors, name=name)
    return out


def _map_serial(
    fn: Callable[[Any], T], items: Sequence[Any], catch_errors: bool
) -> List[T]:
    """串行映射 (退化路径)。"""
    out: List[Any] = []
    for it in items:
        out.append(_call(fn, it, catch_errors=catch_errors, item=it))
    return out


def _call(fn: Callable[[Any], Any], *args, catch_errors: bool, **meta) -> Any:
    """调用单个任务, 按 catch_errors 决定是否收敛异常。"""
    try:
        return fn(*args)
    except Exception as e:  # pragma: no cover - 容错路径
        if not catch_errors:
            raise
        logger.exception("并行任务失败: %s", e)
        return {"success": False, "error": str(e), **meta}


def _resolve(fut: Any, catch_errors: bool, **meta) -> Any:
    """取 future 结果, 按 catch_errors 决定是否收敛异常。"""
    try:
        return fut.result()
    except Exception as e:  # pragma: no cover - 容错路径
        if not catch_errors:
            raise
        logger.exception("并行任务失败: %s", e)
        return {"success": False, "error": str(e), **meta}
