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

"""执行层: 用 snowland-http 封装外部 API 调用, 统一超时与重试逻辑, 快速对接各类工具接口。

工具以「名称 -> 可调用对象」注册, 执行器负责参数透传、超时控制和失败重试,
并对异常做统一收敛, 保证 Agent 主流程不被单点工具故障打断。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class ToolExecutor:
    """工具执行器: 注册 / 执行 / 重试 / 超时。"""

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_interval: float = 1.0,
    ) -> None:
        """初始化执行器。

        Args:
            timeout: 单次调用超时秒数。
            max_retries: 最大重试次数。
            retry_interval: 重试间隔秒数 (指数退避基数)。
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self._tools: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func: Callable[..., Any]) -> None:
        """注册一个工具。

        Args:
            name: 工具名, 决策层通过此名引用。
            func: 可调用对象。
        """
        if not callable(func):
            raise TypeError(f"工具 '{name}' 必须可调用")
        self._tools[name] = func

    def unregister(self, name: str) -> None:
        """注销工具。"""
        self._tools.pop(name, None)

    def list_tools(self) -> list:
        """返回已注册工具名列表。"""
        return list(self._tools)

    def execute(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        """执行工具, 带重试与异常收敛。

        Args:
            name: 工具名。
            kwargs: 透传给工具的参数。

        Returns:
            ``{"success": bool, "name": str, "result": Any, "error": str|None}``
        """
        if name not in self._tools:
            return {
                "success": False,
                "name": name,
                "result": None,
                "error": f"未注册的工具: {name}",
            }

        func = self._tools[name]
        last_error: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                result = self._invoke_with_timeout(func, kwargs)
                return {
                    "success": True,
                    "name": name,
                    "result": result,
                    "error": None,
                }
            except TimeoutError as exc:
                last_error = f"调用超时: {exc}"
            except Exception as exc:  # noqa: BLE001 - 收敛任意工具异常
                last_error = f"{type(exc).__name__}: {exc}"

            logger.warning(
                "工具 '%s' 第 %s/%s 次执行失败: %s",
                name,
                attempt,
                self.max_retries,
                last_error,
            )
            if attempt < self.max_retries:
                time.sleep(self.retry_interval * (2 ** (attempt - 1)))

        return {
            "success": False,
            "name": name,
            "result": None,
            "error": last_error,
        }

    def _invoke_with_timeout(self, func: Callable[..., Any], kwargs: Dict[str, Any]) -> Any:
        """在超时约束内调用工具。

        借助 snowland-http 客户端时, 超时由客户端自身控制; 此处用
        守护线程实现通用超时兜底, 避免同步阻塞主流程。
        """
        import threading

        outcome: Dict[str, Any] = {}

        def target() -> None:
            try:
                outcome["result"] = func(**kwargs)
            except BaseException as exc:  # noqa: BLE001
                outcome["error"] = exc

        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        worker.join(self.timeout)

        if worker.is_alive():
            raise TimeoutError(f"超过 {self.timeout}s 未返回")
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("result")


def build_http_client(
    base_url: str,
    backend: str = "auto",
    max_workers: int = 10,
    max_concurrency: int = 10,
    rate_limit: Optional[Any] = None,
):
    """构建 snowland-http 客户端。

    注意: snowland-http 的 ``HttpClient`` 构造签名为
    ``HttpClient(backend, rate_limit, max_workers, max_concurrency)``;
    ``base_url`` 与 ``timeout`` 并非构造参数, 需在请求级传入
    (参见 :func:`request_json`)。重试由 :class:`ToolExecutor` 统一负责。

    Args:
        base_url: 外部服务基地址, 仅作记录, 请求时由调用方拼接 URL。
        backend: 后端类型: ``auto`` / ``httpx`` / ``requests`` / ``aiohttp`` / ``stdlib``。
        max_workers: 并行工作线程数。
        max_concurrency: 最大并发数。
        rate_limit: 可选 ``RateLimitConfig`` (``max_rate`` 每秒请求数, ``burst`` 突发数)。

    Returns:
        snowland-http ``HttpClient`` 实例。
    """
    from snowland_http import HttpClient

    client = HttpClient(
        backend=backend,
        rate_limit=rate_limit,
        max_workers=max_workers,
        max_concurrency=max_concurrency,
    )
    # base_url 不作为构造参数, 挂载到实例上供 request_json 拼接使用
    client.base_url = base_url.rstrip("/")
    return client


def request_json(
    client: Any,
    method: str,
    path: str,
    timeout: float = 30.0,
    **kwargs: Any,
) -> Any:
    """发起 HTTP 请求并返回 JSON, 统一拼接基地址与超时。

    ``timeout`` 以请求级 kwarg 透传给底层后端 (requests / httpx 等)。

    Args:
        client: :func:`build_http_client` 返回的客户端。
        method: HTTP 方法, 如 ``get`` / ``post``。
        path: 接口路径, 会与 ``client.base_url`` 拼接。
        timeout: 请求超时秒数。
        kwargs: 其他透传参数 (如 ``params`` / ``json`` / ``headers``)。

    Returns:
        反序列化后的 JSON 数据。
    """
    url = path if path.startswith("http") else f"{getattr(client, 'base_url', '')}/{path.lstrip('/')}"
    response = getattr(client, method.lower())(url, timeout=timeout, **kwargs)
    return response.json()
