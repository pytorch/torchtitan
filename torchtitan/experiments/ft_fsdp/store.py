# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Store client with bounded operations."""

import logging
import threading
import time
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import torch.distributed as dist

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.1


class TimedStore(dist.Store):
    """Wraps a store client so no operation blocks forever.

    A TCPStore client waits for a response without a timeout. On MAST, after
    a host failure, nearly every survivor blocked in ``check`` on its existing
    connection while new connections worked. Each operation here runs on a
    helper thread; if it takes longer than ``op_timeout`` the connection is
    abandoned (the stuck thread keeps it) and the operation is retried on a
    new connection, up to ``num_retries`` times. ``add`` with a nonzero amount
    is not idempotent, so it raises ``TimeoutError`` instead of retrying.
    ``wait`` and ``get`` poll ``check`` so they are bounded by their own
    timeout.

    Helper threads are daemons: an abandoned thread may never return and
    must not block interpreter exit.

    Usable under ``PrefixStore``, but keep a reference to the ``TimedStore``:
    the C++ side does not keep a Python store subclass alive. ``clone``
    returns a new ``TimedStore`` with its own connection.
    """

    def __init__(
        self,
        connect: Callable[[], dist.Store],
        *,
        op_timeout: timedelta,
        timeout: timedelta,
        num_retries: int = 3,
    ) -> None:
        super().__init__()
        self._connect = connect
        self._op_timeout = op_timeout
        self._timeout = timeout
        self._num_retries = num_retries
        self._lock = threading.Lock()
        self._client = connect()

    def _run(self, name: str, fn: Callable[[dist.Store], Any], *, retry: bool) -> Any:
        with self._lock:
            for attempt in range(self._num_retries + 1):
                done = threading.Event()
                result: list[Any] = []
                error: list[BaseException] = []
                client = self._client

                def target() -> None:
                    try:
                        result.append(fn(client))
                    except BaseException as e:
                        error.append(e)
                    done.set()

                threading.Thread(
                    target=target, name="ftfsdp-store", daemon=True
                ).start()
                if done.wait(self._op_timeout.total_seconds()):
                    if error:
                        raise error[0]
                    return result[0]
                logger.warning(
                    f"store {name} took over {self._op_timeout.total_seconds()}s "
                    f"(attempt {attempt}); reconnecting"
                )
                self._client = self._connect()
                if not retry:
                    break
            raise TimeoutError(f"store {name} timed out")

    def set(self, key: str, value: str | bytes) -> None:
        self._run("set", lambda c: c.set(key, value), retry=True)

    def get(self, key: str) -> bytes:
        self.wait([key])
        return self._run("get", lambda c: c.get(key), retry=True)

    def add(self, key: str, amount: int) -> int:
        return self._run("add", lambda c: c.add(key, amount), retry=amount == 0)

    def check(self, keys: list[str]) -> bool:
        return self._run("check", lambda c: c.check(keys), retry=True)

    def wait(self, keys: list[str], timeout: timedelta | None = None) -> None:
        timeout = self._timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout.total_seconds()
        while not self.check(keys):
            if time.monotonic() > deadline:
                raise dist.DistStoreError(f"wait for {keys} timed out after {timeout}")
            time.sleep(_POLL_INTERVAL)

    def delete_key(self, key: str) -> bool:
        return self._run("delete_key", lambda c: c.delete_key(key), retry=True)

    def num_keys(self) -> int:
        return self._run("num_keys", lambda c: c.num_keys(), retry=True)

    def set_timeout(self, timeout: timedelta) -> None:
        self._timeout = timeout

    def clone(self) -> "TimedStore":
        return TimedStore(
            self._connect,
            op_timeout=self._op_timeout,
            timeout=self._timeout,
            num_retries=self._num_retries,
        )
