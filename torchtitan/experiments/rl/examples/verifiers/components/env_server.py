# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import os
from dataclasses import dataclass, field
from functools import partial
from queue import Empty
from typing import Any

from verifiers.v1.configs.env import EnvConfig as VerifiersEnvConfig
from verifiers.v1.configs.serve import (
    pool_serve_kwargs,
    ServeConfig as VerifiersServeConfig,
)
from verifiers.v1.serve import env_config_data, serve_env
from verifiers.v1.utils.loaders import resolve_env_config
from verifiers.v1.utils.logging import setup_logging

from torchtitan.config import Configurable
from torchtitan.experiments.rl.examples.verifiers.components.dataset import (
    register_local_taskset_alias,
)


def _run_env_server_process(
    environment: dict[str, Any],
    serve: VerifiersServeConfig,
    local_taskset_module: str | None,
    address_queue: Any,
    death_pipe: Any,
) -> None:
    """Run a Verifiers EnvServer from config data in a spawned process."""

    no_proxy = os.environ.get("no_proxy", "")
    no_proxy = ",".join(filter(None, (no_proxy, "127.0.0.1", "localhost")))
    os.environ["no_proxy"] = no_proxy
    os.environ["NO_PROXY"] = no_proxy

    if local_taskset_module is not None:
        environment["taskset"]["id"] = register_local_taskset_alias(
            local_taskset_module
        )
    env_config = resolve_env_config(environment)
    serve_env(
        **pool_serve_kwargs(serve.pool),
        address=serve.address,
        address_queue=address_queue,
        death_pipe=death_pipe,
        log_setup=partial(setup_logging, "INFO"),
        config_data=env_config_data(env_config),
        max_concurrent=serve.max_concurrent,
    )


class VerifiersEnvServer(Configurable):
    """Locally managed Verifiers EnvServer process."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        environment: VerifiersEnvConfig
        """Typed Verifiers environment and agent configuration."""

        serve: VerifiersServeConfig = field(
            default_factory=lambda: VerifiersServeConfig(address="tcp://127.0.0.1:0")
        )
        """Typed Verifiers worker-pool and bind-address configuration."""

        local_taskset_module: str | None = None
        """Dotted local taskset module to register in the spawned server process."""

        startup_timeout_sec: float = 120.0
        """Maximum time to wait for the server process to publish its address."""

        def __post_init__(self) -> None:
            if self.startup_timeout_sec <= 0:
                raise ValueError("startup_timeout_sec must be positive")

        def to_dict(self) -> dict[str, Any]:
            return {
                "environment": self.environment.model_dump(mode="json"),
                "serve": self.serve.model_dump(mode="json"),
                "local_taskset_module": self.local_taskset_module,
                "startup_timeout_sec": self.startup_timeout_sec,
            }

    def __init__(self, config: Config) -> None:
        self.config = config
        self.process: Any = None
        self.address_queue: Any = None
        self.parent_conn: Any = None
        self.address: str | None = None

    async def start(self) -> str:
        """Start the server and return its resolved ZMQ address."""
        if self.address is not None:
            return self.address

        context = multiprocessing.get_context("spawn")
        address_queue = context.Queue()
        parent_conn, child_conn = context.Pipe()
        process = context.Process(
            target=_run_env_server_process,
            args=(
                env_config_data(self.config.environment),
                self.config.serve,
                self.config.local_taskset_module,
                address_queue,
                child_conn,
            ),
            daemon=False,
        )
        process.start()
        child_conn.close()

        deadline = asyncio.get_running_loop().time() + self.config.startup_timeout_sec
        while True:
            try:
                address = address_queue.get_nowait()
                break
            except Empty:
                if not process.is_alive():
                    exit_code = process.exitcode
                    await self._close_server_process_resources(
                        process=process,
                        address_queue=address_queue,
                        parent_conn=parent_conn,
                    )
                    raise RuntimeError(
                        f"Verifiers EnvServer exited with code {exit_code}"
                    ) from None
                if asyncio.get_running_loop().time() >= deadline:
                    await self._close_server_process_resources(
                        process=process,
                        address_queue=address_queue,
                        parent_conn=parent_conn,
                    )
                    raise TimeoutError(
                        "Verifiers EnvServer did not publish its address within "
                        f"{self.config.startup_timeout_sec} seconds"
                    ) from None
                await asyncio.sleep(0.1)

        self.process = process
        self.address_queue = address_queue
        self.parent_conn = parent_conn
        self.address = address
        return address

    async def close(self) -> None:
        """Stop the EnvServer and release its multiprocessing resources."""
        process = self.process
        address_queue = self.address_queue
        parent_conn = self.parent_conn
        self.process = None
        self.address_queue = None
        self.parent_conn = None
        self.address = None
        await self._close_server_process_resources(
            process=process,
            address_queue=address_queue,
            parent_conn=parent_conn,
        )

    @staticmethod
    async def _close_server_process_resources(
        *,
        process: Any,
        address_queue: Any,
        parent_conn: Any,
    ) -> None:
        if process is not None:
            process.terminate()
            await asyncio.to_thread(process.join, 10)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 5)
            process.close()
        if parent_conn is not None:
            with contextlib.suppress(Exception):
                parent_conn.close()
        if address_queue is not None:
            address_queue.close()
            address_queue.join_thread()
