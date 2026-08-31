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
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

from torchtitan.config import Configurable
from torchtitan.experiments.rl.rollout.verifiers.dataset import _load_taskset_id


def _serve_env_from_config(
    config_path: str,
    address: str,
    address_queue: Any,
    death_pipe: Any,
) -> None:
    """Run a Verifiers EnvServer from a TOML file in a spawned process."""
    import tomllib
    from functools import partial

    from verifiers.v1.configs.serve import pool_serve_kwargs, ServeConfig
    from verifiers.v1.serve import env_config_data, serve_env
    from verifiers.v1.utils.loaders import resolve_env_config
    from verifiers.v1.utils.logging import setup_logging

    no_proxy = os.environ.get("no_proxy", "")
    no_proxy = ",".join(filter(None, (no_proxy, "127.0.0.1", "localhost")))
    os.environ["no_proxy"] = no_proxy
    os.environ["NO_PROXY"] = no_proxy

    with open(config_path, "rb") as file:
        data = tomllib.load(file)
    taskset = data.get("env", {}).get("taskset", {})
    if taskset_id := taskset.get("id"):
        taskset["id"] = _load_taskset_id(taskset_id)
    env_config = resolve_env_config(data.get("env"))
    serve_config = ServeConfig.model_validate(data.get("serve", {}))
    serve_env(
        **pool_serve_kwargs(serve_config.pool),
        legacy=False,
        address=address,
        address_queue=address_queue,
        death_pipe=death_pipe,
        log_setup=partial(setup_logging, "INFO"),
        config_data=env_config_data(env_config),
        max_concurrent=serve_config.max_concurrent,
    )


class VerifiersEnvServer(Configurable):
    """Locally managed Verifiers EnvServer process."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        # TOML file defining the Verifiers environment and server pool.
        config_path: str
        # ZMQ address for the local server; port 0 requests an ephemeral port.
        bind_address: str = "tcp://127.0.0.1:0"
        # Maximum time to wait for the server process to publish its address.
        startup_timeout_sec: float = 120.0

        def __post_init__(self) -> None:
            if not Path(self.config_path).is_file():
                raise ValueError(
                    f"Verifiers EnvServer config does not exist: {self.config_path}"
                )
            if self.startup_timeout_sec <= 0:
                raise ValueError("startup_timeout_sec must be positive")

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
            target=_serve_env_from_config,
            args=(
                self.config.config_path,
                self.config.bind_address,
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
                    await self._close_resources(
                        process=process,
                        address_queue=address_queue,
                        parent_conn=parent_conn,
                    )
                    raise RuntimeError(
                        f"Verifiers EnvServer exited with code {exit_code}"
                    ) from None
                if asyncio.get_running_loop().time() >= deadline:
                    await self._close_resources(
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
        await self._close_resources(
            process=process,
            address_queue=address_queue,
            parent_conn=parent_conn,
        )

    @staticmethod
    async def _close_resources(
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
