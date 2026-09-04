# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import contextlib
import functools
import multiprocessing
import os
from dataclasses import dataclass, field
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


REQUEST_IDS_BY_NODE_INFO_KEY = "torchtitan_request_ids_by_node"
_REQUEST_ID_PATCH_MARKER = "_torchtitan_records_request_id"


def _wrap_commit_to_retain_response_id(commit):
    """Carry the key for TorchTitan generation metadata across processes.

    ``GenerationServer`` and the Verifiers EnvServer run in different
    processes and record different halves of a model call:

    1. ``GenerationServer`` creates and returns ``request_id``, then stores a
       ``VerifiersGenerationMetadata`` value in
       ``generation_metadata[request_id]``.
    2. Verifiers converts that same ``request_id`` field to ``Response.id`` and
       commits the response as an assistant node. Verifiers 0.3.1 does not
       retain ``Response.id`` in its trace, so this wrapper stores it as
       ``trace.info[node] = response.id``.
    3. ``VerifiersRollouter`` later receives the trace and must match each node
       with its ``VerifiersGenerationMetadata`` value.

    Matching the two lists by position is incorrect when concurrent requests
    finish or commit in different orders. This wrapper runs synchronously around
    the commit, when both values are known, and stores
    ``trace.info[node] = response.id``. ``Trace.info`` crosses the EnvServer
    process boundary, so ``VerifiersRollouter`` can use
    ``node -> request_id -> metadata`` instead of relying on list order.
    """

    @functools.wraps(commit)
    def commit_with_request_id(turn, response, tools=None):
        node = commit(turn, response, tools)
        if response.id:
            request_ids = turn.trace.info.setdefault(REQUEST_IDS_BY_NODE_INFO_KEY, {})
            request_ids[str(node)] = response.id
        return node

    setattr(commit_with_request_id, _REQUEST_ID_PATCH_MARKER, True)
    return commit_with_request_id


def _setup_env_server_process() -> None:
    """Configure logging and install TorchTitan's Verifiers compatibility patch."""
    setup_logging("INFO")

    # Verifiers 0.3.1 carries the generation request ID on Response.id but drops
    # it when committing the response to a Trace. Preserve it in Trace.info so
    # policy-version metadata can be joined by identity instead of call order.
    from verifiers.v1.graph import PendingTurn

    if not getattr(PendingTurn.commit, _REQUEST_ID_PATCH_MARKER, False):
        PendingTurn.commit = _wrap_commit_to_retain_response_id(PendingTurn.commit)


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
        log_setup=_setup_env_server_process,
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
