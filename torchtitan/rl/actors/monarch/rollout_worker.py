# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import functools
import socket
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from monarch.actor import Actor, concurrent_endpoint, current_rank

from torchtitan.components.renderer import RendererConfig

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.rollout.rollouter import RolloutWorker
from torchtitan.rl.rollout.types import GenerateFn, RolloutGroup


class RolloutWorkerActor(Actor):
    """Monarch adapter that owns a configurable standalone rollout worker."""

    def __init__(
        self,
        *,
        worker_config: RolloutWorker.Config,
        num_threads: int,
        structured_logger_init_fn: functools.partial[None] | None,
        first_worker_index: int,
    ) -> None:
        if structured_logger_init_fn is not None:
            # Each host's workers form their own proc mesh, so `current_rank()` restarts at 0 on
            # every host. The offset keeps the log rank unique across hosts.
            structured_logger_init_fn(
                source="rl_rollout_worker",
                rank=first_worker_index + current_rank().rank,
            )
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=num_threads)
        )
        # TODO: os.nice(10), inherited by agent subprocesses, so ranks on a shared host win CPU
        #   fights; decide from a multi-node run.
        self._worker = worker_config.build()
        # group id -> the `run_group` call running it here, for `cancel_group`.
        self._group_tasks: dict[int, asyncio.Task] = {}
        # Generate calls the router hasn't answered yet, for `wait_for_generate_calls`.
        self._generate_calls: set[asyncio.Future] = set()

    @concurrent_endpoint
    async def setup_async(
        self,
        *,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
    ) -> None:
        await self._worker.setup_async(
            tokenizer_config=tokenizer_config,
            renderer_config=renderer_config,
            hf_assets_path=hf_assets_path,
        )

    @concurrent_endpoint
    async def run_group(
        self,
        *,
        generate_fn: Any,
        release_session_fn: Any,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: Any,
    ) -> RolloutGroup:
        self._group_tasks[group_id] = asyncio.current_task()
        try:
            return await self._worker.run_group(
                generate_fn=self._tracked(generate_fn),
                release_session_fn=release_session_fn,
                sample=sample,
                group_id=group_id,
                group_size=group_size,
                sampling=sampling,
            )
        finally:
            self._group_tasks.pop(group_id, None)

    def _tracked(self, generate_fn: GenerateFn) -> GenerateFn:
        """Wrap `generate_fn` so a cancelled caller stops waiting while its call stays in
        `self._generate_calls` until the router replies."""
        # TODO(monarch): drop the tracking once a reply to a stopped actor no longer fails its sender.

        async def generate(*args, **kwargs):
            call = asyncio.ensure_future(generate_fn(*args, **kwargs))
            self._generate_calls.add(call)
            call.add_done_callback(self._on_generate_call_done)
            return await asyncio.shield(call)

        return generate

    def _on_generate_call_done(self, call: asyncio.Future) -> None:
        self._generate_calls.discard(call)
        # A cancelled caller never reads the error; read it so asyncio does not log it.
        if not call.cancelled():
            call.exception()

    @concurrent_endpoint
    async def cancel_group(self, group_id: int) -> None:
        """Cancel this group's `run_group` if this worker runs it, and wait until it ended. Monarch
        starts one sender's concurrent calls in order, so a `run_group` sent first is found here.
        A group that never stops (e.g. an env that swallows the cancel) hangs this call."""
        # TODO: group ids must be unique among running groups; every validation pass reuses
        # -1, -2, ..., so two overlapping passes would clash here.
        task = self._group_tasks.get(group_id)
        if task is not None:
            task.cancel()
            await asyncio.wait([task])

    @concurrent_endpoint
    async def wait_for_generate_calls(self) -> None:
        """Return once the router answered every generate call sent from this worker, also the
        calls of cancelled rollouts. `Rollouter.close` calls it before it stops the workers."""
        if self._generate_calls:
            await asyncio.wait(self._generate_calls)

    @concurrent_endpoint
    async def sync_log_step(self, step: int) -> None:
        sl.set_step(step)

    @concurrent_endpoint
    async def hostname(self) -> str:
        return socket.gethostname()

    async def __cleanup__(self, exc: Exception | None) -> None:
        """Monarch runs this on a graceful stop (`mesh.stop()`), after cancelling the endpoint
        calls. A killed process skips it."""
        await self._worker.close()
