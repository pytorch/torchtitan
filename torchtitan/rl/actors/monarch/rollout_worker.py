# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from monarch.actor import Actor, concurrent_endpoint, current_rank

from torchtitan.components.renderer import RendererConfig

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.rollouter import RolloutWorker
from torchtitan.rl.rollout.types import RolloutGroup


class RolloutWorkerActor(Actor):
    """Monarch adapter that owns a configurable standalone rollout worker."""

    def __init__(
        self,
        *,
        worker_config: RolloutWorker.Config,
        num_threads: int,
        structured_logger_init_fn: Callable[..., None] | None,
    ) -> None:
        if structured_logger_init_fn is not None:
            structured_logger_init_fn(
                source="rl_rollout_worker", rank=current_rank().rank
            )
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=num_threads)
        )
        self._worker = worker_config.build()
        self._gc_timer = m.GCTimer("rollout_worker")
        self._loop_lag = m.EventLoopLag("rollout_worker")
        # group id -> the `run_group` call running it here, for `cancel_group`.
        self._group_tasks: dict[int, asyncio.Task] = {}

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
        task = asyncio.current_task()
        self._group_tasks[group_id] = task
        try:
            group = await self._worker.run_group(
                generate_fn=generate_fn,
                release_session_fn=release_session_fn,
                sample=sample,
                group_id=group_id,
                group_size=group_size,
                sampling=sampling,
            )
            # This process's GC time and event-loop lag since the last finished training group;
            # summed over a step's groups. Validation (negative group ids) ignores group metrics,
            # so it keeps them.
            if group_id >= 0:
                group.metrics += [*self._gc_timer.flush(), *self._loop_lag.flush()]
            return group
        finally:
            # Validation reuses group ids, and a `run_group` can outlive a failed `cancel_group`:
            # don't drop a newer call's entry.
            if self._group_tasks.get(group_id) is task:
                del self._group_tasks[group_id]

    @concurrent_endpoint
    async def cancel_group(self, group_id: int) -> None:
        """Cancel this group's `run_group` if this worker runs it, and wait until it ended. Monarch
        starts one sender's concurrent calls in order, so a `run_group` sent first is found here.
        A group that never stops (e.g. an env that swallows the cancel) hangs this call."""
        # TODO: every validation pass reuses group ids -1, -2, ...; if two passes ever overlap,
        # this cancels both groups with one id. Give each pass its own ids first.
        task = self._group_tasks.get(group_id)
        if task is not None:
            task.cancel()
            await asyncio.wait([task])

    @concurrent_endpoint
    async def sync_log_step(self, step: int) -> None:
        sl.set_step(step)
