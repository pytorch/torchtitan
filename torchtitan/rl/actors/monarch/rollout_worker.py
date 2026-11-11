# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from monarch.actor import Actor, concurrent_endpoint, current_rank

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.renderer import RendererConfig

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.rollout.rollouter import BaseRolloutWorker
from torchtitan.rl.rollout.types import RolloutGroup


class RolloutWorkerActor(Actor):
    """Monarch adapter that owns a configurable standalone rollout worker."""

    def __init__(
        self,
        *,
        worker_config: BaseRolloutWorker.Config,
        num_threads: int,
        structured_logger_init_fn: functools.partial[None] | None,
    ) -> None:
        if structured_logger_init_fn is not None:
            structured_logger_init_fn(
                source="rl_rollout_worker", rank=current_rank().rank
            )
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=num_threads)
        )
        self._worker = worker_config.build()
        # group id -> the task running that group's `run_group` here; `cancel_group` cancels it.
        self._group_tasks: dict[int, asyncio.Task] = {}
        # Generate calls sent from this worker that no generator has answered yet.
        self._generate_calls: set[asyncio.Future] = set()

    @concurrent_endpoint
    async def setup_async(
        self,
        *,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        vision_processor_config: VisionProcessor.Config | None = None,
    ) -> None:
        await self._worker.setup_async(
            tokenizer_config=tokenizer_config,
            renderer_config=renderer_config,
            hf_assets_path=hf_assets_path,
            vision_processor_config=vision_processor_config,
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
        # If a generator (or the router) replies to a worker that already stopped, Monarch fails that actor
        # and the job exits 130 (https://github.com/meta-pytorch/monarch/issues/5249, a reply-delivery bug).
        # So each generate call stays in `self._generate_calls` until answered, also after its rollout is
        # cancelled.
        # TODO(monarch): drop this once Monarch drops such a reply instead of failing the sender.
        async def generate(*args, **kwargs):
            call = asyncio.ensure_future(generate_fn(*args, **kwargs))
            self._generate_calls.add(call)
            call.add_done_callback(self._on_generate_call_done)
            # TODO: when the rollout is cancelled, abort its request on the generator too; the fix for
            # Monarch issue 5249 won't. Today the request decodes to its end and can pin its group's
            # cache salt again.
            return await asyncio.shield(call)

        self._group_tasks[group_id] = asyncio.current_task()
        try:
            return await self._worker.run_group(
                generate_fn=generate,
                release_session_fn=release_session_fn,
                sample=sample,
                group_id=group_id,
                group_size=group_size,
                sampling=sampling,
            )
        finally:
            del self._group_tasks[group_id]

    def _on_generate_call_done(self, call: asyncio.Future) -> None:
        self._generate_calls.discard(call)
        # Once its rollout is cancelled, `asyncio.shield` stops watching `call`, so nobody reads its
        # error and asyncio logs "exception was never retrieved". Read it here. A cancelled `call`
        # has no error (`exception()` would raise).
        if not call.cancelled():
            call.exception()

    @concurrent_endpoint
    async def cancel_group(self, group_id: int) -> None:
        """Cancel the group's `run_group` if it still runs on this worker. Returning means the
        group's work and cleanup ended; its generate calls may still wait for answers
        (`wait_for_generate_calls`).

            group still runs            cancel it; return once its `finally` blocks ran
            group already ended         return at once
            group ignores the cancel    return only when the group ends; the Rollouter stops waiting
                                        for this call after _SHUTDOWN_DRAIN_TIMEOUT_S

        Monarch starts one sender's concurrent endpoint calls in the order they were sent, so a
        `run_group` sent before this call is already in `_group_tasks`.
        """
        task = self._group_tasks.get(group_id)
        if task is not None:
            task.cancel()
            await asyncio.wait([task])

    @concurrent_endpoint
    async def wait_for_generate_calls(self) -> None:
        """Return once this worker got every generate answer owed to it when this call started.
        `Rollouter.close` calls it after every `run_group` ended, so no rollout sends a new call.

            rollout sends generate call c        _generate_calls = {c}
            cancel_group cancels the rollout     _generate_calls = {c}   its generator still owes c
            the generator answers c              _generate_calls = {}    this returns
        """
        # TODO: a call the Verifiers GenerationServer starts after this wait began isn't awaited.
        if self._generate_calls:
            await asyncio.wait(self._generate_calls)

    @concurrent_endpoint
    async def sync_log_step(self, step: int) -> None:
        sl.set_step(step)
