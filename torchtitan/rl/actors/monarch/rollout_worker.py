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

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.renderer import RendererConfig

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.observability.metrics.process import ProcessStats
from torchtitan.rl.rollout.rollouter import BaseRolloutWorker
from torchtitan.rl.rollout.types import GenerateFn, ReleaseSessionFn, RolloutGroup


class RolloutWorkerActor(Actor):
    """Monarch adapter that owns a configurable standalone rollout worker."""

    def __init__(
        self,
        *,
        worker_config: BaseRolloutWorker.Config,
        num_threads: int | None,
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
        self._worker = worker_config.build()
        # With children: an env runtime may run each agent as a subprocess.
        self._process_stats = ProcessStats("rollout_worker", children=True)

    @concurrent_endpoint
    async def setup_async(
        self,
        *,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        vision_processor_config: VisionProcessor.Config | None = None,
    ) -> None:
        await self._worker.setup_async(
            generate_fn=generate_fn,
            release_session_fn=release_session_fn,
            tokenizer_config=tokenizer_config,
            renderer_config=renderer_config,
            hf_assets_path=hf_assets_path,
            vision_processor_config=vision_processor_config,
        )

    @concurrent_endpoint
    async def run_group(
        self,
        *,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: Any,
    ) -> RolloutGroup:
        # TODO: once a group's per-token arrays (router replay rows, top-k logprobs) pass about
        #   3 GiB, send RDMA handles in this reply, registered once per rollout and released after
        #   the batcher reads them: Monarch caps a message at 10 GiB.
        return await self._worker.run_group(
            sample=sample,
            group_id=group_id,
            group_size=group_size,
            sampling=sampling,
        )

    @concurrent_endpoint
    async def sync_log_step(self, step: int) -> None:
        sl.set_step(step)

    @concurrent_endpoint
    async def hostname(self) -> str:
        return socket.gethostname()

    @concurrent_endpoint
    async def process_metrics(self) -> list[m.Metric]:
        """This worker's and its agent subprocesses' CPU since the previous call, and their memory."""
        # A /proc scan, 0.18-0.30 s on a host with ~2,600 processes. It runs off the event loop, on
        # one of the `num_threads` default-executor threads that rollouts also use.
        # TODO: one scan per host instead of one per worker.
        return await asyncio.to_thread(self._process_stats.metrics)

    @concurrent_endpoint
    async def close(self) -> None:
        """The Rollouter calls this before it stops the worker: a slow close (e.g. stopping
        sandboxes) can't run in `__cleanup__`, which Monarch cancels after 3 s."""
        await self._worker.close()

    async def __cleanup__(self, exc: Exception | None) -> None:
        """Monarch runs this when the actor stops; a killed process skips it. It closes a worker
        the Rollouter didn't, e.g. after an error."""
        await self._worker.close()
