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

# TODO: load the `grain` and `datasets` packages only where they are used, to save ~450 MiB per rollout worker.
#   This import runs `torchtitan/components/data/__init__.py`, which loads both. The renderer and
#   `rl/rollout/data.py` import them too. First check that a worker doesn't need them.
from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.renderer import RendererConfig

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.rollouter import BaseRolloutWorker
from torchtitan.rl.rollout.types import RolloutGroup
from torchtitan.tools.garbage_collector import GarbageCollector


class RolloutWorkerActor(Actor):
    """Monarch adapter that owns a configurable standalone rollout worker."""

    def __init__(
        self,
        *,
        worker_config: BaseRolloutWorker.Config,
        worker_kwargs: dict[str, Any],
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
        self._worker = worker_config.build(**worker_kwargs)
        # Created at the end of `setup_async`, so setup isn't counted as lag.
        self._event_loop_lag: m.EventLoopLag
        self._thread_pool_lag: m.ThreadPoolLag

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

        # Setup objects live until exit: freeze them, so later full GCs skip them.
        GarbageCollector.freeze(
            "Full GC before freezing the rollout worker's setup objects"
        )

        # Start measuring after setup and the freeze: both hold the loop.
        self._event_loop_lag = m.EventLoopLag("rollout_worker")
        self._thread_pool_lag = m.ThreadPoolLag("rollout_worker")

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
        group = await self._worker.run_group(
            generate_fn=generate_fn,
            release_session_fn=release_session_fn,
            sample=sample,
            group_id=group_id,
            group_size=group_size,
            sampling=sampling,
        )

        # =========== Log worker metrics ===========
        # Validation groups (group_id < 0) get none: the controller drops their metrics, so what is
        # measured during validation goes with this worker's next training group.
        if group_id >= 0:
            group.metrics += self._flush_metrics()

        return group

    def _flush_metrics(self) -> list[m.Metric]:
        """Return this worker's metrics since its previous training group, then reset them."""
        # RAM this worker uses now (VmRSS) and the most it has used (VmHWM), shared libraries counted
        # in full. Each line reads e.g. "VmRSS:  1745628 kB".
        with open("/proc/self/status") as status:
            kib = {
                line.split(":")[0]: int(line.split()[1])
                for line in status
                if line.startswith(("VmRSS:", "VmHWM:"))
            }
        return [
            # The event loop's longest delay, and how many ms late it ran per second
            *self._event_loop_lag.flush(),
            # How late the thread pool starts a call: the longest delay, and ms late per second
            *self._thread_pool_lag.flush(),
            m.Metric("rollout_worker/ram_gib", m.Max(kib["VmRSS"] / 2**20)),
            m.Metric("rollout_worker/peak_ram_gib", m.Max(kib["VmHWM"] / 2**20)),
        ]

    @concurrent_endpoint
    async def sync_log_step(self, step: int) -> None:
        sl.set_step(step)
