# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Overlap the trainer->generator weight handoff with the next training step."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from torchtitan.observability import structured_logger as sl

from torchtitan.rl.components.work_buffer import RolloutGroupWorkBuffer
from torchtitan.rl.observability import metrics as m

if TYPE_CHECKING:
    from torchtitan.rl.distributed.routing.inter_generator import InterGeneratorRouter
    from torchtitan.rl.trainer import Trainer

# dummy no-op for step 0, used in WeightSyncManager
async def _noop() -> None:
    return None


class WeightSyncManager:
    """Overlap the trainer->generator weight sync with the next training step.

     Trainer weight push:
        - Called after optimizer.step()
        - Awaited before next optimizer.step (weights changes then)
    Generator weight pull:
        - Called after push completes.
        - Awaited before next push (weights changes then)

    Trained groups release their active slots only after every generator has
    pulled the new weights. Other slots may open earlier; the router pins a
    cache namespace once per group when its first request is routed.

    Example:
        for step in training_steps:
            fwd_bwd(batch)
            push_metrics = await weight_sync.wait_prev_push()    # before optim mutates the weights
            optimizer_result = await trainer.optimizer_step.call()
            pull_metrics = await weight_sync.wait_prev_pull()  # before the next push overwrites the key
            weight_sync.start_async_push_pull(version=optimizer_result.policy_version)
        await weight_sync.wait_inflight_push_pull()  # finish the last step's sync before validation
    """

    def __init__(
        self,
        *,
        trainer: Trainer,
        generator_router: InterGeneratorRouter,
        group_buffer: RolloutGroupWorkBuffer,
        num_prompts_per_train_step: int,
    ) -> None:
        self._trainer = trainer
        self._generator_router = generator_router
        self._group_buffer = group_buffer
        self._num_prompts_per_train_step = num_prompts_per_train_step

        # Step 0 has no `wait_prev_push/pull`, so we start with a noop task.
        self._trainer_push_task: asyncio.Task = asyncio.create_task(_noop())
        self._generator_pull_task: asyncio.Task = asyncio.create_task(_noop())

        # Wall time of the push and pull of the last completed sync.
        self._last_push_s: float = 0.0
        self._last_pull_s: float = 0.0

    def start_async_push_pull(self, *, version: int) -> None:
        """Fire push -> pull -> buffer-slot release in the background; returns immediately.

        Args:
            version: policy version the generators hold after the pull completes.
        """
        push_task = asyncio.create_task(self._trainer_push())
        self._trainer_push_task = push_task
        self._generator_pull_task = asyncio.create_task(
            self._generator_pull_and_release_buffer_slots(version, push_task)
        )

    async def wait_prev_push(self) -> list[m.Metric]:
        await self._trainer_push_task
        return [
            m.Metric(
                "timing/weight_sync/trainer_push_model_state_dict",
                m.NoReduce(self._last_push_s),
            )
        ]

    async def wait_prev_pull(self) -> list[m.Metric]:
        await self._generator_pull_task
        return [
            m.Metric(
                "timing/weight_sync/generator_pull_model_state_dict",
                m.NoReduce(self._last_pull_s),
            )
        ]

    async def wait_inflight_push_pull(self) -> None:
        """Finish the last in-flight push+pull so generators hold the final weights (e.g. before validation)."""
        await self.wait_prev_push()
        await self.wait_prev_pull()

    async def _trainer_push(self) -> None:
        with sl.log_trace_span("trainer_push_model_state_dict"):
            start = time.perf_counter()
            await self._trainer.push_model_state_dict.call()
            self._last_push_s = time.perf_counter() - start

    async def _generator_pull_and_release_buffer_slots(
        self, version: int, push_task: asyncio.Task
    ) -> None:
        await push_task
        with sl.log_trace_span("generator_pull_model_state_dict"):
            start = time.perf_counter()
            await self._generator_router.pull_model_state_dict.call_one(version)
            self._last_pull_s = time.perf_counter() - start
        # TODO(perf): trained slots wait for all pulls even though each generator
        #   can resume earlier. Per-generator slot release would require routing
        #   new groups to updated generators while preserving the freshness bound.

        # Keep trained slots occupied until every generator has installed the
        # new weights; existing groups retain their cache namespace across pulls.
        await self._group_buffer.release_active_groups(
            self._num_prompts_per_train_step, reason="trained"
        )
