# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Overlap the trainer->generator weight handoff with the next training step."""

import asyncio
import time

from torchtitan.experiments.rl.components.work_buffer import RolloutGroupWorkBuffer
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.routing.inter_generator_router import (
    InterGeneratorRouter,
)
from torchtitan.observability import structured_logger as sl


class WeightSyncManager:
    """Overlap the trainer->generator weight sync with the next training step.

     Trainer weight push:
        - Called after optimizer.step()
        - Awaited before next optimizer.step (weights changes then)
    Generator weight pull:
        - Called after push completes.
        - Awaited before next push (weights changes then)

    Impact on off-policiness: The buffer guarantees that no sample will be born stale,
    as long as we call `self._group_buffer.release_active_groups` after the pull.

    Example:
        for step in training_steps:
            fwd_bwd(batch)
            push_metrics = await weight_sync.wait_prev_trainer_weight_push()    # before optim mutates the weights
            optim_result = await trainer.optim_step.call()
            pull_metrics = await weight_sync.wait_prev_generator_weight_pull()  # before the next push overwrites the key
            weight_sync.start_async_push_pull(version=optim_result.policy_version)
        await weight_sync.wait_inflight_push_pull()  # finish the last step's sync before validation
    """

    def __init__(
        self,
        *,
        trainer,  # PolicyTrainer actor handle
        generator_router: InterGeneratorRouter,
        group_buffer: RolloutGroupWorkBuffer,
        groups_per_train_step: int,
    ) -> None:
        self._trainer = trainer
        self._generator_router = generator_router
        self._group_buffer = group_buffer
        self._groups_per_train_step = groups_per_train_step
        self._trainer_push_task: asyncio.Task | None = None
        self._generator_pull_task: asyncio.Task | None = None

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

    async def wait_prev_trainer_weight_push(self) -> list[m.Metric]:
        if self._trainer_push_task is not None:
            await self._trainer_push_task
        return [
            m.Metric(
                "timing/weight_sync/trainer_push_model_state_dict",
                m.NoReduce(self._last_push_s),
            )
        ]

    async def wait_prev_generator_weight_pull(self) -> list[m.Metric]:
        if self._generator_pull_task is not None:
            await self._generator_pull_task
        return [
            m.Metric(
                "timing/weight_sync/generator_pull_model_state_dict",
                m.NoReduce(self._last_pull_s),
            )
        ]

    async def wait_inflight_push_pull(self) -> None:
        """Finish the last in-flight push+pull so generators hold the final weights (e.g. before validation)."""
        await self.wait_prev_trainer_weight_push()
        await self.wait_prev_generator_weight_pull()

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
            await self._generator_router.pull_model_state_dict(policy_version=version)
            self._last_pull_s = time.perf_counter() - start
        # Born-fresh: admit the next groups only now that the generators are on `version`, so a new
        # rollout starts at the current version (keeps policy_age <= max_offpolicy_steps).
        await self._group_buffer.release_active_groups(
            self._groups_per_train_step, reason="trained"
        )
