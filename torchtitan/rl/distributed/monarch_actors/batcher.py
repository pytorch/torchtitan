# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch actor that turns finished rollout groups into training batches."""

from __future__ import annotations

import asyncio
import dataclasses
from collections import deque
from typing import Any

import monarch
from monarch.actor import Actor, endpoint

from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.rl.components.batcher import Batcher, TrainerStepBatch
from torchtitan.rl.components.training_sample_builder import TrainingSampleBuilder
from torchtitan.rl.distributed.rdma_microbatch import (
    create_cuda_context_to_silence_monarch_log,
    RDMAMicrobatch,
)
from torchtitan.rl.rollout.types import RolloutGroup


class BatcherActor(Actor):
    """Turns finished rollout groups into training batches, in its own process.

    Keeps each batch's tensors until `release_oldest_batch` and returns `RDMAMicrobatch`es, so
    trainer ranks read from this process.

    Why not in the controller, whose busy event loop holds the GIL:
        build on the loop thread:     blocks every rollout while it runs
        pack in `asyncio.to_thread`:  each torch op waits up to 5 ms (the GIL switch interval)

    Args:
        training_sample_builder_config: Built in this process.
        batcher_config: Built in this process with `batcher_kwargs`.
        batcher_kwargs: Runtime kwargs for `batcher_config.build`, e.g. `dp_degree` and `pad_id`.
        enable_rdma: See `AsyncLoopConfig.enable_rdma`.
        output_dir: Where the structured logger writes this process's spans.
        enable_structured_logging: Whether to write those spans.

    Example:
        batcher = this_host().spawn_procs(per_host={"cpus": 1}).spawn(
            "batcher",
            BatcherActor,
            training_sample_builder_config=TrainingSampleBuilder.Config(),
            batcher_config=Batcher.Config(),
            batcher_kwargs=dict(num_prompts_per_train_step=2, dp_degree=1, ...),
            enable_rdma=False,
            output_dir="outputs",
            enable_structured_logging=False,
        )
        await batcher.add_group.call_one(rollout_group=group0)  # -> (None, True)
        await batcher.add_group.call_one(rollout_group=group1)  # -> (TrainerStepBatch of RDMAMicrobatches, True)
        batcher.release_oldest_batch.broadcast()  # once every trainer rank has read group1's batch
    """

    def __init__(
        self,
        *,
        training_sample_builder_config: TrainingSampleBuilder.Config,
        batcher_config: Batcher.Config,
        batcher_kwargs: dict[str, Any],
        enable_rdma: bool,
        output_dir: str,
        enable_structured_logging: bool,
    ) -> None:
        init_logger()
        sl.init_structured_logger(
            source="rl_batcher",
            output_dir=output_dir,
            rank=0,
            enable=enable_structured_logging,
        )
        self._training_sample_builder = training_sample_builder_config.build()
        self._batcher = batcher_config.build(**batcher_kwargs)
        if not enable_rdma:
            # Trainer ranks read this process's buffers over TCP; their own processes keep ibverbs.
            monarch.configure(rdma_disable_ibverbs=True)
        create_cuda_context_to_silence_monarch_log()
        # Batches trainer ranks may still read, oldest first.
        self._batches_being_read: deque[
            tuple[TrainerStepBatch, list[list[RDMAMicrobatch]]]
        ] = deque()
        # The last released batch, already dropped; freed at the next release.
        self._dropped_batch: TrainerStepBatch | None = None

    @endpoint
    async def add_group(
        self, *, rollout_group: RolloutGroup
    ) -> tuple[TrainerStepBatch | None, bool]:
        """Build this group's training samples and add them; returns `(batch | None, group_is_trainable)`.

        Plain `@endpoint`, not `@concurrent_endpoint` like the other actors: each call finishes
        before the next starts, so groups reach the stateful batcher in arrival order. `async`
        because Monarch rejects mixing sync and async endpoints, and `release_oldest_batch` awaits.
        """
        with sl.log_trace_span("training_sample_builder"):
            training_sample_group = self._training_sample_builder.build_from_group(
                rollout_group=rollout_group
            )
        with sl.log_trace_span("batcher_pack"):
            batch, group_is_trainable = self._batcher.add_training_samples(
                training_sample_group=training_sample_group
            )
        if batch is None:
            return batch, group_is_trainable
        # The reply carries RDMAMicrobatches, not large tensors: trainer ranks read their columns here.
        with sl.log_trace_span("register_rdma_microbatches"):
            rdma_microbatches = [
                [RDMAMicrobatch.register(microbatch) for microbatch in row]
                for row in batch.microbatches
            ]
        self._batches_being_read.append((batch, rdma_microbatches))
        return (
            dataclasses.replace(batch, microbatches=rdma_microbatches),
            group_is_trainable,
        )

    @endpoint
    async def release_oldest_batch(self) -> None:
        """Free the oldest batch, once every trainer rank has read it.

        Call once per trained batch, in training order, after `forward_backward` returns.
        """
        batch, rdma_microbatches = self._batches_being_read.popleft()
        await asyncio.gather(
            *(
                rdma_microbatch.drop()
                for row in rdma_microbatches
                for rdma_microbatch in row
            )
        )
        # Keep the tensors until the next release: drop() returns before Monarch 0.6 lets go of
        # their storage, and a free racing that release deadlocks its RDMA cache lock.
        self._dropped_batch = batch

    @endpoint
    async def sync_log_step(self, step: int) -> None:
        """Plain `@endpoint` like `add_group`, so messages stay in order: the step is set before
        the next group's spans."""
        sl.set_step(step)
