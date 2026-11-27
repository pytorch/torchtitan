# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch actor adapter for the RL training batch builder."""

from __future__ import annotations

from monarch.actor import endpoint

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.rl.actors.monarch.actor import Actor
from torchtitan.rl.actors.monarch.rdma_microbatch import MonarchRDMAMicrobatch
from torchtitan.rl.rollout.types import RolloutGroup
from torchtitan.rl.trainer.training_batch_builder import TrainingBatchBuilder, TrainerStepBatch


class _TrainingBatchBuilderActorEndpoints:
    def __init__(
        self,
        config: TrainingBatchBuilder.Config,
        *,
        num_tokens_per_microbatch_per_dp_rank: int,
        max_context_length: int,
        num_prompts_per_train_step: int,
        dp_degree: int,
        num_pp_microbatches: int = 1,
        pad_id: int,
        temperature: float,
        num_topk_logprobs: int,
        max_kept_set_size: int | None,
        output_dir: str,
        enable_structured_logging: bool,
        vision_processor: VisionProcessor.Config | None = None,
    ) -> None:
        init_logger()
        sl.init_structured_logger(
            source="rl_training_batch_builder",
            output_dir=output_dir,
            rank=0,
            enable=enable_structured_logging,
        )
        super().__init__(
            config,
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
            max_context_length=max_context_length,
            num_prompts_per_train_step=num_prompts_per_train_step,
            dp_degree=dp_degree,
            num_pp_microbatches=num_pp_microbatches,
            pad_id=pad_id,
            temperature=temperature,
            num_topk_logprobs=num_topk_logprobs,
            max_kept_set_size=max_kept_set_size,
            register_rdma_microbatch=MonarchRDMAMicrobatch.register,
            vision_processor=vision_processor,
        )

    @endpoint
    async def add_group(
        self, *, rollout_group: RolloutGroup
    ) -> tuple[TrainerStepBatch | None, bool]:
        """Plain `@endpoint`, not `@concurrent_endpoint`: calls run one at a time, so groups reach the
        stateful training batch builder in arrival order. `async` because Monarch rejects mixing sync and async endpoints."""
        return super().add_group(rollout_group=rollout_group)

    @endpoint
    async def maybe_release_oldest_batch(self) -> None:
        await super().maybe_release_oldest_batch()

    @endpoint
    async def sync_log_step(self, step: int) -> None:
        """Plain `@endpoint` like `add_group`, so the step is set before the next group's spans."""
        sl.set_step(step)


class TrainingBatchBuilderActor(Actor, _TrainingBatchBuilderActorEndpoints, TrainingBatchBuilder):
    """Expose a :class:`TrainingBatchBuilder` through Monarch endpoints, in its own process.

    Why not batch in the controller, whose busy event loop holds the GIL:
        on the loop thread:      blocks every rollout while it runs
        in `asyncio.to_thread`:  each GIL release waits up to 5 ms to get it back (the switch interval)

    Example:
        training_batch_builder = this_host().spawn_procs(per_host={"cpus": 1}).spawn(
            "training_batch_builder",
            TrainingBatchBuilderActor,
            TrainingBatchBuilder.Config(),
            num_prompts_per_train_step=2,
            **other_init_kwargs,  # see `_TrainingBatchBuilderActorEndpoints.__init__`
        )
        await training_batch_builder.add_group.call_one(rollout_group=group0)  # -> (None, True)
        await training_batch_builder.add_group.call_one(rollout_group=group1)  # -> (batch of MonarchRDMAMicrobatches, True)
        training_batch_builder.maybe_release_oldest_batch.broadcast()  # once every trainer rank has read group1's batch
    """
