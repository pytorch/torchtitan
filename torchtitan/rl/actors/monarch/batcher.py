# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch actor adapter for the RL batcher."""

from __future__ import annotations

from monarch.actor import Actor, endpoint

from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.rl.actors.monarch.rdma_microbatch import (
    create_cuda_context_to_silence_monarch_log,
    MonarchRDMAMicrobatch,
)
from torchtitan.rl.rollout.types import RolloutGroup
from torchtitan.rl.trainer.batcher import Batcher, TrainerStepBatch


class _BatcherActorEndpoints:
    def __init__(
        self,
        config: Batcher.Config,
        *,
        num_tokens_per_microbatch_per_dp_rank: int,
        max_context_length: int,
        num_prompts_per_train_step: int,
        dp_degree: int,
        pad_id: int,
        temperature: float,
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
        if config.rdma_min_tensor_bytes is not None:
            create_cuda_context_to_silence_monarch_log()
        super().__init__(
            config,
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
            max_context_length=max_context_length,
            num_prompts_per_train_step=num_prompts_per_train_step,
            dp_degree=dp_degree,
            pad_id=pad_id,
            temperature=temperature,
            register_rdma_microbatch=MonarchRDMAMicrobatch.register,
        )

    @endpoint
    async def add_group(
        self, *, rollout_group: RolloutGroup
    ) -> tuple[TrainerStepBatch | None, bool]:
        """Plain `@endpoint`, not `@concurrent_endpoint`: calls run one at a time, so groups reach the
        stateful batcher in arrival order. `async` because Monarch rejects mixing sync and async endpoints."""
        return super().add_group(rollout_group=rollout_group)

    @endpoint
    async def maybe_release_oldest_batch(self) -> None:
        await super().maybe_release_oldest_batch()

    @endpoint
    async def sync_log_step(self, step: int) -> None:
        """Plain `@endpoint` like `add_group`, so the step is set before the next group's spans."""
        sl.set_step(step)


class BatcherActor(Actor, _BatcherActorEndpoints, Batcher):
    """Expose a :class:`Batcher` through Monarch endpoints, in its own process.

    Why not batch in the controller, whose busy event loop holds the GIL:
        on the loop thread:      blocks every rollout while it runs
        in `asyncio.to_thread`:  each GIL release waits up to 5 ms to get it back (the switch interval)

    Example:
        batcher = this_host().spawn_procs(per_host={"cpus": 1}).spawn(
            "batcher",
            BatcherActor,
            Batcher.Config(rdma_min_tensor_bytes=1024),
            num_prompts_per_train_step=2,
            **other_init_kwargs,  # see `_BatcherActorEndpoints.__init__`
        )
        await batcher.add_group.call_one(rollout_group=group0)  # -> (None, True)
        await batcher.add_group.call_one(rollout_group=group1)  # -> (batch of MonarchRDMAMicrobatches, True)
        batcher.maybe_release_oldest_batch.broadcast()  # once every trainer rank has read group1's batch
    """
