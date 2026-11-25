# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch actor adapter for the RL batcher."""

from __future__ import annotations

from monarch.actor import endpoint

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.rl.actors.monarch.actor import Actor
from torchtitan.rl.rollout.types import RolloutGroupSummary
from torchtitan.rl.trainer.batcher import Batcher, StepPlan


class _BatcherActorEndpoints:
    def __init__(
        self,
        config: Batcher.Config,
        *,
        num_tokens_per_microbatch_per_dp_rank: int,
        max_context_length: int,
        num_prompts_per_train_step: int,
        dp_degree: int,
        num_pp_microbatches: int = 1,
        output_dir: str,
        enable_structured_logging: bool,
        vision_processor: VisionProcessor.Config | None = None,
    ) -> None:
        init_logger()
        sl.init_structured_logger(
            source="rl_batcher",
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
            vision_processor=vision_processor,
        )

    @endpoint
    async def add_group(
        self, *, summary: RolloutGroupSummary
    ) -> tuple[StepPlan | None, bool]:
        """Plain `@endpoint`, not `@concurrent_endpoint`: calls run one at a time, so groups reach the
        stateful batcher in arrival order. `async` because Monarch rejects mixing sync and async endpoints."""
        return super().add_group(summary=summary)

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
            Batcher.Config(),
            num_prompts_per_train_step=2,
            **other_init_kwargs,  # see `_BatcherActorEndpoints.__init__`
        )
        await batcher.add_group.call_one(summary=summary0)  # -> (None, True)
        await batcher.add_group.call_one(summary=summary1)  # -> (StepPlan, True)
    """
