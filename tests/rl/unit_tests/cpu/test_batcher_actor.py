# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""BatcherActor, in its own process, returns the in-process builder + batcher's batch as RDMAMicrobatches."""

import asyncio
import dataclasses

import pytest
import torch
from monarch.actor import ActorError, this_host

from torchtitan.rl.components.batcher import Batcher
from torchtitan.rl.components.training_sample_builder import TrainingSampleBuilder
from torchtitan.rl.distributed.monarch_actors.batcher import BatcherActor
from torchtitan.rl.distributed.rdma_microbatch import _TensorRef
from torchtitan.rl.rollout.types import (
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
    RolloutTurnID,
)

_BATCHER_KWARGS = dict(
    num_tokens_per_microbatch_per_dp_rank=256,
    max_context_length=16,
    num_prompts_per_train_step=2,
    dp_degree=2,
    pad_id=0,
    temperature=1.0,
)


def _group(group_id: int, rewards: list[float]) -> RolloutGroup:
    rollouts = [
        Rollout(
            group_id=group_id,
            rollout_id=rollout_id,
            turns=[
                RolloutTurn(
                    rollout_id=RolloutTurnID(
                        group_id=group_id, rollout_id=rollout_id, turn_id=0
                    ),
                    prompt_token_ids=[1, 2, 3],
                    completion_token_ids=[4 + rollout_id, 5],
                    completion_logprobs=[-0.5, -0.25],
                    min_policy_version=0,
                    max_policy_version=0,
                )
            ],
            status=RolloutStatus.COMPLETED,
            reward=reward,
            advantage=reward - sum(rewards) / len(rewards),
        )
        for rollout_id, reward in enumerate(rewards)
    ]
    return RolloutGroup(group_id=group_id, rollouts=rollouts)


def test_batcher_actor_returns_the_in_process_batch(tmp_path) -> None:
    # Group 1 has zero reward spread, so the builder drops it: it rides along but isn't trainable.
    groups = [_group(0, [0.0, 1.0]), _group(1, [1.0, 1.0]), _group(2, [1.0, 0.0])]

    builder = TrainingSampleBuilder.Config().build()
    batcher = Batcher.Config().build(**_BATCHER_KWARGS)
    expected = [
        batcher.add_training_samples(
            training_sample_group=builder.build_from_group(rollout_group=group)
        )
        for group in groups
    ]

    async def run() -> tuple[list, list]:
        mesh = this_host().spawn_procs(per_host={"cpus": 1})
        try:
            actor = mesh.spawn(
                "batcher",
                BatcherActor,
                training_sample_builder_config=TrainingSampleBuilder.Config(),
                batcher_config=Batcher.Config(),
                batcher_kwargs=_BATCHER_KWARGS,
                enable_rdma=False,
                output_dir=str(tmp_path),
                enable_structured_logging=False,
            )
            results = [
                await actor.add_group.call_one(rollout_group=group) for group in groups
            ]
            # Read the RDMAMicrobatches from this process, as a trainer rank would, then free them.
            batch = results[-1][0]
            microbatches = [
                [await rdma_microbatch.read() for rdma_microbatch in row]
                for row in batch.microbatches
            ]
            await actor.release_oldest_batch.call_one()
            # A builder error surfaces at call_one.
            no_advantage = _group(3, [0.0, 1.0])
            no_advantage.rollouts[0].advantage = None
            with pytest.raises(ActorError, match="has no advantage"):
                await actor.add_group.call_one(rollout_group=no_advantage)
            return results, microbatches
        finally:
            await mesh.stop()

    results, microbatches = asyncio.run(run())

    assert [(batch is None, trainable) for batch, trainable in results] == [
        (True, True),
        (True, False),
        (False, True),
    ]
    batch, expected_batch = results[-1][0], expected[-1][0]
    assert batch.group_ids == [0, 1, 2]
    # int64 fields (2 KiB) cross processes by ref; the token counts go by value.
    assert isinstance(batch.microbatches[0][0].fields["input"], _TensorRef)
    assert isinstance(
        batch.microbatches[0][0].fields["loss_token_counts"], torch.Tensor
    )
    # assert_close can't compare metric objects; every other field must match exactly.
    torch.testing.assert_close(
        dataclasses.asdict(
            dataclasses.replace(batch, metrics=[], microbatches=microbatches)
        ),
        dataclasses.asdict(dataclasses.replace(expected_batch, metrics=[])),
        rtol=0,
        atol=0,
    )
