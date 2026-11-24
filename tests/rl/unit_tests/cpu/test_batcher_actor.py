# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""BatcherActor, in its own process, returns the same plans as an in-process Batcher."""

import asyncio
import dataclasses

import numpy as np
import pytest
import torch
from monarch.actor import ActorError, this_host
from torchtitan.rl.actors.monarch.batcher import BatcherActor
from torchtitan.rl.rollout.packed_group import pack_rollout_group
from torchtitan.rl.rollout.types import (
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
    RolloutTurnID,
)

from torchtitan.rl.trainer.batcher import Batcher

_BATCHER_KWARGS = dict(
    num_tokens_per_microbatch_per_dp_rank=256,
    max_context_length=16,
    num_prompts_per_train_step=2,
    dp_degree=2,
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
                    prompt_prefix_len=0,
                    prompt_delta_token_ids=np.array([1, 2, 3], dtype=np.int32),
                    completion_token_ids=np.array([4 + rollout_id, 5], dtype=np.int32),
                    completion_logprobs=np.array([-0.5, -0.25], dtype=np.float32),
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


def test_batcher_actor_returns_the_in_process_plan(tmp_path) -> None:
    # Plan A after group 2, plan B after group 4. Group 1 has zero reward spread, so the
    # builder drops it: it rides along but isn't trainable.
    summaries = [
        pack_rollout_group(group)[0]
        for group in (
            _group(0, [0.0, 1.0]),
            _group(1, [1.0, 1.0]),
            _group(2, [1.0, 0.0]),
            _group(3, [0.0, 1.0]),
            _group(4, [1.0, 0.0]),
        )
    ]

    batcher = Batcher.Config().build(**_BATCHER_KWARGS)
    expected = [batcher.add_group(summary=summary) for summary in summaries]

    async def run() -> list:
        mesh = this_host().spawn_procs(per_host={"cpus": 1})
        try:
            actor = mesh.spawn(
                "batcher",
                BatcherActor,
                Batcher.Config(),
                **_BATCHER_KWARGS,
                output_dir=str(tmp_path),
                enable_structured_logging=False,
            )
            results = [
                await actor.add_group.call_one(summary=summary) for summary in summaries
            ]
            # A builder error surfaces at call_one.
            no_advantage = _group(5, [0.0, 1.0])
            no_advantage.rollouts[0].advantage = None
            with pytest.raises(ActorError, match="has no advantage"):
                await actor.add_group.call_one(
                    summary=pack_rollout_group(no_advantage)[0]
                )
            return results
        finally:
            await mesh.stop()

    results = asyncio.run(run())

    assert [(batch is None, trainable) for batch, trainable in results] == [
        (True, True),
        (True, False),
        (False, True),
        (True, True),
        (False, True),
    ]
    assert [results[2][0].group_ids, results[4][0].group_ids] == [[0, 1, 2], [3, 4]]

    def comparable(batch) -> dict:
        # assert_close can't compare metric objects; every other field must match exactly.
        return dataclasses.asdict(dataclasses.replace(batch, metrics=[]))

    for (batch, _), (expected_batch, _) in zip(results, expected, strict=True):
        if expected_batch is not None:
            torch.testing.assert_close(
                comparable(batch), comparable(expected_batch), rtol=0, atol=0
            )
