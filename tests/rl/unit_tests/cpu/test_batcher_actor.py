# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""BatcherActor, in its own process, returns the same batches as an in-process Batcher."""

import asyncio
import dataclasses

import numpy as np
import pytest
import torch
from monarch.actor import ActorError, this_host
from torchtitan.rl.actors.monarch.batcher import BatcherActor
from torchtitan.rl.actors.monarch.rdma_microbatch import _TensorRef
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


def test_batcher_actor_returns_the_in_process_batch(tmp_path, monkeypatch) -> None:
    # TCP, not ibverbs: works on any CI host. The batcher process inherits the env var.
    monkeypatch.setenv("MONARCH_RDMA_DISABLE_IBVERBS", "1")
    # Batch A after group 2, batch B after group 4. Group 1 has zero reward spread, so the
    # builder drops it: it rides along but isn't trainable.
    groups = [
        _group(0, [0.0, 1.0]),
        _group(1, [1.0, 1.0]),
        _group(2, [1.0, 0.0]),
        _group(3, [0.0, 1.0]),
        _group(4, [1.0, 0.0]),
    ]

    batcher = Batcher.Config().build(**_BATCHER_KWARGS)
    expected = [batcher.add_group(rollout_group=group) for group in groups]

    async def run() -> tuple[list, list, list]:
        mesh = this_host().spawn_procs(per_host={"cpus": 1})
        try:
            spawn_kwargs = dict(
                **_BATCHER_KWARGS,
                output_dir=str(tmp_path),
                enable_structured_logging=False,
            )
            actor = mesh.spawn(
                "batcher",
                BatcherActor,
                Batcher.Config(rdma_min_tensor_bytes=1024),
                **spawn_kwargs,
            )
            by_value = mesh.spawn(
                "batcher_by_value", BatcherActor, Batcher.Config(), **spawn_kwargs
            )
            results = [
                await actor.add_group.call_one(rollout_group=group) for group in groups
            ]
            by_value_results = [
                await by_value.add_group.call_one(rollout_group=group)
                for group in groups
            ]
            # Release A once: B, still held, must read back exactly, as a trainer rank reads it.
            await actor.maybe_release_oldest_batch.call_one()
            microbatch_groups_b = [
                [
                    [await rdma_microbatch.read() for rdma_microbatch in row]
                    for row in group
                ]
                for group in results[4][0].microbatch_groups
            ]
            # A builder error surfaces at call_one.
            no_advantage = _group(5, [0.0, 1.0])
            no_advantage.rollouts[0].advantage = None
            with pytest.raises(ActorError, match="has no advantage"):
                await actor.add_group.call_one(rollout_group=no_advantage)
            return results, by_value_results, microbatch_groups_b
        finally:
            await mesh.stop()

    results, by_value_results, microbatch_groups_b = asyncio.run(run())

    assert [(batch is None, trainable) for batch, trainable in results] == [
        (True, True),
        (True, False),
        (False, True),
        (True, True),
        (False, True),
    ]
    batch_b = results[4][0]
    assert [results[2][0].group_ids, batch_b.group_ids] == [[0, 1, 2], [3, 4]]
    # int64 fields (2 KiB) cross processes by ref; the token counts go by value.
    assert isinstance(batch_b.microbatch_groups[0][0][0].fields["input"], _TensorRef)
    assert isinstance(
        batch_b.microbatch_groups[0][0][0].fields["loss_token_counts"], torch.Tensor
    )

    def comparable(batch) -> dict:
        # assert_close can't compare metric objects; every other field must match exactly.
        return dataclasses.asdict(dataclasses.replace(batch, metrics=[]))

    torch.testing.assert_close(
        comparable(dataclasses.replace(batch_b, microbatch_groups=microbatch_groups_b)),
        comparable(expected[4][0]),
        rtol=0,
        atol=0,
    )
    # Without the threshold, the batches come back by value and match the in-process ones.
    assert [batch is None for batch, _ in by_value_results] == [
        batch is None for batch, _ in expected
    ]
    for (batch, _), (expected_batch, _) in zip(by_value_results, expected, strict=True):
        if expected_batch is not None:
            torch.testing.assert_close(
                comparable(batch), comparable(expected_batch), rtol=0, atol=0
            )
