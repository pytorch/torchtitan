# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.context_parallel import (
    get_token_fragments,
    HeadTailCPLoadBalancer,
    shard_per_token_tensors,
)
from torchtitan.training_engine import TrainingEngine


def _rl_loss_kwargs(num_tokens: int) -> dict:
    tokens = torch.arange(num_tokens)
    return {
        "generator_logprobs": -tokens.float(),
        "temperature": torch.ones(num_tokens),
        "loss_mask": tokens % 3 != 0,
        "advantages": tokens.float() / 10,
    }


def test_per_token_shards_match_the_unpermuted_cp_token_ranges() -> None:
    num_tokens, cp_size = 12, 4
    loss_kwargs = _rl_loss_kwargs(num_tokens)
    fragments = get_token_fragments(num_tokens, cp_size=cp_size, permutation=None)
    for cp_rank in range(cp_size):
        sharded = shard_per_token_tensors(loss_kwargs, cp_rank=cp_rank, cp_size=cp_size)
        ((start, stop),) = fragments[cp_rank]
        for name, value in loss_kwargs.items():
            torch.testing.assert_close(sharded[name], value[start:stop])


def test_concatenated_shards_reconstruct_every_tensor() -> None:
    loss_kwargs = _rl_loss_kwargs(16)
    shards = [
        shard_per_token_tensors(loss_kwargs, cp_rank=r, cp_size=2) for r in range(2)
    ]
    for name, value in loss_kwargs.items():
        torch.testing.assert_close(torch.cat([s[name] for s in shards]), value)


def test_masked_token_sum_over_shards_equals_the_unsharded_sum() -> None:
    """The DAPO/GRPO loss and its metrics are masked per-token sums divided by a
    global count, so summing per-rank shares reproduces the unsharded value."""
    loss_kwargs = _rl_loss_kwargs(24)
    full = (loss_kwargs["advantages"] * loss_kwargs["loss_mask"]).sum()
    shards = [
        shard_per_token_tensors(loss_kwargs, cp_rank=r, cp_size=4) for r in range(4)
    ]
    sharded = sum((s["advantages"] * s["loss_mask"]).sum() for s in shards)
    torch.testing.assert_close(sharded, full)


def test_non_tensor_values_pass_through() -> None:
    sharded = shard_per_token_tensors(
        {"advantages": torch.zeros(8), "global_scale": 2.0}, cp_rank=1, cp_size=2
    )
    assert sharded["global_scale"] == 2.0
    assert sharded["advantages"].shape == (4,)


def test_mismatched_or_indivisible_token_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="share one leading token dimension"):
        shard_per_token_tensors(
            {"a": torch.zeros(8), "b": torch.zeros(6)}, cp_rank=0, cp_size=2
        )
    with pytest.raises(ValueError, match="divisible by the CP degree"):
        shard_per_token_tensors({"a": torch.zeros(6)}, cp_rank=0, cp_size=4)


def test_engine_refuses_a_cp_load_balancer_with_per_token_loss_arguments() -> None:
    engine = SimpleNamespace(
        config=SimpleNamespace(
            parallelism=ParallelismConfig(
                context_parallel_load_balancer=HeadTailCPLoadBalancer.Config()
            )
        )
    )
    with pytest.raises(ValueError, match="context_parallel_load_balancer=None"):
        TrainingEngine._shard_loss_kwargs_for_cp(engine, _rl_loss_kwargs(8))


def test_engine_shards_with_its_cp_mesh_coordinates() -> None:
    cp_mesh = SimpleNamespace(get_local_rank=lambda: 1, size=lambda: 2)
    engine = SimpleNamespace(
        config=SimpleNamespace(parallelism=ParallelismConfig()),
        parallelism_context=SimpleNamespace(get_mesh=lambda axis: cp_mesh),
    )
    loss_kwargs = _rl_loss_kwargs(8)
    sharded = TrainingEngine._shard_loss_kwargs_for_cp(engine, loss_kwargs)
    torch.testing.assert_close(sharded["advantages"], loss_kwargs["advantages"][4:])
