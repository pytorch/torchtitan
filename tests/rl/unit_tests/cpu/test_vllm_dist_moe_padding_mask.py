# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The padding-mask path from vLLM's forward context into the model, without a GPU.

vLLM's padding buffer -> forward context ``is_padding`` -> padding mask passed to the
model's forward. MoE.forward handing the mask to the routed experts is tested in
test_moe.py.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts, DistMoeRuntime
from torchtitan.rl.generator.vllm.model_wrapper import VLLMModelWrapper


class _RecordingModel:
    """Records the kwargs the wrapper passes to the model's forward."""

    def __init__(self) -> None:
        self.seen: list[dict] = []
        self.num_rows: list[int] = []

    def __call__(self, input_ids, attention_metadata=None, positions=None, **kwargs):
        self.seen.append(kwargs)
        self.num_rows.append(input_ids.shape[0])
        return input_ids.float()[:, None].expand(-1, 4).contiguous()


@pytest.fixture(autouse=True)
def forward_context():
    context = SimpleNamespace(dp_metadata=None, is_padding=None)
    with patch(
        "torchtitan.rl.generator.vllm.model_wrapper.get_forward_context",
        return_value=context,
    ):
        yield context


class _RecordingRuntime:
    """Records the per-call Dist-MoE row counts the wrapper sets."""

    def __init__(self) -> None:
        self.num_tokens_per_call: list[int] = []

    def set_num_local_input_tokens_per_call(self, num_tokens: int) -> None:
        self.num_tokens_per_call.append(num_tokens)


def _wrapper(
    *, dist_moe: bool, tp: int = 1, dp: int = 1, expert_sequence_parallel_size: int = 1
) -> VLLMModelWrapper:
    wrapper = cast(Any, object.__new__(VLLMModelWrapper))
    torch.nn.Module.__init__(wrapper)
    wrapper.model = _RecordingModel()

    @contextmanager
    def activate_spmd():
        yield

    wrapper.parallelism_context = SimpleNamespace(
        activate_spmd=activate_spmd, tp=tp, dp_shard=dp
    )
    wrapper._expert_sequence_parallel_size = expert_sequence_parallel_size
    wrapper._dist_moe_runtime = _RecordingRuntime() if dist_moe else None
    return wrapper


def test_forward_passes_vllm_padding_mask(forward_context):
    # CUDA graphs capture vLLM's buffer, so the wrapper must pass it, not a copy.
    forward_context.is_padding = torch.tensor([False, False, False, True, True])
    wrapper = _wrapper(dist_moe=True)
    wrapper.forward(input_ids=torch.arange(5))
    (kwargs,) = wrapper.model.seen
    assert kwargs["padding_mask"] is forward_context.is_padding


def test_forward_pads_the_step_to_a_multiple_of_tp(forward_context):
    # EP over TP 4: 3 real tokens run as 4 rows; the extra row is padding, dropped after.
    forward_context.is_padding = torch.zeros(3, dtype=torch.bool)
    wrapper = _wrapper(dist_moe=True, tp=4, expert_sequence_parallel_size=4)
    hidden_states = wrapper.forward(
        input_ids=torch.arange(3), positions=torch.arange(3)
    )
    (kwargs,) = wrapper.model.seen
    assert wrapper.model.num_rows == [4]
    assert kwargs["padding_mask"].tolist() == [False, False, False, True]
    assert wrapper._dist_moe_runtime.num_tokens_per_call == [1]
    assert hidden_states.shape[0] == 3


def test_forward_sets_the_step_dist_moe_call_size(forward_context):
    forward_context.is_padding = torch.zeros(8, dtype=torch.bool)
    wrapper = _wrapper(dist_moe=True, tp=4)
    wrapper.forward(input_ids=torch.arange(8))
    assert wrapper._dist_moe_runtime.num_tokens_per_call == [2]


@pytest.mark.parametrize(
    "dp,num_tokens_across_dp,expected",
    [
        # Eager step: DP replicas keep their own sizes; all use the largest.
        (4, [8, 64, 4, 16], 16),
        # Graph step: every replica already has the same size.
        (4, [16, 16, 16, 16], 4),
        # Eager step that TP does not divide: the wrapper pads 5 rows to 8.
        (2, [5, 3], 2),
        # DP 1: no DP metadata; the EP group is the TP group, which shares one batch.
        (1, None, 3),
    ],
)
def test_dist_moe_call_size_uses_the_largest_dp_replica(
    dp, num_tokens_across_dp, expected
):
    wrapper = _wrapper(dist_moe=True, tp=4, dp=dp)
    dp_metadata = (
        None
        if num_tokens_across_dp is None
        else SimpleNamespace(
            num_tokens_across_dp_cpu=torch.tensor(num_tokens_across_dp)
        )
    )
    with patch(
        "torchtitan.rl.generator.vllm.model_wrapper.get_forward_context",
        return_value=SimpleNamespace(dp_metadata=dp_metadata),
    ):
        assert wrapper._dist_moe_num_local_input_tokens(12) == expected


def test_forward_without_dist_moe_passes_no_padding_mask(forward_context):
    forward_context.is_padding = torch.zeros(4, dtype=torch.bool)
    wrapper = _wrapper(dist_moe=False)
    wrapper.forward(input_ids=torch.arange(4))
    (kwargs,) = wrapper.model.seen
    assert kwargs == {}


def test_dist_moe_needs_ep_when_tp_splits_tokens():
    # Dist-MoE splits each step over TP, but forward pads to a multiple of TP only with EP.
    wrapper = cast(Any, object.__new__(VLLMModelWrapper))
    torch.nn.Module.__init__(wrapper)
    experts = cast(Any, object.__new__(DistMoeRoutedExperts))
    torch.nn.Module.__init__(experts)
    wrapper.model = torch.nn.Sequential(experts)
    wrapper.parallelism_context = SimpleNamespace(tp=2, ep=1)
    wrapper._expert_sequence_parallel_size = 1
    with pytest.raises(
        ValueError, match="tensor_parallel_degree=2, expert_parallel_degree=1"
    ):
        wrapper._initialize_dist_moe_runtime(
            DistMoeRuntime.Config(inference=True), 16, torch.device("cpu")
        )
