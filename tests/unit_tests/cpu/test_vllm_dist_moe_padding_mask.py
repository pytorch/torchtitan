# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The padding-mask path from vLLM's runner to Dist-MoE's experts, without a GPU.

Runner -> wrapper device scalar -> padding mask passed to the model's forward ->
MoE.forward -> routed experts.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

pytest.importorskip(
    "dist_moe",
    reason="Dist-MoE integration tests require the optional dist_moe package",
)
pytest.importorskip("vllm", reason="the wrapper and runner need vLLM")
from torchtitan.models.common.moe import MoE
from torchtitan.models.kimi_k3.moe import KimiLatentMoE
from torchtitan.rl.model.vllm_worker import TorchTitanGPUModelRunner
from torchtitan.rl.model.vllm_wrapper import VLLMModelWrapper
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class _RecordingModel:
    """Records the kwargs the wrapper passes to the model's forward."""

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def __call__(self, input_ids, attention_metadata=None, positions=None, **kwargs):
        self.seen.append(kwargs)
        return input_ids.float()[:, None].expand(-1, 4).contiguous()


@pytest.fixture(autouse=True)
def _forward_context_without_dp():
    with patch(
        "torchtitan.rl.model.vllm_wrapper.get_forward_context",
        return_value=SimpleNamespace(dp_metadata=None),
    ):
        yield


class _RecordingRuntime:
    """Records the per-call Dist-MoE row counts the wrapper sets."""

    def __init__(self) -> None:
        self.num_tokens_per_call: list[int] = []

    def set_num_local_input_tokens_per_call(self, num_tokens: int) -> None:
        self.num_tokens_per_call.append(num_tokens)


def _wrapper(
    *, num_valid_tokens: torch.Tensor | None, tp: int = 1, dp: int = 1
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
    wrapper._num_valid_tokens = num_valid_tokens
    wrapper._dist_moe_runtime = (
        _RecordingRuntime() if num_valid_tokens is not None else None
    )
    return wrapper


def test_forward_passes_a_padding_mask_built_from_the_published_count():
    wrapper = _wrapper(num_valid_tokens=torch.tensor(3, dtype=torch.int32))
    wrapper.forward(input_ids=torch.arange(5))
    (kwargs,) = wrapper.model.seen
    assert kwargs["padding_mask"].dtype is torch.bool
    assert kwargs["padding_mask"].tolist() == [False, False, False, True, True]


def test_forward_before_any_publish_marks_nothing_as_padding():
    sentinel = torch.full((), torch.iinfo(torch.int32).max, dtype=torch.int32)
    wrapper = _wrapper(num_valid_tokens=sentinel)
    wrapper.forward(input_ids=torch.arange(4))
    (kwargs,) = wrapper.model.seen
    assert not kwargs["padding_mask"].any()


def test_forward_follows_the_scalar_in_place():
    # CUDA-graph replay reads the same memory, so an in-place write must change the mask.
    buffer = torch.full((), 4, dtype=torch.int32)
    wrapper = _wrapper(num_valid_tokens=buffer)
    wrapper.forward(input_ids=torch.arange(4))
    wrapper.set_num_valid_tokens(2)
    wrapper.forward(input_ids=torch.arange(4))
    first, second = wrapper.model.seen
    assert not first["padding_mask"].any()
    assert second["padding_mask"].tolist() == [False, False, True, True]
    assert wrapper._num_valid_tokens is buffer


def test_forward_sets_the_step_dist_moe_call_size():
    wrapper = _wrapper(num_valid_tokens=torch.tensor(3, dtype=torch.int32), tp=4)
    wrapper.forward(input_ids=torch.arange(8))
    assert wrapper._dist_moe_runtime.num_tokens_per_call == [2]


@pytest.mark.parametrize(
    "dp,num_tokens_across_dp,expected",
    [
        # Eager step: DP replicas keep their own sizes; all use the largest.
        (4, [8, 64, 4, 16], 16),
        # Graph step: every replica already has the same size.
        (4, [16, 16, 16, 16], 4),
        # DP 1: no DP metadata; the EP group is the TP group, which shares one batch.
        (1, None, 3),
    ],
)
def test_dist_moe_call_size_uses_the_largest_dp_replica(
    dp, num_tokens_across_dp, expected
):
    wrapper = _wrapper(num_valid_tokens=torch.tensor(0, dtype=torch.int32), tp=4, dp=dp)
    dp_metadata = (
        None
        if num_tokens_across_dp is None
        else SimpleNamespace(
            num_tokens_across_dp_cpu=torch.tensor(num_tokens_across_dp)
        )
    )
    with patch(
        "torchtitan.rl.model.vllm_wrapper.get_forward_context",
        return_value=SimpleNamespace(dp_metadata=dp_metadata),
    ):
        assert wrapper._dist_moe_num_local_input_tokens(12) == expected


def test_forward_without_dist_moe_passes_no_padding_mask():
    wrapper = _wrapper(num_valid_tokens=None)
    wrapper.forward(input_ids=torch.arange(4))
    (kwargs,) = wrapper.model.seen
    assert kwargs == {}
    wrapper.set_num_valid_tokens(2)  # a no-op, not an error


class _Recorder:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def __call__(self, x, scores, ids, counts, **kwargs):
        self.kwargs = kwargs
        return x


def _moe_forward(moe_cls: type[MoE], experts: _Recorder, padding_mask_T: torch.Tensor):
    routed_mask = padding_mask_T.clone()  # stands for the TP-sharded mask
    self = SimpleNamespace(
        _maybe_shard_routed_branch_inputs_across_tp=lambda x, mask: (x, routed_mask),
        router=lambda x, bias, padding_mask_T=None, **kw: (
            torch.zeros(x.shape[0], 2),
            torch.zeros(x.shape[0], 2, dtype=torch.int64),
            torch.zeros(x.shape[0], 4, dtype=torch.bool),
        ),
        expert_bias_E=None,
        routed_experts=experts,
        # KimiLatentMoE projects into and out of the routed experts' latent space.
        routed_down=lambda x: x,
        routed_norm=lambda x: x,
        routed_up=lambda x: x,
        _maybe_zero_fill_routed_output_to_tp_partial=lambda out: out,
        shared_experts=None,
        _maybe_all_reduce_moe_output_across_tp=lambda out: out,
    )
    moe_cls.forward(cast(Any, self), torch.zeros(4, 3), padding_mask_T=padding_mask_T)
    return routed_mask


@pytest.mark.parametrize("moe_cls", [MoE, KimiLatentMoE])
def test_moe_hands_the_routed_mask_to_the_experts(moe_cls):
    experts = _Recorder()
    mask = torch.tensor([False, False, True, True])
    routed = _moe_forward(moe_cls, experts, mask)
    assert experts.kwargs is not None and set(experts.kwargs) == {"padding_mask_T"}
    assert experts.kwargs["padding_mask_T"] is routed


class _Model:
    def __init__(self) -> None:
        self.published: list[int] = []

    def set_num_valid_tokens(self, n: int) -> None:
        self.published.append(n)


def _runner(model) -> TorchTitanGPUModelRunner:
    runner = cast(Any, object.__new__(TorchTitanGPUModelRunner))
    runner.get_model = lambda: model  # type: ignore[method-assign]
    return runner


def test_runner_publishes_the_unpadded_count_before_vllm_pads():
    model = _Model()
    sentinel = (object(), object())
    with patch.object(
        GPUModelRunner, "_determine_batch_execution_and_padding", return_value=sentinel
    ) as parent:
        out = _runner(model)._determine_batch_execution_and_padding(
            31, 3, None, 1, False, force_eager=True
        )
    assert out is sentinel
    assert model.published == [31]
    parent.assert_called_once()
    assert parent.call_args.args[0] == 31
    assert parent.call_args.kwargs == {"force_eager": True}
