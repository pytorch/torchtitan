# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import spmd_types as spmd
import torch
from spmd_types.checker import typecheck
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.utils.checkpoint import checkpoint

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.common.activation import ClampedSwiGLU
from torchtitan.models.gpt_oss.moe import add_grouped_bias, GptOssGroupedLinear


@pytest.fixture
def reset_local_compile():
    LocalCompileConfig().apply()
    torch._dynamo.reset()
    yield
    LocalCompileConfig().apply()


def _reference_swiglu(gate, up, limit: float = 7.0):
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    silu = gate * torch.sigmoid(1.702 * gate)
    return torch.addcmul(silu, silu, up)


def test_swiglu_matches_reference_formula_and_gradients():
    """The configured GPT-OSS activation matches its forward and backward formula."""
    torch.manual_seed(0)
    gate = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    up = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    gate_ref = gate.detach().clone().requires_grad_()
    up_ref = up.detach().clone().requires_grad_()

    out = ClampedSwiGLU.Config(limit=2.0).build()(torch.stack([gate, up], dim=-2))
    out_ref = _reference_swiglu(gate_ref, up_ref, limit=2.0)

    assert torch.equal(out, out_ref)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)

    assert torch.equal(gate.grad, gate_ref.grad)
    assert torch.equal(up.grad, up_ref.grad)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alpha", 0.0),
        ("alpha", float("inf")),
        ("limit", 0.0),
        ("limit", float("nan")),
    ],
)
def test_clamped_swiglu_rejects_invalid_policy(field: str, value: float) -> None:
    """Invalid kernel policy is rejected by the source activation config."""
    with pytest.raises(ValueError, match=f"{field} must be finite and positive"):
        ClampedSwiGLU.Config(**{field: value})


def _repeat_interleave_bias(output_RO, bias_EO, offsets_E, *, scores_R=None):
    """Reference: expand each expert's bias over its rows; padding rows get zeros."""
    counts_E = torch.diff(torch.cat((offsets_E.new_zeros(1), offsets_E)))
    tail = output_RO.shape[0] - offsets_E[-1]
    bias_RO = torch.cat(
        (
            bias_EO.repeat_interleave(counts_E.long(), dim=0),
            bias_EO.new_zeros(tail, bias_EO.shape[-1]),
        )
    )
    if scores_R is not None:
        bias_RO = bias_RO * scores_R.unsqueeze(-1)
    return output_RO + bias_RO


@pytest.mark.parametrize("with_scores", [False, True])
@pytest.mark.parametrize("regions", [[], ["grouped_expert_bias"]])
def test_grouped_bias_matches_repeat_interleave(
    reset_local_compile, regions, with_scores
) -> None:
    torch.manual_seed(0)
    LocalCompileConfig(regions=regions).apply()
    # 3 experts own rows 0-1, (none), 2-4; rows 5-6 are padding.
    offsets_E = torch.tensor([2, 2, 5], dtype=torch.int32)
    tensors = [torch.randn(7, 6), torch.randn(3, 6), torch.rand(7), torch.randn(7, 6)]
    output_RO, bias_EO, scores_R, grad_RO = (t.double() for t in tensors)
    results = []
    for fn in (_repeat_interleave_bias, add_grouped_bias):
        inputs = [
            t.clone().requires_grad_(True) for t in (output_RO, bias_EO, scores_R)
        ]
        out_RO = fn(*inputs[:2], offsets_E, scores_R=inputs[2] if with_scores else None)
        out_RO.backward(grad_RO)
        results.append((out_RO.detach(), *(t.grad for t in inputs)))
    for reference, actual in zip(*results, strict=True):
        torch.testing.assert_close(actual, reference)


def test_grouped_bias_region_in_pipeline_order(reset_local_compile) -> None:
    """PP runs microbatch B's forward before A's backward, with FSDP2's fp32 bias grad.

    Example: forward A (64 rows), forward B (32 rows), backward A replays the region.
    """
    LocalCompileConfig(regions=["grouped_expert_bias"]).apply()
    layer = GptOssGroupedLinear.Config(
        group_size=2, in_features=8, out_features=16, num_linears=2
    ).build()
    for param in layer.parameters():
        param.data = torch.randn_like(param, dtype=torch.bfloat16)
    layer.bias.grad_dtype = torch.float32
    layer.bias.grad = torch.zeros(layer.bias.shape, dtype=torch.float32)

    def forward(rows):
        input_RI = torch.randn(rows, 8, dtype=torch.bfloat16, requires_grad=True)
        offsets_E = torch.tensor([rows // 3, rows - 1], dtype=torch.int32)
        return checkpoint(
            lambda x: layer(x, offsets_E),
            input_RI,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    output_a, output_b = forward(64), forward(32)
    output_a.sum().backward()
    output_b.sum().backward()
    assert layer.bias.grad.dtype == torch.float32


class TestGptOssGroupedLinearSPMD(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @property
    def device_type(self):
        return "cpu"

    @with_comms
    def test_grouped_bias_propagates_spmd_types(self):
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        layer = GptOssGroupedLinear.Config(
            group_size=3, in_features=8, out_features=4
        ).build()
        input_RI = torch.randn(7, 8, dtype=torch.bfloat16)
        offsets_E = torch.tensor([2, 2, 5], dtype=torch.int32)

        with set_current_spmd_mesh(mesh), typecheck(strict_mode="strict", local=True):
            spmd.assert_type(input_RI, {"tp": spmd.V})
            spmd.assert_type(layer.weight, {"tp": spmd.R})
            spmd.assert_type(layer.bias, {"tp": spmd.R})
            spmd.assert_type(offsets_E, {"tp": spmd.V})
            output_RO = layer(input_RI, offsets_E)
            spmd.assert_type(output_RO, {"tp": spmd.V})
