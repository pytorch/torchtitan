# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


pytest.importorskip("torchao")
pytest.importorskip("torchao.prototype.moe_training.nvfp4_training")

from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (  # noqa: E402
    NVFP4Linear as TorchAONVFP4Linear,
)

from torchtitan.quantization._fsdp_tensor import _UnshardedFSDPTensor  # noqa: E402
from torchtitan.quantization.nvfp4 import (  # noqa: E402
    _HARDCODED_SIGN_VECTOR,
    NVFP4Linear,
)
from torchtitan.quantization.nvfp4.tensor import (  # noqa: E402
    _LinearShardedTensorWithNVFP4Compute,
)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0),
        reason="NVFP4 requires SM100 or later",
    ),
]


def _make_nvfp4_linear(*, num_linears: int = 1, bias: bool = True) -> NVFP4Linear:
    linear = (
        NVFP4Linear.Config(
            in_features=128,
            out_features=128,
            num_linears=num_linears,
            bias=bias,
        )
        .build()
        .cuda()
        .bfloat16()
    )
    linear._init_self_buffers(buffer_device=torch.device("cuda"))
    return linear


def _install_unsharded_weight(linear: NVFP4Linear) -> NVFP4Linear:
    sharded_weight = linear.weight
    with torch.no_grad():
        unsharded_weight = _UnshardedFSDPTensor(
            sharded_weight._tensor,
            sharded_weight._build_operands(sharded_weight._tensor),
        )
    linear.weight = nn.Parameter(
        unsharded_weight,
        requires_grad=sharded_weight.requires_grad,
    )
    return linear


@pytest.mark.parametrize("input_shape", [(128, 128), (2, 128, 128)])
def test_nvfp4_linear_matches_torchao(input_shape):
    torch.manual_seed(1)
    linear = _make_nvfp4_linear()
    torchao_linear = TorchAONVFP4Linear(
        128,
        128,
        bias=True,
        device="cuda",
        dtype=torch.bfloat16,
        rht_sign_vector=_HARDCODED_SIGN_VECTOR,
    )
    with torch.no_grad():
        torchao_linear.weight.copy_(linear.weight._tensor)
        torchao_linear.bias.copy_(linear.bias)
        torchao_linear._sr_seed.copy_(linear._sr_seed)

    input_tt = torch.randn(
        *input_shape,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    input_ao = input_tt.detach().clone().requires_grad_()
    grad_output = torch.randn(
        *input_shape[:-1],
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    output_tt = linear(input_tt)
    output_ao = torchao_linear(input_ao)

    rng_state = torch.cuda.get_rng_state()
    output_tt.backward(grad_output)
    torch.cuda.set_rng_state(rng_state)
    output_ao.backward(grad_output)

    torch.testing.assert_close(output_tt, output_ao, rtol=0, atol=0)
    torch.testing.assert_close(input_tt.grad, input_ao.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        linear.weight.grad,
        torchao_linear.weight.grad,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        linear.bias.grad, torchao_linear.bias.grad, rtol=0, atol=0
    )
    assert output_tt._base is None


def test_nvfp4_linear_saves_fsdps_weight_holder():
    linear = _install_unsharded_weight(_make_nvfp4_linear(bias=False))
    input_MK = torch.randn(
        128,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    saved_tensors = []

    def pack_hook(tensor):
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
        linear(input_MK).sum().backward()

    weight_saves = [
        tensor for tensor in saved_tensors if isinstance(tensor, _UnshardedFSDPTensor)
    ]
    assert len(weight_saves) == 1
    assert all(
        tensor is weight_saves[0]
        or tensor.untyped_storage()._cdata
        not in {
            weight_saves[0].operands.weight_qdata_fprop.untyped_storage()._cdata,
            weight_saves[0].operands.weight_scale_fprop.untyped_storage()._cdata,
            weight_saves[0].operands.weight_qdata_dgrad.untyped_storage()._cdata,
            weight_saves[0].operands.weight_scale_dgrad.untyped_storage()._cdata,
            weight_saves[0].operands.weight_amax.untyped_storage()._cdata,
        }
        for tensor in saved_tensors
    )


def test_nvfp4_weight_operands_are_independent_and_refill_in_place():
    linear = _make_nvfp4_linear(bias=False)
    sharded_weight = linear.weight
    assert isinstance(sharded_weight, _LinearShardedTensorWithNVFP4Compute)
    operands = sharded_weight._build_operands(sharded_weight._tensor)
    fields = (
        operands.weight_qdata_fprop,
        operands.weight_scale_fprop,
        operands.weight_qdata_dgrad,
        operands.weight_scale_dgrad,
        operands.weight_amax,
    )
    assert len({tensor.untyped_storage()._cdata for tensor in fields}) == len(fields)
    tensor_ids = tuple(map(id, fields))
    refilled = sharded_weight._build_operands(sharded_weight._tensor, out=operands)
    refilled_fields = (
        refilled.weight_qdata_fprop,
        refilled.weight_scale_fprop,
        refilled.weight_qdata_dgrad,
        refilled.weight_scale_dgrad,
        refilled.weight_amax,
    )
    assert tuple(map(id, refilled_fields)) == tensor_ids


@pytest.mark.parametrize("execution_mode", ["compile", "activation_checkpoint"])
def test_nvfp4_linear_runs_outside_plain_eager(execution_mode):
    linear = _install_unsharded_weight(_make_nvfp4_linear(num_linears=2, bias=False))
    input_BMK = torch.randn(
        2,
        128,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    if execution_mode == "compile":
        output = torch.compile(linear, fullgraph=True)(input_BMK)
    else:
        output = checkpoint(linear, input_BMK, use_reentrant=False)
    output.sum().backward()

    assert output.shape == (2, 128, 2, 128)
    assert input_BMK.grad is not None
    assert linear.weight.grad is not None
