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

from torchao.float8 import Float8LinearConfig  # noqa: E402
from torchao.float8.float8_linear import (  # noqa: E402
    Float8Linear as TorchAOFloat8Linear,
)

from torchtitan.quantization._fsdp_tensor import _UnshardedFSDPTensor  # noqa: E402
from torchtitan.quantization.float8 import Float8Linear  # noqa: E402
from torchtitan.quantization.float8.tensor import (  # noqa: E402
    _LinearShardedTensorWithFloat8Compute,
    _LinearShardedTensorWithFloat8HighPrecisionWeightGradient,
)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 9),
        reason="Float8 requires SM89 or later",
    ),
]


def _make_float8_linear(
    recipe_name: str,
    *,
    num_linears: int = 1,
    bias: bool = True,
) -> Float8Linear:
    return (
        Float8Linear.Config(
            in_features=128,
            out_features=128,
            num_linears=num_linears,
            bias=bias,
            recipe_name=recipe_name,
        )
        .build()
        .cuda()
        .bfloat16()
    )


def _install_unsharded_weight(linear: Float8Linear) -> Float8Linear:
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


@pytest.mark.parametrize("recipe_name", ["rowwise", "rowwise_with_gw_hp"])
@pytest.mark.parametrize("input_shape", [(64, 128), (2, 64, 128)])
def test_float8_linear_matches_torchao(recipe_name, input_shape):
    torch.manual_seed(1)
    linear = _make_float8_linear(recipe_name)
    torchao_linear = TorchAOFloat8Linear(
        128,
        128,
        bias=True,
        config=Float8LinearConfig.from_recipe_name(recipe_name),
        device="cuda",
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        torchao_linear.weight.copy_(linear.weight._tensor)
        torchao_linear.bias.copy_(linear.bias)

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
    output_tt.backward(grad_output)
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


@pytest.mark.parametrize(
    ("recipe_name", "wrapper_cls", "expected_dgrad_scale_shape"),
    [
        ("rowwise", _LinearShardedTensorWithFloat8Compute, (1, 128)),
        (
            "rowwise_with_gw_hp",
            _LinearShardedTensorWithFloat8HighPrecisionWeightGradient,
            (),
        ),
    ],
)
def test_float8_weight_operands_follow_recipe(
    recipe_name,
    wrapper_cls,
    expected_dgrad_scale_shape,
):
    linear = _make_float8_linear(recipe_name, bias=False)
    assert isinstance(linear.weight, wrapper_cls)
    operands = linear.weight._build_operands(linear.weight._tensor)

    assert operands.weight_qdata_fprop_NK.shape == (128, 128)
    assert operands.weight_scale_fprop_N1.shape == (128, 1)
    assert operands.weight_qdata_dgrad_NK.shape == (128, 128)
    assert operands.weight_scale_dgrad.shape == expected_dgrad_scale_shape
    assert (
        operands.weight_qdata_fprop_NK.untyped_storage()._cdata
        != operands.weight_qdata_dgrad_NK.untyped_storage()._cdata
    )


def test_float8_linear_saves_fsdps_weight_holder():
    linear = _install_unsharded_weight(_make_float8_linear("rowwise", bias=False))
    input_MK = torch.randn(
        64,
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
    activation_saves = [
        tensor
        for tensor in saved_tensors
        if not isinstance(tensor, _UnshardedFSDPTensor)
    ]
    assert len(weight_saves) == 1
    assert activation_saves == [input_MK]


@pytest.mark.parametrize("recipe_name", ["rowwise", "rowwise_with_gw_hp"])
@pytest.mark.parametrize("execution_mode", ["compile", "activation_checkpoint"])
def test_float8_linear_runs_outside_plain_eager(recipe_name, execution_mode):
    linear = _install_unsharded_weight(
        _make_float8_linear(recipe_name, num_linears=2, bias=False)
    )
    input_BMK = torch.randn(
        2,
        64,
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

    assert output.shape == (2, 64, 2, 128)
    assert input_BMK.grad is not None
    assert linear.weight.grad is not None
