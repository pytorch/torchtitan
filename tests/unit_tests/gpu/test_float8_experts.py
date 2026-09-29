# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


pytest.importorskip("torchao")
pytest.importorskip("torchao.prototype.moe_training.kernels")

from torchao.prototype.moe_training.config import Float8TrainingOpConfig  # noqa: E402
from torchao.prototype.moe_training.utils import (  # noqa: E402
    _quantize_then_scaled_grouped_mm,
)

from torchtitan.models.common.linear import GroupedLinear  # noqa: E402
from torchtitan.models.gpt_oss.moe import GptOssGroupedLinear  # noqa: E402
from torchtitan.quantization._fsdp_tensor import _UnshardedFSDPTensor  # noqa: E402
from torchtitan.quantization.float8 import _get_float8_grouped_linear_cls  # noqa: E402
from torchtitan.quantization.float8.experts import (  # noqa: E402
    _Float8GroupedMMFunction,
)
from torchtitan.quantization.float8.tensor import (  # noqa: E402
    _GroupedLinearShardedTensorWithFloat8Compute,
    _quantize_float8_grouped_weight,
)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 9),
        reason="Float8 requires SM89 or later",
    ),
]


def _make_float8_grouped_linear():
    float8_cls = _get_float8_grouped_linear_cls(GroupedLinear)
    grouped_linear = (
        float8_cls.Config(
            group_size=4,
            in_features=128,
            out_features=128,
            num_linears=2,
        )
        .build()
        .cuda()
        .bfloat16()
    )
    with torch.no_grad():
        for parameter in grouped_linear.parameters():
            parameter._tensor.normal_(std=0.02)
    return grouped_linear


def _install_unsharded_weights(experts):
    for name, sharded_weight in tuple(experts.named_parameters(recurse=False)):
        with torch.no_grad():
            unsharded_weight = _UnshardedFSDPTensor(
                sharded_weight._tensor,
                sharded_weight._build_operands(sharded_weight._tensor),
            )
        setattr(
            experts,
            name,
            nn.Parameter(
                unsharded_weight,
                requires_grad=sharded_weight.requires_grad,
            ),
        )
    return experts


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_float8_grouped_mm_matches_torchao(weight_dtype):
    torch.manual_seed(1)
    input_tt_RI = torch.randn(
        64,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    input_ao_RI = input_tt_RI.detach().clone().requires_grad_()
    weight_tt_EOI = torch.randn(
        4,
        128,
        128,
        device="cuda",
        dtype=weight_dtype,
        requires_grad=True,
    )
    weight_ao_EOI = weight_tt_EOI.detach().clone().requires_grad_()
    offsets_E = torch.arange(16, 65, 16, device="cuda", dtype=torch.int32)
    with torch.no_grad():
        operands = _quantize_float8_grouped_weight(weight_tt_EOI)

    output_tt_RO = _Float8GroupedMMFunction.apply(
        input_tt_RI,
        weight_tt_EOI,
        operands.weight_qdata_fprop_EIO,
        operands.weight_scale_fprop_E1O,
        operands.weight_qdata_dgrad_EOI,
        operands.weight_scale_dgrad_EI,
        offsets_E,
    )
    output_ao_RO = _quantize_then_scaled_grouped_mm(
        input_ao_RI,
        weight_ao_EOI.bfloat16().transpose(-2, -1),
        config=Float8TrainingOpConfig(),
        offs=offsets_E,
    )
    grad_output_RO = torch.randn_like(output_tt_RO)
    output_tt_RO.backward(grad_output_RO)
    output_ao_RO.backward(grad_output_RO)

    torch.testing.assert_close(output_tt_RO, output_ao_RO, rtol=0, atol=0)
    torch.testing.assert_close(input_tt_RI.grad, input_ao_RI.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        weight_tt_EOI.grad,
        weight_ao_EOI.grad,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("parent_cls", [GroupedLinear, GptOssGroupedLinear])
def test_float8_grouped_linear_wraps_weight(parent_cls):
    """Float8 wraps only the grouped weight, leaving model-specific bias plain."""
    float8_cls = _get_float8_grouped_linear_cls(parent_cls)
    grouped_linear = float8_cls.Config(
        group_size=4,
        in_features=128,
        out_features=128,
        num_linears=2,
    ).build()
    assert isinstance(
        grouped_linear.weight,
        _GroupedLinearShardedTensorWithFloat8Compute,
    )
    if isinstance(grouped_linear, GptOssGroupedLinear):
        assert type(grouped_linear.bias) is nn.Parameter


def test_float8_grouped_weight_operands_refill_in_place():
    weight_EOI = torch.randn(
        4,
        128,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    sharded_weight = _GroupedLinearShardedTensorWithFloat8Compute(weight_EOI)
    operands = sharded_weight._build_operands(weight_EOI)
    fields = (
        operands.weight_qdata_fprop_EIO,
        operands.weight_scale_fprop_E1O,
        operands.weight_qdata_dgrad_EOI,
        operands.weight_scale_dgrad_EI,
    )
    assert len({tensor.untyped_storage()._cdata for tensor in fields}) == len(fields)
    tensor_ids = tuple(map(id, fields))

    refilled = sharded_weight._build_operands(weight_EOI, out=operands)
    refilled_fields = (
        refilled.weight_qdata_fprop_EIO,
        refilled.weight_scale_fprop_E1O,
        refilled.weight_qdata_dgrad_EOI,
        refilled.weight_scale_dgrad_EI,
    )
    assert tuple(map(id, refilled_fields)) == tensor_ids


def test_float8_grouped_mm_saves_fsdps_weight_holder():
    weight_EOI = torch.randn(
        4,
        128,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    sharded_weight = _GroupedLinearShardedTensorWithFloat8Compute(weight_EOI)
    with torch.no_grad():
        unsharded_weight = _UnshardedFSDPTensor(
            weight_EOI,
            sharded_weight._build_operands(weight_EOI),
        )
    weight_EOI = torch.nn.Parameter(unsharded_weight)
    input_RI = torch.randn(
        64,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    offsets_E = torch.arange(16, 65, 16, device="cuda", dtype=torch.int32)
    saved_tensors = []

    def pack_hook(tensor):
        saved_tensors.append(tensor)
        return tensor

    operands = weight_EOI.operands
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
        output_RO = _Float8GroupedMMFunction.apply(
            input_RI,
            weight_EOI,
            operands.weight_qdata_fprop_EIO,
            operands.weight_scale_fprop_E1O,
            operands.weight_qdata_dgrad_EOI,
            operands.weight_scale_dgrad_EI,
            offsets_E,
        )
        output_RO.sum().backward()

    weight_saves = [
        tensor for tensor in saved_tensors if isinstance(tensor, _UnshardedFSDPTensor)
    ]
    assert len(weight_saves) == 1
    assert all(
        tensor is weight_saves[0]
        or tensor.untyped_storage()._cdata
        not in {
            operand.untyped_storage()._cdata
            for operand in (
                weight_saves[0].operands.weight_qdata_fprop_EIO,
                weight_saves[0].operands.weight_scale_fprop_E1O,
                weight_saves[0].operands.weight_qdata_dgrad_EOI,
                weight_saves[0].operands.weight_scale_dgrad_EI,
            )
        }
        for tensor in saved_tensors
    )


@pytest.mark.parametrize("execution_mode", ["compile", "activation_checkpoint"])
def test_float8_grouped_linear_runs_outside_plain_eager(execution_mode):
    """Structured Float8 grouped linear supports compile and checkpoint replay."""
    grouped_linear = _install_unsharded_weights(_make_float8_grouped_linear())
    input_RI = torch.randn(
        64,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    offsets_E = torch.arange(16, 65, 16, device="cuda", dtype=torch.int32)
    if execution_mode == "compile":
        output_R2O = torch.compile(grouped_linear, fullgraph=True)(
            input_RI,
            offsets_E,
        )
    else:
        output_R2O = checkpoint(
            grouped_linear,
            input_RI,
            offsets_E,
            use_reentrant=False,
        )
    output_R2O.sum().backward()

    assert output_R2O.shape == (64, 2, 128)
    assert input_RI.grad is not None
    assert all(parameter.grad is not None for parameter in grouped_linear.parameters())
