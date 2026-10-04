# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

import pytest
import spmd_types as spmd
import torch
from spmd_types.checker import typecheck
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.common.linear import (
    _split_into_bf16_pieces,
    FP32OutputLinear,
    Linear,
)


def test_fp32_output_linear_forward_and_backward_contract_cpu():
    torch.manual_seed(0)
    layer = FP32OutputLinear.Config(
        in_features=8,
        out_features=4,
        bias=True,
    ).build()
    layer = layer.to(dtype=torch.bfloat16)
    input_TD = torch.randn(6, 8, dtype=torch.bfloat16, requires_grad=True)
    grad_output_TE = torch.randn(6, 4, dtype=torch.float32)

    output_TE = layer(input_TD)
    output_TE.backward(grad_output_TE)

    input_ref_TD = input_TD.detach().float().requires_grad_()
    weight_ref_ED = layer.weight.detach().float().requires_grad_()
    bias_ref_E = layer.bias.detach().float().requires_grad_()
    output_ref_TE = input_ref_TD @ weight_ref_ED.T + bias_ref_E
    output_ref_TE.backward(grad_output_TE)

    assert output_TE.dtype is torch.float32
    torch.testing.assert_close(output_TE, output_ref_TE)
    torch.testing.assert_close(input_TD.grad, input_ref_TD.grad.bfloat16())
    torch.testing.assert_close(layer.weight.grad, weight_ref_ED.grad.bfloat16())
    torch.testing.assert_close(layer.bias.grad, bias_ref_E.grad.bfloat16())


@pytest.mark.parametrize(
    ("input_dtype", "weight_dtype"),
    [
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
def test_fp32_output_linear_uses_fp32_if_either_operand_is_fp32(
    input_dtype, weight_dtype
):
    torch.manual_seed(0)
    layer = (
        FP32OutputLinear.Config(
            in_features=8,
            out_features=4,
            bias=True,
        )
        .build()
        .to(dtype=weight_dtype)
    )
    input_TD = torch.randn(6, 8, dtype=input_dtype, requires_grad=True)
    grad_output_TE = torch.randn(6, 4)

    output_TE = layer(input_TD)
    output_TE.backward(grad_output_TE)

    input_ref_TD = input_TD.detach().float().requires_grad_()
    weight_ref_ED = layer.weight.detach().float().requires_grad_()
    bias_ref_E = layer.bias.detach().float().requires_grad_()
    output_ref_TE = input_ref_TD @ weight_ref_ED.T + bias_ref_E
    output_ref_TE.backward(grad_output_TE)

    torch.testing.assert_close(output_TE, output_ref_TE)
    torch.testing.assert_close(input_TD.grad, input_ref_TD.grad.to(input_dtype))
    torch.testing.assert_close(layer.weight.grad, weight_ref_ED.grad.to(weight_dtype))
    torch.testing.assert_close(layer.bias.grad, bias_ref_E.grad.to(weight_dtype))


def test_fp32_output_linear_preserves_linear_state_dict():
    layer = FP32OutputLinear.Config(
        in_features=8,
        out_features=4,
        bias=True,
    ).build()
    assert set(layer.state_dict()) == {"weight", "bias"}


class TestFP32OutputLinearSPMD(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @property
    def device_type(self):
        return "cpu"

    @with_comms
    def test_autograd_function_propagates_router_types(self):
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        layer = FP32OutputLinear.Config(
            in_features=8,
            out_features=4,
            bias=True,
        ).build()
        input_TD = torch.randn(3, 8)

        with set_current_spmd_mesh(mesh), typecheck(strict_mode="strict", local=False):
            spmd.assert_type(input_TD, {"tp": spmd.S(0)})
            spmd.assert_type(layer.weight, {"tp": spmd.R})
            spmd.assert_type(layer.bias, {"tp": spmd.R})
            output_TE = layer(input_TD)
            spmd.assert_type(output_TE, {"tp": spmd.V})


def test_fp32_output_linear_preserves_stacked_output_shape():
    layer = (
        FP32OutputLinear.Config(in_features=8, out_features=4, num_linears=2, bias=True)
        .build()
        .to(torch.bfloat16)
    )
    input_BTD = torch.randn(2, 3, 8, dtype=torch.bfloat16)

    output = layer(input_BTD)

    weight_OD = layer.weight.float().flatten(0, -2)
    expected = (
        input_BTD.float() @ weight_OD.T + layer.bias.float().flatten()
    ).unflatten(-1, (2, 4))
    assert output.shape == (2, 3, 2, 4)
    assert output.dtype is torch.float32
    torch.testing.assert_close(output, expected)


def test_split_into_bf16_pieces_recovers_fp32():
    tensor = torch.randn(1000) * torch.logspace(-8, 8, 1000)

    hi, lo = _split_into_bf16_pieces(tensor, grad_output_pieces=2)

    assert hi.dtype == lo.dtype == torch.bfloat16
    relative_error = ((hi.float() + lo.float()) - tensor).abs() / tensor.abs()
    assert relative_error.max() <= 2**-15
    # One bf16 alone is ~100x worse.
    assert (tensor.bfloat16().float() - tensor).abs().div(tensor.abs()).max() > 2**-10
    # Three pieces are exact.
    pieces = _split_into_bf16_pieces(tensor, grad_output_pieces=3)
    assert sum(piece.double() for piece in pieces).equal(tensor.double())


def test_split_into_bf16_pieces_rounds_to_nearest():
    # 1 + 3 * 2^-9 is 0.75 of a bf16 step above 1, so hi rounds up and mid is negative.
    tensor = torch.tensor([1 + 3 * 2**-9])

    hi, mid, lo = _split_into_bf16_pieces(tensor, grad_output_pieces=3)

    assert hi.item() == 1 + 2**-7
    assert mid.item() == -(2**-9)
    assert lo.item() == 0.0


def test_lora_wraps_fp32_output_linear():
    from torchtitan.config.transform.lora import LinearLoRAHandler

    config = FP32OutputLinear.Config(in_features=8, out_features=16)
    layer = LinearLoRAHandler().make_config(config, rank=4, alpha=8.0).build()
    layer = layer.to(torch.bfloat16)
    # Zero the adapter (param_init does this at model init) so the output is the base projection.
    torch.nn.init.zeros_(layer.lora_b.weight)
    input_TD = torch.randn(3, 8, dtype=torch.bfloat16)

    output_TO = layer(input_TD)

    assert isinstance(layer, FP32OutputLinear)
    assert output_TO.dtype is torch.float32
    torch.testing.assert_close(output_TO, input_TD.float() @ layer.weight.float().T)


def test_lm_head_converter_swaps_only_lm_head():
    from torchtitan.config.transform import LMHeadFP32OutputConverter
    from torchtitan.models.qwen3 import MODEL_FLAVORS

    build_config, max_context_length = MODEL_FLAVORS["0.6B"]
    config = build_config(attn_backend="flex", seq_len=max_context_length)
    lm_head_before = config.lm_head

    LMHeadFP32OutputConverter.Config().build().convert(config)

    swapped = [
        fqn
        for fqn, linear_config, _, _ in config.traverse(Linear.Config)
        if isinstance(linear_config, FP32OutputLinear.Config)
    ]
    assert swapped == ["lm_head"]
    for field in fields(lm_head_before):
        assert getattr(config.lm_head, field.name) == getattr(
            lm_head_before, field.name
        )
    # The head keeps 2 pieces: summed over the vocab, a third doesn't help.
    assert config.lm_head.higher_precision_bwd is False
    # Converting an already converted head is a no-op.
    converted = config.lm_head
    LMHeadFP32OutputConverter.Config().build().convert(config)
    assert config.lm_head == converted
    # The flag is configurable.
    config.lm_head = lm_head_before
    converter = LMHeadFP32OutputConverter.Config(higher_precision_bwd=True).build()
    converter.convert(config)
    assert config.lm_head.higher_precision_bwd is True

    config.lm_head = None
    with pytest.raises(ValueError, match="lm_head"):
        LMHeadFP32OutputConverter.Config().build().convert(config)
