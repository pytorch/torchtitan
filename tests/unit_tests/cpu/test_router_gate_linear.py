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

from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.common.linear import RouterGateLinear


def test_router_gate_linear_forward_and_backward_contract_cpu():
    torch.manual_seed(0)
    layer = RouterGateLinear.Config(
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
def test_router_gate_linear_uses_fp32_if_either_operand_is_fp32(
    input_dtype, weight_dtype
):
    torch.manual_seed(0)
    layer = (
        RouterGateLinear.Config(
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


def test_router_gate_linear_preserves_linear_state_dict():
    layer = RouterGateLinear.Config(
        in_features=8,
        out_features=4,
        bias=True,
    ).build()
    assert set(layer.state_dict()) == {"weight", "bias"}


class TestRouterGateLinearSPMD(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @property
    def device_type(self):
        return "cpu"

    @with_comms
    def test_autograd_function_propagates_router_types(self):
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        layer = RouterGateLinear.Config(
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
