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
from torchtitan.models.common.linear.full_precision_linear import FullPrecisionLinear


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_full_precision_linear_computes_in_fp32(input_dtype):
    torch.manual_seed(0)
    layer = FullPrecisionLinear.Config(in_features=8, out_features=4, bias=True).build()
    input_TD = torch.randn(6, 8, dtype=input_dtype, requires_grad=True)
    grad_output_TO = torch.randn(6, 4)

    output_TO = layer(input_TD)
    output_TO.backward(grad_output_TO)

    input_ref_TD = input_TD.detach().float().requires_grad_()
    weight_ref_OD = layer.weight.detach().requires_grad_()
    bias_ref_O = layer.bias.detach().requires_grad_()
    output_ref_TO = input_ref_TD @ weight_ref_OD.T + bias_ref_O
    output_ref_TO.backward(grad_output_TO)

    assert output_TO.dtype is torch.float32
    torch.testing.assert_close(output_TO, output_ref_TO)
    # The input gradient comes back in the input's dtype; the weight's stays fp32.
    assert input_TD.grad.dtype is input_dtype
    torch.testing.assert_close(input_TD.grad, input_ref_TD.grad.to(input_dtype))
    assert layer.weight.grad.dtype is torch.float32
    torch.testing.assert_close(layer.weight.grad, weight_ref_OD.grad)
    torch.testing.assert_close(layer.bias.grad, bias_ref_O.grad)


def test_full_precision_linear_rejects_a_lower_precision_weight():
    # A bf16 weight (preserve_dtype_params overridden, or a bulk cast) fails loudly
    # instead of running the matmul in bf16.
    layer = FullPrecisionLinear.Config(in_features=8, out_features=4).build()
    layer = layer.to(torch.bfloat16)

    with pytest.raises(RuntimeError, match="same dtype"):
        layer(torch.randn(6, 8, dtype=torch.bfloat16))


class TestFullPrecisionLinearSPMD(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @property
    def device_type(self):
        return "cpu"

    @with_comms
    def test_propagates_spmd_types(self):
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        layer = FullPrecisionLinear.Config(
            in_features=8,
            out_features=4,
            bias=True,
        ).build()
        input_TD = torch.randn(3, 8, dtype=torch.bfloat16)

        with set_current_spmd_mesh(mesh), typecheck(strict_mode="strict", local=False):
            spmd.assert_type(input_TD, {"tp": spmd.S(0)})
            spmd.assert_type(layer.weight, {"tp": spmd.R})
            spmd.assert_type(layer.bias, {"tp": spmd.R})
            output_TO = layer(input_TD)
            spmd.assert_type(output_TO, {"tp": spmd.S(0)})
