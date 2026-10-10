# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Production-shape fixtures for the shared-expert numerical and FSDP tests."""

import torch
from torch import nn

from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.quantization._fsdp_tensor import _UnshardedFSDPTensor
from torchtitan.quantization.mxfp8.linear import (
    InputActivationFormatForBackward,
    MXFP8Linear,
)
from torchtitan_recipes.overrides.fused_dsv3_shared_expert import (
    fused_dsv3_shared_expert,
)
from torchtitan_recipes.overrides.fused_swiglu import FusedSwiGLU


MODES = {
    "native": (False, False),
    "forward_fused": (True, False),
    "backward_fused": (False, True),
    "both_fused": (True, True),
}


def shared_expert_config(
    *,
    input_save_format: InputActivationFormatForBackward = "bf16",
    hidden_save_format: InputActivationFormatForBackward = "mxfp8",
):
    return FeedForward.Config(
        w13=MXFP8Linear.Config(
            in_features=7168,
            out_features=2048,
            num_linears=2,
            input_activation_format_for_backward=input_save_format,
        ),
        w2=MXFP8Linear.Config(
            in_features=2048,
            out_features=7168,
            input_activation_format_for_backward=hidden_save_format,
        ),
        activation_fn=FusedSwiGLU.Config(),
    )


def make_case(
    seed,
    mode,
    *,
    input_save_format: InputActivationFormatForBackward = "bf16",
    hidden_save_format: InputActivationFormatForBackward = "mxfp8",
    requires_grad=(True, True, True),
    grad_dtype=None,
    prepared_weights=True,
):
    """Nonzero production-shape inputs; stand in for FSDP weight preparation."""
    torch.manual_seed(seed)
    config = shared_expert_config(
        input_save_format=input_save_format, hidden_save_format=hidden_save_format
    )
    forward, backward = MODES[mode]
    if forward or backward:
        config = fused_dsv3_shared_expert(
            config, forward_quant=forward, backward_quant=backward
        )
    with torch.device("cuda"):
        module = config.build().bfloat16()
    for linear, needs_grad in zip((module.w13, module.w2), requires_grad[1:]):
        sharded_weight = linear.weight
        with torch.no_grad():
            sharded_weight._tensor.normal_(0, 0.02)
            if prepared_weights:
                linear.weight = nn.Parameter(
                    _UnshardedFSDPTensor(
                        sharded_weight._tensor,
                        sharded_weight._build_operands(sharded_weight._tensor),
                    ),
                    requires_grad=needs_grad,
                )
        linear.weight.requires_grad_(needs_grad)
        if grad_dtype is not None:
            linear.weight.grad_dtype = grad_dtype
    x_TD = torch.randn(
        1,
        4096,
        7168,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=requires_grad[0],
    )
    dy_TD = torch.randn_like(x_TD)
    return module, x_TD, dy_TD


def assert_bits_equal(expected, actual):
    if expected is None or actual is None:
        assert expected is actual
        return
    assert (expected.shape, expected.dtype) == (actual.shape, actual.dtype)
    expected_bits = (
        expected.detach().resolve_neg().contiguous().reshape(-1).view(torch.uint8)
    )
    actual_bits = (
        actual.detach().resolve_neg().contiguous().reshape(-1).view(torch.uint8)
    )
    assert torch.equal(expected_bits, actual_bits), (
        f"{int((expected_bits != actual_bits).sum())} mismatched bytes in "
        f"{tuple(expected.shape)} {expected.dtype}"
    )
