# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from torchtitan.config import apply_overrides, OverrideConfig

from torchtitan.config.override import _REGISTRY
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import MicrobatchWiseLoadBalanceLoss
from torchtitan.models.deepseek_v3 import build_model_config
from torchtitan_recipes.overrides import fused_dsv3_router_gate as fusion
from torchtitan_recipes.overrides.fused_dsv3_router import FusedDSv3Router
from torchtitan_recipes.overrides.fused_dsv3_seqwise_loss import FusedDSv3SeqwiseLoss


_REGISTERED_OVERRIDES = {
    key: value
    for key, value in _REGISTRY.items()
    if key.startswith("torchtitan_recipes.overrides.fused_dsv3_")
}


@pytest.fixture(autouse=True)
def _registered_overrides():
    with patch.dict(_REGISTRY, _REGISTERED_OVERRIDES):
        yield


def test_override_keeps_native_backward_by_default():
    cfg = HiMidLoLinear.Config(
        in_features=7168, out_features=256, backward_mode="hi_mid_lo"
    )
    with pytest.warns(UserWarning, match="roofline acceptance"):
        replacement = fusion.fused_dsv3_router_gate(cfg)
    assert type(replacement) is fusion.FusedDSv3RouterGate.Config
    assert not replacement.use_fused_backward
    assert replacement.backward_mode == cfg.backward_mode
    assert not fusion.ACCEPTED


def test_override_selects_gates_and_preserves_other_linears():
    model = build_model_config("671B", seq_len=4096)
    model.lm_head = HiMidLoLinear.Config(in_features=7168, out_features=129280)
    lm_head = model.lm_head
    with pytest.warns(UserWarning, match="roofline acceptance"):
        apply_overrides(
            OverrideConfig(
                imports=[
                    (
                        "torchtitan_recipes.overrides.fused_dsv3_router_gate.fused_dsv3_router_gate",
                        {"experimental": True},
                    )
                ]
            ),
            model,
        )
    router = model.layers[3].moe.router
    assert type(router.gate) is fusion.FusedDSv3RouterGate.Config
    assert router.gate.use_fused_backward
    assert router.gate.backward_mode == "hi_mid_lo"
    assert type(router.aux_loss) is MicrobatchWiseLoadBalanceLoss.Config
    assert model.lm_head is lm_head


@pytest.mark.parametrize("seqwise_loss", [False, True])
def test_composition_uses_one_parent_claim(seqwise_loss):
    model = build_model_config("671B", seq_len=4096)
    with pytest.warns(UserWarning, match="roofline acceptance"):
        apply_overrides(
            OverrideConfig(
                imports=[
                    (
                        "torchtitan_recipes.overrides.fused_dsv3_router_gate.fused_dsv3_router_with_gate",
                        {"experimental": True, "seqwise_loss": seqwise_loss},
                    )
                ]
            ),
            model,
        )
    router = model.layers[3].moe.router
    assert type(router) is FusedDSv3Router.Config
    assert type(router.gate) is fusion.FusedDSv3RouterGate.Config
    assert type(router.aux_loss) is (
        FusedDSv3SeqwiseLoss.Config
        if seqwise_loss
        else MicrobatchWiseLoadBalanceLoss.Config
    )
    assert (router.gate.in_features, router.gate.out_features, router.top_k) == (
        7168,
        256,
        8,
    )


@pytest.mark.parametrize("num_linears", [1, 2])
@pytest.mark.parametrize("mode", ["hi_mid", "hi_mid_lo"])
def test_cpu_fallback_preserves_forward_backward_and_state(num_linears, mode):
    torch.manual_seed(42)
    cfg = HiMidLoLinear.Config(
        in_features=16,
        out_features=8,
        num_linears=num_linears,
        bias=True,
        backward_mode=mode,
    )
    native = cfg.build()
    with pytest.warns(UserWarning, match="roofline acceptance"):
        candidate = fusion.fused_dsv3_router_gate(cfg, experimental=True).build()
    candidate.load_state_dict(native.state_dict())
    inputs = torch.randn(2, 5, 16)
    outputs, gradients = [], []
    with patch.object(
        fusion, "dgrad_op", side_effect=AssertionError("CPU uses native backward")
    ), patch.object(
        fusion, "wgrad_op", side_effect=AssertionError("CPU uses native backward")
    ):
        for model in (native, candidate):
            x = inputs.clone().requires_grad_()
            output = model(x)
            output.square().sum().backward()
            outputs.append(output)
            gradients.append((x.grad, model.weight.grad, model.bias.grad))
    assert torch.equal(*outputs)
    for expected, actual in zip(*gradients):
        assert torch.equal(expected, actual)
    assert native.state_dict().keys() == candidate.state_dict().keys()


def test_fake_operators_preserve_gradient_dtypes_without_cutedsl():
    with patch.object(
        fusion, "_CUTEDSL_IMPORT_ERROR", ImportError("unavailable")
    ), FakeTensorMode():
        g = torch.empty(4096, 256, device="cuda")
        x = torch.empty(4096, 7168, device="cuda", dtype=torch.bfloat16)
        w = torch.empty(256, 7168, device="cuda", dtype=torch.bfloat16)
        dx = fusion.dgrad_op(g, w, 3)
        dw = fusion.wgrad_op(g, x, 3)
        assert dx.shape == x.shape and dx.dtype == torch.bfloat16
        assert dw.shape == w.shape and dw.dtype == torch.float32
        assert dx.is_contiguous() and dw.is_contiguous()
