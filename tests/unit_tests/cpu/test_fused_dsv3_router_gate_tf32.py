# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.config import apply_overrides, Configurable, OverrideConfig
from torchtitan_recipes.overrides import fused_dsv3_router_gate_tf32 as fusion


_HAS_NATIVE_GATE = fusion.RouterGateLinear is not None and fusion._NATIVE_MM is not None
_requires_native = pytest.mark.skipif(
    not _HAS_NATIVE_GATE, reason="The native TF32 gate belongs to the pinned fbpackage"
)


def _config(*, bias=False):
    from torchtitan.models.common.router_gate import RouterGateLinear, RouterGemmConfig

    return RouterGateLinear.Config(
        in_features=7168,
        out_features=256,
        bias=bias,
        forward=RouterGemmConfig(
            input_dtype=torch.bfloat16,
            compute_mode="bf16",
            output_dtype=torch.float32,
        ),
        backward=RouterGemmConfig(
            input_dtype=torch.float32,
            compute_mode="tf32",
            output_dtype=torch.float32,
        ),
    )


def test_missing_native_tf32_policy_is_explicit():
    with patch.object(fusion, "_NATIVE_MM", None):
        with pytest.raises(RuntimeError, match="pinned fbpackage"):
            fusion.fused_dsv3_router_gate_tf32(None, experimental=True)


def test_fake_operators_keep_bf16_gradients_and_distinct_names():
    with patch.object(
        fusion, "_CUTEDSL_IMPORT_ERROR", ImportError("unavailable")
    ), FakeTensorMode():
        gradient_TE = torch.empty((4096, 256), device="cuda", dtype=torch.float32)
        weight_ED = torch.empty((256, 7168), device="cuda", dtype=torch.bfloat16)
        input_TD = torch.empty((4096, 7168), device="cuda", dtype=torch.bfloat16)

        def backward(gradient_TE, weight_ED, input_TD):
            return fusion.dgrad_op(gradient_TE, weight_ED), fusion.wgrad_op(
                gradient_TE, input_TD
            )

        grad_input_TD, grad_weight_ED = backward(gradient_TE, weight_ED, input_TD)
        assert grad_input_TD.shape == input_TD.shape
        assert grad_weight_ED.shape == weight_ED.shape
        assert grad_input_TD.dtype == grad_weight_ED.dtype == torch.bfloat16
        assert grad_input_TD.is_contiguous() and grad_weight_ED.is_contiguous()
        graph = make_fx(backward)(gradient_TE, weight_ED, input_TD)
        assert [
            str(node.target) for node in graph.graph.nodes if node.op == "call_function"
        ] == [
            "torchtitan.dsv3_router_gate_tf32_dgrad.default",
            "torchtitan.dsv3_router_gate_tf32_wgrad.default",
        ]
        assert not fusion._supports_gate_backward(gradient_TE, weight_ED, wgrad=False)


@_requires_native
@pytest.mark.parametrize("experimental", [False, True])
def test_override_preserves_native_policy_and_state(experimental):
    config = _config(bias=True)
    with pytest.warns(UserWarning, match="roofline acceptance"):
        replacement = fusion.fused_dsv3_router_gate_tf32(
            config, experimental=experimental
        )
    assert not fusion.ACCEPTED
    assert replacement.use_fused_backward is experimental
    assert replacement.forward == config.forward
    assert replacement.backward == config.backward
    with torch.device("meta"):
        native = config.build()
        candidate = replacement.build()
    assert (
        list(native.state_dict()) == list(candidate.state_dict()) == ["weight", "bias"]
    )
    assert native.weight.shape == candidate.weight.shape == (256, 7168)


@_requires_native
def test_override_selects_moe_gates_without_replacing_other_linears():
    @dataclass(kw_only=True, slots=True)
    class RouterConfig(Configurable.Config):
        gate: Configurable.Config

    @dataclass(kw_only=True, slots=True)
    class MoEConfig(Configurable.Config):
        router: RouterConfig

    @dataclass(kw_only=True, slots=True)
    class LayerConfig(Configurable.Config):
        moe: MoEConfig

    @dataclass(kw_only=True, slots=True)
    class ModelConfig(Configurable.Config):
        layer: LayerConfig
        lm_head: Configurable.Config

    root = ModelConfig(
        layer=LayerConfig(moe=MoEConfig(router=RouterConfig(gate=_config()))),
        lm_head=_config(),
    )
    lm_head = root.lm_head
    with pytest.warns(UserWarning, match="roofline acceptance"):
        replaced = apply_overrides(
            OverrideConfig(
                imports=[
                    (
                        "torchtitan_recipes.overrides.fused_dsv3_router_gate_tf32.fused_dsv3_router_gate_tf32",
                        {"experimental": True},
                    )
                ]
            ),
            root,
        )
    assert len(replaced) == 1
    assert isinstance(root.layer.moe.router.gate, fusion.FusedDSv3TF32RouterGate.Config)
    assert root.layer.moe.router.gate.use_fused_backward
    assert root.lm_head is lm_head
