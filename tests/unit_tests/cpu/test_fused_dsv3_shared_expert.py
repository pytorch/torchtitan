# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

import pytest
import torch


pytest.importorskip("torchao")

from torchtitan.config import apply_overrides, derive, OverrideConfig  # noqa: E402
from torchtitan.models.common.activation import SwiGLU  # noqa: E402
from torchtitan.models.common.feed_forward import FeedForward  # noqa: E402
from torchtitan.models.deepseek_v3 import build_model_config  # noqa: E402
from torchtitan.quantization.mxfp8.linear import MXFP8Linear  # noqa: E402
from torchtitan.quantization.utils import get_quantized_linear  # noqa: E402
from torchtitan_recipes.overrides.fused_dsv3_shared_expert import (  # noqa: E402
    ACCEPTED,
    fused_dsv3_shared_expert,
    FusedDSv3SharedExpert,
)
from torchtitan_recipes.overrides.fused_swiglu import FusedSwiGLU  # noqa: E402


def _config():
    return FeedForward.Config(
        w13=MXFP8Linear.Config(
            in_features=7168,
            out_features=2048,
            num_linears=2,
            input_activation_format_for_backward="bf16",
        ),
        w2=MXFP8Linear.Config(
            in_features=2048,
            out_features=7168,
            input_activation_format_for_backward="mxfp8",
        ),
        activation_fn=FusedSwiGLU.Config(),
    )


def test_shared_expert_is_explicitly_experimental_and_off_by_default():
    config = _config()
    assert ACCEPTED is False
    assert fused_dsv3_shared_expert(config) is config


@pytest.mark.parametrize(
    "forward,backward", [(True, False), (False, True), (True, True)]
)
def test_shared_expert_config_preserves_linears_and_save_policies(forward, backward):
    config = _config()
    fused = fused_dsv3_shared_expert(
        config, forward_quant=forward, backward_quant=backward
    )
    assert type(fused) is FusedDSv3SharedExpert.Config
    assert fused.shared_forward_quant is forward
    assert fused.shared_backward_quant is backward
    assert fused.w13 is config.w13
    assert fused.activation_fn is config.activation_fn
    assert fused.w2.input_activation_format_for_backward == "mxfp8"
    assert fused.w2.sharding_config == config.w2.sharding_config
    assert fused.w2.param_init == config.w2.param_init


def test_shared_expert_preserves_checkpoint_keys_and_physical_weight_shapes():
    config = _config()
    with torch.device("meta"):
        native = config.build()
        fused = fused_dsv3_shared_expert(
            config, forward_quant=True, backward_quant=True
        ).build()
    expected = {
        name: (value.shape, value.dtype) for name, value in native.state_dict().items()
    }
    actual = {
        name: (value.shape, value.dtype) for name, value in fused.state_dict().items()
    }
    assert actual == expected
    assert actual["w13.weight"][0] == (2, 2048, 7168)
    assert actual["w2.weight"][0] == (7168, 2048)


def test_shared_expert_requires_the_existing_fused_activation_contract():
    config = dataclasses.replace(_config(), activation_fn=SwiGLU.Config())
    assert fused_dsv3_shared_expert(config, forward_quant=True) is config


def test_shared_expert_leaves_the_dense_ffn_shape_native():
    config = _config()
    config.w13.out_features = 18432
    config.w2.in_features = 18432
    assert fused_dsv3_shared_expert(config, forward_quant=True) is config


@pytest.mark.parametrize(
    "router_override", ["fused_dsv3_router", "fused_dsv3_seqwise_loss"]
)
def test_shared_expert_override_composes_with_the_existing_stack(router_override):
    config = build_model_config("671B", seq_len=4096)
    shared = config.layers[25].moe.shared_experts
    for name in ("w13", "w2"):
        linear = getattr(shared, name)
        cls = get_quantized_linear(MXFP8Linear, linear._owner)
        setattr(shared, name, derive(linear, cls.Config))
    shared.activation_fn = FusedSwiGLU.Config()
    dense = config.layers[0].feed_forward
    apply_overrides(
        OverrideConfig(
            imports=[
                f"torchtitan_recipes.overrides.{router_override}.{router_override}",
                (
                    "torchtitan_recipes.overrides.fused_dsv3_shared_expert.fused_dsv3_shared_expert",
                    {"forward_quant": True, "backward_quant": True},
                ),
            ]
        ),
        config,
    )
    assert type(config.layers[25].moe.shared_experts) is FusedDSv3SharedExpert.Config
    assert config.layers[0].feed_forward is dense
    assert config.layers[25].moe.shared_experts.w13.in_features == 7168
    assert config.layers[25].moe.shared_experts.w2.out_features == 7168


def test_shared_expert_fake_outputs_preserve_production_layouts():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torchtitan_recipes.overrides._dsv3_shared_expert.ops import (
        shared_expert_backward_op,
        shared_expert_forward_op,
    )

    with FakeTensorMode():
        x = torch.empty(4096, 7168, device="cuda", dtype=torch.float8_e4m3fn)
        w13 = torch.empty(4096, 7168, device="cuda", dtype=x.dtype)
        w2 = torch.empty(7168, 2048, device="cuda", dtype=x.dtype)
        x_scale = torch.empty(917504, device="cuda", dtype=torch.float8_e8m0fnu)
        w2_scale = torch.empty(458752, device="cuda", dtype=x_scale.dtype)
        forward = shared_expert_forward_op(x, w13, x_scale, x_scale)
        assert [tuple(t.shape) for t in forward] == [
            (4096, 2048),
            (4096, 4096),
            (4096, 2048),
            (4096, 2048),
            (262144,),
            (262144,),
        ]
        assert forward[0].dtype == forward[1].dtype == torch.bfloat16
        assert forward[2].stride() == (2048, 1)
        assert forward[3].stride() == (1, 4096)
        backward = shared_expert_backward_op(x, w2, x_scale, w2_scale, forward[1])
        assert [tuple(t.shape) for t in backward] == [
            (4096, 4096),
            (4096, 4096),
            (524288,),
            (524288,),
        ]
        assert backward[0].stride() == (4096, 1)
        assert backward[1].stride() == (1, 4096)
        with pytest.raises(ValueError, match="validated"):
            shared_expert_forward_op(x[:2048], w13, x_scale, x_scale)
