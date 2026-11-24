# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import pytest

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    FullPrecisionLinearTransform,
    LMHeadFP32OutputTransform,
    transform_model_config_,
)
from torchtitan.models.common.linear.full_precision_linear import FullPrecisionLinear
from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.deepseek_v3 import build_model_config


def test_swaps_every_hi_mid_lo_linear():
    model = build_model_config(
        "debugmodel", parallelism=ParallelismConfig(), attn_backend="flex", seq_len=128
    )
    router_gates = [
        (fqn, gate)
        for fqn, gate, _, _ in model.traverse(HiMidLoLinear.Config)
        if fqn.endswith("router.gate")
    ]
    assert router_gates

    # Listed out of order: the fp32 lm_head must exist before it is swapped.
    model = transform_model_config_(
        model, [FullPrecisionLinearTransform(), LMHeadFP32OutputTransform()]
    )

    assert not list(model.traverse(HiMidLoLinear.Config))
    assert isinstance(model.lm_head, FullPrecisionLinear.Config)
    converted = {
        fqn: config for fqn, config, _, _ in model.traverse(FullPrecisionLinear.Config)
    }
    for fqn, gate in router_gates:
        assert converted[fqn].in_features == gate.in_features
        assert converted[fqn].out_features == gate.out_features
        assert converted[fqn].bias == gate.bias
        assert converted[fqn].param_init == gate.param_init
        assert converted[fqn].preserve_dtype_params == ("weight",)


def test_rejects_subclasses():
    @dataclass(kw_only=True, slots=True)
    class SubclassConfig(HiMidLoLinear.Config):
        soft_cap: float = 30.0

    with pytest.raises(ValueError, match="only converts a plain HiMidLoLinear.Config"):
        FullPrecisionLinearTransform().transform(
            SubclassConfig(in_features=8, out_features=4)
        )
