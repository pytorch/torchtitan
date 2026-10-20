# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models import (
    deepseek_v3,
    deepseek_v4,
    gpt_oss,
    kimi_k2_7,
    kimi_k3,
    muse_spark,
    qwen3,
    qwen3_5,
)
from torchtitan.models.common.moe import TokenChoiceTopKRouter


# One MoE debug flavor per model whose routers subclass TokenChoiceTopKRouter.
@pytest.mark.parametrize(
    "model, flavor",
    [
        (qwen3, "debugmodel_moe"),
        (qwen3_5, "debugmodel_moe"),
        (deepseek_v3, "debugmodel"),
        (deepseek_v4, "debugmodel"),
        (gpt_oss, "debugmodel"),
        (kimi_k2_7, "debugmodel"),
        (kimi_k3, "debugmodel"),
        (muse_spark, "debugmodel"),
    ],
    ids=lambda value: getattr(value, "__name__", value),
)
def test_moe_models_compile_the_router_region(model, flavor: str) -> None:
    """Each MoE model lists moe_router_topk, uses TokenChoiceTopKRouter subclasses, and lists only registered regions."""
    config = model.build_model_config(flavor, parallelism=ParallelismConfig())
    routers = [
        layer.feed_forward.router
        for layer in config.layers
        if hasattr(layer.feed_forward, "router")
    ]

    assert "moe_router_topk" in config.local_compile_regions
    assert routers
    assert all(isinstance(r, TokenChoiceTopKRouter.Config) for r in routers)
    try:
        # Raises on a name no local_compile function registered.
        apply_local_compile(config.local_compile_regions)
    finally:
        # The binding is process-wide: don't leave compiled regions on for later tests.
        apply_local_compile([])
