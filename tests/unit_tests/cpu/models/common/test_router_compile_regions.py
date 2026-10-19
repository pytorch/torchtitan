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


# One MoE debug flavor per model whose forward runs TokenChoiceTopKRouter's compiled region.
_MOE_DEBUG_CONFIGS = {
    "qwen3": lambda: qwen3.build_model_config(
        "debugmodel_moe", parallelism=ParallelismConfig()
    ),
    "qwen3_5": lambda: qwen3_5.build_model_config(
        "debugmodel_moe", parallelism=ParallelismConfig()
    ),
    "deepseek_v3": lambda: deepseek_v3.build_model_config(
        "debugmodel", parallelism=ParallelismConfig()
    ),
    "deepseek_v4": lambda: deepseek_v4.build_model_config(
        "debugmodel", parallelism=ParallelismConfig()
    ),
    "gpt_oss": lambda: gpt_oss.build_model_config(
        "debugmodel", parallelism=ParallelismConfig()
    ),
    "kimi_k2_7": lambda: kimi_k2_7.build_model_config(
        "debugmodel", parallelism=ParallelismConfig()
    ),
    "kimi_k3": lambda: kimi_k3.build_model_config(
        "debugmodel", parallelism=ParallelismConfig()
    ),
    "muse_spark": lambda: muse_spark.build_model_config("debugmodel"),
}


@pytest.mark.parametrize("model", sorted(_MOE_DEBUG_CONFIGS))
def test_moe_models_compile_the_router_region(model: str) -> None:
    """Each MoE model lists moe_router_topk, its routers run that region, and its names are registered."""
    config = _MOE_DEBUG_CONFIGS[model]()
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


def test_router_region_lists() -> None:
    """Kimi K2.5 inherits DeepSeek-V3's list, which has moe_router_topk; Muse lists it."""
    deepseek_v3_regions = deepseek_v3.build_model_config(
        "debugmodel", parallelism=ParallelismConfig()
    ).local_compile_regions
    kimi_regions = kimi_k2_7.build_model_config(
        "Kimi-K2.5", parallelism=ParallelismConfig()
    ).local_compile_regions
    muse_regions = muse_spark.build_model_config("816B-A42B").local_compile_regions

    assert "moe_router_topk" in deepseek_v3_regions
    assert kimi_regions == deepseek_v3_regions
    assert "moe_router_topk" in muse_regions
