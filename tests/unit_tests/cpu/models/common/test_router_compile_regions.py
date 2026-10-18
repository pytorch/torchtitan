# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models import deepseek_v3, kimi_k2_7


def test_router_region_lists() -> None:
    """Kimi adds moe_router_topk to DeepSeek-V3's list, which leaves it out."""
    deepseek_v3_regions = deepseek_v3.build_model_config(
        "debugmodel", parallelism=ParallelismConfig()
    ).local_compile_regions
    kimi_regions = kimi_k2_7.build_model_config(
        "Kimi-K2.5", parallelism=ParallelismConfig()
    ).local_compile_regions

    assert "moe_router_topk" not in deepseek_v3_regions
    assert kimi_regions == [*deepseek_v3_regions, "moe_router_topk"]
