# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
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


# One MoE debug flavor per model on the shared router.
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
def test_moe_models_list_only_registered_regions(model, flavor: str) -> None:
    """Each MoE model's default regions, moe_router_topk included, are registered."""
    config = model.build_model_config(flavor, parallelism=ParallelismConfig())
    try:
        # Raises on a name no local_compile function registered.
        LocalCompileConfig(regions=config.local_compile.regions).apply()
    finally:
        # The binding is process-wide: don't leave compiled regions on for later tests.
        LocalCompileConfig(regions=[]).apply()
