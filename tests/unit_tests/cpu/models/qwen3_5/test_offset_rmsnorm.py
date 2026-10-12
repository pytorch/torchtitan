# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_recipes.tests.models.qwen3_5 import qwen35_debugmodel
from torchtitan_recipes.tests.models.qwen3_6 import qwen36_debugmodel
from torchtitan_recipes.tests.models.qwen3_8 import qwen38_debugmodel


def test_qwen35_architecture_enables_offset_rmsnorm_local_compile() -> None:
    for config_factory in (qwen35_debugmodel, qwen36_debugmodel, qwen38_debugmodel):
        config = config_factory()
        assert config.model.local_compile_regions == [
            "loss",
            "fused_binary_activation",
            "gated_rmsnorm",
            "offset_rmsnorm",
            "partial_rope",
            "shared_expert_gate",
            "fp32_to_bf16_split",
        ]
