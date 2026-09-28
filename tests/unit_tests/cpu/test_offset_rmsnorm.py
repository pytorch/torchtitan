# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

from torchtitan.models.qwen3_5 import model_registry, Qwen35Model
from torchtitan.models.qwen3_5.model import OffsetRMSNorm
from torchtitan.models.qwen3_5.sharding import set_qwen35_sharding_config


def test_all_offset_rmsnorms_are_local_spmd_regions() -> None:
    config = cast(
        Qwen35Model.Config,
        model_registry("debugmodel", enable_sp=True, attn_backend="flex"),
    )
    set_qwen35_sharding_config(config, enable_sp=True, enable_ep=False)

    norms = [norm for _, norm, _, _ in config.traverse(OffsetRMSNorm.Config)]
    assert norms
    for norm in norms:
        assert norm.sharding_config is not None
        assert norm.sharding_config.local_spmd
        assert norm.sharding_config.in_src_shardings is not None
        assert "input" in norm.sharding_config.in_src_shardings
