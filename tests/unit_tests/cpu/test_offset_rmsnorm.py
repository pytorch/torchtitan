# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.qwen3_5.config_registry import qwen35_debugmodel
from torchtitan.models.qwen3_6.config_registry import qwen36_debugmodel
from torchtitan.models.qwen3_8.config_registry import qwen38_debugmodel


def test_qwen35_architecture_enables_offset_rmsnorm_local_compile() -> None:
    for config_factory in (qwen35_debugmodel, qwen36_debugmodel, qwen38_debugmodel):
        config = config_factory()
        assert config.compile is not None
        assert config.compile.regions == [
            "gated_rmsnorm",
            "loss",
            "offset_rmsnorm",
        ]
