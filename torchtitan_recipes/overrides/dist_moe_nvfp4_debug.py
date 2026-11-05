# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Debugging-only NVFP4 override for Dist-MoE expert modules.

Requires a Dist-MoE build with native-BF16-weight NVFP4 training support. This
opt-in adapter is for local integration debugging, not production training or
numerical validation. Dense layers, parameters, and sharding remain unchanged.
NVFP4 tensor-core forward uses the selected package's BF16 STE backward path.

Activate on a Dist-MoE RL recipe with:

    --override torchtitan_recipes.overrides.dist_moe_nvfp4_debug.nvfp4_experts

Both trainer and generator experts use the staged NVFP4 pipeline. Nothing is
changed unless this override is explicitly selected.
"""

import logging
from dataclasses import dataclass, field

import dist_moe

from torchtitan.config import derive, override
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts

logger = logging.getLogger(__name__)


def _nvfp4_policy() -> dist_moe.BlockScaledConfig:
    """Use native NVFP4 forward with the package's training backward path."""
    return dist_moe.BlockScaledConfig(
        format=dist_moe.BlockScaledFormat.NVFP4,
        pipeline="staged",
    )


class NVFP4Experts(DistMoeRoutedExperts):
    """Debugging adapter retaining ordinary weights and Dist-MoE execution."""

    @dataclass(kw_only=True, slots=True)
    class Config(DistMoeRoutedExperts.Config):
        """Add an NVFP4 policy without changing parameters or sharding."""

        block_scaled_config: dist_moe.BlockScaledConfig = field(
            default_factory=_nvfp4_policy
        )

        def __post_init__(self) -> None:
            """Reject an incompatible format or BF16-only tuning override."""
            DistMoeRoutedExperts.Config.__post_init__(self)
            if self.block_scaled_config.format is not dist_moe.BlockScaledFormat.NVFP4:
                raise ValueError("This debugging override requires NVFP4")
            if self.bf16_grouped_gemm_preset is not None:
                raise ValueError("NVFP4 does not use a BF16 FPROP preset")

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.block_scaled_config = config.block_scaled_config
        logger.info(
            "Debug NVFP4 expert module: policy=%s pipeline=%s package=%s",
            self.block_scaled_config.format,
            self.block_scaled_config.pipeline,
            dist_moe.__file__,
        )


@override(
    target=DistMoeRoutedExperts.Config,
    fqns=["trainer.model.*", "generator.model.*"],
    exact=True,
    description="Debugging-only NVFP4 experts with native weights in both RL roles",
)
def nvfp4_experts(config: DistMoeRoutedExperts.Config) -> NVFP4Experts.Config:
    """Preserve the selected toy's shapes and enable NVFP4 for debugging."""
    return derive(config, NVFP4Experts.Config)
