# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in CUDA stream overlap for MoE shared experts."""

from dataclasses import replace

from torchtitan.config import override
from torchtitan.models.common.moe import MoE


@override(target=MoE.Config)
def shared_expert_overlap(config: MoE.Config) -> MoE.Config:
    """Enable shared/routed overlap without changing parameter ownership."""
    if config.shared_experts is None:
        return config
    return replace(config, overlap_shared_experts=True)
