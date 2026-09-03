# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel Muse Glimmer recipes."""

from torchtitan.models.common.cp_attention import AllGatherCPFlexAttention
from torchtitan.models.muse_glimmer.config_registry import muse_glimmer_30b
from torchtitan.protocols.module import Module
from torchtitan.trainer import Trainer
from torchtitan.transforms import apply_transforms, ContextParallelTransform


def _muse_glimmer_30b_cp(*, kernel: type[Module], cp_degree: int) -> Trainer.Config:
    config = muse_glimmer_30b()
    config.parallelism.context_parallel_degree = cp_degree
    return apply_transforms(config, [ContextParallelTransform.Config(kernel=kernel)])


def muse_glimmer_30b_allgather_cp8() -> Trainer.Config:
    """Muse Glimmer 30B with all-gather CP degree 8."""
    return _muse_glimmer_30b_cp(kernel=AllGatherCPFlexAttention, cp_degree=8)
