# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel Muse Glimmer recipes."""

from torchtitan.models.common.cp_attention import (
    AllGatherCPFlexAttention,
    UlyssesCPFlexAttention,
    use_cp_kernel,
)
from torchtitan.models.muse_glimmer.config_registry import muse_glimmer_30b
from torchtitan.protocols.module import Module
from torchtitan.trainer import Trainer


def _muse_glimmer_30b_cp(*, kernel: type[Module], cp_degree: int) -> Trainer.Config:
    config = muse_glimmer_30b()
    use_cp_kernel(config, kernel)
    config.parallelism.context_parallel_degree = cp_degree
    return config


def muse_glimmer_30b_allgather_cp8() -> Trainer.Config:
    """Muse Glimmer 30B with all-gather CP degree 8."""
    return _muse_glimmer_30b_cp(kernel=AllGatherCPFlexAttention, cp_degree=8)


def muse_glimmer_30b_ulysses_cp2() -> Trainer.Config:
    """Muse Glimmer 30B with Ulysses CP degree 2.

    Degree 2, not 8, because Ulysses shards KV heads over the CP axis and this
    model has only 2 of them. Higher degrees are rejected at config time.
    """
    config = _muse_glimmer_30b_cp(kernel=UlyssesCPFlexAttention, cp_degree=2)
    # Head-sharded attention has no per-rank sequence imbalance to balance.
    config.parallelism.context_parallel_load_balancer = None
    return config
