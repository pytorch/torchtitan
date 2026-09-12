# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel Muse Glimmer recipes."""

from torchtitan.config.transform import apply_transforms, ContextParallelTransform
from torchtitan.models.common.cp_attention import (
    KVAllGatherCPFlexInnerAttention,
    UlyssesCPFlexInnerAttention,
)
from torchtitan.models.muse_glimmer.config_registry import muse_glimmer_30b
from torchtitan.protocols.module import Module
from torchtitan.trainer import Trainer


def _muse_glimmer_30b_cp(
    *,
    inner_attention: type[Module],
    cp_degree: int,
    load_balancer: str | None = "headtail",
) -> Trainer.Config:
    config = muse_glimmer_30b()
    config.parallelism.context_parallel_degree = cp_degree
    config.parallelism.context_parallel_load_balancer = load_balancer
    return apply_transforms(
        config,
        [ContextParallelTransform(inner_attention=inner_attention)],
    )


def muse_glimmer_30b_allgather_cp8() -> Trainer.Config:
    """Muse Glimmer 30B with all-gather CP degree 8."""
    return _muse_glimmer_30b_cp(
        inner_attention=KVAllGatherCPFlexInnerAttention, cp_degree=8
    )


def muse_glimmer_30b_ulysses_cp2() -> Trainer.Config:
    """Muse Glimmer 30B with Ulysses CP degree 2.

    The model has two KV heads, which limits Ulysses CP to degree 2.
    """
    return _muse_glimmer_30b_cp(
        inner_attention=UlyssesCPFlexInnerAttention,
        cp_degree=2,
        # Ulysses does not support token reordering.
        load_balancer=None,
    )
