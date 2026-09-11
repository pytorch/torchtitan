# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the ``b200`` integration test suite."""

from torchtitan.trainer import Trainer

from torchtitan_recipes.tests import _use_spmd_types


def kimi_k3_debugmodel_mm_fsdp2() -> Trainer.Config:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    return config


def llama3_debugmodel_mxfp8_fsdp2() -> Trainer.Config:
    from torchtitan.models.llama3.config_registry import llama3_debugmodel_mxfp8

    config = llama3_debugmodel_mxfp8()
    config.parallelism.data_parallel_shard_degree = 2
    return config


def kimi_k3_debugmodel_mm_allgather_kv_cp2() -> Trainer.Config:
    from torchtitan.config.transform import (
        apply_transforms,
        ContextParallelTransform,
        KDAContextParallelTransform,
    )
    from torchtitan.models.common.cp_attention import KVAllGatherCPFlexInnerAttention
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = "headtail"
    return apply_transforms(
        config,
        [
            ContextParallelTransform(inner_attention=KVAllGatherCPFlexInnerAttention),
            KDAContextParallelTransform(),
        ],
    )


def kimi_k3_debugmodel_mm_ulysses_cp2() -> Trainer.Config:
    from torchtitan.config.transform import (
        apply_transforms,
        ContextParallelTransform,
        KDAContextParallelTransform,
    )
    from torchtitan.models.common.cp_attention import UlyssesCPFlexInnerAttention
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = None
    return apply_transforms(
        config,
        [
            ContextParallelTransform(inner_attention=UlyssesCPFlexInnerAttention),
            KDAContextParallelTransform(),
        ],
    )
