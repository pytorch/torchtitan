# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the ``b200`` integration test suite."""

from dist_moe import VmmConfig
from torchtitan.components.optimizer import default_adamw
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests import _set_spmd_typechecking
from torchtitan_recipes.tests.multimodal import set_rank_conditional_image_presence


def kimi_k3_debugmodel_mm() -> Trainer.Config:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    # DistMuon rejects TP-produced _StridedShard storage, so the TP coverage
    # keeps AdamW; kimi_k3_debugmodel_mm_muon covers the default optimizer.
    config.optimizer = default_adamw(lr=8e-4)
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.enable_sequence_parallel = True
    config.parallelism.expert_parallel_degree = 2
    set_rank_conditional_image_presence(config)
    return config


def kimi_k3_debugmodel_mm_muon() -> Trainer.Config:
    """Per-head DistMuon with FSDP and EP."""
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = 2
    return config


def llama3_debugmodel_mxfp8_fsdp2() -> Trainer.Config:
    from torchtitan.models.llama3.config_registry import llama3_debugmodel_mxfp8

    config = llama3_debugmodel_mxfp8()
    config.parallelism.data_parallel_shard_degree = 2
    return config


def llama3_debugmodel_nvfp4_fsdp2() -> Trainer.Config:
    from torchtitan.config import CompileConfig
    from torchtitan.models.llama3.config_registry import llama3_debugmodel_nvfp4

    config = llama3_debugmodel_nvfp4(seq_len=2048)
    config.compile = CompileConfig(components=["model"])
    config.parallelism.data_parallel_shard_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    return config


def deepseek_v3_debugmodel_dist_moe_bf16_fsdp2_ep2() -> Trainer.Config:
    from torchtitan.models.deepseek_v3.config_registry import (
        deepseek_v3_debugmodel_dist_moe_bf16,
    )

    config = deepseek_v3_debugmodel_dist_moe_bf16(
        seq_len=128,
        device_scratch_capacity_factor=2.0,
    )
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.training.steps = 4
    config.checkpointer = None
    return config


def deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2() -> Trainer.Config:
    from torchtitan.models.deepseek_v3.config_registry import (
        deepseek_v3_debugmodel_dist_moe_mxfp8,
    )

    config = deepseek_v3_debugmodel_dist_moe_mxfp8(
        seq_len=128,
        device_scratch_capacity_factor=2.0,
    )
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.training.steps = 4
    config.checkpointer = None
    return config


def deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2_vmm() -> Trainer.Config:
    """Exercise host-backed VMM scratch preallocation with MXFP8 DistMoE."""
    from torchtitan.components.dist_moe import DistMoeRoutedExperts

    config = deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2()
    experts = list(config.model.traverse(DistMoeRoutedExperts.Config))
    assert experts, "the VMM integration recipe requires routed experts"
    for _, expert, _, _ in experts:
        assert isinstance(expert, DistMoeRoutedExperts.Config)
        expert.vmm = VmmConfig(total_scratch_capacity_factor=4.0, prefetch=True)
    return config
