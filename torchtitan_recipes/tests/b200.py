# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the ``b200`` integration test suite."""

from torchtitan.components.optimizer import default_adamw
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests import _set_spmd_typechecking


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


def kimi_k3_debugmodel_pp4_vp4() -> Trainer.Config:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 4
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.num_pp_microbatches = 4
    # 16 stages: one layer per stage from layer 5 on, the head alone on the last.
    config.parallelism.pipeline_parallel_module_fqns_per_model_part = [
        ["vision_encoder", "tok_embeddings", "layers.0"],
        ["layers.1", "layers.2"],
        ["layers.3", "layers.4"],
        *[[f"layers.{i}"] for i in range(5, 17)],
        ["norm", "lm_head", "output_res_proj", "output_res_norm"],
    ]
    return config


def kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2() -> Trainer.Config:
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    # Type checking stays off under pipeline parallelism, as the other pipeline
    # recipes have it.
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.enable_sequence_parallel = True
    config.parallelism.expert_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.parallelism.num_pp_microbatches = 4
    # DistMuon does not support tensor parallelism yet (#3353), so this cell
    # keeps AdamW the way kimi_k3_debugmodel_mm does.
    config.optimizer = default_adamw(lr=8e-4)
    return config
