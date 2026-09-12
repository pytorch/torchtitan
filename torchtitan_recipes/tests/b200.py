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


def kimi_k3_debugmodel_pp2_vp2() -> Trainer.Config:
    # The smallest cell that exercises the block transport: two ranks, two
    # stages each. 26 units (24 layers, the embedding and the head) over four
    # stages is uneven, a stage boundary falls inside a block, and with two
    # stages per rank a rank receives a block it already holds -- which is what
    # the rank cache and the gradient deposits are for. The shared debug model,
    # unchanged by this PR.
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.num_pp_microbatches = 4
    return config


def kimi_k3_debugmodel_pp8_vp4() -> Trainer.Config:
    # The stress cell, on a deeper model of its own: 35 units (33 layers, the
    # embedding and the head) over 32 stages, so the split is uneven and the
    # last stage holds the head alone. The depth travels with this recipe, not
    # with the shared "debugmodel" flavor. No layers_per_stage reaches 32 stages
    # for 35 units, so the recipe spells out core's split for that count, with
    # the vision tower and the AttnRes aggregation where Kimi K3's entry pins them.
    from torchtitan.distributed.pipeline_parallel import (
        _generate_llm_fqn_per_model_part,
    )
    from torchtitan.models.kimi_k3 import model_registry
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    config.model_spec = model_registry("debugmodel_33_layers")
    config.parallelism.pipeline_parallel_degree = 8
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.num_pp_microbatches = 8
    split = _generate_llm_fqn_per_model_part(
        8 * 4, 33, last_stage_modules=("output_res_proj", "output_res_norm")
    )
    split[0].insert(0, "vision_encoder")
    config.parallelism.module_fqns_per_model_part = split
    return config
