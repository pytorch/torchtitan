# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

from torchtitan.components.data import ConcatThenSplitPackingConfig
from torchtitan.components.quantization import (
    MXFP8GroupedExpertsConverter,
    MXFP8LinearConverter,
)
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.deepseek_v3 import model_registry as deepseek_v3_model_registry
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_16b,
    deepseek_v3_16b_dist_moe_bf16,
    deepseek_v3_16b_minimal_async_ep,
    deepseek_v3_671b,
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_minimal_async_ep,
)

from . import model_registry


def graph_trainer_deepseek_v3_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(deepseek_v3_debugmodel(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_debugmodel_mxfp8() -> GraphTrainer.Config:
    base = deepseek_v3_debugmodel()
    # Quantize dense and moe gemms to mxfp8
    base.model_spec = deepseek_v3_model_registry(
        "debugmodel",
        converters=[
            MXFP8LinearConverter.Config(
                model_compile_enabled=True,
                fqns=["attention", "shared_experts", "feed_forward"],
            ),
            MXFP8GroupedExpertsConverter.Config(
                model_compile_enabled=True,
                pad_multiple=128,
            ),
        ],
    )
    config = to_graph_trainer_config(base, model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_debugmodel_hybridep() -> GraphTrainer.Config:
    config = to_graph_trainer_config(deepseek_v3_debugmodel(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    config.model_spec = model_registry(
        "debugmodel",
        moe_comm_backend="hybridep",
        non_blocking_capacity_factor=1.0,
    )
    return config


def graph_trainer_deepseek_v3_debugmodel_minimal_async_ep() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        deepseek_v3_debugmodel_minimal_async_ep(),
        model_registry,
    )
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_debugmodel_eager_pp() -> GraphTrainer.Config:
    """Test-only FlexAttention baseline that runs through eager pipeline parallelism."""
    config = graph_trainer_deepseek_v3_debugmodel()
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        components=["loss"],
        mode=None,
    )
    config.model_spec = replace(config.model_spec, pipelining_fn=pipeline_llm)
    return config


def graph_trainer_deepseek_v3_16b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(deepseek_v3_16b(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_16b_minimal_async_ep() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        deepseek_v3_16b_minimal_async_ep(),
        model_registry,
    )
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config


def graph_trainer_deepseek_v3_16b_dist_moe_bf16() -> GraphTrainer.Config:
    """Build the DSV3 16B GraphTrainer config with BF16 DistMoE."""
    base = deepseek_v3_16b_dist_moe_bf16()
    base.hf_assets_path = "./tests/assets/tokenizer"
    base.parallelism = replace(
        base.parallelism,
        data_parallel_shard_degree=2,
        expert_parallel_degree=2,
    )
    base.dataloader.dataset = ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
    config = to_graph_trainer_config(
        base,
        model_registry,
    )
    config.override.imports = [
        "torchtitan.overrides.helion_rope.helion_complex_rope",
    ]
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        components=[],
        inductor_compilation="regional",
        memory_policy="full",
        require_cudagraph=True,
        fsdp_contiguous_module_fqns=["layers.*.moe.routed_experts"],
    )
    return config


def graph_trainer_deepseek_v3_16b_coda() -> GraphTrainer.Config:
    base = deepseek_v3_16b_minimal_async_ep()
    base.hf_assets_path = "./tests/assets/tokenizer"
    base.parallelism = replace(
        base.parallelism,
        data_parallel_shard_degree=2,
        expert_parallel_degree=2,
    )
    base.dataloader.dataset = ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
    base.model_spec = deepseek_v3_model_registry(
        "16B",
        attn_backend="flex_flash",
        moe_comm_backend="minimal_async_ep",
    )
    base.override.imports = [
        "torchtitan.overrides.fused_swiglu.fused_grouped_experts",
        "torchtitan.overrides.helion_rope.helion_complex_rope",
    ]
    config = to_graph_trainer_config(base, model_registry)
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        enable_coda=True,
        inductor_compilation="regional",
        memory_policy="full",
        numerics_changing_optim=True,
    )
    return config


def graph_trainer_deepseek_v3_16b_sdpa() -> GraphTrainer.Config:
    config = graph_trainer_deepseek_v3_16b()
    config.parallelism.context_parallel_load_balancer = "headtail"
    config.model_spec = model_registry("16B", attn_backend="sdpa")
    return config


def graph_trainer_deepseek_v3_671b() -> GraphTrainer.Config:
    config = to_graph_trainer_config(deepseek_v3_671b(), model_registry)
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config
