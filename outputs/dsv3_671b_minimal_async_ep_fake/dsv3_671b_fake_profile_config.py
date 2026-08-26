# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.config import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.deepseek_v3 import (
    model_registry as graph_trainer_model_registry,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.deepseek_v3 import model_registry
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_671b,
    enable_fused_swiglu,
)
from torchtitan.trainer import Trainer


def dsv3_671b_minimal_async_ep_fake_profile() -> Trainer.Config:
    # Varlen attention otherwise prioritizes cuDNN over the registered FA4
    # provider. Disable only the cuDNN SDPA path for this FA4 benchmark.
    torch.backends.cuda.enable_cudnn_sdp(False)
    config = deepseek_v3_671b()
    config.model_spec = model_registry(
        "671B",
        attn_backend="varlen",
        moe_comm_backend="minimal_async_ep",
    )
    enable_fused_swiglu(config)
    config.override.imports.append("torchtitan.overrides.fused_mla.fused_mla")
    config.hf_assets_path = "./tests/assets/tokenizer"
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.training.dtype = "bfloat16"
    config.training.num_tokens_per_microbatch_per_dp_rank = 8 * 4096
    config.training.max_context_length = 4096
    config.training.steps = 20
    config.training.disable_cuda_graphs = True
    config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=256,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=64,
        enable_sequence_parallel=False,
    )
    config.activation_checkpoint = FullAC.Config()
    config.compile.enable = False
    config.debug.moe_force_load_balance = True
    config.metrics.log_freq = 1
    return config


def dsv3_671b_minimal_async_ep_fake_graph_profile() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        dsv3_671b_minimal_async_ep_fake_profile(),
        graph_trainer_model_registry,
    )
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        memory_policy="full",
        disable_passes=[
            "joint_transformer_block_bucketing_reordering_pass",
            "cudagraph_pass",
        ],
    )
    return config
