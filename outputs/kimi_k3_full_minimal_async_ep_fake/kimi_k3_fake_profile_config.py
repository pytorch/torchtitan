# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel
from torchtitan.trainer import Trainer


def kimi_k3_full_minimal_async_ep_fake_profile() -> Trainer.Config:
    config = kimi_k3_debugmodel()
    config.model_spec = model_registry(
        "Kimi-K3",
        attn_backend="flex",
        moe_comm_backend="minimal_async_ep",
    )
    config.hf_assets_path = "./tests/assets/tokenizer"
    config.dataloader.collator.max_images_per_batch = 128
    config.training.dtype = "bfloat16"
    config.training.num_tokens_per_microbatch_per_dp_rank = 4096
    config.training.num_tokens_per_train_step = 4096 * 256
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
        spmd_backend="partial_dtensor",
    )
    config.activation_checkpoint = FullAC.Config()
    config.compile.enable = False
    config.debug.moe_force_load_balance = True
    config.metrics.log_freq = 1
    config.checkpoint.enable = False
    return config
