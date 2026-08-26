# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_16b_minimal_async_ep,
)
from torchtitan.trainer import Trainer


def dsv3_16b_minimal_async_ep_fake_profile() -> Trainer.Config:
    config = deepseek_v3_16b_minimal_async_ep()
    config.hf_assets_path = "./tests/assets/tokenizer"
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.training.dtype = "bfloat16"
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.max_context_length = 512
    config.training.steps = 2
    config.training.disable_cuda_graphs = True
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.expert_parallel_degree = 8
    config.activation_checkpoint = FullAC.Config()
    config.compile.enable = False
    config.debug.moe_force_load_balance = True
    config.metrics.log_freq = 1
    return config
