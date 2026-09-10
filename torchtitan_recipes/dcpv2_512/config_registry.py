# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Matched DeepSeek-V3 671B checkpoint producers on 256 H100 GPUs.

The package name is retained from the earlier 512-GPU checkpoint panel so its
module path remains stable. The current producer topology uses 256 GPUs.
"""

from dataclasses import replace

from torchtitan.components.checkpointer import BaseCheckpointManager, CheckpointManager
from torchtitan.components.checkpointer.torch_checkpointing import (
    TorchCheckpointingManager,
)
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceStreamingSource,
)
from torchtitan.config import DebugConfig, ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_671b,
    deepseek_v3_debugmodel,
)
from torchtitan.trainer import Trainer


CHECKPOINT_INTERVAL = 50
TRAINING_STEPS = 151
HF_ASSETS = "/mnt/mffuse/deepseek-v3/DeepSeek-V3.1-Base"
C4_PATH = "/mnt/mffuse/c4"


def _producer_topology() -> ParallelismConfig:
    """Return the 256-rank DP256, TP1, EP64 producer topology."""
    return ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=256,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=64,
        enable_sequence_parallel=False,
    )


def _base_producer() -> Trainer.Config:
    """Build the shared producer configuration before selecting a backend."""
    config = deepseek_v3_671b()
    config.hf_assets_path = HF_ASSETS
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    config.dataloader.dataset = ConcatThenSplitPackingConfig(
        dataset=replace(
            DATASETS["c4"],
            source=HuggingFaceStreamingSource.Config(
                path=C4_PATH,
                name="en",
                split="train",
            ),
        )
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = 4 * 4096
    config.training.max_context_length = 4096
    config.training.steps = TRAINING_STEPS
    config.training.dtype = "bfloat16"
    config.training.disable_cuda_graphs = True
    config.lr_scheduler.total_steps = 10000
    config.parallelism = _producer_topology()
    config.debug = DebugConfig(
        seed=42,
        deterministic=True,
        moe_force_load_balance=True,
    )
    config.metrics.log_freq = 1
    config.metrics.enable_tensorboard = True
    config.activation_checkpoint = FullAC.Config()
    return config


def dsv3_671b_save_dcp() -> Trainer.Config:
    """Produce interval-50 native DCP checkpoints on 256 H100 GPUs."""
    config = _base_producer()
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=CHECKPOINT_INTERVAL,
        async_mode="async_with_pinned_mem",
        keep_latest_k=2,
        initial_load_model_only=False,
        last_save_model_only=True,
        last_save_in_hf=True,
        export_dtype="bfloat16",
    )
    return config


def dsv3_671b_save_dcpv2() -> Trainer.Config:
    """Produce matched interval-50 DCPv2 checkpoints on 256 H100 GPUs."""
    config = _base_producer()
    config.checkpoint = TorchCheckpointingManager.Config(
        enable=True,
        interval=CHECKPOINT_INTERVAL,
        keep_latest_k=2,
        initial_load_model_only=False,
        last_save_model_only=True,
        last_save_in_hf=True,
        export_dtype="bfloat16",
    )
    return config


def _debug_smoke(
    checkpoint_config: BaseCheckpointManager.Config,
) -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.training.steps = 3
    config.training.dtype = "bfloat16"
    config.training.disable_cuda_graphs = True
    config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=8,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=2,
        enable_sequence_parallel=False,
    )
    config.debug = DebugConfig(
        seed=42,
        deterministic=True,
        moe_force_load_balance=True,
    )
    config.activation_checkpoint = FullAC.Config()
    config.checkpoint = checkpoint_config
    return config


def dsv3_debug_save_dcp() -> Trainer.Config:
    """Exercise periodic and final native DCP saves on one H100 host."""
    return _debug_smoke(
        CheckpointManager.Config(
            enable=True,
            interval=2,
            async_mode="async_with_pinned_mem",
            keep_latest_k=2,
            initial_load_model_only=False,
            last_save_model_only=True,
            last_save_in_hf=True,
            export_dtype="bfloat16",
        )
    )


def dsv3_debug_save_dcpv2() -> Trainer.Config:
    """Exercise periodic and final DCPv2 saves on one H100 host."""
    return _debug_smoke(
        TorchCheckpointingManager.Config(
            enable=True,
            interval=2,
            keep_latest_k=2,
            initial_load_model_only=False,
            last_save_model_only=True,
            last_save_in_hf=True,
            export_dtype="bfloat16",
        )
    )
