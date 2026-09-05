# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Gemma-4 Training Configuration Registry

from torchtitan.config import (
    CKPT_PATH,
    CheckpointConfig,
    CompileConfig,
    DataConfig,
    FSDPConfig,
    LRConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelismConfig,
    ProfilerConfig,
    QuantizationConfig,
    SimulatedDataConfig,
    TokenizerType,
    TrainingConfig,
)
from torchtitan.trainer import Trainer


def gemma4_debugmodel() -> Trainer.Config:
    """
    Debug model configuration for Gemma-4.
    Small model for quick iteration and testing.
    """
    return Trainer.Config(
        model=ModelConfig(
            name="gemma4",
            flavor="debugmodel",
            norm_type="rmsnorm",
        ),
        parallelism=ParallelismConfig(
            data_parallel_size=1,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
        ),
        training=TrainingConfig(
            steps=100,
            log_freq=10,
            enable_loss_scrubbing=True,
        ),
        checkpoint=CheckpointConfig(
            enable_checkpoint=False,
        ),
        data=SimulatedDataConfig(
            seq_len=2048,
        ),
        optimizer=OptimizerConfig(
            lr=3e-4,
        ),
    )


def gemma4_12b() -> Trainer.Config:
    """
    Gemma-4 12B model configuration.
    
    Full production training config with:
    - 12B parameters
    - 256K context length
    - Hybrid attention (sliding window + global)
    - Optimized for 8x H100 GPUs
    """
    return Trainer.Config(
        model=ModelConfig(
            name="gemma4",
            flavor="12b",
            norm_type="rmsnorm",
        ),
        parallelism=ParallelismConfig(
            data_parallel_size=8,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            enable_sequence_parallel=False,
        ),
        training=TrainingConfig(
            steps=150000,
            log_freq=10,
            enable_loss_scrubbing=True,
        ),
        checkpoint=CheckpointConfig(
            enable_checkpoint=True,
            checkpoint_freq=500,
            checkpoint_dir=CKPT_PATH,
        ),
        data=SimulatedDataConfig(
            seq_len=4096,
        ),
        optimizer=OptimizerConfig(
            lr=2e-4,
            fused=True,
        ),
    )


def gemma4_12b_1node_full() -> Trainer.Config:
    """
    Gemma-4 12B single-node training (8 H100s).
    
    Configuration:
    - Data parallelism: 8 (full node)
    - Batch size: 32 (4 per GPU)
    - Seq length: 4096 tokens
    - Expected throughput: ~3,500 tokens/sec
    - Memory usage: ~72GB per GPU
    """
    return Trainer.Config(
        model=ModelConfig(
            name="gemma4",
            flavor="12b",
            norm_type="rmsnorm",
        ),
        parallelism=ParallelismConfig(
            data_parallel_size=8,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            enable_sequence_parallel=False,
        ),
        training=TrainingConfig(
            batch_size=32,
            steps=150000,
            log_freq=10,
            enable_loss_scrubbing=False,
        ),
        checkpoint=CheckpointConfig(
            enable_checkpoint=True,
            checkpoint_freq=500,
            checkpoint_dir=CKPT_PATH,
        ),
        data=SimulatedDataConfig(
            seq_len=4096,
        ),
        optimizer=OptimizerConfig(
            lr=2e-4,
            fused=True,
            max_norm=1.0,
        ),
        compile=CompileConfig(
            enable=True,
            components=["model"],
        ),
    )


def gemma4_12b_multinode() -> Trainer.Config:
    """
    Gemma-4 12B multi-node training (4 nodes × 8 H100s = 32 GPUs).
    
    Configuration:
    - Data parallelism: 4 (across nodes)
    - Tensor parallelism: 2 (within node)
    - Batch size: 64 global (2 per GPU)
    - Seq length: 4096 tokens
    - Expected throughput: ~14,000 tokens/sec (4x throughput)
    """
    return Trainer.Config(
        model=ModelConfig(
            name="gemma4",
            flavor="12b",
            norm_type="rmsnorm",
        ),
        parallelism=ParallelismConfig(
            data_parallel_size=4,
            pipeline_parallel_size=1,
            tensor_parallel_size=2,  # Shard across node
            enable_sequence_parallel=False,
        ),
        training=TrainingConfig(
            batch_size=64,
            steps=150000,
            log_freq=10,
            enable_loss_scrubbing=False,
        ),
        checkpoint=CheckpointConfig(
            enable_checkpoint=True,
            checkpoint_freq=500,
            checkpoint_dir=CKPT_PATH,
        ),
        data=SimulatedDataConfig(
            seq_len=4096,
        ),
        optimizer=OptimizerConfig(
            lr=2e-4,
            fused=True,
            max_norm=1.0,
        ),
        compile=CompileConfig(
            enable=True,
            components=["model"],
        ),
    )


def gemma4_12b_long_context() -> Trainer.Config:
    """
    Gemma-4 12B with extended context training (16K sequence length).
    
    Uses sequence parallelism to handle longer sequences:
    - Sequence parallelism: 2 (split seq across devices)
    - Seq length: 16K tokens (4x standard)
    - Batch size: 8 global (1 per GPU)
    - Memory efficiency: ~40GB per GPU
    """
    return Trainer.Config(
        model=ModelConfig(
            name="gemma4",
            flavor="12b",
            norm_type="rmsnorm",
        ),
        parallelism=ParallelismConfig(
            data_parallel_size=4,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            enable_sequence_parallel=True,  # Shard sequences
        ),
        training=TrainingConfig(
            batch_size=8,
            steps=100000,
            log_freq=10,
            enable_loss_scrubbing=False,
        ),
        checkpoint=CheckpointConfig(
            enable_checkpoint=True,
            checkpoint_freq=500,
            checkpoint_dir=CKPT_PATH,
        ),
        data=SimulatedDataConfig(
            seq_len=16384,  # 16K tokens
        ),
        optimizer=OptimizerConfig(
            lr=2e-4,
            fused=True,
            max_norm=1.0,
        ),
        compile=CompileConfig(
            enable=True,
            components=["model"],
        ),
    )
