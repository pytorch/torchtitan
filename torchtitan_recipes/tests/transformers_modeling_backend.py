# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base Transformers modeling backend configurations used by tests."""

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.context_parallel import PTRRFlexAttentionCPLoadBalancer
from torchtitan.experiments.transformers_modeling_backend import build_model_config
from torchtitan.experiments.transformers_modeling_backend.tokenizer import (
    HFBackendTokenizer,
)
from torchtitan.hf_datasets.text_datasets import ChatProcessor, DATASETS
from torchtitan.models.common.config_utils import DEFAULT_DEBUG_MODEL_SEQ_LEN
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer


def transformers_modeling_backend_debugmodel(
    seq_len: int = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    deterministic: bool = False,
) -> Trainer.Config:
    hf_model = "Qwen/Qwen3-4B-Instruct-2507"
    model_config = build_model_config(
        "debugmodel",
        seq_len=seq_len,
        hf_model=hf_model,
        deterministic=deterministic,
    )
    return Trainer.Config(
        loss=CrossEntropyLoss.Config(global_vocab_size=model_config.vocab_size),
        hf_assets_path="./tests/assets/tokenizer",
        debug=DebugConfig(print_config=True, deterministic=deterministic),
        model=model_config,
        profiler=Profiler.Config(profile_freq=5),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
            max_context_length=seq_len,
            steps=10,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="1F1B",
        ),
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
    )


def transformers_modeling_backend_debugmodel_compile(
    seq_len: int = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    deterministic: bool = False,
) -> Trainer.Config:
    config = transformers_modeling_backend_debugmodel(
        seq_len=seq_len,
        deterministic=deterministic,
    )
    return config


def transformers_modeling_backend_debugmodel_moe(
    seq_len: int = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    deterministic: bool = False,
) -> Trainer.Config:
    hf_model = "Qwen/Qwen3-30B-A3B"
    model_config = build_model_config(
        "debugmodel_moe",
        seq_len=seq_len,
        hf_model=hf_model,
        deterministic=deterministic,
    )
    return Trainer.Config(
        loss=CrossEntropyLoss.Config(global_vocab_size=model_config.vocab_size),
        hf_assets_path="./tests/assets/tokenizer",
        debug=DebugConfig(print_config=True, deterministic=deterministic),
        model=model_config,
        profiler=Profiler.Config(profile_freq=5),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
            max_context_length=seq_len,
            steps=10,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="1F1B",
        ),
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
    )


def transformers_modeling_backend_sft_debugmodel(
    seq_len: int = 1024,
) -> Trainer.Config:
    """SFT debug config for the transformers backend."""

    def process_sample(sample):
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]

    hf_model = "Qwen/Qwen3-4B-Instruct-2507"
    model_config = build_model_config(
        "sft_debugmodel", seq_len=seq_len, hf_model=hf_model
    )
    return Trainer.Config(
        loss=CrossEntropyLoss.Config(global_vocab_size=model_config.vocab_size),
        hf_assets_path="./tests/assets/tokenizer",
        model=model_config,
        tokenizer=HFBackendTokenizer.Config(),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            # Keep this small: this debug model uses the full Qwen3 vocab
            # (~152k), so cross-entropy materializes a num_tokens * vocab
            # logits tensor. 16384 tokens is ~9GB in fp32 and OOMs the 22GB
            # CI GPUs.
            num_tokens_per_microbatch_per_dp_rank=1 * seq_len,
            max_context_length=seq_len,
            steps=10,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=FirstFitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceRandomAccessSource.Config(
                        path="json",
                        split="train",
                        load_dataset_kwargs={
                            "data_files": "tests/assets/sft_test/data.json",
                        },
                    ),
                    processor=ChatProcessor.Config(messages_fn=process_sample),
                    post_filters=(lambda sample: sample is not None,),
                ),
            ),
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="1F1B",
        ),
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
    )


def transformers_backend_moe_fsdp_tp_ep_cp() -> Trainer.Config:
    config = transformers_modeling_backend_debugmodel_moe()
    config.parallelism.data_parallel_shard_degree = -1
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = (
        PTRRFlexAttentionCPLoadBalancer.Config()
    )
    config.training.disable_cuda_graphs = True
    config.training.steps = 2
    return config


def transformers_backend_dense_fsdp_tp_pp() -> Trainer.Config:
    config = transformers_modeling_backend_debugmodel()
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 2
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.disable_cuda_graphs = True
    config.training.steps = 2
    return config


def transformers_backend_dense_cp_pp() -> Trainer.Config:
    config = transformers_modeling_backend_debugmodel()
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.context_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 2
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.parallelism.context_parallel_load_balancer = (
        PTRRFlexAttentionCPLoadBalancer.Config()
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.disable_cuda_graphs = True
    config.training.steps = 2
    return config


def transformers_backend_sft() -> Trainer.Config:
    config = transformers_modeling_backend_sft_debugmodel()
    config.training.steps = 2
    return config
