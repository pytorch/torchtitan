# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from datasets import load_dataset

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
    TextCollator,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.components.quantization import Float8LinearConverter
from torchtitan.components.validate import Validator
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.hf_datasets.text_datasets import (
    ChatProcessor,
    DATASETS,
    HuggingFaceTextProcessor,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer

from . import model_registry


def llama3_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    dataset = DATASETS["c4_test"]
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_spec,
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceRandomAccessSource.Config(
                        path=dataset.path,
                        loader=dataset.loader,
                    ),
                    process=HuggingFaceTextProcessor.Config(
                        text_processor=dataset.sample_processor,
                    ),
                ),
            ),
            collator=TextCollator.Config(),
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(pipeline_parallel_schedule="Interleaved1F1B"),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
        validator=Validator.Config(
            freq=5,
            steps=10,
            dataloader=GrainDataLoader.Config(
                dataset=ConcatThenSplitPackingConfig(
                    dataset=SingleDatasetConfig(
                        source=HuggingFaceRandomAccessSource.Config(
                            path=dataset.path,
                            loader=dataset.loader,
                        ),
                        process=HuggingFaceTextProcessor.Config(
                            text_processor=dataset.sample_processor,
                        ),
                    ),
                ),
                collator=TextCollator.Config(),
            ),
        ),
    )


def llama3_debugmodel_varlen_attn() -> Trainer.Config:
    config = llama3_debugmodel()
    config.model_spec = model_registry("debugmodel", attn_backend="varlen")
    return config


def llama3_debugmodel_float8() -> Trainer.Config:
    config = llama3_debugmodel()
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            Float8LinearConverter.Config(model_compile_enabled=model_compile_enabled),
        ],
    )
    return config


def llama3_debugmodel_float8_emulate_lora() -> Trainer.Config:
    from torchtitan.components.lora import LoRAConverter

    config = llama3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            Float8LinearConverter.Config(
                emulate=True,
                model_compile_enabled=False,
            ),
            LoRAConverter.Config(
                rank=8, alpha=16.0, target_modules=["wq", "wkv", "wo"]
            ),
        ],
    )
    return config


def llama3_debugmodel_ce_loss() -> Trainer.Config:
    """Debug model with standard (non-chunked) CrossEntropyLoss."""
    config = llama3_debugmodel()
    assert config.model_spec is not None
    config.loss = CrossEntropyLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model_spec),
    )
    return config


def llama3_8b() -> Trainer.Config:
    model_spec = model_registry("8B")
    dataset = DATASETS["c4"]
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Llama-3.1-8B",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_spec,
        optimizer=default_adamw(lr=3e-4),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=1000,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceStreamingSource.Config(
                        path=dataset.path,
                        loader=dataset.loader,
                    ),
                    process=HuggingFaceTextProcessor.Config(
                        text_processor=dataset.sample_processor,
                    ),
                ),
            ),
            collator=TextCollator.Config(),
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=SelectiveAC.Config(),
        validator=Validator.Config(
            freq=500,
            steps=1200,
        ),
    )


def llama3_70b() -> Trainer.Config:
    model_spec = model_registry("70B")
    dataset = DATASETS["c4"]
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Llama-3.1-70B",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_spec,
        optimizer=default_adamw(lr=1.5e-4),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=8192,
            steps=1000,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceStreamingSource.Config(
                        path=dataset.path,
                        loader=dataset.loader,
                    ),
                    process=HuggingFaceTextProcessor.Config(
                        text_processor=dataset.sample_processor,
                    ),
                ),
            ),
            collator=TextCollator.Config(),
        ),
        parallelism=ParallelismConfig(
            tensor_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=FullAC.Config(),
        validator=Validator.Config(
            freq=500,
            steps=1200,
        ),
    )


def llama3_405b() -> Trainer.Config:
    compile_config = CompileConfig(enable=True)
    dataset = DATASETS["c4"]
    model_spec = model_registry(
        "405B",
        converters=[
            Float8LinearConverter.Config(
                filter_fqns=["output"],
                model_compile_enabled=(
                    compile_config.enable and "model" in compile_config.components
                ),
            ),
        ],
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Llama-3.1-405B",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_spec,
        optimizer=default_adamw(lr=8e-5),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=8192,
            steps=3000,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceStreamingSource.Config(
                        path=dataset.path,
                        loader=dataset.loader,
                    ),
                    process=HuggingFaceTextProcessor.Config(
                        text_processor=dataset.sample_processor,
                    ),
                ),
            ),
            collator=TextCollator.Config(),
        ),
        parallelism=ParallelismConfig(
            tensor_parallel_degree=8,
            enable_async_tensor_parallel=True,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=FullAC.Config(),
        compile=compile_config,
        validator=Validator.Config(
            freq=500,
            steps=1200,
        ),
    )


def sft_debugmodel() -> Trainer.Config:
    """SFT debug config with Llama3 debugmodel and local test data."""

    def process_sample(sample):
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]

    model_spec = model_registry("debugmodel", attn_backend="flex")

    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_spec,
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=FirstFitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceRandomAccessSource.Config(
                        path="json",
                        loader=partial(
                            load_dataset,
                            data_files="tests/assets/sft_test/data.json",
                            split="train",
                        ),
                    ),
                    process=ChatProcessor.Config(sample_processor=process_sample),
                    filters=(lambda sample: sample is not None,),
                ),
            ),
            collator=TextCollator.Config(),
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )
