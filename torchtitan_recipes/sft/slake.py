# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SLAKE medical VQA LoRA SFT recipes."""

import functools
import os
from dataclasses import replace
from typing import Any, cast, Literal

from renderers import CustomRendererConfig, Message

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.checkpointer.torch_checkpointing import (
    TorchCheckpointingManager,
)
from torchtitan.components.data import (
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.renderer import from_renderers, RendererConfig
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.components.validator import Validator
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import apply_transforms, LoRATransform
from torchtitan.models.muse_glimmer import (
    build_model_config as build_muse_glimmer_model_config,
    MUSE_GLIMMER_30B_SPECIAL_TOKENS,
    MUSE_GLIMMER_30B_VISION_PROCESSOR_KWARGS,
)
from torchtitan.models.muse_spark import (
    build_model_config as build_muse_spark_model_config,
    MUSE_SPARK_VISION_PROCESSOR_KWARGS,
)
from torchtitan.models.muse_spark.model import MuseSparkModel
from torchtitan.models.muse_spark.renderer import MuseSparkRendererConfig
from torchtitan.observability.metrics import TensorBoardLogger
from torchtitan.trainer import Trainer

from torchtitan_recipes.models.muse_glimmer import muse_glimmer_30b
from torchtitan_recipes.models.muse_spark import (
    apply_gb300_lora_dist_moe,
    muse_spark_816b_a42b,
    MUSE_SPARK_LORA_DECODER_TARGETS,
    muse_spark_mm_dataloader,
    use_flex_flash_attention,
)


_SLAKE_DATASET_PATH = "./assets/datasets/slake"
# The shortest English SLAKE sample and the English validation microbatches of
# 4096 tokens per DP rank depend on the chat format, so they are per model.
# Validation steps are the largest per-rank count, counted on CPU with the
# recipe's data path. Re-measure when the DP degree changes.
# Muse Spark: 219 tokens (64 image tokens); 25 to 32 microbatches per rank at DP 8.
_MUSE_SPARK_MIN_SAMPLE_TOKENS = 219
_MUSE_SPARK_VALIDATION_STEPS = 32
# Muse Glimmer: 142 tokens (64 image tokens); 46 to 53 microbatches per rank at
# DP 4.
_MUSE_GLIMMER_MIN_SAMPLE_TOKENS = 142
_MUSE_GLIMMER_VALIDATION_STEPS = 53
_MUSE_GLIMMER_LORA_DECODER_TARGETS = ["wqkv", "wo", "o_gate", "w13", "w2"]
# Both models use the Muse Spark NaViT vision tower. conv1_linear (the patch
# embedding) stays frozen.
_LORA_VISION_TARGETS = [
    "vision_encoder.block.attn.wq",
    "vision_encoder.block.attn.wk",
    "vision_encoder.block.attn.wv",
    "vision_encoder.block.attn.proj",
    "vision_encoder.block.mlp.fc1",
    "vision_encoder.block.mlp.fc2",
    "vision_adapter.c_fc",
    "vision_adapter.c_proj",
    "vision_projection",
]
_SlakeVariant = Literal["lora_decoder", "lora_vision_decoder"]
# SLAKE answers are single words or short phrases. Without this LLaVA 1.5 prompt
# the model answers in full sentences, so the end-of-turn token after a one-word
# label is very unlikely. Eval and RL prompts must use the same text. RL that keeps
# direct answers needs it too. RL that should reason before answering conflicts
# with this SFT: drop the prompt there.
_SLAKE_ANSWER_INSTRUCTION = "Answer the question using a single word or phrase."


def _slake_messages(row: dict[str, Any], *, root: str) -> list[Message]:
    """Return a user turn with the image, question and answer instruction, and
    the answer turn."""
    image_path = os.path.join(root, "imgs", row["img_name"])
    return cast(
        list[Message],
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {
                        "type": "text",
                        "text": f"{row['question']}\n{_SLAKE_ANSWER_INSTRUCTION}",
                    },
                ],
            },
            {"role": "assistant", "content": str(row["answer"])},
        ],
    )


def _slake_dataset(root: str, *, renderer: RendererConfig) -> SingleDatasetConfig:
    """Return English SLAKE train rows as chat samples. ``renderer`` decides which
    tokens get loss.

    ``root`` holds ``train.json``, ``validation.json`` and ``imgs/``.
    """
    from torchtitan.components.data.multimodal.mm_datasets import (
        MultiModalChatProcessor,
    )

    # TODO: Drop train rows whose image is a duplicate or an adjacent slice of a
    # validation or test image (same scan series), so test cases do not leak.
    return SingleDatasetConfig(
        source=HuggingFaceRandomAccessSource.Config(
            path="json",
            split="train",
            load_dataset_kwargs={"data_files": os.path.join(root, "train.json")},
        ),
        pre_filters=(lambda row: row["q_lang"] == "en",),
        processor=MultiModalChatProcessor.Config(
            messages_fn=functools.partial(_slake_messages, root=root),
            renderer=renderer,
        ),
        post_filters=(lambda sample: sample is not None,),
    )


def _use_slake_sft_data(
    config: Trainer.Config,
    loader: GrainDataLoader.Config,
    *,
    root: str,
    min_sample_tokens: int,
    validation_steps: int,
) -> None:
    """Set packed SLAKE training data and English validation loss.

    ``loader`` reads ``_slake_dataset(root, ...)``; ``config.vision_processor``
    holds the model's image settings.
    Validation loss runs at step 1 and every 10 steps; training loss alone hides
    overfitting on this small dataset.
    """
    from torchtitan.components.data.multimodal.mm_datasets import MMSamplePackingConfig

    num_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
    config.dataloader = replace(
        loader,
        dataset=MMSamplePackingConfig(dataset=loader.dataset),
        collator=replace(
            loader.collator,
            max_images_per_microbatch=num_tokens // min_sample_tokens,
        ),
    )
    # Same dataset config as training with validation.json as the source.
    train_dataset = loader.dataset
    assert isinstance(train_dataset, SingleDatasetConfig)
    validation_dataset = replace(
        train_dataset,
        source=replace(
            train_dataset.source,
            load_dataset_kwargs={"data_files": os.path.join(root, "validation.json")},
        ),
    )
    config.validator = Validator.Config(
        freq=10,
        # One step is one microbatch per DP rank. Steps cover the largest rank
        # share once; ranks with fewer microbatches wrap around (repeat=True).
        steps=validation_steps,
        dataloader=replace(
            config.dataloader,
            dataset=MMSamplePackingConfig(dataset=validation_dataset),
        ),
    )


def _use_muse_spark_slake_sft_data(config: Trainer.Config) -> None:
    """Train on English SLAKE with vision as the only dataset modality.

    ``MUSE_SPARK_SLAKE_PATH`` overrides the SLAKE root. See ``_use_slake_sft_data``
    for packing and validation.
    """
    model_config = config.model
    assert isinstance(model_config, MuseSparkModel.Config)
    model_config = replace(model_config, dataset_modalities=("vision",))
    config.model = model_config
    config.vision_processor = VisionProcessor.Config(
        **MUSE_SPARK_VISION_PROCESSOR_KWARGS
    )
    root = os.environ.get("MUSE_SPARK_SLAKE_PATH", _SLAKE_DATASET_PATH)
    # low (Reasoning strength 8) is the lowest level the published Muse Spark chat
    # template supports. With the short-answer prompt in the user turn, training
    # also teaches the model to answer to=user directly. RL that should reason
    # before answering conflicts with this: drop the prompt there, or stop training
    # the to=user header here.
    dataset = _slake_dataset(
        root,
        renderer=MuseSparkRendererConfig(reasoning_effort="low", current_date=""),
    )
    _use_slake_sft_data(
        config,
        muse_spark_mm_dataloader(model_config, dataset),
        root=root,
        min_sample_tokens=_MUSE_SPARK_MIN_SAMPLE_TOKENS,
        validation_steps=_MUSE_SPARK_VALIDATION_STEPS,
    )


def _slake_lora_optim() -> Optim.Config:
    return Optim.Config(
        optimizer=OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(
                    pattern=r".*",
                    lr=1e-4,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=0.0,
                )
            ]
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=10,
            decay_ratio=0.1,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
    )


def _slake_lora_transform(
    variant: _SlakeVariant, *, decoder_targets: list[str]
) -> LoRATransform:
    targets = list(decoder_targets)
    if variant == "lora_vision_decoder":
        targets += _LORA_VISION_TARGETS
    return LoRATransform(rank=8, alpha=16.0, target_modules=targets)


def _muse_spark_816b_a42b_slake_sft(variant: _SlakeVariant) -> Trainer.Config:
    """Configure English SLAKE LoRA SFT from released Muse Spark Open weights.

    Stage ``meta-models/Muse-Spark-Open-816B-A42B`` at
    ``./assets/hf/Muse-Spark-Open-816B-A42B``, or set
    ``MUSE_SPARK_HF_ASSETS_PATH``. See ``_use_muse_spark_slake_sft_data`` for the
    data path.
    Learning rates are untuned starting points.
    """
    seq_len = 4096  # The longest English SLAKE sample is 1553 tokens.
    # LoRA runs on 8 GB300 GPUs (Dist-MoE and FA4 need SM100). A fake-PG run
    # (random init, forced balanced routing) of this layout used 254 GiB
    # allocated / 261 GiB reserved of 276.6 GiB per GPU. Use dp_shard=16 and
    # EP=16 for more headroom.
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=8,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        expert_parallel_degree=8,
        pipeline_parallel_degree=1,
        enable_sequence_parallel=False,
    )
    config = muse_spark_816b_a42b(seq_len, parallelism=parallelism)
    config.model = build_muse_spark_model_config(
        "816B-A42B_mm",
        parallelism=config.parallelism,
        seq_len=seq_len,
        attn_backend="flex",
    )
    config.hf_assets_path = os.environ.get(
        "MUSE_SPARK_HF_ASSETS_PATH", config.hf_assets_path
    )
    # One pass over the English train split is 28.7 steps of 32 x 4096 tokens.
    # LoRA vision + decoder validation loss was lowest after about 2 passes (step
    # 60 of a 90-step run) and rose after that.
    config.training.steps = 60
    config.metrics.loggers = [TensorBoardLogger.Config()]
    _use_muse_spark_slake_sft_data(config)
    # LoRA freezes the router gates that the default optimizer groups match.
    config.optim = _slake_lora_optim()
    use_flex_flash_attention(config.model)
    checkpointer = config.checkpointer
    assert isinstance(checkpointer, TorchCheckpointingManager.Config)
    checkpointer.folder = f"muse_spark_slake_{variant}"
    return apply_gb300_lora_dist_moe(
        config,
        _slake_lora_transform(variant, decoder_targets=MUSE_SPARK_LORA_DECODER_TARGETS),
    )


def muse_spark_816b_a42b_slake_lora_decoder_sft() -> Trainer.Config:
    return _muse_spark_816b_a42b_slake_sft("lora_decoder")


def muse_spark_816b_a42b_slake_lora_vision_decoder_sft() -> Trainer.Config:
    return _muse_spark_816b_a42b_slake_sft("lora_vision_decoder")


def _muse_glimmer_30b_slake_sft(variant: _SlakeVariant) -> Trainer.Config:
    """Configure English SLAKE LoRA SFT from the released Muse Glimmer 30B weights.

    Stage the HF checkpoint at ``./assets/hf/Muse-Glimmer-30B``, or set
    ``MUSE_GLIMMER_HF_ASSETS_PATH``. ``MUSE_GLIMMER_SLAKE_PATH`` overrides the SLAKE
    root. See ``_use_slake_sft_data`` for packing and validation. Learning rates
    are untuned starting points.
    """
    seq_len = 4096  # The longest English SLAKE sample is 1476 tokens.
    config = muse_glimmer_30b(seq_len)
    # LoRA on the dense 30B model runs with FSDP on one node of 4 GB300 GPUs.
    config.parallelism = ParallelismConfig(
        data_parallel_shard_degree=4,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
    )
    config.model = build_muse_glimmer_model_config(
        "30B_mm", seq_len=seq_len, parallelism=config.parallelism
    )
    config.hf_assets_path = os.environ.get(
        "MUSE_GLIMMER_HF_ASSETS_PATH", config.hf_assets_path
    )
    config.tokenizer = MultiModalTokenizer.Config(**MUSE_GLIMMER_30B_SPECIAL_TOKENS)
    config.vision_processor = VisionProcessor.Config(
        **MUSE_GLIMMER_30B_VISION_PROCESSOR_KWARGS
    )
    # No remat: peak 116 GiB of 276.6 GiB per GPU on 4 GB300, about 28% faster
    # than full remat.
    config.activation_remat = None
    # A step is 32 x 4096 tokens, as in the Muse Spark SLAKE recipes. One pass over
    # the English train split is 26.8 steps (859 microbatches at DP 4), so 90 steps
    # is about 3.4 passes. In a 90-step run validation loss went from 2.73 at step
    # 1 to 0.169 at step 90; a 150-step run plateaued after step 70 and did not
    # improve.
    config.training.num_tokens_per_train_step = 32 * seq_len
    config.training.steps = 90
    config.training.disable_accelerator_graphs = True
    config.metrics.log_freq = 1
    config.metrics.loggers = [TensorBoardLogger.Config()]
    config.optim = _slake_lora_optim()
    config.checkpointer = cast(
        CheckpointManager.Config,
        TorchCheckpointingManager.Config(
            folder=f"muse_glimmer_slake_{variant}",
            interval=2000,
            initial_load_model_only=True,
            initial_load_in_hf=True,
            last_save_model_only=False,
            keep_latest_k=2,
        ),
    )
    root = os.environ.get("MUSE_GLIMMER_SLAKE_PATH", _SLAKE_DATASET_PATH)
    dataset = _slake_dataset(
        root,
        renderer=from_renderers(
            CustomRendererConfig(
                import_path=(
                    "torchtitan.models.muse_glimmer.renderer.MuseGlimmerRenderer"
                ),
                reasoning_strength="low",
                # The chat template always renders a date line; pin it so the
                # prompt does not change from day to day.
                current_date="2026-01-04",
            )
        ),
    )
    loader = GrainDataLoader.Config(
        dataset=dataset,
        collator=MultiModalCollator.Config(build_mrope_positions=False),
    )
    _use_slake_sft_data(
        config,
        loader,
        root=root,
        min_sample_tokens=_MUSE_GLIMMER_MIN_SAMPLE_TOKENS,
        validation_steps=_MUSE_GLIMMER_VALIDATION_STEPS,
    )
    transform = _slake_lora_transform(
        variant, decoder_targets=_MUSE_GLIMMER_LORA_DECODER_TARGETS
    )
    return apply_transforms(config, [transform])


def muse_glimmer_30b_slake_lora_decoder_sft() -> Trainer.Config:
    return _muse_glimmer_30b_slake_sft("lora_decoder")


def muse_glimmer_30b_slake_lora_vision_decoder_sft() -> Trainer.Config:
    return _muse_glimmer_30b_slake_sft("lora_vision_decoder")
