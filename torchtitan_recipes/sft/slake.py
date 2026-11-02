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

from renderers import Message

from torchtitan.components.checkpointer.torch_checkpointing import (
    TorchCheckpointingManager,
)
from torchtitan.components.data import (
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.validator import Validator
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LoRATransform
from torchtitan.models.muse_spark import (
    build_model_config,
    MUSE_SPARK_VISION_PROCESSOR_KWARGS,
)
from torchtitan.models.muse_spark.model import MuseSparkModel
from torchtitan.models.muse_spark.renderer import MuseSparkRendererConfig
from torchtitan.observability.metrics import TensorBoardLogger
from torchtitan.trainer import Trainer

from torchtitan_recipes.models.muse_spark import (
    apply_gb300_lora_dist_moe,
    muse_spark_816b_a42b,
    MUSE_SPARK_LORA_DECODER_TARGETS,
    muse_spark_mm_dataloader,
    use_flex_flash_attention,
)


_SLAKE_DATASET_PATH = "./assets/datasets/slake"
# The shortest English SLAKE sample is 219 tokens (64 image tokens).
_MUSE_SPARK_MIN_SAMPLE_TOKENS = 219
# At DP 8 each rank packs 25 to 32 English validation microbatches of 4096 tokens
# (214 in total), counted on CPU with the recipe's data path. Re-measure when the
# DP degree changes.
_MUSE_SPARK_VALIDATION_STEPS = 32
# conv1_linear (the patch embedding) stays frozen.
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
# the model answers in full sentences, so <|eot|> after a one-word label is very
# unlikely. Training also teaches it to answer to=user directly when the prompt
# is present. Eval and RL prompts must use the same text. RL that keeps direct
# answers needs it too. RL that should reason before answering conflicts with this
# SFT: drop the prompt there, or stop training the to=user header here.
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


def _slake_dataset(root: str) -> SingleDatasetConfig:
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
            # low (Reasoning strength 8) is the lowest level the published Muse Spark
            # chat template supports.
            renderer=MuseSparkRendererConfig(reasoning_effort="low", current_date=""),
        ),
        post_filters=(lambda sample: sample is not None,),
    )


def _use_slake_sft_data(config: Trainer.Config) -> None:
    """Train on English SLAKE with vision as the only dataset modality.

    Validation loss on the English validation split runs at step 1 and every 10
    steps; training loss alone hides overfitting on this small dataset.
    ``MUSE_SPARK_SLAKE_PATH`` overrides the SLAKE root, which holds
    ``train.json``, ``validation.json`` and ``imgs/``.
    """
    from torchtitan.components.data.multimodal.mm_datasets import MMSamplePackingConfig

    model_config = config.model
    assert isinstance(model_config, MuseSparkModel.Config)
    model_config = replace(model_config, dataset_modalities=("vision",))
    config.model = model_config
    config.vision_processor = VisionProcessor.Config(
        **MUSE_SPARK_VISION_PROCESSOR_KWARGS
    )
    root = os.environ.get("MUSE_SPARK_SLAKE_PATH", _SLAKE_DATASET_PATH)
    loader = muse_spark_mm_dataloader(model_config, _slake_dataset(root))
    num_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
    config.dataloader = replace(
        loader,
        dataset=MMSamplePackingConfig(dataset=loader.dataset),
        collator=replace(
            loader.collator,
            max_images_per_microbatch=num_tokens // _MUSE_SPARK_MIN_SAMPLE_TOKENS,
        ),
    )
    # Same dataset config as training (processor, image resize and normalization)
    # with validation.json as the source.
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
        steps=_MUSE_SPARK_VALIDATION_STEPS,
        dataloader=replace(
            config.dataloader,
            dataset=MMSamplePackingConfig(dataset=validation_dataset),
        ),
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


def _slake_lora_transform(variant: _SlakeVariant) -> LoRATransform:
    targets = list(MUSE_SPARK_LORA_DECODER_TARGETS)
    if variant == "lora_vision_decoder":
        targets += _LORA_VISION_TARGETS
    return LoRATransform(rank=8, alpha=16.0, target_modules=targets)


def _muse_spark_816b_a42b_slake_sft(variant: _SlakeVariant) -> Trainer.Config:
    """Configure English SLAKE LoRA SFT from released Muse Spark Open weights.

    Stage ``meta-models/Muse-Spark-Open-816B-A42B`` at
    ``./assets/hf/Muse-Spark-Open-816B-A42B``, or set
    ``MUSE_SPARK_HF_ASSETS_PATH``. See ``_use_slake_sft_data`` for the data path.
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
    config.model = build_model_config(
        "816B-A42B_mm",
        parallelism=config.parallelism,
        seq_len=seq_len,
        moe_comm_backend="standard",
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
    _use_slake_sft_data(config)
    # LoRA freezes the router gates that the default optimizer groups match.
    config.optim = _slake_lora_optim()
    use_flex_flash_attention(config.model)
    checkpointer = config.checkpointer
    assert isinstance(checkpointer, TorchCheckpointingManager.Config)
    checkpointer.folder = f"muse_spark_slake_{variant}"
    return apply_gb300_lora_dist_moe(config, _slake_lora_transform(variant))


def muse_spark_816b_a42b_slake_lora_decoder_sft() -> Trainer.Config:
    return _muse_spark_816b_a42b_slake_sft("lora_decoder")


def muse_spark_816b_a42b_slake_lora_vision_decoder_sft() -> Trainer.Config:
    return _muse_spark_816b_a42b_slake_sft("lora_vision_decoder")
