# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset-specific defaults shared by training recipes."""

import functools
import logging
import math
import os
from dataclasses import replace
from typing import Any, cast

from renderers import Message

from torchtitan.components.data import (
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.renderer import RendererConfig
from torchtitan.components.validator import Validator
from torchtitan.observability.logging import warn_once
from torchtitan.trainer import Trainer


logger = logging.getLogger(__name__)


def c4_document_capacity(num_tokens_per_microbatch: int) -> int:
    """Estimate C4 document capacity from the per-rank token budget."""
    warn_once(
        logger,
        "C4 document capacity is an estimate. Tune dataloader.max_num_documents "
        "for your workload to balance attention overhead and packing padding.",
    )
    # Estimate 16 document slots per 8K tokens, with at least 16 for short batches.
    return max(16, (num_tokens_per_microbatch + 511) // 512)


# SLAKE medical VQA.

SLAKE_DATASET_PATH = "./assets/datasets/slake"
# SLAKE answers are single words or short phrases. Without this LLaVA 1.5 prompt
# the model answers in full sentences, so the end-of-turn token after a one-word
# label is very unlikely. Eval and RL prompts must use the same text. RL that keeps
# direct answers needs it too. RL that should reason before answering conflicts
# with this SFT: drop the prompt there.
_SLAKE_ANSWER_INSTRUCTION = "Answer the question using a single word or phrase."


def slake_messages(row: dict[str, Any], *, root: str) -> list[Message]:
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


def slake_dataset(root: str, *, renderer: RendererConfig) -> SingleDatasetConfig:
    """Return English SLAKE train rows as chat samples with loss on the answer only.

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
            messages_fn=functools.partial(slake_messages, root=root),
            renderer=renderer,
        ),
        post_filters=(lambda sample: sample is not None,),
    )


def use_slake_sft_data(
    config: Trainer.Config,
    loader: GrainDataLoader.Config,
    *,
    root: str,
    min_sample_tokens: int,
    num_validation_microbatches: int,
) -> None:
    """Set packed SLAKE training data and English validation loss.

    ``loader`` reads ``slake_dataset(root, ...)`` with the model's image settings.
    Validation loss runs at step 1 and every 10 steps; training loss alone hides
    overfitting on this small dataset. ``min_sample_tokens`` is the shortest
    English sample and ``num_validation_microbatches`` is the number of packed
    validation microbatches, both measured with the model's data path.
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
    parallelism = config.parallelism
    dp_degree = (
        parallelism.data_parallel_replicate_degree
        * parallelism.data_parallel_shard_degree
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
        # One validation step is one microbatch per DP rank; cover the split once.
        steps=math.ceil(num_validation_microbatches / dp_degree),
        dataloader=replace(
            config.dataloader,
            dataset=MMSamplePackingConfig(dataset=validation_dataset),
        ),
    )
