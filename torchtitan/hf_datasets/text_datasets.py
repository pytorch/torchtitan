# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Text and chat recipes for the Grain data pipeline."""

from typing import Literal

from torchtitan.components.data.collators import TextCollator
from torchtitan.components.data.dataset import SingleDatasetConfig, TextToTokenSequence
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.components.data.sources import (
    HuggingFaceStreamingSource,
    IndexedJsonlSource,
)


def c4_text_dataloader(
    dataset: Literal["c4", "c4_test", "c4_validation"] = "c4_test",
) -> GrainDataLoader.Config:
    """Return the built-in C4 pretraining pipeline."""
    if dataset == "c4_test":
        source = IndexedJsonlSource.Config(
            patterns=("tests/assets/c4_test/data.json",),
        )
    else:
        source = HuggingFaceStreamingSource.Config(
            path="allenai/c4",
            load_dataset_kwargs={
                "name": "en",
                "split": "train" if dataset == "c4" else "validation",
            },
        )

    return GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(
            dataset=SingleDatasetConfig(
                source=source,
                process=TextToTokenSequence.Config(),
            ),
        ),
        collator=TextCollator.Config(),
    )
