# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torchtitan.experiments.vlm.datasets.mm_datasets import (
    HuggingFaceMultiModalDataLoader,
)
from torchtitan.trainer import Trainer


@dataclass(kw_only=True, slots=True)
class MultiModalTrainerConfig(Trainer.Config):
    dataloader: HuggingFaceMultiModalDataLoader.Config = field(
        default_factory=HuggingFaceMultiModalDataLoader.Config
    )
