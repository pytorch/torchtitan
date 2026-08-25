# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import numpy as np

from torchtitan.components.data import TextSequence
from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.hf_datasets.text_datasets import TextProcessor


class KimiK3TextProcessor(TextProcessor):
    """Truncate each document independently so KDA state never crosses documents."""

    @dataclass(kw_only=True, slots=True)
    class Config(TextProcessor.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        super().__init__(config, context=context)
        self._max_context_length = context.max_context_length
        self._num_tokens_per_batch = context.num_tokens_per_batch

    def __call__(
        self,
        sample: dict[str, Any],
        rng: np.random.Generator,
    ) -> TextSequence | None:
        sequence = super().__call__(sample, rng)
        if sequence is None:
            return None
        max_length = min(self._max_context_length, self._num_tokens_per_batch)
        return TextSequence(
            input_ids=sequence.input_ids[:max_length],
            labels=sequence.labels[:max_length],
        )
