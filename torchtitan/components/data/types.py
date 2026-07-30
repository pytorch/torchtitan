# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared data-pipeline types."""

from dataclasses import dataclass

import grain.python as grain

from torchtitan.components.tokenizer import BaseTokenizer


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetBuildContext:
    """Runtime values shared while building the data pipeline."""

    tokenizer: BaseTokenizer
    seq_len: int
    local_batch_size: int
    read_options: grain.ReadOptions


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetIterationPolicy:
    """Controls dataset order, repetition, and data-parallel ownership."""

    seed: int
    shuffle: bool
    repeat: bool
    dp_rank: int
    dp_world_size: int
    streaming_shuffle_buffer_size: int
