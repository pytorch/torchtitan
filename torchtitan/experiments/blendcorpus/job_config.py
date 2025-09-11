# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class Model:
    tokenizer_backend: str = "sptoken"

# --- BEGIN BlendCorpus dataclass ---
@dataclass
class BlendCorpus:
    """Optional settings specific to the BlendCorpus data loader.

    These map to a TOML section named [blendcorpus]. If present, your
    adapter can read them from cfg.blendcorpus.*
    """

    # File list or directory of shards
    data_file_list: str | None = None
    """Path to a text file with one shard path per line, or a directory."""

    # Sequence length and batching (can override training.seq_len / local_batch_size)
    seq_length: int | None = None
    """Optional override for sequence length. If None, use training.seq_len."""

    micro_batch_size: int | None = None
    """Optional override for per-rank batch size. If None, use training.local_batch_size."""

    # Loader behavior
    num_workers: int = 2
    """Number of DataLoader workers."""

    split: str = "98,1,1"
    """Train/valid/test split in percentages."""

    dataloader_type: str = "single"
    """Loader type hint (e.g., 'single', 'repeating')."""

    shuffle: bool = True
    """Whether to shuffle the order of shards."""

    shuffle_sample_in_corpus: bool = True
    """Whether to shuffle samples within corpus."""

    blend_sample_in_corpus: bool = True
    """Whether to shuffle samples within corpus."""

    append_eod: bool = True
    """Append EOD token at the end of each sample when collating."""

    provide_attention_mask: bool = False
    """Whether the adapter should compute and return attention masks."""

    eod_token_id: int | None = None
    """Optional explicit EOD token id; if None the adapter/tokenizer decides."""

    data_cache_path: str = None
# --- END BlendCorpus dataclass ---


@dataclass
class JobConfig:
    """
    Extend the tyro parser with custom config classes for Flux model.
    """

    model: Model = field(default_factory=Model)
    blendcorpus: BlendCorpus = field(default_factory=BlendCorpus)
