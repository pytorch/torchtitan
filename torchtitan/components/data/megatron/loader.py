# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan dataloader over lm-engine's Megatron datasets.

Adapts lm-engine's ``get_megatron_gpt_dataloaders``
(``lm_engine/training/data/pretraining.py``) to the ``BaseDataLoader`` contract:
the same datasets and ``MegatronBatchSampler`` produce each rank's samples,
which are flattened into the 1-D ``TrainerBatch`` layout ``TextCollator`` emits.

Shape suffix legend:
  B = samples per local micro batch, S = sequence length, T = B * S tokens
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import torch
import tyro

from torchtitan.components.data.collators import HAS_PIN_MEMORY, TrainerBatch
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer

from . import utils
from .builder import build
from .sampler import MegatronBatchSampler
from .utils import Split

__all__ = ["MegatronDataLoader"]

_STATE_VERSION = 1


class MegatronDataLoader(BaseDataLoader):
    """Megatron ``.bin``/``.idx`` token data with lm-engine's sample order.

    Each sample is ``sequence_length + 1`` tokens drawn across document
    boundaries (Megatron's concat-then-split), shifted into input and labels.
    A sample is one causal attention segment, as in lm-engine.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        data_paths: Annotated[list[str], tyro.conf.Suppress]
        """Megatron dataset prefixes (``<prefix>.bin`` / ``<prefix>.idx``); an
        ``msc://`` prefix reads through the multi-storage client."""

        sequence_length: Annotated[int, tyro.conf.Suppress]

        num_samples: Annotated[int, tyro.conf.Suppress]
        """Global samples to index for ``split_name``; lm-engine derives this
        from training steps * global batch size."""

        data_weights: Annotated[list[float] | None, tyro.conf.Suppress] = None
        """Sampling weight per prefix (BlendedDataset); None concatenates the
        datasets in proportion to their size (ConcatenatedDataset)."""

        split: Annotated[str, tyro.conf.Suppress] = "969,30,1"
        """train,valid,test ratios over each dataset's sequences."""

        split_name: Annotated[
            Literal["train", "valid", "test"], tyro.conf.Suppress
        ] = "train"

        data_cache_path: Annotated[str | None, tyro.conf.Suppress] = None
        """Where the index cache is written; defaults to ``<prefix>/cache``."""

        node_uses_local_storage: Annotated[bool, tyro.conf.Suppress] = False
        """Build caches (and download ``msc://`` indices) once per node rather
        than once globally."""

        seed: int = 42
        num_workers: Annotated[int, tyro.conf.Suppress] = 2

        fim_rate: Annotated[float, tyro.conf.Suppress] = 0.0
        """Probability of a fill-in-the-middle transform per document segment."""

        fim_spm_rate: Annotated[float, tyro.conf.Suppress] = 0.5
        """Probability that a FIM transform uses SPM rather than PSM order."""

        repeat: Annotated[bool, tyro.conf.Suppress] = True
        """Restart from the first sample after the last one; False stops."""

        def __post_init__(self) -> None:
            BaseDataLoader.Config.__post_init__(self)
            if not self.data_paths:
                raise ValueError("data_paths must name at least one dataset prefix")
            if self.data_weights is not None and len(self.data_weights) != len(
                self.data_paths
            ):
                raise ValueError(
                    f"data_weights has {len(self.data_weights)} entries for "
                    f"{len(self.data_paths)} data_paths"
                )
            if self.sequence_length <= 0 or self.num_samples <= 0:
                raise ValueError("sequence_length and num_samples must be positive")
            for name, rate in (
                ("fim_rate", self.fim_rate),
                ("fim_spm_rate", self.fim_spm_rate),
            ):
                if not 0.0 <= rate <= 1.0:
                    raise ValueError(f"{name} must be in [0, 1], got {rate}")

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        max_context_length: int,
        num_tokens_per_batch: int,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if config.sequence_length > max_context_length:
            raise ValueError(
                f"sequence_length ({config.sequence_length}) exceeds "
                f"max_context_length ({max_context_length})"
            )
        if num_tokens_per_batch % config.sequence_length != 0:
            raise ValueError(
                f"num_tokens_per_batch ({num_tokens_per_batch}) must be a multiple "
                f"of sequence_length ({config.sequence_length})"
            )
        self._micro_batch_size = num_tokens_per_batch // config.sequence_length
        # Each sample is one document segment.
        if (
            config.max_num_documents is not None
            and config.max_num_documents < self._micro_batch_size
        ):
            raise ValueError(
                f"max_num_documents ({config.max_num_documents}) is below the "
                f"{self._micro_batch_size} samples in each batch"
            )
        self.max_num_documents = config.max_num_documents

        if config.fim_rate > 0 and config.num_workers > 0:
            start_method = torch.multiprocessing.get_start_method()
            if start_method != "fork":
                try:
                    pickle.dumps(tokenizer)
                except Exception as e:
                    raise ValueError(
                        f"FIM with num_workers > 0 sends the tokenizer to worker "
                        f"processes, but {type(tokenizer).__qualname__} is not "
                        f"picklable under the {start_method!r} start method; set "
                        "num_workers=0"
                    ) from e

        self._config = config
        self._dp_world_size = dp_world_size
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"
        self._consumed_samples = 0
        self._iterator: Iterator[TrainerBatch] | None = None

        if utils._HELPERS is None:
            utils.compile_helpers()

        split_index = Split[config.split_name].value
        sizes = [0] * len(Split)
        sizes[split_index] = config.num_samples
        if config.data_weights is None:
            blend = list(config.data_paths)
        else:
            blend = [
                item
                for weight, path in zip(config.data_weights, config.data_paths)
                for item in (str(weight), path)
            ]

        self._dataset = build(
            sizes=sizes,
            sequence_length=config.sequence_length,
            tokenizer=tokenizer,
            node_uses_local_storage=config.node_uses_local_storage,
            random_seed=config.seed,
            blend=blend,
            split=config.split,
            path_to_cache=config.data_cache_path,
            fim_rate=config.fim_rate,
            fim_spm_rate=config.fim_spm_rate,
        )[split_index]
        if self._dataset is None:
            raise ValueError(
                f"split {config.split!r} gives the {config.split_name!r} split no data"
            )

        self._samples_per_global_batch = self._micro_batch_size * dp_world_size
        # The sampler drops the last partial global batch.
        self._num_usable_samples = (
            len(self._dataset) // self._samples_per_global_batch
        ) * self._samples_per_global_batch
        if self._num_usable_samples == 0:
            raise ValueError(
                f"the {config.split_name!r} split has {len(self._dataset)} samples, "
                f"fewer than one global batch of {self._samples_per_global_batch}"
            )

    def _batches(self) -> Iterator[TrainerBatch]:
        sequence_length = self._config.sequence_length
        # Every sample restarts at position 0: one causal segment per sample.
        positions_T = torch.arange(sequence_length, dtype=torch.int64).repeat(
            self._micro_batch_size
        )
        if HAS_PIN_MEMORY:
            positions_T = positions_T.pin_memory()
        while True:
            dataloader = torch.utils.data.DataLoader(
                self._dataset,
                batch_sampler=MegatronBatchSampler(
                    total_samples=len(self._dataset),
                    consumed_samples=self._consumed_samples,
                    micro_batch_size=self._micro_batch_size,
                    num_replicas=self._dp_world_size,
                    rank=self._dp_rank,
                ),
                num_workers=self._config.num_workers,
                pin_memory=HAS_PIN_MEMORY,
            )
            for batch in dataloader:
                tokens_BS1 = batch["text"]
                input_T = tokens_BS1[:, :-1].flatten()
                labels_T = tokens_BS1[:, 1:].flatten()
                # Counted when handed to the trainer (not when prefetched), so a
                # checkpoint resumes at the next unseen global batch.
                self._consumed_samples += self._samples_per_global_batch
                yield {
                    "input": input_T,
                    "labels": labels_T,
                    "positions": positions_T,
                    "padding_mask": torch.zeros_like(input_T, dtype=torch.bool),
                    "num_valid_tokens": labels_T.numel(),
                }
            if not self._config.repeat:
                return
            self._consumed_samples = 0

    def __iter__(self) -> Iterator[TrainerBatch]:
        if self._consumed_samples >= self._num_usable_samples:
            if not self._config.repeat:
                return iter(())
            self._consumed_samples = 0
        if self._iterator is None:
            self._iterator = self._batches()
        return self._iterator

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "dp_world_size": self._dp_world_size,
            self._rank_id: {"consumed_samples": self._consumed_samples},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if state_dict.get("version") != _STATE_VERSION:
            raise ValueError(
                f"unsupported MegatronDataLoader state version {state_dict.get('version')}"
            )
        if state_dict.get("dp_world_size") != self._dp_world_size:
            raise ValueError(
                f"checkpoint was saved with dp_world_size "
                f"{state_dict.get('dp_world_size')}, but this run uses "
                f"{self._dp_world_size}"
            )
        if self._rank_id not in state_dict:
            raise ValueError(f"checkpoint has no dataloader state for {self._rank_id}")
        self.close()
        self._consumed_samples = state_dict[self._rank_id]["consumed_samples"]

    def close(self) -> None:
        # Dropping the generator releases its DataLoader and worker processes.
        self._iterator = None
