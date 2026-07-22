# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.tools.logging import logger


@dataclass(frozen=True, slots=True)
class _TokenFile:
    path: Path
    num_tokens: int
    start: int


class PreTokenizedTextDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_path: str,
        seq_len: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.seq_len = seq_len
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.infinite = infinite
        self._epoch = 0
        self._sequence_idx = dp_rank

        metadata_path = (
            self.dataset_path
            if self.dataset_path.is_file()
            else self.dataset_path / "metadata.json"
        )
        with metadata_path.open() as f:
            metadata = json.load(f)

        dtype = metadata.get("dtype", "uint32")
        if dtype != "uint32":
            raise ValueError(
                f"PreTokenizedTextDataset only supports dtype='uint32', got {dtype!r}"
            )

        self._token_files = self._load_token_files(metadata, metadata_path)
        self._file_starts = [token_file.start for token_file in self._token_files]
        self._mapped_files: dict[int, torch.Tensor] = {}

        self.num_tokens = int(metadata["num_tokens"])
        file_token_count = sum(token_file.num_tokens for token_file in self._token_files)
        if file_token_count != self.num_tokens:
            raise ValueError(
                f"metadata num_tokens={self.num_tokens} does not match "
                f"data_files total={file_token_count}"
            )

        self.num_sequences = max(0, (self.num_tokens - 1) // self.seq_len)
        if self.num_sequences == 0:
            raise ValueError(
                f"Pre-tokenized dataset at {metadata_path} has {self.num_tokens} "
                f"tokens, which is too small for seq_len={self.seq_len}"
            )

        logger.info(
            f"Loaded pre-tokenized dataset from {metadata_path} with "
            f"{self.num_tokens} tokens, {self.num_sequences} sequences, "
            f"and {len(self._token_files)} token files"
        )

    @staticmethod
    def _resolve_data_path(metadata_path: Path, data_file: str) -> Path:
        data_path = Path(data_file)
        if not data_path.is_absolute():
            data_path = metadata_path.parent / data_path
        return data_path

    @classmethod
    def _load_token_files(
        cls, metadata: dict[str, Any], metadata_path: Path
    ) -> list[_TokenFile]:
        token_files: list[_TokenFile] = []
        start = 0

        if "data_files" in metadata:
            for data_file in metadata["data_files"]:
                num_tokens = int(data_file["num_tokens"])
                token_files.append(
                    _TokenFile(
                        path=cls._resolve_data_path(metadata_path, data_file["data_file"]),
                        num_tokens=num_tokens,
                        start=start,
                    )
                )
                start += num_tokens
        else:
            token_files.append(
                _TokenFile(
                    path=cls._resolve_data_path(metadata_path, metadata["data_file"]),
                    num_tokens=int(metadata["num_tokens"]),
                    start=0,
                )
            )

        if not token_files:
            raise ValueError(f"No token files found in metadata at {metadata_path}")
        return token_files

    def _get_mapped_file(self, file_idx: int) -> torch.Tensor:
        tokens = self._mapped_files.get(file_idx)
        if tokens is not None:
            return tokens

        token_file = self._token_files[file_idx]
        tokens = torch.from_file(
            str(token_file.path),
            shared=False,
            size=token_file.num_tokens,
            dtype=torch.int32,
        )
        self._mapped_files[file_idx] = tokens
        return tokens

    def _read_tokens(self, start: int, end: int) -> torch.Tensor:
        file_idx = bisect.bisect_right(self._file_starts, start) - 1
        chunks: list[torch.Tensor] = []

        while start < end:
            token_file = self._token_files[file_idx]
            file_end = token_file.start + token_file.num_tokens
            take_end = min(end, file_end)
            tokens = self._get_mapped_file(file_idx)
            local_start = start - token_file.start
            local_end = take_end - token_file.start
            chunks.append(tokens[local_start:local_end])
            start = take_end
            file_idx += 1

        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks)

    def _next_rank_sequence_idx(self, sequence_idx: int) -> int:
        remainder = sequence_idx % self.dp_world_size
        if remainder <= self.dp_rank:
            return sequence_idx + self.dp_rank - remainder
        return sequence_idx + self.dp_world_size - remainder + self.dp_rank

    def __iter__(self):
        sequence_idx = self._next_rank_sequence_idx(self._sequence_idx)
        while True:
            while sequence_idx < self.num_sequences:
                start = sequence_idx * self.seq_len
                end = start + self.seq_len + 1
                x = self._read_tokens(start, end).long()
                self._sequence_idx = sequence_idx + self.dp_world_size
                sequence_idx = self._sequence_idx

                input_ids = x[:-1]
                labels = x[1:]
                positions = torch.arange(self.seq_len, dtype=torch.long)
                yield {"input": input_ids, "positions": positions}, labels

            if not self.infinite:
                logger.warning(
                    f"Pre-tokenized dataset {self.dataset_path} has run out of data"
                )
                break

            self._epoch += 1
            self._sequence_idx = self.dp_rank
            sequence_idx = self._sequence_idx
            logger.warning(
                f"Pre-tokenized dataset {self.dataset_path} is being re-looped "
                f"(epoch {self._epoch})"
            )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._sequence_idx = state_dict["sequence_idx"]
        self._epoch = state_dict["epoch"]

    def state_dict(self) -> dict[str, Any]:
        return {
            "sequence_idx": self._sequence_idx,
            "epoch": self._epoch,
        }


class PreTokenizedTextDataLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        infinite: bool = True
        """Whether to loop the dataset infinitely."""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps: int | None = 1,
        **kwargs,
    ):
        del tokenizer, kwargs

        if config.dataset_path is None:
            raise ValueError(
                "PreTokenizedTextDataLoader requires config.dataset_path to point "
                "to a pre-tokenized dataset directory or metadata.json file"
            )

        ds = PreTokenizedTextDataset(
            dataset_path=config.dataset_path,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
        )

        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "snapshot_every_n_steps": snapshot_every_n_steps,
            "batch_size": local_batch_size,
        }

        super().__init__(
            ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )
