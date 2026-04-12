"""Data loading: local text dataset + distributed dataloader."""

import inspect
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any

import torch
from datasets import Dataset, load_dataset, load_from_disk
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from src.components.tokenizer import BaseTokenizer
from src.config import JobConfig
from src.logging import logger


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_text_dataset(path: str):
    """Load a text dataset from a local path.

    Supports:
    - HF datasets saved via `Dataset.save_to_disk(path)` (detected by `dataset_info.json`)
    - Directories of parquet / jsonl / txt files (auto-detected by HF `load_dataset`)
    """
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "dataset_info.json")):
        return load_from_disk(path)
    return load_dataset(path, split="train")


def extract_text(sample: dict[str, Any]) -> str:
    """Pull the text field out of a sample. Supports common field names."""
    for key in ("text", "content", "raw_content"):
        if key in sample:
            return sample[key]
    raise KeyError(
        f"Could not find a text field in sample; tried 'text', 'content', 'raw_content'. "
        f"Sample keys: {list(sample.keys())}"
    )


# ---------------------------------------------------------------------------
# Streaming text dataset
# ---------------------------------------------------------------------------


class TextDataset(IterableDataset, Stateful):
    """Tokenizes text samples and yields packed fixed-length sequences for LM training."""

    def __init__(
        self,
        dataset_path: str,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        logger.info(f"Loading dataset from {dataset_path}")
        ds = load_text_dataset(dataset_path)

        self.dataset_path = dataset_path
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    def _get_data_iter(self):
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))
        return iter(self._data)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                sample_text = extract_text(sample)
                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=True, add_eos=True
                )
                self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(f"Dataset at {self.dataset_path} has run out of data")
                break
            else:
                self._sample_idx = 0
                logger.warning(f"Dataset at {self.dataset_path} is being re-looped")
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]
        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict: dict[str, Any] = {"token_buffer": self._token_buffer}
        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            _state_dict["data"] = self._data.state_dict()
        return _state_dict


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------


class DataloaderExhaustedError(Exception):
    """Indicates dataloader exhaustion (not StopIteration, see PEP 479)."""

    pass


class BaseDataLoader(Stateful, ABC):
    @abstractmethod
    def __iter__(self): ...


# pyrefly: ignore [inconsistent-inheritance]
class ParallelAwareDataloader(StatefulDataLoader, BaseDataLoader):
    dp_rank: int
    dp_world_size: int

    def __init__(self, dataset: IterableDataset, dp_rank: int, dp_world_size: int, **kwargs):
        self._validate_kwargs(kwargs)
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"
        super().__init__(dataset, **kwargs)

    @staticmethod
    def _validate_kwargs(kwargs: dict[str, Any]) -> None:
        if "dataset" in kwargs:
            raise ValueError(
                "'dataset' should not be passed in kwargs; "
                "it must be provided as the first positional argument."
            )
        sig = inspect.signature(StatefulDataLoader.__init__)
        valid_kwargs = frozenset(
            name for name in sig.parameters.keys() if name not in ("self", "dataset")
        )
        invalid_kwargs = set(kwargs.keys()) - valid_kwargs
        if invalid_kwargs:
            raise ValueError(
                f"Invalid dataloader kwargs: {invalid_kwargs}. "
                f"Valid kwargs are: {sorted(valid_kwargs)}"
            )
        if kwargs.get("num_workers", 0) == 0:
            kwargs.pop("persistent_workers", None)
            kwargs.pop("prefetch_factor", None)

    def state_dict(self) -> dict[str, Any]:
        return {
            self._rank_id: pickle.dumps(super().state_dict()),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self.dp_rank}, "
                f"expected key {self._rank_id}"
            )
            return
        assert self.dp_world_size == state_dict["world_size"], (
            "dp_degree is inconsistent before and after checkpoint, "
            "dataloader resharding is not supported yet."
        )
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_text_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
    dataset_path: str | None = None,
) -> ParallelAwareDataloader:
    dataset_path = dataset_path or job_config.training.dataset_path
    if not dataset_path:
        raise ValueError("dataset_path must be set")
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    hf_ds = TextDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    dataloader_kwargs = {
        **asdict(job_config.training.dataloader),
        "batch_size": batch_size,
    }

    return ParallelAwareDataloader(
        hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )
