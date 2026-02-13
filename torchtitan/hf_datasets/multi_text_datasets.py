# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict
from functools import partial
from typing import Any, Callable, Literal

from datasets import IterableDataset, load_dataset
from datasets.distributed import split_dataset_by_node
from datasets.iterable_dataset import _interleave_iterable_datasets

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.hf_datasets import MultiDatasetConfig
from torchtitan.tools.logging import logger

from .text_datasets import HuggingFaceTextDataset


def _load_c4_dataset(dataset_path: str, split: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, name="en", split=split, streaming=True)


def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]


# Add your dataset here - more information at docs/datasets.md
DATASETS = {
    "c4": MultiDatasetConfig(
        paths=["allenai/c4", "allenai/c4"],
        weights=[1, 100],
        loader=partial(_load_c4_dataset, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_test": MultiDatasetConfig(
        paths=["tests/assets/c4_test", "tests/assets/c4_test"],
        weights=[1, 100],
        loader=lambda path: load_dataset(path, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_validation": MultiDatasetConfig(
        paths=["allenai/c4", "allenai/c4"],
        weights=[1, 100],
        loader=partial(_load_c4_dataset, split="validation"),
        sample_processor=_process_c4_text,
    ),
}


def _validate_datasets(
    dataset_name: str, datasets_paths: list[str] | None = None
) -> tuple[list[str], Callable, Callable]:
    """Validate datasets name and paths."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    paths = config.paths
    if datasets_paths:
        paths = [
            dataset_path or config_path
            for dataset_path, config_path in zip(datasets_paths, config.paths)
        ]
    logger.info(f"Preparing {dataset_name} dataset from {paths}")
    return paths, config.loader, config.sample_processor


def create_interleaved_parallel_dataset(
    datasets_paths: list[str],
    datasets_weights: list[float] | None,
    datasets_loader: Callable,
    seed: int | None = None,
    stopping_strategy: Literal[
        "first_exhausted", "all_exhausted", "all_exhausted_without_replacement"
    ] = "all_exhausted",
    dp_rank: int | None = None,
    dp_world_size: int | None = None,
) -> IterableDataset:
    """
    Create an IterableDataset that:
    1. Interleaves multiple datasets using weights as probabilities
    2. Is data parallel friendly

    datasets are supposed to be shuffled already
    Args:
        datasets_paths: List of datasets paths
        datasets_weights: Sampling probabilities for each dataset (will be normalized)
        datasets_loader: function to load dataset
        seed: Random seed for sampling
        stopping_strategy: When to stop iteration
        dp_rank: Rank of current process
        dp_world_size: Total number of processes

    Returns:
        IterableDataset ready for distributed training
    """
    # Load all datasets
    datasets = [datasets_loader(path) for path in datasets_paths]
    # Calculate probabilities
    probabilities = None
    if datasets_weights is not None:
        total = sum(datasets_weights)
        probabilities = [w / total for w in datasets_weights]

    # Interleave datasets
    interleaved_ds = _interleave_iterable_datasets(
        datasets=datasets,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )

    # Apply distributed sharding
    if dp_rank is not None and dp_world_size is not None:
        distributed_ds = split_dataset_by_node(
            interleaved_ds, rank=dp_rank, world_size=dp_world_size
        )
        return distributed_ds

    return interleaved_ds


class HuggingFaceTextMultiDataset(HuggingFaceTextDataset):
    def __init__(
        self,
        dataset_name: str,
        datasets_paths: list[str] | None,
        datasets_weights: list[float] | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        seed: int | None = None,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        paths, datasets_loader, text_processor = _validate_datasets(
            dataset_name, datasets_paths
        )
        self.dataset_name = dataset_name
        self._data = create_interleaved_parallel_dataset(
            datasets_paths=paths,
            datasets_weights=datasets_weights,
            datasets_loader=datasets_loader,
            seed=seed,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []


def build_text_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets.

    Args:
        dp_world_size: Data parallelism world size.
        dp_rank: Data parallelism rank.
        tokenizer: Tokenizer to use for encoding text.
        job_config: Job configuration containing dataset and DataLoader settings.
        infinite: Whether to loop the dataset infinitely.
    """
    dataset_name = job_config.training.data.name
    datasets_paths = job_config.training.data.paths
    datasets_weights = job_config.training.data.weights
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len
    seed = job_config.debug.seed

    hf_ds = HuggingFaceTextMultiDataset(
        dataset_name=dataset_name,
        datasets_paths=datasets_paths,
        datasets_weights=datasets_weights,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        seed=seed,
    )
    dataloader_kwargs = {
        **asdict(job_config.training.data.dataloader),
        "batch_size": batch_size,
    }

    return ParallelAwareDataloader(
        hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )


def build_text_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = False,
) -> ParallelAwareDataloader:
    """Build a validation data loader for HuggingFace datasets.

    Args:
        dp_world_size: Data parallelism world size.
        dp_rank: Data parallelism rank.
        tokenizer: Tokenizer to use for encoding text.
        job_config: Job configuration containing dataset and DataLoader settings.
        infinite: Whether to loop the dataset infinitely.
    """
    dataset_name = job_config.validation.data.name
    datasets_paths = job_config.validation.data.paths
    datasets_weights = job_config.validation.data.weights
    batch_size = job_config.validation.local_batch_size
    seq_len = job_config.validation.seq_len
    seed = job_config.debug.seed

    hf_ds = HuggingFaceTextMultiDataset(
        dataset_name=dataset_name,
        datasets_paths=datasets_paths,
        datasets_weights=datasets_weights,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        seed=seed,
    )

    dataloader_kwargs = {
        **asdict(job_config.validation.data.dataloader),
        "batch_size": batch_size,
    }

    return ParallelAwareDataloader(
        hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )
