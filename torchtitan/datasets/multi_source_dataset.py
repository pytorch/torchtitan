# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer, build_hf_tokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger


@dataclass
class SourceConfig:
    path: str
    weight: float
    loader: Optional[Callable] = None
    sample_processor: Optional[Callable] = None


def _process_text_file(sample: Dict[str, Any]):
    if isinstance(sample, dict) and "text" in sample:
        return sample["text"]


class MultiSourceDataset(IterableDataset, Stateful):
    """A dataset that allows sampling from multiple sources with configurable weights."""

    def __init__(
        self,
        sources: List[SourceConfig],
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        seed: int = 42,
    ) -> None:
        self.sources = sources
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.seed = seed

        total_weight = sum(source.weight for source in self.sources)
        self.normalized_weights = [
            source.weight / total_weight for source in self.sources
        ]

        self._datasets = []
        self._data_iterators = []
        self._sample_processors = []

        for idx, source in enumerate(self.sources):
            logger.info(
                f"Loading dataset from {source.path} with weight {source.weight:.4f}"
            )
            print(f"Loading dataset from {source.path} with weight {source.weight:.4f}")

            try:
                if Path(source.path).exists():
                    ds = load_dataset(
                        "text",
                        data_files=f"{source.path}/*.jsonl",
                        split="train",
                        streaming=True,
                    )
                    ds_split = split_dataset_by_node(ds, dp_rank, dp_world_size)
                    self._datasets.append(ds_split)

                    processor = source.sample_processor or _process_text_file
                    self._sample_processors.append(processor)
            except Exception as e:
                logger.error(f"Failed to load dataset from {source.dataset}: {e}")
                raise

        self._sample_idx = 0
        self._token_buffer: List[int] = []
        self._current_source_idx = 0
        self._rng = random.Random(seed + dp_rank)

    def _select_source(self) -> int:
        return self._rng.choices(
            range(len(self.sources)), weights=self.normalized_weights
        )[0]

    def _get_sample_from_source(self, source_idx: int) -> str:
        try:
            if source_idx >= len(self._datasets):
                logger.warning(f"Source index {source_idx} out of range, using 0")
                source_idx = 0

            # if we don't have a data iterator for this source index, add it
            if len(self._data_iterators) <= source_idx:
                self._data_iterators.extend(
                    [None] * (source_idx + 1 - len(self._data_iterators))
                )

            if self._data_iterators[source_idx] is None:
                self._data_iterators[source_idx] = iter(self._datasets[source_idx])

            sample = next(self._data_iterators[source_idx])
            processor = self._sample_processors[source_idx]
            return processor(sample)

        except StopIteration:
            if self.infinite:
                # Reset iterator and try again
                self._data_iterators[source_idx] = iter(self._datasets[source_idx])
                sample = next(self._data_iterators[source_idx])
                processor = self._sample_processors[source_idx]
                return processor(sample)
            else:
                raise

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            try:
                source_idx = self._select_source()

                sample_text = self._get_sample_from_source(source_idx)

                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=True, add_eos=True
                )
                self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input_tokens = x[:-1]
                    label_tokens = x[1:]
                    yield {"input": input_tokens}, label_tokens
            except StopIteration:
                if not self.infinite:
                    logger.warning("All datasets have run out of data")
                    break
                else:
                    # Reset all iterators for infinite mode
                    self._data_iterators = [None] * len(self._datasets)
                    self._sample_idx = 0
                    logger.warning("Datasets are being re-looped")

    def state_dict(self):
        return {
            "token_buffer": self._token_buffer,
            "sample_idx": self._sample_idx,
            "current_source_idx": self._current_source_idx,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict.get("token_buffer", [])
        self._sample_idx = state_dict.get("sample_idx", 0)
        self._current_source_idx = state_dict.get("current_source_idx", 0)
        if "rng_state" in state_dict:
            self._rng.setstate(state_dict["rng_state"])


def build_multi_source_dataloader(
    job_config: JobConfig,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    datasets = job_config.training.dataset
    datasets = datasets.split(",")
    sources = []
    for ds in datasets:
        dataset_path, weight = ds.split(":")
        sources.append(SourceConfig(path=dataset_path, weight=float(weight)))

    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len
    seed = job_config.training.seed

    dataset = MultiSourceDataset(
        sources=sources,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        seed=seed if seed is not None else 42,
    )

    return ParallelAwareDataloader(
        dataset=dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )


def test_multi_source_dataloader():
    dataset = "/mnt/wsfuse/htouvron/cwm_32b_data/20250318/github_neo/github_neo_repo_dfs_v3:27,/mnt/wsfuse/htouvron/cwm_32b_data/20250318/llama3/llama3_fiction_fair_copy:0.8,/mnt/wsfuse/htouvron/cwm_32b_data/llama2/arxiv:0.4,/mnt/wsfuse/htouvron/cwm_32b_data/llama2/stackexchange:0.7,/mnt/wsfuse/htouvron/cwm_32b_data/20250214/proof_pile_2/proof_pile_2_open_web_math_train:0.25,/mnt/wsfuse/htouvron/cwm_32b_data/20250214/proof_pile_2/proof_pile_2_algebraic_stack_train:0.15,/mnt/wsfuse/htouvron/cwm_32b_data/20250214/proof_pile_2/proof_pile_2_arxiv_train:0.4,/mnt/wsfuse/htouvron/cwm_32b_data/20250214/dm_mathematics/dm_math_train_easy:0.02,/mnt/wsfuse/htouvron/cwm_32b_data/20250214/dm_mathematics/dm_math_train_medium:0.02,/mnt/wsfuse/htouvron/cwm_32b_data/20250214/dm_mathematics/dm_math_train_hard:0.02,/mnt/wsfuse/htouvron/cwm_32b_data/20250214/aqua_train:0.0002,/mnt/wsfuse/htouvron/cwm_32b_data/20250318/llama3/llama3_galactica_fair_copy:1,/mnt/wsfuse/htouvron/cwm_32b_data/20250318/llama3/llama3_mobius_wiki_all_en_fair_copy:0.1,/mnt/wsfuse/htouvron/cwm_32b_data/20250318/llama3/llama3_math_v1_data_decontaminated_fair_copy:0.8,/mnt/wsfuse/htouvron/cwm_32b_data/20250318/llama3/llama3_math_decontaminated_fair_copy:0.8,/mnt/wsfuse/htouvron/cwm_32b_data/20250318/llama3/llama3_wikipedia_fair_copy:1,/mnt/wsfuse/htouvron/cwm_32b_data/dclm_baseline_1_0:55"
    # dataset = "/mnt/wsfuse/htouvron/cwm_32b_data/20250318/github_neo/github_neo_repo_dfs_v3:0.1,/mnt/wsfuse/htouvron/cwm_32b_data/20250318/llama3/llama3_fiction_fair_copy:0.8,/mnt/wsfuse/htouvron/cwm_32b_data/llama2/arxiv:0.4,/mnt/wsfuse/htouvron/cwm_32b_data/llama2/stackexchange:0.7,/mnt/wsfuse/htouvron/cwm_32b_data/20250214/proof_pile_2/proof_pile_2_open_web_math_train:0.25,/mnt/wsfuse/htouvron/cwm_32b_data/20250214/proof_pile_2/proof_pile_2_algebraic_stack_train:0.15,/mnt/wsfuse/htouvron/cwm_32b_data/20250214/proof_pile_2/proof_pile_2_arxiv_train:0.4,/mnt/wsfuse/htouvron/cwm_32b_data/20250214/dm_mathematics/dm_math_train_easy:0.02,/mnt/wsfuse/htouvron/cwm_32b_data/20250214/dm_mathematics/dm_math_train_medium:0.02,/mnt/wsfuse/htouvron/cwm_32b_data/20250214/dm_mathematics/dm_math_train_hard:0.02,/mnt/wsfuse/htouvron/cwm_32b_data/20250214/aqua_train:0.0002,/mnt/wsfuse/htouvron/cwm_32b_data/20250318/llama3/llama3_galactica_fair_copy:1,/mnt/wsfuse/htouvron/cwm_32b_data/20250318/llama3/llama3_mobius_wiki_all_en_fair_copy:0.1,/mnt/wsfuse/htouvron/cwm_32b_data/20250318/llama3/llama3_math_v1_data_decontaminated_fair_copy:0.8,/mnt/wsfuse/htouvron/cwm_32b_data/20250318/llama3/llama3_math_decontaminated_fair_copy:0.8,/mnt/wsfuse/htouvron/cwm_32b_data/20250318/llama3/llama3_wikipedia_fair_copy:1,/mnt/wsfuse/htouvron/cwm_32b_data/dclm_baseline_1_0:55"
    job_config = JobConfig()
    job_config.training.dataset = dataset
    job_config.model.hf_assets_path = "assets/hf/Llama-3.1-70B/Llama-3.1-8B"
    dp_rank = 0
    dp_world_size = 1
    tokenizer = build_hf_tokenizer(job_config)
    dataloader = build_multi_source_dataloader(
        job_config, dp_world_size, dp_rank, tokenizer
    )
    print(next(iter(dataloader)))


test_multi_source_dataloader()
