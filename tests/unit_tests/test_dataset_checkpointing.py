# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from datasets import load_dataset
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import ConfigManager
from torchtitan.hf_datasets import DatasetConfig, MultiDatasetConfig
from torchtitan.hf_datasets.multi_text_datasets import (
    build_text_dataloader as build_text_dataloader_multi,
    DATASETS as DATASETS_MULTI,
)
from torchtitan.hf_datasets.text_datasets import build_text_dataloader, DATASETS


class TestDatasetCheckpointing(unittest.TestCase):
    def setUp(self):
        DATASETS["c4_test_streaming"] = DatasetConfig(
            path="tests/assets/c4_test",
            loader=lambda path: load_dataset(path, split="train").to_iterable_dataset(
                num_shards=4
            ),
            sample_processor=lambda sample: sample["text"],
        )

    def tearDown(self):
        del DATASETS["c4_test_streaming"]

    def test_c4_resumption(self):
        for dataset_name in ["c4_test", "c4_test_streaming"]:
            for world_size in [2, 4]:
                for rank in range(world_size):
                    batch_size = 1
                    seq_len = 1024

                    dl = self._build_dataloader(
                        dataset_name, batch_size, seq_len, world_size, rank
                    )

                    it = iter(dl)
                    for _ in range(250):
                        next(it)
                    state = dl.state_dict()

                    # Create new dataloader, restore checkpoint, and check if next data yielded is the same as above
                    dl_resumed = self._build_dataloader(
                        dataset_name, batch_size, seq_len, world_size, rank
                    )
                    dl_resumed.load_state_dict(state)
                    it_resumed = iter(dl_resumed)

                    for _ in range(500):
                        expected_input_ids, expected_labels = next(it)
                        input_ids, labels = next(it_resumed)
                        assert torch.equal(
                            input_ids["input"], expected_input_ids["input"]
                        )
                        assert torch.equal(labels, expected_labels)

    def _build_dataloader(self, dataset_name, batch_size, seq_len, world_size, rank):
        tokenizer = HuggingFaceTokenizer("./tests/assets/tokenizer")
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--training.data.name",
                dataset_name,
                "--training.local_batch_size",
                str(batch_size),
                "--training.seq_len",
                str(seq_len),
            ]
        )

        return build_text_dataloader(
            tokenizer=tokenizer,
            dp_world_size=world_size,
            dp_rank=rank,
            job_config=config,
        )


class TestMultiDatasetCheckpointing(TestDatasetCheckpointing):
    def setUp(self):
        DATASETS_MULTI["c4_test_streaming"] = MultiDatasetConfig(
            paths=["tests/assets/c4_test", "tests/assets/c4_test"],
            weights=[1, 3],
            loader=lambda path: load_dataset(path, split="train").to_iterable_dataset(
                num_shards=4
            ),
            sample_processor=lambda sample: sample["text"],
        )

    def tearDown(self):
        del DATASETS_MULTI["c4_test_streaming"]

    def _build_dataloader(self, dataset_name, batch_size, seq_len, world_size, rank):
        tokenizer = HuggingFaceTokenizer("./tests/assets/tokenizer")
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--training.data.name",
                dataset_name,
                "--training.local_batch_size",
                str(batch_size),
                "--training.seq_len",
                str(seq_len),
            ]
        )

        return build_text_dataloader_multi(
            tokenizer=tokenizer,
            dp_world_size=world_size,
            dp_rank=rank,
            job_config=config,
        )

    def test_cross_run_reproducibility(self):
        for world_size in [2, 4]:
            for rank in range(world_size):
                batch_size = 1
                seq_len = 1024

                dl1 = self._build_dataloader(
                    "c4_test", batch_size, seq_len, world_size, rank
                )
                dl2 = self._build_dataloader(
                    "c4_test", batch_size, seq_len, world_size, rank
                )

                it1 = iter(dl1)
                it2 = iter(dl2)

                for _ in range(100):
                    input_ids1, labels1 = next(it1)
                    input_ids2, labels2 = next(it2)
                    assert torch.equal(
                        input_ids1["input"], input_ids2["input"]
                    ), f"Cross-run reproducibility failed at world_size={world_size}, rank={rank}"
                    assert torch.equal(
                        labels1, labels2
                    ), f"Cross-run reproducibility failed at world_size={world_size}, rank={rank}"
