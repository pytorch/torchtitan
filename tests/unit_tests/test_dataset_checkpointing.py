# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import unittest
from torchtitan.config_manager import JobConfig
from torchtitan.datasets.hf_datasets import build_hf_dataloader, DATASETS, DatasetConfig
from torchtitan.datasets.tokenizer.tiktoken import TikTokenizer
from datasets import load_dataset


class TestDatasetCheckpointing(unittest.TestCase):
    def setUp(self):
        DATASETS["c4_test"] = DatasetConfig(
            path="tests/assets/c4_test",
            loader=lambda path: load_dataset(path, split="train"),
            text_processor=lambda sample: sample["text"],
        )
        DATASETS["c4_test_streaming"] = DatasetConfig(
            path="tests/assets/c4_test",
            loader=lambda path: load_dataset(path, split="train").to_iterable_dataset(num_shards=2),
            text_processor=lambda sample: sample["text"],
        )

    def test_c4_resumption(self):
        for dataset_name in ["c4_test", "c4_test_streaming"]:
            world_size = 4
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
                    assert torch.equal(input_ids["input"], expected_input_ids["input"])
                    assert torch.equal(labels, expected_labels)


    def _build_dataloader(self, dataset_name, batch_size, seq_len, world_size, rank):
        tokenizer = TikTokenizer("./tests/assets/test_tiktoken.model")
        config = JobConfig()
        config.parse_args(
            [
                "--training.dataset",
                dataset_name,
                "--training.batch_size",
                str(batch_size),
                "--training.seq_len",
                str(seq_len),
            ]
        )

        return build_hf_dataloader(
            tokenizer=tokenizer,
            dp_world_size=world_size,
            dp_rank=rank,
            job_config=config,
        )
