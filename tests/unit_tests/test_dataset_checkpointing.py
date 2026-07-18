# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import grain.python as grain
import torch
from datasets import load_dataset

from torchtitan.components.data.collators import TextCollator
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.components.data.sources import (
    HuggingFaceRandomAccessSource,
    HuggingFaceStreamingSource,
)
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextProcessor


_DATA_PATH = "tests/assets/c4_test"
_TOKENIZER_PATH = "tests/assets/tokenizer"


def _load_c4_map(path):
    return load_dataset(path, split="train").select(range(8))


def _load_c4_stream(path):
    return _load_c4_map(path).to_iterable_dataset(num_shards=2)


def _process_text(sample):
    return sample["text"]


class TestDatasetCheckpointing(unittest.TestCase):
    def test_c4_resumption(self):
        for source_type in (
            HuggingFaceRandomAccessSource,
            HuggingFaceStreamingSource,
        ):
            for rank in range(2):
                with self.subTest(source_type=source_type, rank=rank):
                    dataloader = self._build_dataloader(source_type, rank)
                    iterator = iter(dataloader)

                    # Eight source rows make fewer than 40 packed rows per rank,
                    # so this crosses at least one repeat boundary.
                    for _ in range(40):
                        next(iterator)
                    state = dataloader.state_dict()

                    resumed = self._build_dataloader(source_type, rank)
                    resumed.load_state_dict(state)
                    resumed_iterator = iter(resumed)

                    for _ in range(8):
                        expected_inputs, expected_labels = next(iterator)
                        actual_inputs, actual_labels = next(resumed_iterator)
                        self.assertTrue(
                            torch.equal(
                                actual_inputs["input"], expected_inputs["input"]
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                actual_inputs["positions"],
                                expected_inputs["positions"],
                            )
                        )
                        self.assertTrue(torch.equal(actual_labels, expected_labels))

    def _build_dataloader(self, source_type, rank):
        loader = (
            _load_c4_map
            if source_type is HuggingFaceRandomAccessSource
            else _load_c4_stream
        )
        config = GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=source_type.Config(
                        path=_DATA_PATH,
                        loader=loader,
                    ),
                    process=HuggingFaceTextProcessor.Config(
                        text_processor=_process_text,
                    ),
                ),
            ),
            collator=TextCollator.Config(),
            seed=42,
            shuffle=True,
            repeat=True,
            read_options=grain.ReadOptions(
                num_threads=1,
                prefetch_buffer_size=1,
            ),
            batch_prefetch_buffer_size=1,
        )
        return config.build(
            dp_world_size=2,
            dp_rank=rank,
            tokenizer=HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH),
            seq_len=128,
            local_batch_size=1,
        )


if __name__ == "__main__":
    unittest.main()
