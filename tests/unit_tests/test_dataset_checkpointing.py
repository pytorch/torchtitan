# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.components.data.collators import TextCollator
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.components.data.sources import (
    HuggingFaceRandomAccessSource,
    HuggingFaceStreamingSource,
)
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.text_datasets import TextProcessor


_DATA_PATH = "tests/assets/sft_test/data.json"
_TOKENIZER_PATH = "tests/assets/tokenizer"


def _process_text(sample):
    return f"{sample['question']} {sample['answer']}"


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
        config = GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=source_type.Config(
                        path="json",
                        split="train",
                        load_dataset_kwargs={
                            "data_files": _DATA_PATH,
                        },
                    ),
                    processor=TextProcessor.Config(
                        text_fn=_process_text,
                    ),
                    post_filters=(lambda sample: sample is not None,),
                ),
            ),
            collator=TextCollator.Config(),
            seed=42,
            shuffle=True,
            repeat=True,
            num_prefetch_batches=1,
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
