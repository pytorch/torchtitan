# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import replace

import torch

from torchtitan.components.data import GrainDataLoader
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.hf_datasets.multimodal.mm_datasets import (
    MM_DATASETS,
    MMSamplePackingConfig,
)


_TOKENIZER_PATH = "tests/assets/tokenizer"

_TOKENIZER_CONFIG = MultiModalTokenizer.Config(
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    vision_start_token="<|vision_start|>",
    vision_end_token="<|vision_end|>",
    pad_token="<|endoftext|>",
)


_TOKENIZER = _TOKENIZER_CONFIG.build(tokenizer_path=_TOKENIZER_PATH)


class TestMMDatasetCheckpointing(unittest.TestCase):
    """Test save/load for multimodal dataset, mirroring test_dataset_checkpointing.py."""

    def _build_dataloader(self, batch_size, seq_len, world_size, rank):
        dataset = MM_DATASETS["cc12m-test"]
        assert dataset.processor is not None
        dataset = replace(
            dataset,
            processor=replace(
                dataset.processor,
                min_pixels=784,
                max_pixels=200000,
            ),
        )
        dataset = MMSamplePackingConfig(dataset=dataset, num_packing_bins=2)
        dl_config = GrainDataLoader.Config(
            dataset=dataset,
            collator=MultiModalCollator.Config(),
            streaming_shuffle_buffer_size=128,
        )

        return dl_config.build(
            dp_world_size=world_size,
            dp_rank=rank,
            tokenizer=_TOKENIZER,
            seq_len=seq_len,
            local_batch_size=batch_size,
        )

    def test_cc12m_resumption(self):
        dl = self._build_dataloader(
            batch_size=1,
            seq_len=512,
            world_size=1,
            rank=0,
        )
        it = iter(dl)
        next(it)
        state = dl.state_dict()

        dl_resumed = self._build_dataloader(
            batch_size=1,
            seq_len=512,
            world_size=1,
            rank=0,
        )
        dl_resumed.load_state_dict(state)
        it_resumed = iter(dl_resumed)

        for _ in range(2):
            expected_input, expected_labels = next(it)
            input_dict, labels = next(it_resumed)
            assert torch.equal(input_dict["input"], expected_input["input"])
            assert torch.equal(labels, expected_labels)
            assert torch.equal(input_dict["positions"], expected_input["positions"])
            for key in ["pixel_values", "grid_thw"]:
                expected = expected_input[key]
                resumed = input_dict[key]
                assert (expected is None) == (resumed is None)
                if expected is not None:
                    assert torch.equal(expected, resumed)

        dl.close()
        dl_resumed.close()


if __name__ == "__main__":
    unittest.main()
