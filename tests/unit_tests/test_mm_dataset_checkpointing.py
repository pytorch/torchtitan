# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.sources import HuggingFaceStreamingSource
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.hf_datasets.multimodal.mm_collator import QwenMultimodalCollator
from torchtitan.hf_datasets.multimodal.mm_datasets import (
    QwenCC12MProcessor,
    QwenMultimodalPackingConfig,
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
        dl_config = GrainDataLoader.Config(
            dataset=QwenMultimodalPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceStreamingSource.Config(
                        path="tests/assets/cc12m_test",
                        load_dataset_kwargs={
                            "split": "train",
                            "data_files": {"train": "*.tar"},
                        },
                    ),
                    process=QwenCC12MProcessor.Config(
                        patch_size=16,
                        temporal_patch_size=2,
                        spatial_merge_size=2,
                        min_pixels=784,
                        max_pixels=200000,
                        image_mean=(0.5, 0.5, 0.5),
                        image_std=(0.5, 0.5, 0.5),
                    ),
                ),
                max_images_per_batch=128,
                max_patches_per_batch=128 * (200000 // (16 * 16)),
                patch_size=16,
                temporal_patch_size=2,
            ),
            collator=QwenMultimodalCollator.Config(
                patch_size=16,
                temporal_patch_size=2,
                spatial_merge_size=2,
                build_mrope_positions=True,
            ),
            batch_prefetch_buffer_size=1,
            repeat=True,
            shuffle=False,
        )

        return dl_config.build(
            dp_world_size=world_size,
            dp_rank=rank,
            tokenizer=_TOKENIZER,
            seq_len=seq_len,
            local_batch_size=batch_size,
        )

    def test_cc12m_resumption(self):
        for world_size in [1, 2]:
            for rank in range(world_size):
                batch_size = 1
                seq_len = 4096

                dl = self._build_dataloader(batch_size, seq_len, world_size, rank)

                it = iter(dl)
                for _ in range(5):
                    next(it)
                state = dl.state_dict()

                # Create new dataloader, restore checkpoint, verify subsequent
                # batches match
                dl_resumed = self._build_dataloader(
                    batch_size, seq_len, world_size, rank
                )
                dl_resumed.load_state_dict(state)
                it_resumed = iter(dl_resumed)

                for _ in range(10):
                    expected_input, expected_labels = next(it)
                    input_dict, labels = next(it_resumed)
                    assert torch.equal(
                        input_dict["input"], expected_input["input"]
                    ), f"input_ids mismatch (world_size={world_size}, rank={rank})"
                    assert torch.equal(
                        labels, expected_labels
                    ), f"labels mismatch (world_size={world_size}, rank={rank})"
                    assert torch.equal(
                        input_dict["positions"], expected_input["positions"]
                    ), f"positions mismatch (world_size={world_size}, rank={rank})"
                    for key in ["pixel_values", "grid_thw"]:
                        exp_v = expected_input[key]
                        res_v = input_dict[key]
                        assert (exp_v is None) == (
                            res_v is None
                        ), f"{key} None mismatch (world_size={world_size}, rank={rank})"
                        if exp_v is not None:
                            assert exp_v.shape == res_v.shape, (
                                f"{key} shape mismatch: {exp_v.shape} vs {res_v.shape} "
                                f"(world_size={world_size}, rank={rank})"
                            )


if __name__ == "__main__":
    unittest.main()
