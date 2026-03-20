# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.mm_datasets import MMDataLoader


class TestMMDatasetCheckpointing(unittest.TestCase):
    """Test save/load for multimodal dataset, mirroring test_dataset_checkpointing.py."""

    def _build_dataloader(self, batch_size, seq_len, world_size, rank):
        tokenizer = HuggingFaceTokenizer.Config().build(
            tokenizer_path="tests/assets/tokenizer"
        )
        dl_config = MMDataLoader.Config(
            dataset="cc12m-test",
            patch_size=14,
            temporal_patch_size=2,
            spatial_merge_size=2,
            min_pixels=784,
            max_pixels=200000,
        )

        return MMDataLoader(
            dl_config,
            dp_world_size=world_size,
            dp_rank=rank,
            tokenizer=tokenizer,
            seq_len=seq_len,
            local_batch_size=batch_size,
        )

    def test_cc12m_resumption(self):
        # cc12m_test has 32 samples; with world_size=2, each rank gets 16
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
                    # Verify pixel_values shapes match (values not compared)
                    for key in ["pixel_values", "grid_thw"]:
                        exp_v = expected_input[key]
                        res_v = input_dict[key]
                        assert (exp_v is None) == (
                            res_v is None
                        ), f"{key} None mismatch (world_size={world_size}, rank={rank})"
                        if exp_v is not None:
                            print(exp_v.shape)
                            assert exp_v.shape == res_v.shape, (
                                f"{key} shape mismatch: {exp_v.shape} vs {res_v.shape} "
                                f"(world_size={world_size}, rank={rank})"
                            )


if __name__ == "__main__":
    unittest.main()
