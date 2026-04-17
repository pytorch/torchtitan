# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from datasets import load_dataset
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import (
    DATASETS,
    HuggingFaceTextDataLoader,
    HuggingFaceTextDataset,
)


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
                    # consume and trigger re-looping
                    for _ in range(2050):
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
                        assert torch.equal(
                            input_ids["positions"],
                            expected_input_ids["positions"],
                        )
                        assert torch.equal(labels, expected_labels)

    def _build_dataloader(self, dataset_name, batch_size, seq_len, world_size, rank):
        tokenizer_config = HuggingFaceTokenizer.Config()
        dl_config = HuggingFaceTextDataLoader.Config(
            dataset=dataset_name, infinite=True
        )

        return dl_config.build(
            dp_world_size=world_size,
            dp_rank=rank,
            tokenizer=tokenizer_config.build(tokenizer_path="./tests/assets/tokenizer"),
            seq_len=seq_len,
            local_batch_size=batch_size,
        )

    def test_map_style_shuffle_on_reloop(self):
        """Re-looping a map-style (``Dataset``) source should change order every
        epoch; leaving it as-is meant the model kept seeing identical batches
        (https://github.com/pytorch/torchtitan/issues/2733).

        Validates three things end-to-end without having to drain a full epoch
        of c4_test (which would be slow for a unit test):
          1. After an epoch boundary, ``_data`` is a *shuffled* copy of
             ``_original_data`` — not the same object.
          2. ``state_dict()`` carries ``epoch`` so resume knows the shuffle
             seed.
          3. ``load_state_dict()`` replays the same ``shuffle(seed=42+epoch)``
             so a resumed run observes the identical sample order.
        """

        def _build_ds():
            return HuggingFaceTextDataset(
                dataset_name="c4_test",
                dataset_path=None,
                tokenizer=HuggingFaceTokenizer.Config().build(
                    tokenizer_path="./tests/assets/tokenizer"
                ),
                seq_len=128,
                dp_rank=0,
                dp_world_size=1,
                infinite=True,
            )

        # 1) Simulate an epoch boundary directly — exercising the re-loop branch
        # without having to drain 200k batches of c4_test.
        ds = _build_ds()
        original = ds._data
        # Manually fast-forward to the last sample so the next iteration wraps.
        ds._sample_idx = len(ds._data)
        ds._epoch = 0
        it = iter(ds)
        # One next() is enough to trip the exhaustion branch and advance epoch.
        next(it)
        assert ds._epoch == 1, f"expected epoch=1 after wrap, got {ds._epoch}"
        assert ds._data is not original, (
            "map-style re-loop must re-shuffle _data from _original_data; "
            "identity check failed"
        )

        # 2) state_dict persists epoch so resume can replay the shuffle.
        state = ds.state_dict()
        assert state.get("epoch") == 1, f"state_dict missing epoch: {state.keys()}"

        # 3) load_state_dict on a fresh instance reproduces the shuffled view.
        ds_resumed = _build_ds()
        ds_resumed.load_state_dict(state)
        assert ds_resumed._epoch == 1
        assert ds_resumed._data is not ds_resumed._original_data
        # Same seed → same first-N sample ids (datasets.shuffle is deterministic).
        assert list(ds._data[:5]["text"]) == list(ds_resumed._data[:5]["text"])

        # 4) Backward compatibility: old checkpoints without "epoch" still load
        # and default to 0 (epoch-0 path is unshuffled, so no behavior change).
        legacy_state = {
            "inputs_buffer": [],
            "positions_buffer": [],
            "sample_idx": 0,
        }
        ds_legacy = _build_ds()
        ds_legacy.load_state_dict(legacy_state)
        assert ds_legacy._epoch == 0
        assert ds_legacy._data is ds_legacy._original_data
