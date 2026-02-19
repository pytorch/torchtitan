# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from datasets import load_dataset
from torchtitan.config import ConfigManager
from torchtitan.hf_datasets import DatasetConfig


class TestFluxDataLoader(unittest.TestCase):
    def setUp(self):
        # Import here to avoid circular import during test collection
        from torchtitan.models.flux.flux_datasets import (
            _cc12m_wds_data_processor,
            DATASETS,
            FluxDataLoader,
        )

        # Store reference for use in tearDown
        self._DATASETS = DATASETS
        self._cc12m_wds_data_processor = _cc12m_wds_data_processor
        self._FluxDataLoader = FluxDataLoader

        self._DATASETS["cc12m-test-iterable"] = DatasetConfig(
            path="tests/assets/cc12m_test",
            loader=lambda path: load_dataset(
                path, split="train", data_files={"train": "*tar"}
            ).to_iterable_dataset(num_shards=4),
            sample_processor=self._cc12m_wds_data_processor,
        )

    def tearDown(self):
        del self._DATASETS["cc12m-test-iterable"]

    def test_load_dataset(self):
        # The test checks for the correct tensor shapes during the first num_steps
        # The next num_steps ensure the loaded from checkpoint dataloader generates tokens and labels correctly
        for world_size in [2]:
            for rank in range(world_size):
                dataset_name = "cc12m-test-iterable"
                batch_size = 1

                num_steps = 15

                # TODO: if num_steps * batch_size * world_size is larger than the number of samples
                # in the dataset, then the test will fail, due to huggingface's
                # non-resumption when checkpointing after the first epoch

                # Load flux config via --module/--config
                config_manager = ConfigManager()
                config = config_manager.parse_args(
                    [
                        "--module",
                        "flux",
                        "--config",
                        "flux_debugmodel",
                        "--training.local_batch_size",
                        str(batch_size),
                        "--dataloader.img_size",
                        str(256),
                        "--dataloader.dataset",
                        dataset_name,
                        "--dataloader.classifier_free_guidance_prob",
                        "0.447",
                        "--dataloader.encoder.test_mode",
                        "--encoder.test_mode",
                        "--encoder.t5_encoder",
                        "tests/assets/flux_test_encoders/t5-v1_1-xxl",
                        "--encoder.clip_encoder",
                        "tests/assets/flux_test_encoders/clip-vit-large-patch14",
                    ]
                )

                dl = config.dataloader.build(
                    dp_world_size=world_size,
                    dp_rank=rank,
                    local_batch_size=batch_size,
                )

                it = iter(dl)

                for i in range(0, num_steps):
                    input_data, labels = next(it)

                    assert (
                        len(input_data) == 3
                    )  # (clip_encodings, t5_encodings, prompt)
                    assert labels.shape == (batch_size, 3, 256, 256)
                    assert input_data["clip_tokens"].shape == (
                        batch_size,
                        77,
                    )
                    assert input_data["t5_tokens"].shape == (
                        batch_size,
                        256,
                    )

                state = dl.state_dict()

                # Create new dataloader, restore checkpoint, and check if next data yielded is the same as above
                dl_resumed = config.dataloader.build(
                    dp_world_size=world_size,
                    dp_rank=rank,
                    local_batch_size=batch_size,
                )
                dl_resumed.load_state_dict(state)
                it_resumed = iter(dl_resumed)

                for i in range(num_steps):
                    # Set torch manual seed before each dataloader iteration to ensure consistent randomness
                    # across dataloaders for testing purposes.
                    torch.manual_seed(i)
                    expected_input_ids, expected_labels = next(it)
                    torch.manual_seed(i)
                    input_ids, labels = next(it_resumed)

                    assert torch.equal(
                        input_ids["clip_tokens"], expected_input_ids["clip_tokens"]
                    )
                    assert torch.equal(
                        input_ids["t5_tokens"], expected_input_ids["t5_tokens"]
                    )
                    assert torch.equal(labels, expected_labels)
