# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from datasets import load_dataset
from torchtitan.components.data import (
    GrainDataLoader,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
)
from torchtitan.config import ConfigManager
from torchtitan.hf_datasets import DatasetConfig


class TestFluxDataLoader(unittest.TestCase):
    def setUp(self):
        # Import here to avoid circular import during test collection
        from torchtitan.models.flux.flux_datasets import (
            _cc12m_wds_data_processor,
            DATASETS,
            FluxCollator,
            FluxSampleProcessor,
        )

        # Store reference for use in tearDown
        self._DATASETS = DATASETS
        self._cc12m_wds_data_processor = _cc12m_wds_data_processor
        self._FluxCollator = FluxCollator
        self._FluxSampleProcessor = FluxSampleProcessor

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
                        "--tokenizer.test_mode",
                        "--tokenizer.t5_tokenizer_path",
                        "tests/assets/tokenizer",
                        "--tokenizer.clip_tokenizer_path",
                        "tests/assets/tokenizer",
                        "--encoder.random_init",
                        "--encoder.t5_encoder",
                        "tests/assets/flux_test_encoders/t5-v1_1-xxl",
                        "--encoder.clip_encoder",
                        "tests/assets/flux_test_encoders/clip-vit-large-patch14",
                    ]
                )
                dataset = self._DATASETS[dataset_name]
                config.dataloader = GrainDataLoader.Config(
                    dataset=SingleDatasetConfig(
                        source=HuggingFaceStreamingSource.Config(
                            path=dataset.path,
                            loader=dataset.loader,
                        ),
                        process=self._FluxSampleProcessor.Config(
                            data_processor=dataset.sample_processor,
                            img_size=256,
                            prompt_dropout_prob=0.447,
                        ),
                        filters=(lambda sample: sample is not None,),
                    ),
                    collator=self._FluxCollator.Config(),
                    shuffle=False,
                )

                # Build the tokenizer container from config
                tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)

                dl = config.dataloader.build(
                    dp_world_size=world_size,
                    dp_rank=rank,
                    local_batch_size=batch_size,
                    tokenizer=tokenizer,
                    seq_len=config.training.seq_len,
                )

                it = iter(dl)

                for i in range(0, num_steps):
                    input_data, labels = next(it)

                    assert (
                        len(input_data) == 3
                    )  # (clip_encodings, t5_encodings, prompt)
                    assert labels.shape == (batch_size, 3, 256, 256)
                    assert input_data["clip"].shape == (
                        batch_size,
                        77,
                    )
                    assert input_data["t5"].shape == (
                        batch_size,
                        256,
                    )

                state = dl.state_dict()

                # Create new dataloader, restore checkpoint, and check if next data yielded is the same as above
                dl_resumed = config.dataloader.build(
                    dp_world_size=world_size,
                    dp_rank=rank,
                    local_batch_size=batch_size,
                    tokenizer=tokenizer,
                    seq_len=config.training.seq_len,
                )
                dl_resumed.load_state_dict(state)
                it_resumed = iter(dl_resumed)

                for i in range(num_steps):
                    expected_input_ids, expected_labels = next(it)
                    input_ids, labels = next(it_resumed)

                    assert torch.equal(input_ids["clip"], expected_input_ids["clip"])
                    assert torch.equal(input_ids["t5"], expected_input_ids["t5"])
                    assert torch.equal(labels, expected_labels)
