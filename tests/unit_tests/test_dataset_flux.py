# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchtitan.components.data import GrainDataLoader
from torchtitan.config import ConfigManager
from torchtitan.models.flux.config_registry import (
    flux_debugmodel,
    flux_dev,
    flux_schnell,
)
from torchtitan.models.flux.flux_datasets import FluxSampleProcessor


class TestFluxDataLoader(unittest.TestCase):
    def setUp(self):
        # Import here to avoid circular import during test collection
        from torchtitan.models.flux.flux_datasets import DATASETS, FluxCollator

        self._dataset = DATASETS["cc12m-test"]
        self._collator = FluxCollator.Config()

    def test_collator_moves_image_to_labels(self):
        rows = [
            {
                "t5": torch.tensor([1, 2]),
                "clip": torch.tensor([3]),
                "prompt": "first",
                "image": torch.full((3, 2, 2), 1.0),
            },
            {
                "t5": torch.tensor([4, 5]),
                "clip": torch.tensor([6]),
                "prompt": "second",
                "image": torch.full((3, 2, 2), 2.0),
            },
        ]

        model_inputs, labels = self._collator.build(context=None)(rows)

        self.assertNotIn("image", model_inputs)
        self.assertEqual(model_inputs["prompt"], ["first", "second"])
        self.assertTrue(
            torch.equal(labels, torch.stack([row["image"] for row in rows]))
        )

    def test_recipe_sequence_length_matches_dataset_geometry(self):
        for recipe, expected_seq_len in (
            (flux_debugmodel, 512),
            (flux_dev, 768),
            (flux_schnell, 512),
        ):
            with self.subTest(recipe=recipe.__name__):
                config = recipe()
                processor = config.dataloader.dataset.processor

                self.assertIsInstance(processor, FluxSampleProcessor.Config)
                self.assertEqual(processor.img_size, 256)
                self.assertEqual(config.training.seq_len, expected_seq_len)

    def test_validation_timestep_preserves_sample(self):
        from torchtitan.models.flux.flux_datasets import _add_validation_timestep

        sample = {
            "t5": torch.tensor([1, 2]),
            "clip": torch.tensor([3]),
            "prompt": "caption",
            "image": torch.ones(3, 2, 2),
        }

        timed = _add_validation_timestep(3, sample)

        self.assertEqual(timed["timestep"], 3.5 / 8)
        self.assertNotIn("timestep", sample)
        for key in sample:
            self.assertIs(timed[key], sample[key])

    def test_load_dataset(self):
        # The test checks for the correct tensor shapes during the first num_steps
        # The next num_steps ensure the loaded from checkpoint dataloader generates tokens and labels correctly
        for world_size in [2]:
            for rank in range(world_size):
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
                config.dataloader = GrainDataLoader.Config(
                    dataset=self._dataset,
                    collator=self._collator,
                    shuffle=False,
                    streaming_shuffle_buffer_size=128,
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
