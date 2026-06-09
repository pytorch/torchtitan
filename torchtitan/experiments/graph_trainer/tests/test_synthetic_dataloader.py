# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.experiments.graph_trainer.synthetic_dataloader import (
    SyntheticDataLoader,
    SyntheticDataset,
)


class TestSyntheticDataset(unittest.TestCase):
    def test_yields_correct_shapes(self):
        ds = SyntheticDataset(vocab_size=100, seq_len=32)
        it = iter(ds)
        input_dict, labels = next(it)

        self.assertIn("input", input_dict)
        self.assertEqual(input_dict["input"].shape, (32,))
        self.assertEqual(labels.shape, (32,))

    def test_token_ids_in_range(self):
        vocab_size = 50
        ds = SyntheticDataset(vocab_size=vocab_size, seq_len=64)
        it = iter(ds)

        for _ in range(10):
            input_dict, labels = next(it)
            self.assertTrue((input_dict["input"] >= 0).all())
            self.assertTrue((input_dict["input"] < vocab_size).all())
            self.assertTrue((labels >= 0).all())
            self.assertTrue((labels < vocab_size).all())

    def test_dtype_is_long(self):
        ds = SyntheticDataset(vocab_size=100, seq_len=16)
        it = iter(ds)
        input_dict, labels = next(it)

        self.assertEqual(input_dict["input"].dtype, torch.int64)
        self.assertEqual(labels.dtype, torch.int64)

    def test_invalid_vocab_size_raises(self):
        with self.assertRaises(ValueError):
            SyntheticDataset(vocab_size=0, seq_len=16)

        with self.assertRaises(ValueError):
            SyntheticDataset(vocab_size=-1, seq_len=16)

    def test_state_dict_roundtrip(self):
        ds = SyntheticDataset(vocab_size=100, seq_len=16)
        it = iter(ds)
        for _ in range(5):
            next(it)

        state = ds.state_dict()
        self.assertEqual(state["step"], 5)

        ds2 = SyntheticDataset(vocab_size=100, seq_len=16)
        ds2.load_state_dict(state)
        self.assertEqual(ds2._step, 5)

    def test_infinite_iteration(self):
        """Verify the dataset yields indefinitely."""
        ds = SyntheticDataset(vocab_size=100, seq_len=8)
        it = iter(ds)
        for _ in range(1000):
            input_dict, labels = next(it)
            self.assertEqual(input_dict["input"].shape, (8,))


class TestSyntheticDataLoader(unittest.TestCase):
    def test_config_defaults(self):
        config = SyntheticDataLoader.Config()
        self.assertEqual(config.dataset, "synthetic")
        self.assertEqual(config.vocab_size, 0)

    def test_zero_vocab_size_raises(self):
        config = SyntheticDataLoader.Config(vocab_size=0)
        with self.assertRaises(ValueError):
            SyntheticDataLoader(
                config,
                dp_world_size=1,
                dp_rank=0,
                seq_len=32,
                local_batch_size=2,
            )

    def test_dataloader_yields_batches(self):
        config = SyntheticDataLoader.Config(vocab_size=100)
        dl = SyntheticDataLoader(
            config,
            dp_world_size=1,
            dp_rank=0,
            seq_len=32,
            local_batch_size=4,
        )

        it = iter(dl)
        batch = next(it)
        input_dict, labels = batch

        self.assertIn("input", input_dict)
        # Batched shape: (batch_size, seq_len)
        self.assertEqual(input_dict["input"].shape, (4, 32))
        self.assertEqual(labels.shape, (4, 32))


class TestUseSyntheticDataloader(unittest.TestCase):
    def test_replaces_dataloader_config(self):
        from types import SimpleNamespace

        from torchtitan.experiments.graph_trainer.configs import (
            use_synthetic_dataloader,
        )

        # Create a minimal config-like object with model_spec
        config = SimpleNamespace(
            model_spec=SimpleNamespace(
                model=SimpleNamespace(vocab_size=32000),
            ),
            dataloader=SimpleNamespace(dataset="c4_test"),
        )

        use_synthetic_dataloader(config)

        self.assertIsInstance(config.dataloader, SyntheticDataLoader.Config)
        self.assertEqual(config.dataloader.vocab_size, 32000)
        self.assertEqual(config.dataloader.dataset, "synthetic")


if __name__ == "__main__":
    unittest.main()
