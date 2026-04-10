# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader


class DummyDataset(IterableDataset):
    """A simple dummy dataset for testing."""

    def __iter__(self):
        for i in range(100):
            yield {"input": i}, i


class DummyTokenizer(BaseTokenizer):
    """A dummy tokenizer for testing that implements BaseTokenizer interface."""

    def __init__(self):
        super().__init__()
        self.eos_id = 2
        self.bos_id = 1

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        # Simple encoding: convert each character to its ASCII value
        tokens = [ord(c) for c in text]
        if add_bos:
            tokens.insert(0, self.bos_id)  # BOS token
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        # Simple decoding: convert ASCII values back to characters
        return "".join(chr(t) for t in token_ids if t > 2)

    def get_vocab_size(self) -> int:
        return 256  # ASCII range


class TestParallelAwareDataloader(unittest.TestCase):
    def test_dataloader_yields_correct_batches(self):
        """Test that the dataloader correctly yields batched data from the dataset."""
        dataset = DummyDataset()
        batch_size = 4

        dataloader = ParallelAwareDataloader(
            dataset,
            dp_rank=0,
            dp_world_size=1,
            batch_size=batch_size,
        )

        batches = list(dataloader)

        # DummyDataset yields 100 items, so we expect 25 batches of size 4
        self.assertEqual(len(batches), 25)

        # Check first batch structure and values
        first_batch_input, first_batch_label = batches[0]
        self.assertEqual(len(first_batch_input["input"]), batch_size)
        self.assertEqual(len(first_batch_label), batch_size)

        # Verify first batch contains expected values (0, 1, 2, 3)
        self.assertEqual(first_batch_input["input"].tolist(), [0, 1, 2, 3])
        self.assertEqual(first_batch_label.tolist(), [0, 1, 2, 3])

        # Check last batch
        last_batch_input, last_batch_label = batches[-1]
        self.assertEqual(last_batch_input["input"].tolist(), [96, 97, 98, 99])
        self.assertEqual(last_batch_label.tolist(), [96, 97, 98, 99])

    def test_validate_kwargs_rejects_invalid_kwargs(self):
        """Test that passing invalid kwargs raises ValueError."""
        dataset = DummyDataset()

        with self.assertRaises(ValueError) as context:
            ParallelAwareDataloader(
                dataset,
                dp_rank=0,
                dp_world_size=1,
                invalid_arg=42,
            )

        self.assertIn("Invalid dataloader kwargs", str(context.exception))
        self.assertIn("invalid_arg", str(context.exception))

    def test_config_batch_size_overwritten_by_explicit_batch_size(self):
        """Test that batch_size in config kwargs is overwritten by explicit batch_size."""
        dataset = DummyDataset()

        config_kwargs = {"batch_size": 2, "num_workers": 0}

        explicit_batch_size = 8

        # Merge kwargs with explicit args taking precedence (same pattern as in dataset files)
        dataloader_kwargs = {
            **config_kwargs,
            "batch_size": explicit_batch_size,
        }

        dataloader = ParallelAwareDataloader(
            dataset,
            dp_rank=0,
            dp_world_size=1,
            **dataloader_kwargs,
        )

        # Verify that batch_size is the explicit one, not the config one
        self.assertEqual(dataloader.batch_size, explicit_batch_size)

    def test_build_dataloader_with_trainer_config(self):
        """Verify batch_size from training.local_batch_size is correctly used."""
        tokenizer = DummyTokenizer()

        dl_config = HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
            num_workers=2,
        )

        dataloader = HuggingFaceTextDataLoader(
            dl_config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            seq_len=512,
            local_batch_size=8,
        )

        self.assertEqual(dataloader.batch_size, 8)
        self.assertEqual(dataloader.num_workers, 2)

    def test_positions_matching_sequences(self):
        tokenizer = DummyTokenizer()

        dl_config = HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
            num_workers=0,
            infinite=False,
        )

        dataloader = HuggingFaceTextDataLoader(
            dl_config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            seq_len=(seq_len := 512),
            local_batch_size=8,
        )

        for batch, _ in zip(map(lambda x: x[0], dataloader), range(10)):
            batch_input_ids = batch["input"]
            batch_positions = batch["positions"]
            for input_ids, positions in zip(batch_input_ids, batch_positions):
                for i, (tok, pos) in enumerate(zip(input_ids, positions)):
                    # pos is less then seq_len
                    self.assertLess(pos.item(), seq_len)
                    self.assertGreaterEqual(pos.item(), 0)
                    if i == 0:
                        # First token should always have position 0
                        self.assertEqual(pos.item(), 0)
                    if i > 0 and pos.item() > 0:
                        # Position should increment by 1 for each subsequent token
                        self.assertEqual(pos.item(), positions[i - 1].item() + 1)
                    if tok == tokenizer.eos_id and i < len(input_ids) - 1:
                        # After EOS, positions should reset to 0
                        self.assertEqual(positions[i + 1].item(), 0)
                    if tok == tokenizer.bos_id and i > 0:
                        # BOS token should have position 0
                        self.assertEqual(pos.item(), 0)


if __name__ == "__main__":
    unittest.main()
