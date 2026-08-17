# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest

import torch
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets.text_datasets import (
    HFDataSource,
    HuggingFaceTextDataLoader,
    InterleavedHuggingFaceTextDataLoader,
)


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

    def test_load_state_dict_missing_rank_warning_includes_rank_id(self):
        """The missing-rank warning must interpolate the actual rank key."""
        dataloader = ParallelAwareDataloader(
            DummyDataset(),
            dp_rank=0,
            dp_world_size=1,
            batch_size=4,
        )
        # Non-empty state that lacks this rank's key hits the warning branch.
        state_dict = {"dp_rank_1": b"", "world_size": 1}

        with self.assertLogs(level="WARNING") as cm:
            dataloader.load_state_dict(state_dict)

        output = "\n".join(cm.output)
        self.assertIn(dataloader._rank_id, output)
        self.assertNotIn("{self._rank_id}", output)

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

    def test_build_packed_token_dataloader(self):
        """Verify the DP-rank token budget is returned without batching."""
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
            max_seq_len=512,
            num_tokens_per_batch=4096,
        )

        self.assertIsNone(dataloader.batch_size)
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
            max_seq_len=(max_seq_len := 512),
            num_tokens_per_batch=4096,
        )

        for batch, _ in itertools.islice(dataloader, 10):
            batch_input_ids = batch["input"]
            batch_positions = batch["positions"]
            self.assertEqual(batch_input_ids.shape, (4096,))
            self.assertEqual(batch_positions.shape, (4096,))
            for i, (tok, pos) in enumerate(
                zip(batch_input_ids, batch_positions, strict=True)
            ):
                self.assertLess(pos.item(), max_seq_len)
                self.assertGreaterEqual(pos.item(), 0)
                if i % max_seq_len == 0:
                    self.assertEqual(pos.item(), 0)
                if i > 0 and pos.item() > 0:
                    self.assertEqual(pos.item(), batch_positions[i - 1].item() + 1)
                if tok == tokenizer.eos_id and i < len(batch_input_ids) - 1:
                    self.assertEqual(batch_positions[i + 1].item(), 0)
                if tok == tokenizer.bos_id and i > 0:
                    self.assertEqual(pos.item(), 0)

class TestInterleavedHuggingFaceTextDataLoader(unittest.TestCase):
    def _make_config(self, **kwargs) -> InterleavedHuggingFaceTextDataLoader.Config:
        defaults = dict(
            sources=[
                HFDataSource(dataset="c4_test", weight=1.0, infinite=False),
                HFDataSource(dataset="c4_test", weight=1.0, infinite=False),
            ],
            seed=42,
            num_workers=0,
        )
        defaults.update(kwargs)
        return InterleavedHuggingFaceTextDataLoader.Config(**defaults)

    def test_rejects_empty_sources(self):
        with self.assertRaises(ValueError) as ctx:
            InterleavedHuggingFaceTextDataLoader.Config(sources=[], seed=42)
        self.assertIn("At least one source", str(ctx.exception))

    def test_rejects_mixed_infinite(self):
        with self.assertRaises(ValueError) as ctx:
            InterleavedHuggingFaceTextDataLoader.Config(
                sources=[
                    HFDataSource(dataset="c4_test", weight=1.0, infinite=True),
                    HFDataSource(dataset="c4_test", weight=1.0, infinite=False),
                ],
                seed=42,
            )
        self.assertIn("infinite", str(ctx.exception))

    def test_construction_batch_size_and_num_workers(self):
        """Verify token chunks per batch and workers are configured."""
        config = self._make_config(num_workers=2)
        dataloader = InterleavedHuggingFaceTextDataLoader(
            config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=DummyTokenizer(),
            max_seq_len=512,
            num_tokens_per_batch=4 * 512,
        )
        self.assertEqual(dataloader.batch_size, 4)
        self.assertEqual(dataloader.num_workers, 2)

    def test_yields_input_and_positions_keys(self):
        """Batches must contain 'input' and 'positions' keys, matching single-source format."""
        config = self._make_config()
        dataloader = InterleavedHuggingFaceTextDataLoader(
            config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=DummyTokenizer(),
            max_seq_len=512,
            num_tokens_per_batch=2 * 512,
        )
        batch_input, batch_label = next(iter(dataloader))
        self.assertIn("input", batch_input)
        self.assertIn("positions", batch_input)
        self.assertEqual(batch_input["input"].shape, (2 * 512,))
        self.assertEqual(batch_input["positions"].shape, (2 * 512,))
        self.assertEqual(batch_label.shape, (2 * 512,))

    def test_single_source_equivalent_to_huggingfacetextdataloader(self):
        """A single-source interleaved dataloader must produce the same batch
        shape as HuggingFaceTextDataLoader with the same config."""
        tokenizer = DummyTokenizer()
        seq_len = 512
        local_batch_size = 4

        single_dl = HuggingFaceTextDataLoader(
            HuggingFaceTextDataLoader.Config(
                dataset="c4_test", num_workers=0, infinite=False
            ),
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            max_seq_len=seq_len,
            num_tokens_per_batch=local_batch_size * seq_len,
        )

        interleaved_dl = InterleavedHuggingFaceTextDataLoader(
            self._make_config(
                sources=[HFDataSource(dataset="c4_test", weight=1.0, infinite=False)],
            ),
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            max_seq_len=seq_len,
            num_tokens_per_batch=local_batch_size * seq_len,
        )

        single_batch_input, single_batch_labels = next(iter(single_dl))
        interleaved_batch_input, interleaved_batch_labels = next(iter(interleaved_dl))

        self.assertTrue(
            bool(
                torch.equal(
                    single_batch_input["input"],
                    interleaved_batch_input["input"],
                )
            )
        )
        self.assertTrue(
            bool(
                torch.equal(
                    single_batch_input["positions"],
                    interleaved_batch_input["positions"],
                )
            )
        )
        self.assertTrue(
            bool(
                torch.equal(
                    single_batch_labels,
                    interleaved_batch_labels,
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
