# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

from copy import deepcopy

from datasets import Dataset

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.text_datasets import ChatDataset

# Path to the test tokenizer and fixture data
_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
_TOKENIZER_PATH = os.path.join(_ASSETS_DIR, "tokenizer")
_DATA_PATH = os.path.join(_ASSETS_DIR, "sft_test", "data.json")


def _process_sample(sample):
    """Convert a test data sample into [user, assistant] messages."""
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]


def _load_tokenizer():
    return HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)


def _load_dataset():
    return Dataset.from_json(_DATA_PATH)


class TestChatDatasetLabelMasking(unittest.TestCase):
    """Prompt tokens should be masked (IGNORE_INDEX), assistant tokens should not."""

    def test_prompt_masked_response_unmasked(self):
        tokenizer = _load_tokenizer()
        ds = _load_dataset()
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=_process_sample,
            seq_len=2048,
            infinite=False,
        )

        batch, labels = next(iter(chat_ds))
        input_ids = batch["input"]
        label_ids = labels

        self.assertEqual(input_ids.shape, label_ids.shape)
        self.assertEqual(input_ids.shape[0], 2048)

        # Some labels at the start should be IGNORE_INDEX (prompt masking)
        masked = (label_ids == IGNORE_INDEX).nonzero(as_tuple=True)[0]
        unmasked = (label_ids != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        self.assertGreater(len(masked), 0, "Expected some masked prompt labels")
        self.assertGreater(len(unmasked), 0, "Expected some unmasked response labels")

        # All masked positions should precede all unmasked non-padding positions.
        # The unmasked region is the response, then padding follows with IGNORE_INDEX.
        # Find first unmasked position and last contiguous unmasked position.
        first_unmasked = unmasked[0].item()
        self.assertGreater(first_unmasked, 0, "First token label should be masked")


class TestChatDatasetShiftedTokens(unittest.TestCase):
    """input_ids = tokens[:-1], label_ids = tokens[1:]."""

    def test_shifted_by_one(self):
        tokenizer = _load_tokenizer()
        ds = _load_dataset()
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=_process_sample,
            seq_len=2048,
            infinite=False,
        )

        batch, labels = next(iter(chat_ds))
        input_ids = batch["input"]
        label_ids = labels

        # Tokenize the first sample directly to get ground truth tokens
        sample = ds[0]
        messages = _process_sample(sample)
        full_text = tokenizer.apply_chat_template(messages)
        # Chat templates already include end tokens, so no add_eos
        full_tokens = tokenizer.encode(full_text, add_bos=True, add_eos=False)

        expected_input = full_tokens[:-1]
        expected_label = full_tokens[1:]

        # The non-padded portion of input_ids should match expected_input
        seq_len_actual = len(expected_input)
        self.assertEqual(
            input_ids[:seq_len_actual].tolist(),
            expected_input,
        )
        # The non-masked, non-padded portion of label_ids that corresponds to
        # the response should come from full_tokens[1:]
        # Just verify the response portion matches
        prompt_text = tokenizer.apply_chat_template(
            messages[:1], add_generation_prompt=True
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        response_start = len(prompt_tokens) - 1
        self.assertGreaterEqual(response_start, 0)
        self.assertNotEqual(
            label_ids[response_start].item(),
            IGNORE_INDEX,
            "First assistant token should not be masked",
        )
        self.assertEqual(
            label_ids[response_start:seq_len_actual].tolist(),
            expected_label[response_start:],
        )


class TestChatDatasetGreedyPacking(unittest.TestCase):
    """Multiple short samples packed into one sequence with small seq_len."""

    def test_packing_multiple_samples(self):
        tokenizer = _load_tokenizer()
        ds = _load_dataset()
        # seq_len=256 should fit multiple of the shortest samples (effective_len ~79)
        seq_len = 256
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=_process_sample,
            seq_len=seq_len,
            infinite=False,
        )

        batches = list(chat_ds)
        # With 10 samples of lengths 79-123, they should pack into fewer than 10 batches
        self.assertGreater(len(batches), 0)
        self.assertLess(len(batches), 10)

        # Each batch should have seq_len tokens
        for batch, labels in batches:
            self.assertEqual(batch["input"].shape[0], seq_len)
            self.assertEqual(labels.shape[0], seq_len)
            self.assertIn("positions", batch)
            self.assertEqual(batch["positions"].shape[0], seq_len)


class TestChatDatasetPerDocumentPositions(unittest.TestCase):
    """Positions reset to 0 at each document boundary in packed mode."""

    def test_positions_reset_at_boundaries(self):
        tokenizer = _load_tokenizer()
        ds = _load_dataset()
        seq_len = 256
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=_process_sample,
            seq_len=seq_len,
            infinite=False,
        )

        batch, _ = next(iter(chat_ds))
        positions = batch["positions"]

        # Positions should start at 0
        self.assertEqual(positions[0].item(), 0)

        # Find where positions reset to 0 (document boundaries)
        resets = (positions[1:] == 0).nonzero(as_tuple=True)[0]
        # With seq_len=256 and samples of ~79 tokens, at least one reset
        self.assertGreater(
            len(resets), 0, "Expected at least one position reset (document boundary)"
        )

        # Between resets, positions should be consecutive (0, 1, 2, ...)
        pos_list = positions.tolist()
        for i in range(1, len(pos_list)):
            if pos_list[i] == 0:
                # Document boundary: reset is fine
                continue
            self.assertEqual(
                pos_list[i],
                pos_list[i - 1] + 1,
                f"Positions should be consecutive at index {i}, "
                f"got {pos_list[i - 1]} -> {pos_list[i]}",
            )


class TestChatDatasetDropOnOverflow(unittest.TestCase):
    """Samples exceeding seq_len are silently dropped."""

    def test_all_dropped_with_tiny_seq_len(self):
        tokenizer = _load_tokenizer()
        ds = _load_dataset()
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=_process_sample,
            seq_len=32,
            infinite=False,
        )

        batches = list(chat_ds)
        self.assertEqual(len(batches), 0, "All samples should be dropped at seq_len=32")


class TestChatDatasetMessageValidation(unittest.TestCase):
    """Non-[user, assistant] messages raise ValueError."""

    def test_wrong_first_role(self):
        tokenizer = _load_tokenizer()

        def bad_processor(sample):
            return [
                {"role": "system", "content": "You are helpful."},
                {"role": "assistant", "content": "OK"},
            ]

        ds = Dataset.from_list([{"question": "hi", "answer": "bye"}])
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=bad_processor,
            seq_len=2048,
            infinite=False,
        )

        with self.assertRaises(ValueError, msg="system role should raise"):
            next(iter(chat_ds))

    def test_wrong_second_role(self):
        tokenizer = _load_tokenizer()

        def bad_processor(sample):
            return [
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "hello again"},
            ]

        ds = Dataset.from_list([{"question": "hi", "answer": "bye"}])
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=bad_processor,
            seq_len=2048,
            infinite=False,
        )

        with self.assertRaises(ValueError, msg="two user messages should raise"):
            next(iter(chat_ds))

    def test_three_messages(self):
        tokenizer = _load_tokenizer()

        def bad_processor(sample):
            return [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "bye"},
            ]

        ds = Dataset.from_list([{"question": "hi", "answer": "bye"}])
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=bad_processor,
            seq_len=2048,
            infinite=False,
        )

        with self.assertRaises(ValueError, msg="3 messages should raise"):
            next(iter(chat_ds))


class TestChatDatasetCheckpointing(unittest.TestCase):
    """state_dict / load_state_dict round-trips correctly."""

    def test_state_dict_round_trip(self):
        tokenizer = _load_tokenizer()
        ds = _load_dataset()
        seq_len = 128
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=_process_sample,
            seq_len=seq_len,
            infinite=False,
        )

        # Consume one packed batch
        it = iter(chat_ds)
        next(it)

        state = chat_ds.state_dict()

        # Verify state has expected keys
        self.assertIn("sample_idx", state)
        self.assertIn("epoch", state)
        self.assertIn("inputs_buffer", state)
        self.assertIn("labels_buffer", state)
        self.assertIn("positions_buffer", state)
        self.assertGreater(state["sample_idx"], 0)
        self.assertEqual(state["epoch"], 0)

        # Restore and verify the dataset can produce valid packed batches
        chat_ds2 = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=_process_sample,
            seq_len=seq_len,
            infinite=False,
        )
        chat_ds2.load_state_dict(state)

        self.assertEqual(chat_ds2._sample_idx, state["sample_idx"])
        self.assertEqual(chat_ds2._epoch, state["epoch"])

        remaining = list(chat_ds2)
        self.assertGreater(len(remaining), 0, "Restored dataset should produce batches")
        for batch, labels in remaining:
            self.assertEqual(batch["input"].shape[0], seq_len)
            self.assertEqual(batch["positions"].shape[0], seq_len)
            self.assertEqual(labels.shape[0], seq_len)

    def test_yield_same_data_multi_epoch(self):
        tokenizer = _load_tokenizer()
        ds = _load_dataset()
        seq_len = 128
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=_process_sample,
            seq_len=seq_len,
            infinite=True,
        )

        # Consume at least 2 epochs
        it = iter(chat_ds)
        for i in range(25):
            next(it)

        state = deepcopy(chat_ds.state_dict())

        # Restore
        chat_ds2 = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=_process_sample,
            seq_len=seq_len,
            infinite=True,
        )
        chat_ds2.load_state_dict(state)

        # verify yield gives same input data
        # test assertion seveal times in order to empty potential input buffer.
        it2 = iter(chat_ds2)
        for _ in range(10):
            self.assertEqual(
                next(it)[0]["input"].tolist(), next(it2)[0]["input"].tolist()
            )


class TestChatDatasetInfiniteLooping(unittest.TestCase):
    """Dataset re-shuffles and continues after exhausting data."""

    def test_infinite_produces_more_than_dataset_size(self):
        tokenizer = _load_tokenizer()
        ds = _load_dataset()
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=_process_sample,
            seq_len=2048,
            infinite=True,
        )

        # The dataset has 10 samples. Consuming 15 requires at least one re-loop.
        it = iter(chat_ds)
        samples = [next(it) for _ in range(15)]
        self.assertEqual(len(samples), 15)

        # After the first 10, the epoch counter should have incremented
        self.assertGreaterEqual(chat_ds._epoch, 1)

    def test_infinite_packed(self):
        tokenizer = _load_tokenizer()
        ds = _load_dataset()
        seq_len = 256
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=_process_sample,
            seq_len=seq_len,
            infinite=True,
        )

        # Consume enough packed batches to exceed the 10-sample dataset
        it = iter(chat_ds)
        batches = [next(it) for _ in range(20)]
        self.assertEqual(len(batches), 20)
        self.assertGreaterEqual(chat_ds._epoch, 1)


if __name__ == "__main__":
    unittest.main()
