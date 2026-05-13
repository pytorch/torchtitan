# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

from copy import deepcopy

import torch
from datasets import Dataset

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.text_datasets import ChatDataLoader, ChatDataset
from torchtitan.models.common.attention import get_document_mask_mod

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
        chat_ds = ChatDataset(
            dataset=_load_dataset(),
            tokenizer=tokenizer,
            sample_processor=_process_sample,
            seq_len=2048,
            infinite=False,
        )

        batch, labels = next(iter(chat_ds))
        input_ids = batch["input"]
        label_ids = labels

        # Tokenize the first sample directly to get ground truth tokens. ChatDataset shuffles the
        # dataset internally at init, and ChatDataset._original_data is the post-shuffle dataset.
        sample = chat_ds._original_data[0]
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
        chat_ds_resumed = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=_process_sample,
            seq_len=seq_len,
            infinite=False,
        )
        chat_ds_resumed.load_state_dict(state)

        self.assertEqual(chat_ds_resumed._sample_idx, state["sample_idx"])
        self.assertEqual(chat_ds_resumed._epoch, state["epoch"])

        remaining = list(chat_ds_resumed)
        self.assertGreater(len(remaining), 0, "Restored dataset should produce batches")
        for batch, labels in remaining:
            self.assertEqual(batch["input"].shape[0], seq_len)
            self.assertEqual(batch["positions"].shape[0], seq_len)
            self.assertEqual(labels.shape[0], seq_len)

    def test_yield_same_data_multi_epoch(self):
        def _build_dataloader(streaming, batch_size, seq_len, world_size, rank):
            tokenizer_config = HuggingFaceTokenizer.Config()
            dl_config = ChatDataLoader.Config(
                dataset_path="json",
                load_dataset_kwargs={
                    "data_files": _DATA_PATH,
                    "split": "train",
                    "streaming": streaming,
                },
                sample_processor=_process_sample,
                infinite=True,
            )

            return dl_config.build(
                dp_world_size=world_size,
                dp_rank=rank,
                tokenizer=tokenizer_config.build(tokenizer_path=_TOKENIZER_PATH),
                seq_len=seq_len,
                local_batch_size=batch_size,
            )

        for streaming in [True, False]:
            for world_size in [2, 4]:
                for rank in range(world_size):
                    batch_size = 1
                    seq_len = 128
                    dl = _build_dataloader(
                        streaming, batch_size, seq_len, world_size, rank
                    )

                    # Consume at least 2 epochs
                    it = iter(dl)
                    for _ in range(8):
                        next(it)

                    state = deepcopy(dl.state_dict())
                    # Restore
                    dl_resumed = _build_dataloader(
                        streaming, batch_size, seq_len, world_size, rank
                    )
                    dl_resumed.load_state_dict(state)
                    # verify yield gives same input data
                    # test assertion seveal times in order to empty potential input buffer.
                    it_resumed = iter(dl_resumed)

                    expected, expected_labels = next(it)
                    input_ids, labels = next(it_resumed)
                    for _ in range(3):
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
                        self.assertEqual(
                            next(it)[0]["input"].tolist(),
                            next(it_resumed)[0]["input"].tolist(),
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


class TestDocumentMaskBlocksCrossDocAttention(unittest.TestCase):
    """Verify that position-based document masks block cross-document attention."""

    def test_packed_samples_block_cross_document_attention(self):
        tokenizer = _load_tokenizer()
        ds = _load_dataset()
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=_process_sample,
            seq_len=2048,
            infinite=False,
        )

        r0 = chat_ds._tokenize_sample(ds[0])
        r1 = chat_ds._tokenize_sample(ds[1])
        self.assertIsNotNone(r0)
        self.assertIsNotNone(r1)
        input_ids_0, _ = r0
        input_ids_1, _ = r1

        packed = input_ids_0 + input_ids_1
        boundary = len(input_ids_0)
        positions = torch.tensor(
            [list(range(len(input_ids_0))) + list(range(len(input_ids_1)))]
        )

        mask_mod = get_document_mask_mod(positions)
        b, h = torch.tensor(0), torch.tensor(0)

        self.assertFalse(
            mask_mod(b, h, torch.tensor(boundary), torch.tensor(boundary - 1)).item(),
        )
        self.assertFalse(
            mask_mod(b, h, torch.tensor(len(packed) - 1), torch.tensor(0)).item(),
        )
        self.assertTrue(
            mask_mod(b, h, torch.tensor(boundary - 1), torch.tensor(0)).item(),
        )
        self.assertTrue(
            mask_mod(
                b, h, torch.tensor(len(packed) - 1), torch.tensor(boundary)
            ).item(),
        )


if __name__ == "__main__":
    unittest.main()
