# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from datasets import load_dataset

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.experiments.sft.dataset import SFTDataset


def _process_gsm8k_default(sample):
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]


def _make_tokenizer() -> HuggingFaceTokenizer:
    """Load the test tokenizer (has a chat template configured)."""
    return HuggingFaceTokenizer(tokenizer_path="./tests/assets/tokenizer")


def _load_test_dataset():
    return load_dataset("json", data_files="tests/assets/sft_test/data.json", split="train")


def _make_dataset(**kwargs) -> SFTDataset:
    """Create an SFTDataset with test defaults."""
    defaults = {
        "dataset": _load_test_dataset(),
        "tokenizer": _make_tokenizer(),
        "sample_processor": _process_gsm8k_default,
        "seq_len": 512,
    }
    defaults.update(kwargs)
    return SFTDataset(**defaults)


class TestSFTDatasetLabelMasking(unittest.TestCase):
    """Test that prompt tokens are masked and response tokens have correct IDs."""

    def test_label_masking(self):
        ds = _make_dataset(pack_sequences=False)

        # Get a single sample
        sample = next(iter(ds))
        input_ids, labels = sample[0]["input"], sample[1]

        self.assertEqual(input_ids.shape, (512,))
        self.assertEqual(labels.shape, (512,))

        # Find the boundary between IGNORE_INDEX and real labels
        # The first non-IGNORE_INDEX label should be a response token
        mask = labels != IGNORE_INDEX
        self.assertTrue(mask.any(), "There should be at least some non-masked labels")

        # The beginning of the labels should be IGNORE_INDEX (prompt is masked)
        self.assertEqual(
            labels[0].item(),
            IGNORE_INDEX,
            "First label should be IGNORE_INDEX (prompt token)",
        )

        # Find first non-masked position
        first_response_idx = mask.nonzero(as_tuple=True)[0][0].item()
        self.assertGreater(
            first_response_idx,
            0,
            "Response should not start at position 0 (prompt should be masked)",
        )

        # All labels before the first response token should be IGNORE_INDEX
        self.assertTrue(
            (labels[:first_response_idx] == IGNORE_INDEX).all(),
            "All prompt labels should be IGNORE_INDEX",
        )

        # Response labels should be valid token IDs (not IGNORE_INDEX),
        # until we hit padding (which is also IGNORE_INDEX)
        response_labels = labels[first_response_idx:]
        # Find first padding position in response (if any)
        response_mask = response_labels != IGNORE_INDEX
        if not response_mask.all():
            # There's padding after the response
            last_response_idx = response_mask.nonzero(as_tuple=True)[0][-1].item()
            response_labels = response_labels[: last_response_idx + 1]

        # All non-padding response labels should be >= 0
        self.assertTrue(
            (response_labels >= 0).all(),
            "Response labels should be valid token IDs",
        )

    def test_labels_match_shifted_tokens(self):
        """Verify that response labels are the next-token-prediction targets."""
        tokenizer = _make_tokenizer()
        ds = _make_dataset(tokenizer=tokenizer, pack_sequences=False)

        # Manually tokenize the first sample using the same processor
        sample_data = {"question": "What is 2 + 3?", "answer": "2 + 3 = 5. #### 5"}
        messages = _process_gsm8k_default(sample_data)

        full_text = tokenizer.apply_chat_template(messages)
        full_tokens = tokenizer.encode(full_text, add_bos=True, add_eos=True)

        # Get dataset output
        sample = next(iter(ds))
        input_ids = sample[0]["input"]
        labels = sample[1]

        # The non-padded portion of input_ids should match full_tokens[:-1]
        actual_len = len(full_tokens) - 1  # -1 for the shift
        # Non-padded input should match
        for i in range(actual_len):
            self.assertEqual(
                input_ids[i].item(),
                full_tokens[i],
                f"Input token mismatch at position {i}",
            )


class TestSFTDatasetPacking(unittest.TestCase):
    """Test sequence packing behavior."""

    def test_packing_multiple_examples(self):
        """Short examples should be packed into a single sequence."""
        ds = _make_dataset(seq_len=2048, pack_sequences=True)

        sample = next(iter(ds))
        input_ids, labels = sample[0]["input"], sample[1]

        self.assertEqual(input_ids.shape[0], 2048)
        self.assertEqual(labels.shape[0], 2048)

        # With a 2048 seq_len and short test examples, multiple examples
        # should be packed. Check that there are multiple regions of
        # non-IGNORE_INDEX labels (one per packed example)
        mask = labels != IGNORE_INDEX
        non_masked = mask.nonzero(as_tuple=True)[0]

        if len(non_masked) > 1:
            # Look for gaps in the non-masked positions (boundaries between examples)
            diffs = non_masked[1:] - non_masked[:-1]
            # If there are multiple packed examples, there should be gaps > 1
            # (from the prompt masking of the second example)
            has_gaps = (diffs > 1).any()
            self.assertTrue(
                has_gaps,
                "Multiple examples should be packed, creating gaps in response labels",
            )

    def test_packing_padding(self):
        """Padding positions should use EOS for input and IGNORE_INDEX for labels."""
        tokenizer = _make_tokenizer()
        eos_id = tokenizer.eos_id

        ds = _make_dataset(
            tokenizer=tokenizer, seq_len=2048, pack_sequences=True, infinite=False
        )

        # Consume all sequences
        sequences = list(ds)
        self.assertGreater(len(sequences), 0)

        # The last sequence likely has padding
        last_input, last_labels = sequences[-1][0]["input"], sequences[-1][1]

        # Find padding region: after the last real token
        # Check that padding labels are IGNORE_INDEX
        padding_mask = last_labels == IGNORE_INDEX

        # There should be some padding in the last sequence
        # (unless the data perfectly fills the last sequence, which is unlikely)
        if padding_mask.any():
            # Get the trailing padding region
            non_padding = (last_labels != IGNORE_INDEX).nonzero(as_tuple=True)[0]
            if len(non_padding) > 0:
                last_real = non_padding[-1].item()
                if last_real < len(last_labels) - 1:
                    # Verify trailing padding
                    trailing_input = last_input[last_real + 1 :]
                    trailing_labels = last_labels[last_real + 1 :]

                    # All trailing labels should be IGNORE_INDEX
                    self.assertTrue(
                        (trailing_labels == IGNORE_INDEX).all(),
                        "Trailing padding labels should be IGNORE_INDEX",
                    )

                    # All trailing input should be EOS
                    if eos_id is not None:
                        self.assertTrue(
                            (trailing_input == eos_id).all(),
                            "Trailing padding input should be EOS tokens",
                        )


class TestSFTDatasetDropOverLength(unittest.TestCase):
    """Test that examples exceeding seq_len are dropped (not truncated)."""

    def test_over_length_examples_dropped(self):
        # seq_len=16 is shorter than any tokenized sample, so all are dropped
        ds = _make_dataset(seq_len=16, pack_sequences=False, infinite=False)
        with self.assertLogs(level="DEBUG") as cm:
            sequences = list(ds)
        self.assertEqual(len(sequences), 0, "All over-length examples should be dropped")
        drop_logs = [m for m in cm.output if "Dropping sample" in m and "seq_len" in m]
        self.assertEqual(len(drop_logs), 10, "Should log a drop message for each sample")

    def test_short_examples_kept(self):
        # seq_len=512 is longer than test samples, so all are kept
        ds = _make_dataset(seq_len=512, pack_sequences=False, infinite=False)
        sequences = list(ds)
        self.assertEqual(len(sequences), 10, "All short examples should be kept")


class TestSFTDatasetNoPacking(unittest.TestCase):
    """Test that without packing, each example is yielded independently."""

    def test_no_packing(self):
        ds = _make_dataset(pack_sequences=False, infinite=False)

        sequences = list(ds)
        # Should have one sequence per sample (10 samples in test data)
        self.assertEqual(len(sequences), 10)

        for input_dict, labels in sequences:
            self.assertEqual(input_dict["input"].shape[0], 512)
            self.assertEqual(labels.shape[0], 512)

    def test_no_packing_each_example_has_response_labels(self):
        """Each unpacked example should have its own response labels."""
        ds = _make_dataset(pack_sequences=False, infinite=False)

        for input_dict, labels in ds:
            mask = labels != IGNORE_INDEX
            self.assertTrue(
                mask.any(),
                "Each example should have at least some response labels",
            )


class TestSFTDatasetCheckpointing(unittest.TestCase):
    """Test state_dict / load_state_dict round-trip."""

    def test_checkpoint_roundtrip(self):
        tokenizer = _make_tokenizer()

        ds1 = _make_dataset(
            tokenizer=tokenizer, pack_sequences=True, infinite=False
        )

        # Consume some samples
        it = iter(ds1)
        first_batch = next(it)
        second_batch = next(it)

        # Save state
        state = ds1.state_dict()

        # Create a new dataset and load state
        ds2 = _make_dataset(
            tokenizer=tokenizer, pack_sequences=True, infinite=False
        )
        ds2.load_state_dict(state)

        # The remaining data from ds2 should match ds1
        remaining_ds1 = list(ds1)
        remaining_ds2 = list(ds2)

        # They should produce the same remaining sequences
        self.assertEqual(len(remaining_ds1), len(remaining_ds2))
        for (in1, lb1), (in2, lb2) in zip(remaining_ds1, remaining_ds2):
            torch.testing.assert_close(in1["input"], in2["input"])
            torch.testing.assert_close(lb1, lb2)


class TestSFTDatasetShuffling(unittest.TestCase):
    """Test that data is shuffled between epochs."""

    def test_shuffle_between_epochs(self):
        """Second epoch should see a different sample order than the first."""
        ds = _make_dataset(seq_len=512, pack_sequences=False, infinite=True)

        num_samples = 10  # test dataset has 10 samples
        it = iter(ds)

        # Collect input_ids for two full epochs
        epoch1 = [next(it)[0]["input"] for _ in range(num_samples)]
        epoch2 = [next(it)[0]["input"] for _ in range(num_samples)]

        # The sets of sequences should be the same (same data), but order
        # should differ due to shuffling. Compare as tuples for hashability.
        epoch1_order = [tuple(t.tolist()) for t in epoch1]
        epoch2_order = [tuple(t.tolist()) for t in epoch2]

        # Same data
        self.assertEqual(sorted(epoch1_order), sorted(epoch2_order))
        # Different order
        self.assertNotEqual(
            epoch1_order, epoch2_order,
            "Second epoch should be shuffled into a different order",
        )

    def test_epoch_persisted_in_state_dict(self):
        ds = _make_dataset(seq_len=512, pack_sequences=False, infinite=True)

        # Consume one full epoch (10 samples) plus one more to trigger re-loop.
        # The epoch counter increments after the inner for-loop exhausts,
        # which happens when we request the 11th sample.
        it = iter(ds)
        for _ in range(11):
            next(it)
        # After re-loop, epoch should be incremented
        state = ds.state_dict()
        self.assertIn("epoch", state)
        self.assertEqual(state["epoch"], 1)

        # Restore into a fresh dataset
        ds2 = _make_dataset(seq_len=512, pack_sequences=False, infinite=True)
        ds2.load_state_dict(state)
        self.assertEqual(ds2._epoch, 1)


class TestSFTDatasetChatTemplate(unittest.TestCase):
    """Test that the chat template is applied correctly."""

    def test_chat_template_applied(self):
        tokenizer = _make_tokenizer()

        # Manually apply chat template to verify format
        messages = [
            {"role": "user", "content": "What is 2 + 3?"},
            {"role": "assistant", "content": "2 + 3 = 5. #### 5"},
        ]
        formatted = tokenizer.apply_chat_template(messages)

        # The test tokenizer uses <|im_start|>role\ncontent<|im_end|> format
        self.assertIn("user", formatted)
        self.assertIn("What is 2 + 3?", formatted)
        self.assertIn("assistant", formatted)
        self.assertIn("2 + 3 = 5. #### 5", formatted)

    def test_prompt_only_template(self):
        """Verify prompt-only rendering with add_generation_prompt."""
        tokenizer = _make_tokenizer()

        prompt_messages = [{"role": "user", "content": "What is 2 + 3?"}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True
        )

        # Should contain user content but not assistant response
        self.assertIn("What is 2 + 3?", prompt_text)
        self.assertNotIn("2 + 3 = 5", prompt_text)


if __name__ == "__main__":
    unittest.main()
