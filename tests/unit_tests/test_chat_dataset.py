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
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    get_document_mask_mod,
)

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


def _get_full_tokens(tokenizer, messages):
    full_text = tokenizer.apply_chat_template(messages).rstrip("\n")
    full_tokens = tokenizer.encode(full_text, add_bos=True, add_eos=False)
    if full_tokens[-1] != tokenizer.eos_id:
        full_tokens.append(tokenizer.eos_id)
    return full_tokens


class TestChatDatasetShiftedTokens(unittest.TestCase):
    """input_ids = tokens[:-1], assistant labels align with tokens[1:]."""

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
        full_tokens = _get_full_tokens(tokenizer, messages)

        expected_input = full_tokens[:-1]
        expected_label = full_tokens[1:]

        # The non-padded portion of input_ids should match expected_input
        seq_len_actual = len(expected_input)
        self.assertEqual(
            input_ids[:seq_len_actual].tolist(),
            expected_input,
        )

        prompt_text = tokenizer.apply_chat_template(
            messages[:1], add_generation_prompt=True
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        response_start = len(prompt_tokens) - 1
        self.assertGreaterEqual(response_start, 0)
        self.assertTrue(
            torch.all(label_ids[:response_start] == IGNORE_INDEX).item(),
            "Prompt tokens should be masked",
        )
        self.assertEqual(
            label_ids[response_start:seq_len_actual].tolist(),
            expected_label[response_start:],
        )
        self.assertTrue(torch.all(label_ids[seq_len_actual:] == IGNORE_INDEX).item())


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
    """Invalid conversation structures raise ValueError."""

    def test_system_then_assistant_only(self):
        """system + assistant without user should raise."""
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

        with self.assertRaises(ValueError):
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

    def test_ends_with_user(self):
        """Conversation ending with user (odd turns) should raise."""
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

        with self.assertRaises(ValueError, msg="ending with user should raise"):
            next(iter(chat_ds))

    def test_single_message(self):
        tokenizer = _load_tokenizer()

        def bad_processor(sample):
            return [{"role": "user", "content": "hi"}]

        ds = Dataset.from_list([{"question": "hi", "answer": "bye"}])
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=bad_processor,
            seq_len=2048,
            infinite=False,
        )

        with self.assertRaises(ValueError, msg="single message should raise"):
            next(iter(chat_ds))

    def test_valid_multiturn(self):
        """Multi-turn [user, assistant, user, assistant] should NOT raise."""
        tokenizer = _load_tokenizer()

        def good_processor(sample):
            return [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "bye"},
                {"role": "assistant", "content": "goodbye"},
            ]

        ds = Dataset.from_list([{"question": "hi", "answer": "bye"}])
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=good_processor,
            seq_len=2048,
            infinite=False,
        )

        # Should not raise
        batch, labels = next(iter(chat_ds))
        self.assertEqual(batch["input"].shape[0], 2048)

    def test_valid_system_prefix(self):
        """[system, user, assistant] should NOT raise."""
        tokenizer = _load_tokenizer()

        def good_processor(sample):
            return [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]

        ds = Dataset.from_list([{"question": "hi", "answer": "bye"}])
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=good_processor,
            seq_len=2048,
            infinite=False,
        )

        # Should not raise
        batch, labels = next(iter(chat_ds))
        self.assertEqual(batch["input"].shape[0], 2048)


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


class TestChatDatasetAssistantOnlyTemplates(unittest.TestCase):
    def _get_supervised_output(
        self,
        *,
        chat_template: str,
        messages: list[dict[str, str]],
    ) -> tuple[HuggingFaceTokenizer, list[int], str]:
        tokenizer = _load_tokenizer()
        tokenizer.set_chat_template(chat_template)

        chat_ds = ChatDataset(
            dataset=Dataset.from_list([{"id": 1}]),
            tokenizer=tokenizer,
            sample_processor=lambda sample, messages=messages: messages,
            seq_len=512,
            infinite=False,
        )

        _, labels = next(iter(chat_ds))
        supervised_token_ids = [
            token for token in labels.tolist() if token != IGNORE_INDEX
        ]
        supervised_text = tokenizer.decode(
            supervised_token_ids, skip_special_tokens=False
        )
        return tokenizer, supervised_token_ids, supervised_text

    def _get_supervised_text(
        self,
        *,
        chat_template: str,
        messages: list[dict[str, str]],
    ) -> str:
        _, _, supervised_text = self._get_supervised_output(
            chat_template=chat_template,
            messages=messages,
        )

        return supervised_text

    def test_debugmodel_template_supervises_content_only(self):
        tokenizer, supervised_token_ids, supervised_text = self._get_supervised_output(
            chat_template=(
                "{{ bos_token }}{% for msg in messages %}"
                "{{ msg.role }}\n{{ msg.content }}{{ eos_token }}"
                "{% endfor %}"
            ),
            messages=[
                {"role": "user", "content": "Q?"},
                {"role": "assistant", "content": "4"},
            ],
        )

        self.assertEqual(supervised_text, "4<|end_of_text|>")
        self.assertEqual(supervised_token_ids[-1], tokenizer.eos_id)
        self.assertNotIn("assistant\n", supervised_text)

    def test_gpt_oss_template_supervises_assistant_message_content_only(self):
        assistant_text = (
            "<|channel|>analysis<|message|>Simple arithmetic.<|end|>\n"
            "<|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|return|>"
        )
        tokenizer, supervised_token_ids, supervised_text = self._get_supervised_output(
            chat_template=(
                "{{ bos_token }}{% for msg in messages %}"
                "{% if msg.role == 'user' %}"
                "<|start|>user<|message|>{{ msg.content }}<|end|>"
                "{% else %}"
                "{{ msg.content }}"
                "{% endif %}"
                "{{ eos_token }}"
                "{% endfor %}"
            ),
            messages=[
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": assistant_text},
            ],
        )

        self.assertIn(
            "<|channel|>analysis<|message|>",
            supervised_text,
        )
        self.assertIn(
            "Simplearithmetic.<|end|>",
            supervised_text,
        )
        self.assertIn(
            "<|start|>assistant<|channel|>final<|message|>",
            supervised_text,
        )
        self.assertIn(
            "2+2=4.<|return|>",
            supervised_text,
        )
        self.assertEqual(supervised_token_ids[-1], tokenizer.eos_id)
        self.assertNotIn("What is 2 + 2?", supervised_text)

    def test_qwen3_template_supervises_assistant_content_only(self):
        supervised_text = self._get_supervised_text(
            chat_template=(
                "{{ bos_token }}{% for msg in messages %}"
                "{% if msg.role == 'user' %}"
                "<|im_start|>user\n{{ msg.content }}<|im_end|>\n"
                "{% else %}"
                "<|im_start|>assistant\n"
                "{% if msg.reasoning_content is defined %}"
                "<think>\n{{ msg.reasoning_content }}\n</think>\n\n"
                "{% endif %}"
                "{{ msg.content }}<|im_end|>\n"
                "{% endif %}"
                "{% endfor %}"
            ),
            messages=[
                {"role": "user", "content": "Q1?"},
                {
                    "role": "assistant",
                    "content": "4",
                },
                {"role": "user", "content": "Q2?"},
                {
                    "role": "assistant",
                    "reasoning_content": "calc2",
                    "content": "6",
                },
            ],
        )

        self.assertIn("4", supervised_text)
        self.assertIn("calc2", supervised_text)
        self.assertIn("6", supervised_text)
        self.assertNotIn("<|im_start|>assistant", supervised_text)
        self.assertNotIn("<|im_end|>", supervised_text)
        self.assertNotIn("<|im_start|>userQ1?<|im_end|>", supervised_text)
        self.assertNotIn("<|im_start|>userQ2?<|im_end|>", supervised_text)

    def test_llama3_template_supervises_assistant_content_only(self):
        supervised_text = self._get_supervised_text(
            chat_template=(
                "{{ bos_token }}{% for msg in messages %}"
                "<|start_header_id|>{{ msg.role }}<|end_header_id|>\n\n"
                "{{ msg.content }}<|eot_id|>"
                "{% endfor %}"
            ),
            messages=[
                {"role": "user", "content": "Q?"},
                {"role": "assistant", "content": "4"},
            ],
        )

        self.assertEqual(supervised_text, "4")
        self.assertNotIn(
            "<|start_header_id|>assistant<|end_header_id|>", supervised_text
        )
        self.assertNotIn("<|eot_id|>", supervised_text)
        self.assertNotIn(
            "<|start_header_id|>user<|end_header_id|>Q?<|eot_id|>",
            supervised_text,
        )

    def test_llama4_template_supervises_assistant_content_only(self):
        supervised_text = self._get_supervised_text(
            chat_template=(
                "{{ bos_token }}{% for msg in messages %}"
                "<|header_start|>{{ msg.role }}<|header_end|>\n\n"
                "{{ msg.content }}<|eot|>"
                "{% endfor %}"
            ),
            messages=[
                {"role": "user", "content": "Q?"},
                {"role": "assistant", "content": "4"},
            ],
        )

        self.assertEqual(supervised_text, "4")
        self.assertNotIn("<|header_start|>assistant<|header_end|>", supervised_text)
        self.assertNotIn("<|eot|>", supervised_text)
        self.assertNotIn(
            "<|header_start|>user<|header_end|>Q?<|eot|>",
            supervised_text,
        )

    def test_deepseek_v3_template_supervises_assistant_content_only(self):
        supervised_text = self._get_supervised_text(
            chat_template=(
                "{{ bos_token }}{% for msg in messages %}"
                "{% if msg.role == 'user' %}"
                "<｜User｜>{{ msg.content }}"
                "{% else %}"
                "<｜Assistant｜>{{ msg.content }}<｜end▁of▁sentence｜>"
                "{% endif %}"
                "{% endfor %}"
            ),
            messages=[
                {"role": "user", "content": "Q?"},
                {"role": "assistant", "content": "4"},
            ],
        )

        self.assertEqual(supervised_text, "4")
        self.assertNotIn("<Assistant>", supervised_text)
        self.assertNotIn("<endofsentence>", supervised_text)
        self.assertNotIn("<User>Q?", supervised_text)


class TestChatDatasetPositionBoundaries(unittest.TestCase):
    def test_positions_keep_turns_together_and_packed_conversations_apart(self):
        tokenizer = _load_tokenizer()
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "6"},
        ]

        full_text = tokenizer.apply_chat_template(messages).rstrip("\n")
        full_tokens = tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != tokenizer.eos_id:
            full_tokens.append(tokenizer.eos_id)
        sample_len = len(full_tokens) - 1

        ds = Dataset.from_list([{"id": 1}, {"id": 2}])
        chat_ds = ChatDataset(
            dataset=ds,
            tokenizer=tokenizer,
            sample_processor=lambda sample: messages,
            seq_len=sample_len * 2,
            infinite=False,
        )
        spans = chat_ds._get_assistant_spans(messages, full_tokens)

        batch, _ = next(iter(chat_ds))
        positions = batch["positions"].unsqueeze(0)
        doc_mask = get_document_mask_mod(positions=positions)

        self.assertTrue(
            doc_mask(
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor(spans[1][0]),
                torch.tensor(spans[0][0]),
            ).item(),
            "Assistant turn 2 should attend to turn 1 within the same conversation",
        )
        self.assertFalse(
            doc_mask(
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor(sample_len + spans[0][0]),
                torch.tensor(spans[0][0]),
            ).item(),
            "Packed conversations should not attend across position resets",
        )

        metadata = create_varlen_metadata_for_document(positions=positions)
        self.assertEqual(metadata.cu_seq_q.tolist(), [0, sample_len, sample_len * 2])


if __name__ == "__main__":
    unittest.main()
