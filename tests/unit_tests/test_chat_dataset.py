# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

from copy import deepcopy

import grain.python as grain
import numpy as np
import torch
from datasets import Dataset
from torch.nn.attention.flex_attention import and_masks
from torchtitan.components.data.collators import TextCollator

from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import FirstFitPackingConfig
from torchtitan.components.data.sources import HuggingFaceRandomAccessSource
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.text_datasets import ChatProcessor
from torchtitan.models.common.attention import (
    BaseAttention,
    FlexAttention,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
)
from torchtitan.models.common.decoder import Decoder


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


def _runtime(seq_len):
    return DatasetBuildContext(
        tokenizer=_load_tokenizer(),
        seq_len=seq_len,
        local_batch_size=1,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
    )


def _build_processor(seq_len=2048, messages_fn=_process_sample):
    return ChatProcessor.Config(messages_fn=messages_fn).build(
        context=_runtime(seq_len)
    )


def _build_rows(seq_len):
    dataset = FirstFitPackingConfig(
        dataset=SingleDatasetConfig(
            source=HuggingFaceRandomAccessSource.Config(
                path="json",
                split="train",
                load_dataset_kwargs={
                    "data_files": _DATA_PATH,
                },
            ),
            processor=ChatProcessor.Config(messages_fn=_process_sample),
            post_filters=(lambda sample: sample is not None,),
        )
    )
    return dataset.build(
        context=_runtime(seq_len),
        dataset_iteration_policy=DatasetIterationPolicy(
            seed=42,
            shuffle=False,
            repeat=False,
            dp_rank=0,
            dp_world_size=1,
            streaming_shuffle_buffer_size=1_000,
        ),
    )


def _build_dataloader(seq_len=128, world_size=1, rank=0):
    config = GrainDataLoader.Config(
        dataset=FirstFitPackingConfig(
            dataset=SingleDatasetConfig(
                source=HuggingFaceRandomAccessSource.Config(
                    path="json",
                    split="train",
                    load_dataset_kwargs={
                        "data_files": _DATA_PATH,
                    },
                ),
                processor=ChatProcessor.Config(messages_fn=_process_sample),
                post_filters=(lambda sample: sample is not None,),
            )
        ),
        collator=TextCollator.Config(),
        seed=42,
        shuffle=True,
        repeat=True,
        num_prefetch_batches=1,
    )
    return config.build(
        dp_world_size=world_size,
        dp_rank=rank,
        tokenizer=_load_tokenizer(),
        seq_len=seq_len,
        local_batch_size=1,
    )


class TestChatDatasetLabelMasking(unittest.TestCase):
    """Prompt tokens should be masked (IGNORE_INDEX), assistant tokens should not."""

    def test_prompt_masked_response_unmasked(self):
        sequence = _build_processor()(
            _load_dataset()[0],
            np.random.default_rng(0),
        )
        _, label_ids = TextCollator.Config().build(context=_runtime(2048))([sequence])
        label_ids = label_ids[0]

        masked = (label_ids == IGNORE_INDEX).nonzero(as_tuple=True)[0]
        unmasked = (label_ids != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        self.assertGreater(len(masked), 0, "Expected some masked prompt labels")
        self.assertGreater(len(unmasked), 0, "Expected some unmasked response labels")
        self.assertGreater(unmasked[0].item(), 0, "First token label should be masked")


class TestChatDatasetShiftedTokens(unittest.TestCase):
    """The processor creates next-token pairs before collation."""

    def test_shifted_by_one(self):
        tokenizer = _load_tokenizer()
        sample = _load_dataset()[0]
        messages = _process_sample(sample)
        token_sequence = _build_processor()(sample, np.random.default_rng(0))
        inputs, labels = TextCollator.Config().build(context=_runtime(2048))(
            [token_sequence]
        )

        full_text = tokenizer.apply_chat_template(messages).rstrip("\n")
        full_tokens = tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != tokenizer.eos_id:
            full_tokens.append(tokenizer.eos_id)

        self.assertEqual(
            inputs["input"][0].tolist()[: len(full_tokens) - 1],
            full_tokens[:-1],
        )

        prompt_text = tokenizer.apply_chat_template(
            messages[:1], add_generation_prompt=True
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        response_start = len(prompt_tokens) - 1
        self.assertNotEqual(labels[0, response_start], IGNORE_INDEX)
        self.assertEqual(
            labels[0, response_start : len(full_tokens) - 1].tolist(),
            full_tokens[1:][response_start:],
        )
        self.assertEqual(labels[0, len(full_tokens) - 1], IGNORE_INDEX)


class TestChatDatasetGreedyPacking(unittest.TestCase):
    """Multiple short samples packed into one sequence with small seq_len."""

    def test_packing_multiple_samples(self):
        seq_len = 256
        sequences = list(_build_rows(seq_len))

        # With 10 samples of lengths 79-123, they should pack into fewer than 10 batches
        self.assertGreater(len(sequences), 0)
        self.assertLess(len(sequences), 10)

        collator = TextCollator.Config().build(context=_runtime(seq_len))
        for sequence in sequences:
            batch, labels = collator([sequence])
            self.assertEqual(batch["input"].shape, (1, seq_len))
            self.assertEqual(labels.shape, (1, seq_len))
            self.assertIn("positions", batch)
            self.assertEqual(batch["positions"].shape, (1, seq_len))


class TestChatDatasetPerDocumentPositions(unittest.TestCase):
    """Positions reset to 0 at each document boundary in packed mode."""

    def test_positions_reset_at_boundaries(self):
        sequence = next(iter(_build_rows(seq_len=256)))
        batch, _ = TextCollator.Config().build(context=_runtime(256))([sequence])
        positions = batch["positions"][0]

        self.assertEqual(positions[0].item(), 0)
        resets = (positions[1:] == 0).nonzero(as_tuple=True)[0]
        self.assertGreater(
            len(resets), 0, "Expected at least one position reset (document boundary)"
        )

        pos_list = positions.tolist()
        for index in range(1, len(pos_list)):
            if pos_list[index] == 0:
                continue
            self.assertEqual(
                pos_list[index],
                pos_list[index - 1] + 1,
                f"Positions should be consecutive at index {index}, "
                f"got {pos_list[index - 1]} -> {pos_list[index]}",
            )


class TestChatDatasetDropOnOverflow(unittest.TestCase):
    """Samples exceeding seq_len are silently dropped."""

    def test_all_dropped_with_tiny_seq_len(self):
        self.assertEqual(
            len(list(_build_rows(seq_len=32))),
            0,
            "All samples should be dropped at seq_len=32",
        )


class TestChatDatasetMessageValidation(unittest.TestCase):
    """Non-[user, assistant] messages raise ValueError."""

    def test_invalid_messages(self):
        invalid_messages = (
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "assistant", "content": "OK"},
            ],
            [
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "hello again"},
            ],
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "bye"},
            ],
        )
        for messages in invalid_messages:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                processor = _build_processor(
                    messages_fn=lambda _sample, value=messages: value
                )
                processor({}, np.random.default_rng(0))


class TestChatDatasetCheckpointing(unittest.TestCase):
    """state_dict / load_state_dict round-trips correctly."""

    def test_yield_same_data_after_resume_on_each_rank(self):
        for rank in range(2):
            with self.subTest(rank=rank):
                dataloader = _build_dataloader(world_size=2, rank=rank)
                iterator = iter(dataloader)
                # The source has 10 rows, so 12 batches necessarily cross a repeat.
                for _ in range(12):
                    next(iterator)

                state = deepcopy(dataloader.state_dict())
                resumed = _build_dataloader(world_size=2, rank=rank)
                resumed.load_state_dict(state)
                resumed_iterator = iter(resumed)

                for _ in range(4):
                    expected_inputs, expected_labels = next(iterator)
                    actual_inputs, actual_labels = next(resumed_iterator)
                    self.assertTrue(
                        torch.equal(actual_inputs["input"], expected_inputs["input"])
                    )
                    self.assertTrue(
                        torch.equal(
                            actual_inputs["positions"], expected_inputs["positions"]
                        )
                    )
                    self.assertTrue(torch.equal(actual_labels, expected_labels))


class TestDocumentMaskBlocksCrossDocAttention(unittest.TestCase):
    """Verify that position-based document masks block cross-document attention."""

    def test_packed_samples_block_cross_document_attention(self):
        processor = _build_processor()
        dataset = _load_dataset()
        input_ids_0 = processor(dataset[0], np.random.default_rng(0)).input_ids[:-1]
        input_ids_1 = processor(dataset[1], np.random.default_rng(0)).input_ids[:-1]

        packed = np.concatenate((input_ids_0, input_ids_1))
        boundary = len(input_ids_0)
        positions = torch.tensor(
            [list(range(len(input_ids_0))) + list(range(len(input_ids_1)))]
        )

        mask_mod = get_document_mask_mod(positions)
        batch, head = torch.tensor(0), torch.tensor(0)

        self.assertFalse(
            mask_mod(
                batch, head, torch.tensor(boundary), torch.tensor(boundary - 1)
            ).item(),
        )
        self.assertFalse(
            mask_mod(
                batch, head, torch.tensor(len(packed) - 1), torch.tensor(0)
            ).item(),
        )
        self.assertTrue(
            mask_mod(batch, head, torch.tensor(boundary - 1), torch.tensor(0)).item(),
        )
        self.assertTrue(
            mask_mod(
                batch,
                head,
                torch.tensor(len(packed) - 1),
                torch.tensor(boundary),
            ).item(),
        )

    def test_packed_document_mask_composes_with_causal_mask(self):
        for positions in (
            torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2]]),
            torch.tensor(
                [
                    [0, 1, 2, 0, 1, 0, 1, 2],
                    [0, 1, 0, 1, 2, 3, 0, 1],
                ]
            ),
        ):
            causal_mask = get_causal_mask_mod()
            document_mask = get_document_mask_mod(positions)
            packed_mask = and_masks(
                causal_mask,
                get_efficient_causal_mask_mod_for_packed_document(positions),
            )
            head = torch.tensor(0)

            for batch in range(positions.shape[0]):
                batch_tensor = torch.tensor(batch)
                for query_index in range(positions.shape[1]):
                    query_tensor = torch.tensor(query_index)
                    for key_value_index in range(positions.shape[1]):
                        key_value_tensor = torch.tensor(key_value_index)
                        expected = causal_mask(
                            batch_tensor, head, query_tensor, key_value_tensor
                        ) & document_mask(
                            batch_tensor, head, query_tensor, key_value_tensor
                        )
                        self.assertEqual(
                            packed_mask(
                                batch_tensor,
                                head,
                                query_tensor,
                                key_value_tensor,
                            ).item(),
                            expected.item(),
                        )

    def test_decoder_block_causal_flex_mask_supports_multiple_samples(self):
        positions = torch.tensor(
            [
                [0, 1, 2, 0, 1, 0, 1, 2],
                [0, 1, 0, 1, 2, 3, 0, 1],
            ],
            dtype=torch.int32,
        )
        attn_config = BaseAttention.Config(
            n_heads=1,
            inner_attention=FlexAttention.Config(block_size=4),
        )

        decoder = Decoder.__new__(Decoder)
        mask = decoder._create_flex_attention_mask_for_document(positions, attn_config)

        self.assertEqual(mask.shape, (positions.shape[0], 1, 8, 8))


if __name__ == "__main__":
    unittest.main()
