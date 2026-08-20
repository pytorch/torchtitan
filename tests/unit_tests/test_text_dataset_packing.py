# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import torch

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.text_datasets import DATASETS

_TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "tokenizer")


def _build_dataloader(seq_len: int) -> GrainDataLoader:
    return GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        shuffle=False,
        num_prefetch_batches=0,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH),
        seq_len=seq_len,
        local_batch_size=1,
    )


class TestTextDatasetPacking(unittest.TestCase):
    """Greedy packing must emit exactly seq_len tokens with in-range positions.

    Inputs and labels are shifted per document at tokenization time, so a
    packed sample is seq_len long (not seq_len + 1). Emitting one extra token
    pushes the largest position to seq_len, which is one past the last entry of
    a RoPE cache sized at max_seq_len == seq_len and only surfaces as an async
    device-side assert deep inside the model.
    """

    def test_positions_are_contiguous_per_document_runs(self):
        dataloader = _build_dataloader(256)
        try:
            iterator = iter(dataloader)
            for _ in range(100):
                input_dict, _labels = next(iterator)
                positions = input_dict["positions"][0]
                steps = positions[1:] - positions[:-1]
                # Each position either continues the current document (+1) or
                # restarts a new one (back to 0).
                self.assertTrue(bool(torch.all((steps == 1) | (positions[1:] == 0))))
        finally:
            dataloader.close()

    def test_no_cross_document_targets(self):
        """The last token of a document must never predict the next document."""
        tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
        dataloader = _build_dataloader(256)
        interior_doc_starts = 0
        try:
            iterator = iter(dataloader)
            for _ in range(100):
                input_dict, labels = next(iterator)
                input_ids = input_dict["input"][0]
                positions = input_dict["positions"][0]
                labels = labels[0]

                # EOS closes a document and is never fed back in; BOS opens one and
                # is never a target.
                self.assertFalse(bool(torch.any(input_ids == tokenizer.eos_id)))
                self.assertFalse(bool(torch.any(labels == tokenizer.bos_id)))

                starts = (positions == 0).nonzero().flatten()
                starts = starts[starts > 0]
                interior_doc_starts += len(starts)
                self.assertTrue(bool(torch.all(input_ids[starts] == tokenizer.bos_id)))
                # The token right before a document start predicts that document's
                # own EOS, not the next document's first token.
                self.assertTrue(bool(torch.all(labels[starts - 1] == tokenizer.eos_id)))
        finally:
            dataloader.close()

        # Guard against the assertions above passing vacuously.
        self.assertGreater(interior_doc_starts, 0)


class TestTextDatasetBufferCheckpointing(unittest.TestCase):
    def test_packing_state_round_trips(self):
        dataloader = _build_dataloader(256)
        try:
            iterator = iter(dataloader)
            for _ in range(5):
                next(iterator)
            state = dataloader.state_dict()
            expected = [next(iterator) for _ in range(5)]
        finally:
            dataloader.close()

        resumed = _build_dataloader(256)
        try:
            resumed.load_state_dict(state)
            resumed_iterator = iter(resumed)
            actual = [next(resumed_iterator) for _ in range(5)]
        finally:
            resumed.close()

        for (expected_inputs, expected_labels), (actual_inputs, actual_labels) in zip(
            expected, actual, strict=True
        ):
            self.assertTrue(
                torch.equal(expected_inputs["input"], actual_inputs["input"])
            )
            self.assertTrue(
                torch.equal(expected_inputs["positions"], actual_inputs["positions"])
            )
            self.assertTrue(torch.equal(expected_labels, actual_labels))


if __name__ == "__main__":
    unittest.main()
