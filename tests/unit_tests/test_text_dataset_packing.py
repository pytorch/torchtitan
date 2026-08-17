# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import torch

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataset

_TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "tokenizer")


def _build_dataset(seq_len: int) -> HuggingFaceTextDataset:
    return HuggingFaceTextDataset(
        dataset_name="c4_test",
        dataset_path=None,
        tokenizer=HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH),
        seq_len=seq_len,
        dp_rank=0,
        dp_world_size=1,
        infinite=True,
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
        it = iter(_build_dataset(256))
        for _ in range(100):
            positions = next(it)[0]["positions"]
            steps = positions[1:] - positions[:-1]
            # Each position either continues the current document (+1) or
            # restarts a new one (back to 0).
            self.assertTrue(bool(torch.all((steps == 1) | (positions[1:] == 0))))

    def test_no_cross_document_targets(self):
        """The last token of a document must never predict the next document."""
        tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
        ds = _build_dataset(256)
        it = iter(ds)
        interior_doc_starts = 0
        for _ in range(100):
            input_dict, labels = next(it)
            input_ids = input_dict["input"]
            positions = input_dict["positions"]

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

        # Guard against the assertions above passing vacuously.
        self.assertGreater(interior_doc_starts, 0)


class TestTextDatasetBufferCheckpointing(unittest.TestCase):
    def test_labels_buffer_round_trips(self):
        ds = _build_dataset(256)
        it = iter(ds)
        for _ in range(5):
            next(it)
        # Leave a partial sample in the buffers to checkpoint.
        self.assertGreater(len(ds._inputs_buffer), 0)

        state = ds.state_dict()
        self.assertIn("labels_buffer", state)

        resumed = _build_dataset(256)
        resumed.load_state_dict(state)
        self.assertEqual(resumed._inputs_buffer, ds._inputs_buffer)
        self.assertEqual(resumed._labels_buffer, ds._labels_buffer)
        self.assertEqual(resumed._positions_buffer, ds._positions_buffer)


if __name__ == "__main__":
    unittest.main()
