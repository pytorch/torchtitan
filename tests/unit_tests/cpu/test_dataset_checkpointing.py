# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import torch
import torch.distributed.checkpoint as dcp

from torchtitan.components.data.collators import TextCollator
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.components.data.sources import (
    HuggingFaceRandomAccessSource,
    HuggingFaceStreamingSource,
)
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.text_datasets import TextProcessor


_DATA_PATH = "tests/assets/sft_test/data.json"
_TOKENIZER_PATH = "tests/assets/tokenizer"


def _process_text(sample):
    return f"{sample['question']} {sample['answer']}"


class TestDatasetCheckpointing(unittest.TestCase):
    def test_c4_resumption(self):
        for source_type in (
            HuggingFaceRandomAccessSource,
            HuggingFaceStreamingSource,
        ):
            for rank in range(2):
                with self.subTest(source_type=source_type, rank=rank):
                    dataloader = self._build_dataloader(source_type, rank)
                    iterator = iter(dataloader)

                    # Eight source rows make fewer than 40 packed rows per rank,
                    # so this crosses at least one repeat boundary.
                    for _ in range(40):
                        next(iterator)
                    state = dataloader.state_dict()

                    resumed = self._build_dataloader(source_type, rank)
                    resumed.load_state_dict(state)
                    resumed_iterator = iter(resumed)

                    for _ in range(8):
                        expected_inputs = next(iterator)
                        actual_inputs = next(resumed_iterator)
                        self.assertTrue(
                            torch.equal(actual_inputs.input, expected_inputs.input)
                        )
                        self.assertTrue(
                            torch.equal(
                                actual_inputs.positions,
                                expected_inputs.positions,
                            )
                        )
                        self.assertTrue(
                            torch.equal(actual_inputs.labels, expected_inputs.labels)
                        )

    def test_streaming_state_survives_dcp_at_any_cursor_position(self):
        """A DCP checkpoint written at one cursor position must load at another.

        Grain returns the Hugging Face cursor as a nested dict whose key set
        depends on where in the shard it stopped, and DCP turns every leaf of a
        nested dict into its own storage key. Two positions therefore wrote two
        different key sets, and loading one into the other raised

            RuntimeError: Missing key in checkpoint state_dict: dataloader.
            dp_rank_0.parent_state.parent.parent_window_start_state.hf.
            examples_iterable.previous_state.

        An in-process ``state_dict``/``load_state_dict`` round trip cannot see
        this, which is why ``test_c4_resumption`` passes either way. The
        assertion on the storage metadata is the part that pins the fix: one
        opaque entry per rank, so the key set cannot vary with position.
        """
        rank = 0
        # Two positions far enough apart to sit in different shard states; the
        # eight source rows mean the second has crossed a repeat boundary.
        saved_states = []
        for num_batches in (3, 40):
            dataloader = self._build_dataloader(HuggingFaceStreamingSource, rank)
            iterator = iter(dataloader)
            for _ in range(num_batches):
                next(iterator)
            saved_states.append(dataloader.state_dict())

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            dcp.save(saved_states[0], checkpoint_id=checkpoint_dir, no_dist=True)

            metadata = dcp.FileSystemReader(checkpoint_dir).read_metadata()
            rank_keys = [
                key
                for key in metadata.state_dict_metadata
                if key.startswith(f"dp_rank_{rank}")
            ]
            self.assertEqual(rank_keys, [f"dp_rank_{rank}"])

            # The loading side is at the *other* position, which is what used to
            # fail: its in-memory state carried keys the checkpoint never wrote.
            target = self._build_dataloader(HuggingFaceStreamingSource, rank)
            target_iterator = iter(target)
            for _ in range(40):
                next(target_iterator)
            target_state = target.state_dict()
            dcp.load(target_state, checkpoint_id=checkpoint_dir, no_dist=True)

        target.load_state_dict(target_state)

        expected = self._build_dataloader(HuggingFaceStreamingSource, rank)
        expected.load_state_dict(saved_states[0])
        expected_iterator = iter(expected)
        resumed_iterator = iter(target)
        for _ in range(4):
            expected_inputs = next(expected_iterator)
            actual_inputs = next(resumed_iterator)
            self.assertTrue(torch.equal(actual_inputs.input, expected_inputs.input))

    def _build_dataloader(self, source_type, rank):
        config = GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=source_type.Config(
                        path="json",
                        split="train",
                        load_dataset_kwargs={
                            "data_files": _DATA_PATH,
                        },
                    ),
                    processor=TextProcessor.Config(
                        text_fn=_process_text,
                    ),
                    post_filters=(lambda sample: sample is not None,),
                ),
            ),
            collator=TextCollator.Config(),
            seed=42,
            shuffle=True,
            repeat=True,
            num_prefetch_microbatches=1,
        )
        return config.build(
            dp_world_size=2,
            dp_rank=rank,
            tokenizer=HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH),
            max_context_length=128,
            num_tokens_per_microbatch=128,
        )


if __name__ == "__main__":
    unittest.main()
