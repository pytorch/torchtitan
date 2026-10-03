# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pyarrow as pa
import pyarrow.parquet as pq
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


def _write_multishard_parquet(tmp_path):
    data_files = []
    for shard in range(2):
        path = tmp_path / f"shard-{shard}.parquet"
        texts = [
            f"shard {shard} document {index} "
            + "checkpointable streaming data " * (1 + index % 5)
            for index in range(128)
        ]
        pq.write_table(pa.table({"text": texts}), path, row_group_size=8)
        data_files.append(str(path))
    return data_files


def _build_streaming_parquet_loader(data_files):
    config = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(
            dataset=SingleDatasetConfig(
                source=HuggingFaceStreamingSource.Config(
                    path="parquet",
                    split="train",
                    load_dataset_kwargs={
                        "data_files": {"train": data_files},
                    },
                ),
                processor=TextProcessor.Config(),
                post_filters=(lambda sample: sample is not None,),
            ),
        ),
        collator=TextCollator.Config(),
        seed=42,
        shuffle=False,
        repeat=True,
        num_prefetch_microbatches=1,
    )
    return config.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH),
        max_context_length=128,
        num_tokens_per_microbatch=128,
    )


def _assert_same_microbatch(actual, expected):
    assert torch.equal(actual.input, expected.input)
    assert torch.equal(actual.positions, expected.positions)
    assert torch.equal(actual.labels, expected.labels)


def _hf_state_leaves(state):
    leaves = []
    if isinstance(state, dict):
        for key, value in state.items():
            if key == "hf":
                leaves.append(value)
            else:
                leaves.extend(_hf_state_leaves(value))
    return leaves


def test_dcp_resumption_with_multishard_parquet(tmp_path):
    """DCP must restore a streaming loader after Hugging Face state grows keys.

    ``examples_iterable.previous_state`` is None before iteration and a dict
    afterward. That change stays inside the cursor's ``hf`` bytes leaf. The
    rest of the Grain tree stays a nested dict so other sources can reshard.
    """
    data_files = _write_multishard_parquet(tmp_path)
    original = _build_streaming_parquet_loader(data_files)
    restored = None
    try:
        iterator = iter(original)
        for _ in range(4):
            next(iterator)

        rank_state = original.state_dict()["dp_rank_0"]
        assert isinstance(rank_state, dict)
        hf_leaves = _hf_state_leaves(rank_state)
        assert len(hf_leaves) == 1
        assert isinstance(hf_leaves[0], bytes)

        checkpoint_id = tmp_path / "checkpoint"
        dcp.save({"dataloader": original}, checkpoint_id=checkpoint_id)

        restored = _build_streaming_parquet_loader(data_files)
        dcp.load({"dataloader": restored}, checkpoint_id=checkpoint_id)

        restored_iterator = iter(restored)
        for _ in range(4):
            expected = next(iterator)
            actual = next(restored_iterator)
            _assert_same_microbatch(actual, expected)
    finally:
        original.close()
        if restored is not None:
            restored.close()


if __name__ == "__main__":
    unittest.main()
