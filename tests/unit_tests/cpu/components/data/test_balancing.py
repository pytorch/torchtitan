# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest

import grain.python as grain
import torch

from torchtitan.components.data.balancing import (
    document_attention_cost,
    LocalBalancer,
    MicrobatchBalancingIterDataset,
    packed_attention_cost,
)
from torchtitan.components.data.collators import TextCollator
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.components.data.sources import HuggingFaceRandomAccessSource
from torchtitan.components.data.text_datasets import TextProcessor
from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.components.tokenizer import HuggingFaceTokenizer


def _microbatch(positions: list[int]) -> TokenizedTrainingMicrobatch:
    n = len(positions)
    zeros = torch.zeros(n, dtype=torch.int64)
    return TokenizedTrainingMicrobatch(
        input=zeros,
        labels=zeros,
        positions=torch.tensor(positions),
        padding_mask=torch.zeros(n, dtype=torch.bool),
        loss_token_counts=torch.tensor(n),
        routing_token_counts=torch.tensor([n]),
    )


def _iter(values: list[int], window_size: int):
    balancer = LocalBalancer.Config(window_size=window_size, cost_fn=float).build()
    source = grain.MapDataset.source(values).to_iter_dataset()
    return iter(MicrobatchBalancingIterDataset(source, balancer=balancer))


class TestBalancing(unittest.TestCase):
    def test_document_attention_cost(self):
        r = lambda n: list(range(n))  # positions of one segment
        # Documents 3 and 5, then a padding tail of 8 (max_context_length 8):
        # the same cost from positions as from lengths.
        mb = _microbatch(r(3) + r(5) + r(8))
        self.assertEqual(document_attention_cost(mb), 9 + 25 + 64)
        self.assertEqual(
            document_attention_cost(mb),
            packed_attention_cost([3, 5], num_tokens=16, max_context_length=8),
        )
        # per_sample_pad_multiple 4: documents of 3 and 5 padded to 4 and 8 (the
        # padding continues their positions), then a tail of 4.
        self.assertEqual(document_attention_cost(_microbatch(r(4) + r(8) + r(4))), 96)

    def test_each_window_heaviest_first(self):
        out = list(_iter([3, 9, 1, 4, 2, 8, 5, 7, 6], window_size=4))
        self.assertEqual(out, [9, 4, 3, 1, 8, 7, 5, 2, 6])

    def test_resume_mid_window(self):
        values = list(range(10))
        reference = list(_iter(values, window_size=4))
        for consumed in range(len(values)):
            it = _iter(values, window_size=4)
            for _ in range(consumed):
                next(it)
            resumed = _iter(values, window_size=4)
            resumed.set_state(it.get_state())
            self.assertEqual(list(resumed), reference[consumed:])

    def test_loader(self):
        def build():
            config = GrainDataLoader.Config(
                dataset=ConcatThenSplitPackingConfig(
                    dataset=SingleDatasetConfig(
                        source=HuggingFaceRandomAccessSource.Config(
                            path="json",
                            split="train",
                            load_dataset_kwargs={
                                "data_files": "tests/assets/sft_test/data.json"
                            },
                        ),
                        processor=TextProcessor.Config(
                            text_fn=lambda s: f"{s['question']} {s['answer']}"
                        ),
                    ),
                ),
                collator=TextCollator.Config(),
                max_num_documents=64,
                balancer=LocalBalancer.Config(window_size=4),
            )
            return config.build(
                dp_world_size=2,
                dp_rank=0,
                tokenizer=HuggingFaceTokenizer(tokenizer_path="tests/assets/tokenizer"),
                max_context_length=128,
                num_tokens_per_microbatch=128,
            )

        loader = build()
        it = iter(loader)
        window = list(itertools.islice(it, 4))
        costs = [document_attention_cost(mb) for mb in window]
        self.assertEqual(costs, sorted(costs, reverse=True))

        # Resume mid-window from the loader checkpoint.
        next(it)
        state = loader.state_dict()
        expected = list(itertools.islice(it, 6))
        resumed = build()
        resumed.load_state_dict(state)
        for exp, got in zip(expected, iter(resumed)):
            self.assertTrue(torch.equal(got.input, exp.input))


if __name__ == "__main__":
    unittest.main()
