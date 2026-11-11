# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import itertools
import random
import unittest

import grain.python as grain
import torch

from torchtitan.components.data.balancing import (
    document_attention_cost,
    HeaviestFirstIterDataset,
    HeaviestFirstOrderingConfig,
    vision_encoder_cost,
)
from torchtitan.components.data.collators import TextCollator
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.components.data.sources import HuggingFaceRandomAccessSource
from torchtitan.components.data.text_datasets import TextProcessor
from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.components.tokenizer import HuggingFaceTokenizer

SEQ = 32
_DATA_PATH = "tests/assets/sft_test/data.json"
_TOKENIZER_PATH = "tests/assets/tokenizer"


def _microbatch(doc_lens: list[int]) -> TokenizedTrainingMicrobatch:
    """A SEQ-token microbatch with the given documents, then padding."""
    num_tokens = sum(doc_lens)
    positions = torch.arange(SEQ)
    if doc_lens:
        positions[:num_tokens] = torch.cat([torch.arange(n) for n in doc_lens])
    padding_mask = torch.ones(SEQ, dtype=torch.bool)
    padding_mask[:num_tokens] = False
    return TokenizedTrainingMicrobatch(
        input=torch.zeros(SEQ, dtype=torch.int64),
        labels=torch.zeros(SEQ, dtype=torch.int64),
        positions=positions,
        padding_mask=padding_mask,
        loss_token_counts=torch.tensor(num_tokens),
        routing_token_counts=torch.tensor([num_tokens]),
    )


class TestDocumentAttentionCost(unittest.TestCase):
    def test_sum_of_squared_document_lengths(self):
        self.assertEqual(document_attention_cost(_microbatch([3, 2, 7])), 9 + 4 + 49)

    def test_padding_is_ignored(self):
        # Padding positions restart at 0 but must not count as a document.
        self.assertEqual(document_attention_cost(_microbatch([5])), 25)
        self.assertEqual(document_attention_cost(_microbatch([])), 0)

    def test_token_weight_counts_non_padding_tokens(self):
        cost = document_attention_cost(_microbatch([3, 2]), token_weight=10.0)
        self.assertEqual(cost, 9 + 4 + 10.0 * 5)

    def test_rejects_untokenized_microbatch(self):
        with self.assertRaises(ValueError):
            document_attention_cost(object())  # type: ignore[arg-type]


class TestVisionEncoderCost(unittest.TestCase):
    def test_images_and_videos(self):
        mb = _microbatch([4, 4])
        mb.model_kwargs = {
            # Two images: 1x2x3 and 1x4x4 patches.
            "grid_thw": torch.tensor([[1, 2, 3], [1, 4, 4]]),
            # One video: 3 frames of 2x2 patches, attended per frame.
            "grid_thw_videos": torch.tensor([[3, 2, 2]]),
        }
        patches, segments_sq = 6 + 16 + 12, 36 + 256 + 3 * 16
        cost = functools.partial(vision_encoder_cost, patch_attention_weight=0.0)
        self.assertEqual(cost(mb, patch_weight=1.0), patches)
        cost = functools.partial(vision_encoder_cost, patch_weight=0.0)
        self.assertEqual(cost(mb, patch_attention_weight=1.0), segments_sq)
        self.assertEqual(
            vision_encoder_cost(
                mb,
                patch_weight=2.0,
                patch_attention_weight=0.5,
                document_attention_weight=3.0,
            ),
            2.0 * patches + 0.5 * segments_sq + 3.0 * 32,
        )

    def test_text_only_microbatch(self):
        cost = vision_encoder_cost(
            _microbatch([SEQ]), patch_weight=1.0, patch_attention_weight=1.0
        )
        self.assertEqual(cost, 0.0)


def _ordered(values, window_size):
    dataset = HeaviestFirstIterDataset(
        grain.MapDataset.source(values).to_iter_dataset(),
        window_size=window_size,
        cost_fn=lambda x: x[0],
    )
    return iter(dataset)


class TestHeaviestFirstIterDataset(unittest.TestCase):
    def test_sorts_each_window_and_keeps_every_element(self):
        values = [(c, i) for i, c in enumerate([3, 9, 1, 9, 4, 2, 8, 5, 7, 6])]
        out = list(_ordered(values, window_size=4))
        self.assertEqual(sorted(out), sorted(values))
        windows = [out[0:4], out[4:8], out[8:]]
        self.assertEqual([len(w) for w in windows], [4, 4, 2])
        for window, expected in zip(windows, [values[0:4], values[4:8], values[8:]]):
            # Heaviest first; equal costs keep their input order.
            self.assertEqual(
                window, sorted(expected, key=lambda v: (-v[0], values.index(v)))
            )

    def test_resume_mid_window(self):
        values = [(random.Random(i).random(), i) for i in range(23)]
        reference = list(_ordered(values, window_size=8))
        for consumed in range(len(values)):
            with self.subTest(consumed=consumed):
                it = _ordered(values, window_size=8)
                for _ in range(consumed):
                    next(it)
                state = it.get_state()
                resumed = _ordered(values, window_size=8)
                resumed.set_state(state)
                self.assertEqual(list(resumed), reference[consumed:])

    def test_aligned_windows_never_slower_than_packed_order(self):
        """Sum over steps of the slowest rank's cost, across simulated ranks."""
        rng = random.Random(0)
        for _ in range(50):
            num_ranks, num_steps = rng.randint(2, 16), 8 * rng.randint(1, 4)
            per_rank = [
                [(rng.paretovariate(1.2), i) for i in range(num_steps)]
                for _ in range(num_ranks)
            ]
            ordered = [list(_ordered(v, window_size=8)) for v in per_rank]

            def total(streams):
                return sum(max(s[k][0] for s in streams) for k in range(num_steps))

            self.assertLessEqual(total(ordered), total(per_rank) + 1e-9)


class TestLoaderIntegration(unittest.TestCase):
    def _build(self, heaviest_first):
        config = GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceRandomAccessSource.Config(
                        path="json",
                        split="train",
                        load_dataset_kwargs={"data_files": _DATA_PATH},
                    ),
                    processor=TextProcessor.Config(
                        text_fn=lambda s: f"{s['question']} {s['answer']}",
                    ),
                ),
            ),
            collator=TextCollator.Config(),
            max_num_documents=64,
            num_prefetch_microbatches=1,
            heaviest_first=heaviest_first,
        )
        return config.build(
            dp_world_size=2,
            dp_rank=0,
            tokenizer=HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH),
            max_context_length=128,
            num_tokens_per_microbatch=128,
        )

    def test_reorders_within_windows_and_resumes(self):
        window = 4
        ordering = HeaviestFirstOrderingConfig(
            window_size=window,
            cost_fn=functools.partial(document_attention_cost, token_weight=1.0),
        )
        packed = list(itertools.islice(iter(self._build(None)), 4 * window))
        loader = self._build(ordering)
        it = iter(loader)
        ordered = [next(it) for _ in range(4 * window)]
        for w in range(4):
            got = ordered[w * window : (w + 1) * window]
            ref = packed[w * window : (w + 1) * window]
            costs = [ordering.cost_fn(mb) for mb in got]
            self.assertEqual(costs, sorted(costs, reverse=True))
            self.assertEqual(
                sorted(mb.input.tolist() for mb in got),
                sorted(mb.input.tolist() for mb in ref),
            )

        # Resume mid-window from the loader checkpoint.
        for _ in range(window + 1):
            next(it)
        state = loader.state_dict()
        expected = [next(it) for _ in range(2 * window)]
        resumed = self._build(ordering)
        resumed.load_state_dict(state)
        resumed_it = iter(resumed)
        for exp in expected:
            self.assertTrue(torch.equal(next(resumed_it).input, exp.input))


if __name__ == "__main__":
    unittest.main()
