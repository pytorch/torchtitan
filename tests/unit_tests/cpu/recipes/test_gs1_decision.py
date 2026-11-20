# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import grain
import numpy as np
import torch

from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan_recipes.decision.base import (
    LETTERS,
    CandidateDistributionLoss,
    DecisionSampleProcessor,
)
from torchtitan_recipes.decision.prepare_gs1_data import flatten_record


class _Tokenizer(BaseTokenizer):
    def encode(self, text: str, **kwargs) -> list[int]:
        del kwargs
        if text == "\n":
            return [40]
        if text.startswith(" ") and len(text) == 2 and text[1] in LETTERS:
            return [10 + LETTERS.index(text[1])]
        return [1, 2, 3]

    def decode(self, token_ids, **kwargs) -> str:
        del kwargs
        return "".join(str(token_id) for token_id in token_ids)

    def get_vocab_size(self) -> int:
        return 64


def _sample() -> dict:
    return {
        "state": '{"balance":18}',
        "kind": "choice",
        "instructions": "Choose the lower-cost job.",
        "candidate_ids": ["job_a", "job_b", "job_c"],
        "options": ["Cost 2", "Cost 12", "Cost 5"],
        "target_probs": [0.2, 0.3, 0.5],
    }


def test_native_distribution_record_flattens_by_candidate_identity() -> None:
    record = {
        "request": {
            "state": {"balance": 18},
            "questions": {
                "choice": {
                    "type": "choice",
                    "instructions": "Pick one.",
                    "criteria": {"a": "A", "b": "B"},
                }
            },
        },
        "targets": {
            "choice": {"candidate_ids": ["b", "a"], "distribution": [0.25, 0.75]}
        },
    }
    rows = flatten_record(record)
    assert rows[0]["target_probs"] == [0.75, 0.25]
    assert json.loads(rows[0]["state"]) == {"balance": 18}


def test_preparation_skips_multimodal_states() -> None:
    record = {
        "request": {
            "state": [{"type": "image", "path": "image.png"}],
            "questions": {},
        },
        "targets": {},
    }
    assert flatten_record(record) == []


def test_structured_type_metadata_is_text() -> None:
    record = {
        "request": {
            "state": {"metadata": {"type": "image"}},
            "questions": {
                "q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}
            },
        },
        "targets": {"q": {"candidate_ids": ["a", "b"], "distribution": [0.5, 0.5]}},
    }
    assert len(flatten_record(record)) == 1


def test_processor_carries_candidate_count_and_samples_target_label() -> None:
    context = DatasetBuildContext(
        tokenizer=_Tokenizer(),
        max_context_length=64,
        num_tokens_per_microbatch=64,
        read_options=grain.ReadOptions(num_threads=1),
    )
    processor = DecisionSampleProcessor(
        DecisionSampleProcessor.Config(), context=context
    )
    sequence = processor(_sample(), np.random.default_rng(3))
    assert sequence.labels[-2] == -3
    assert sequence.labels[-1] in {10, 11, 12}
    assert sequence.input_ids[-1] == 40


def test_processor_drops_overlength_rows() -> None:
    context = DatasetBuildContext(
        tokenizer=_Tokenizer(),
        max_context_length=2,
        num_tokens_per_microbatch=2,
        read_options=grain.ReadOptions(num_threads=1),
    )
    processor = DecisionSampleProcessor(DecisionSampleProcessor.Config(), context=context)
    assert processor(_sample(), np.random.default_rng(0)) is None


def test_sampled_nll_brier_has_soft_target_expectation() -> None:
    candidate_ids = list(range(10, 36))
    weight = 0.7
    loss_fn = CandidateDistributionLoss(
        CandidateDistributionLoss.Config(
            candidate_token_ids=candidate_ids, brier_weight=weight
        )
    )
    logits = torch.zeros((1, 40))
    logits[0, 10:13] = torch.tensor([1.0, -0.5, 0.25])
    probs = torch.softmax(logits[0, 10:13], dim=-1)
    target = torch.tensor([0.2, 0.3, 0.5])
    observed = 0.0
    for index, probability in enumerate(target):
        loss, _ = loss_fn(
            logits,
            torch.tensor([candidate_ids[index]]),
            torch.tensor(1.0),
            candidate_counts=torch.tensor([3]),
        )
        observed += float(probability * loss)
    expected_nll = -(target * probs.log()).sum()
    expected_brier = probs.square().sum() + 1 - 2 * (target * probs).sum()
    torch.testing.assert_close(
        torch.tensor(observed), expected_nll + weight * expected_brier
    )
