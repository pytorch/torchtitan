# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import random

import grain.python as grain
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.components.loss import CrossEntropyLoss, IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention.cp_attention import UlyssesVarlenInnerAttention
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan_recipes.decision.evaluate import fit_inverse_temperature
from torchtitan_recipes.decision.prepare_data import flatten_record, LETTERS, render
from torchtitan_recipes.decision.qwen3 import (
    AnswerTokenLoss,
    CANDIDATE_LABEL_START,
    CandidateDistributionLoss,
    CandidateLetterProcessor,
    DecisionLetterProcessor,
    qwen3_1_7b_decision,
)

_TOKENIZER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "assets", "tokenizer"
)


def _processor(max_context_length: int = 2048) -> DecisionLetterProcessor:
    context = DatasetBuildContext(
        tokenizer=HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH),
        max_context_length=max_context_length,
        num_tokens_per_microbatch=max_context_length,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
    )
    return DecisionLetterProcessor(DecisionLetterProcessor.Config(), context=context)


def test_only_answer_letter_is_supervised() -> None:
    processor = _processor()
    sample = {
        "prompt": "Context:\nhi\nOptions:\nA) Yes\nB) No\nAnswer:",
        "keys": ["yes", "no"],
        "target_probs": [0.0, 1.0],
    }
    seq = processor(sample, np.random.default_rng(0))
    prompt = processor._tokenizer.encode(sample["prompt"], add_bos=False, add_eos=False)
    (answer_id,) = processor._tokenizer.encode(" B", add_bos=False, add_eos=False)

    assert seq.input_ids.tolist() == prompt
    assert seq.labels.tolist() == [IGNORE_INDEX] * (len(prompt) - 1) + [answer_id]


def test_misaligned_targets_raise() -> None:
    with pytest.raises(ValueError, match="aligned target probabilities"):
        _processor()(
            {"prompt": "Answer:", "keys": ["yes", "no"], "target_probs": [1.0]},
            np.random.default_rng(0),
        )


def test_overlong_prompt_raises() -> None:
    # "x" * n is n tokens with the test tokenizer: 16 fills the context, 17 overflows.
    processor = _processor(max_context_length=16)
    sample = {"prompt": "x" * 16, "keys": ["a", "b"], "target_probs": [1.0, 0.0]}
    processor(sample, np.random.default_rng(0))
    with pytest.raises(ValueError, match="max_context_length"):
        processor({**sample, "prompt": "x" * 17}, np.random.default_rng(0))


def test_answer_token_loss_matches_full_cross_entropy() -> None:
    torch.manual_seed(0)
    num_tokens, dim, vocab = 32, 8, 16
    lm_head = torch.nn.Linear(dim, vocab, bias=False)
    hidden_TD = torch.randn(num_tokens, dim, requires_grad=True)
    labels_T = torch.full((num_tokens,), IGNORE_INDEX)
    labels_T[[5, 17, 31]] = torch.tensor([3, 9, 1])

    loss_fn = AnswerTokenLoss(
        AnswerTokenLoss.Config(loss_fn=CrossEntropyLoss.Config(), max_num_supervised=8)
    )
    loss_fn.set_lm_head(lm_head)
    loss, _ = loss_fn(hidden_TD, labels_T)
    (grad,) = torch.autograd.grad(loss, hidden_TD)

    expected = F.cross_entropy(
        lm_head(hidden_TD), labels_T, ignore_index=IGNORE_INDEX, reduction="sum"
    )
    (expected_grad,) = torch.autograd.grad(expected, hidden_TD)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(grad, expected_grad)


def test_answer_token_loss_rejects_too_many_supervised_positions() -> None:
    lm_head = torch.nn.Linear(8, 16, bias=False)
    loss_fn = AnswerTokenLoss(
        AnswerTokenLoss.Config(loss_fn=CrossEntropyLoss.Config(), max_num_supervised=2)
    )
    loss_fn.set_lm_head(lm_head)
    labels_T = torch.tensor([1, 2, 3, IGNORE_INDEX])
    with pytest.raises(RuntimeError, match="max_num_supervised"):
        loss_fn(torch.randn(4, 8), labels_T)


@pytest.mark.parametrize(
    "question, gold_key, gold_desc, fixed_order",
    [
        pytest.param(
            {"type": "noul", "instructions": "Refund?", "label": True},
            "true",
            "Yes",
            None,
            id="noul",
        ),
        pytest.param(
            {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {f"k{i}": f"desc {i}" for i in range(6)},
                "label": "k4",
            },
            "k4",
            "desc 4",
            None,
            id="choice",
        ),
        pytest.param(
            {
                "type": "score",
                "instructions": "How urgent?",
                "criteria": ["low", "mid", "high"],
                "label": 2,
            },
            "2",
            "high",
            ("0", "1", "2"),
            id="score",
        ),
    ],
)
def test_render_gold_letter_names_labeled_option(
    question: dict, gold_key: str, gold_desc: str, fixed_order: tuple | None
) -> None:
    rng = random.Random(0)
    orders = set()
    for _ in range(20):
        row = render("state", question, rng)
        gold = row["target_probs"].index(1.0)
        letter = LETTERS[gold]
        assert f"\n{letter}) {gold_desc}\n" in row["prompt"]
        assert row["keys"][gold] == gold_key
        orders.add(tuple(row["keys"]))
    # choice and noul are shuffled against letter-position priors; score stays ordered.
    if fixed_order:
        assert orders == {fixed_order}
    else:
        assert len(orders) > 1


def test_context_parallel_recipe_uses_ulysses_varlen_attention() -> None:
    config = qwen3_1_7b_decision(ParallelismConfig(context_parallel_degree=2))
    assert isinstance(config.model, Qwen3Model.Config)
    attention_types = {
        type(layer.attention.inner_attention) for layer in config.model.layers
    }
    assert attention_types == {UlyssesVarlenInnerAttention.Config}


class _Tokenizer(BaseTokenizer):
    def encode(self, text: str, **kwargs) -> list[int]:
        del kwargs
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
        "type": "choice",
        "prompt": "Context:\nhi\nOptions:\nA) 2\nB) 12\nC) 5\nAnswer:",
        "instructions": "Choose the lower-cost job.",
        "keys": ["job_a", "job_b", "job_c"],
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
    rows = flatten_record(record, random.Random(0))
    assert dict(zip(rows[0]["keys"], rows[0]["target_probs"], strict=True)) == {
        "a": 0.75,
        "b": 0.25,
    }
    assert json.loads(rows[0]["state"]) == {"balance": 18}


@pytest.mark.parametrize(
    ("state", "eligible"),
    [
        ([{"type": "image", "path": "image.png"}], False),
        ("x" * 4001, False),
        ({"metadata": {"type": "image"}}, True),
        ({"frames": [{"type": "image", "path": "frame.png"}]}, False),
    ],
    ids=["media", "long-state", "metadata", "nested-media"],
)
def test_preparation_filters_states(state, eligible: bool) -> None:
    record = {
        "request": {
            "state": state,
            "questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
        },
        "targets": {"q": {"candidate_ids": ["a", "b"], "distribution": [0.5, 0.5]}},
    }
    assert len(flatten_record(record, random.Random(0))) == int(eligible)


def _distribution_processor(max_context_length: int) -> DecisionLetterProcessor:
    return CandidateLetterProcessor(
        CandidateLetterProcessor.Config(),
        context=DatasetBuildContext(
            tokenizer=_Tokenizer(),
            max_context_length=max_context_length,
            num_tokens_per_microbatch=max_context_length,
            read_options=grain.ReadOptions(num_threads=1),
        ),
    )


def test_processor_carries_candidate_count_and_samples_target_label() -> None:
    sequence = _distribution_processor(64)(_sample(), np.random.default_rng(3))
    assert sequence is not None
    assert (sequence.labels != -100).sum() == 1
    assert sequence.labels[-1] in {
        CANDIDATE_LABEL_START - offset for offset in (26, 27, 28)
    }
    assert sequence.input_ids[-1] == 3


def test_distribution_processor_rejects_overlength_rows() -> None:
    with pytest.raises(ValueError, match="max_context_length"):
        _distribution_processor(2)(_sample(), np.random.default_rng(0))


def test_sampled_nll_brier_has_soft_target_expectation() -> None:
    weight = 0.7
    loss_fn = CandidateDistributionLoss(
        CandidateDistributionLoss.Config(brier_weight=weight)
    )
    loss_fn.set_tokenizer(_Tokenizer())
    with pytest.raises(RuntimeError, match="encoded option counts"):
        loss_fn(torch.zeros((1, 40)), torch.tensor([10]))
    logits = torch.zeros((1, 40))
    logits[0, 10:13] = torch.tensor([1.0, -0.5, 0.25])
    probs = torch.softmax(logits[0, 10:13], dim=-1)
    target = torch.tensor([0.2, 0.3, 0.5])
    observed = 0.0
    for index, probability in enumerate(target):
        loss, _ = loss_fn(
            logits,
            torch.tensor([CANDIDATE_LABEL_START - 26 - index]),
            torch.tensor(1.0),
        )
        observed += float(probability * loss)
    expected_nll = -(target * probs.log()).sum()
    expected_brier = probs.square().sum() + 1 - 2 * (target * probs).sum()
    torch.testing.assert_close(
        torch.tensor(observed), expected_nll + weight * expected_brier
    )


@pytest.mark.parametrize("target", [0.49, 0.5, 0.51, 0.9, 1.0])
def test_temperature_fit_handles_optima_outside_the_old_grid(target) -> None:
    logits = [torch.tensor([0.0, 4.0])]
    rows = [{"target_probs": [1 - target, target]}]
    scale = fit_inverse_temperature(logits, rows)
    if target == 1:
        assert scale == float("inf")
    else:
        expected = torch.logit(torch.tensor(max(target, 0.5), dtype=torch.float64))
        assert scale == pytest.approx(float(expected) / 4)
