# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import random

import grain.python as grain
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.components.loss import CrossEntropyLoss, IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention.cp_attention import UlyssesVarlenInnerAttention
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan_recipes.decision.prepare_data import LETTERS, render
from torchtitan_recipes.decision.qwen3 import (
    AnswerTokenLoss,
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
        "answer": " B",
    }
    seq = processor(sample, np.random.default_rng(0))
    prompt = processor._tokenizer.encode(sample["prompt"], add_bos=False, add_eos=False)
    (answer_id,) = processor._tokenizer.encode(" B", add_bos=False, add_eos=False)

    assert seq.input_ids.tolist() == prompt
    assert seq.labels.tolist() == [IGNORE_INDEX] * (len(prompt) - 1) + [answer_id]


def test_multi_token_answer_raises() -> None:
    with pytest.raises(ValueError, match="one token"):
        _processor()({"prompt": "Answer:", "answer": " AB"}, np.random.default_rng(0))


def test_overlong_prompt_raises() -> None:
    # "x" * n is n tokens with the test tokenizer: 16 fills the context, 17 overflows.
    processor = _processor(max_context_length=16)
    processor({"prompt": "x" * 16, "answer": " A"}, np.random.default_rng(0))
    with pytest.raises(ValueError, match="max_context_length"):
        processor({"prompt": "x" * 17, "answer": " A"}, np.random.default_rng(0))


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
        letter = LETTERS[row["gold"]]
        assert row["answer"] == " " + letter
        assert f"\n{letter}) {gold_desc}\n" in row["prompt"]
        assert row["keys"][row["gold"]] == gold_key
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
