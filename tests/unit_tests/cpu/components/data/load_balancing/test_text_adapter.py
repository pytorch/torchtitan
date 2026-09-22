# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from torchtitan.components.data.collators import HAS_PIN_MEMORY, TextCollator
from torchtitan.components.data.dataset import TextSequence
from torchtitan.components.data.load_balancing.text import (
    QuadraticAttentionCost,
    TokenizedTextPackingAdapter,
)
from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.components.loss import IGNORE_INDEX


def _microbatch() -> TokenizedTrainingMicrobatch:
    return TokenizedTrainingMicrobatch(
        input=torch.tensor([10, 11, 12, 20, 21, 30, 0, 0]),
        labels=torch.tensor(
            [11, 12, IGNORE_INDEX, 21, 22, 31, IGNORE_INDEX, IGNORE_INDEX]
        ),
        positions=torch.tensor([0, 1, 2, 0, 1, 0, 0, 1]),
        padding_mask=torch.tensor(
            [False, False, False, False, False, False, True, True]
        ),
        num_valid_tokens=5,
    )


def _adapter(
    *, max_num_documents: int | None = 3, expect_pinned_memory: bool = False
) -> TokenizedTextPackingAdapter:
    return TokenizedTextPackingAdapter.Config().build(
        num_tokens_per_microbatch=8,
        max_context_length=4,
        max_num_documents=max_num_documents,
        expect_pinned_memory=expect_pinned_memory,
    )


def test_inspect_canonical_packed_microbatch_and_compute_cost() -> None:
    metadata = _adapter().inspect_microbatch(_microbatch())
    cost_model = QuadraticAttentionCost.Config().build()

    assert metadata.segment_lengths == (3, 2, 1)
    assert metadata.num_tokens == 8
    assert metadata.num_non_padding_tokens == 6
    assert metadata.num_documents == 3
    assert metadata.payload_bytes == 200
    assert cost_model.estimate(metadata) == 14


def test_inspects_actual_text_collator_output() -> None:
    rows = [
        TextSequence(
            input_ids=np.arange(length),
            labels=np.arange(length),
        )
        for length in (3, 2, 1)
    ]
    batch = TextCollator.Config().build(
        context=SimpleNamespace(
            num_tokens_per_microbatch=8,
            max_context_length=4,
        )
    )(rows)

    metadata = _adapter(expect_pinned_memory=HAS_PIN_MEMORY).inspect_microbatch(batch)

    assert metadata.segment_lengths == (3, 2, 1)


@pytest.mark.parametrize("field", ["input", "labels", "positions"])
def test_rejects_non_int64_tensor(field: str) -> None:
    batch = _microbatch()
    value = getattr(batch, field).to(torch.int32)

    with pytest.raises(ValueError, match=f"{field} must have dtype torch.int64"):
        _adapter().inspect_microbatch(replace(batch, **{field: value}))


def test_rejects_non_boolean_padding_mask() -> None:
    batch = _microbatch()

    with pytest.raises(ValueError, match="padding_mask must have dtype torch.bool"):
        _adapter().inspect_microbatch(
            replace(batch, padding_mask=batch.padding_mask.to(torch.int64))
        )


@pytest.mark.parametrize("field", ["input", "labels", "positions", "padding_mask"])
def test_rejects_noncanonical_shape(field: str) -> None:
    batch = _microbatch()
    value = getattr(batch, field).reshape(2, 4)

    with pytest.raises(ValueError, match=f"{field} must have shape \\(8,\\)"):
        _adapter().inspect_microbatch(replace(batch, **{field: value}))


def test_rejects_non_cpu_tensor() -> None:
    batch = _microbatch()

    with pytest.raises(ValueError, match="input must be on CPU"):
        _adapter().inspect_microbatch(
            replace(batch, input=torch.empty(8, dtype=torch.int64, device="meta"))
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "padding_mask",
            torch.tensor([False, True, False, False, False, False, True, True]),
            "padding_mask must be one trailing suffix",
        ),
        (
            "input",
            torch.tensor([10, 11, 12, 20, 21, 30, 7, 0]),
            "padding input tokens must be zero",
        ),
        (
            "labels",
            torch.tensor([11, 12, IGNORE_INDEX, 21, 22, 31, 7, IGNORE_INDEX]),
            "padding labels must equal IGNORE_INDEX",
        ),
        (
            "positions",
            torch.tensor([0, 1, 2, 0, 1, 0, 0, 3]),
            "padding positions are not canonical",
        ),
        (
            "positions",
            torch.tensor([0, 1, 3, 0, 1, 0, 0, 1]),
            "real-token positions are not canonical",
        ),
    ],
)
def test_rejects_noncanonical_packed_content(
    field: str, value: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _adapter().inspect_microbatch(replace(_microbatch(), **{field: value}))


def test_rejects_empty_microbatch() -> None:
    batch = _microbatch()

    with pytest.raises(ValueError, match="at least one non-padding token"):
        _adapter().inspect_microbatch(
            replace(
                batch,
                input=torch.zeros(8, dtype=torch.int64),
                labels=torch.full((8,), IGNORE_INDEX, dtype=torch.int64),
                positions=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
                padding_mask=torch.ones(8, dtype=torch.bool),
                num_valid_tokens=0,
            )
        )


def test_rejects_incorrect_valid_token_count() -> None:
    with pytest.raises(ValueError, match="num_valid_tokens does not match"):
        _adapter().inspect_microbatch(replace(_microbatch(), num_valid_tokens=6))


def test_rejects_model_kwargs() -> None:
    with pytest.raises(ValueError, match="model_kwargs must be empty"):
        _adapter().inspect_microbatch(
            replace(_microbatch(), model_kwargs={"unsupported": torch.tensor(1)})
        )


def test_rejects_more_than_configured_document_capacity() -> None:
    with pytest.raises(ValueError, match="3 documents exceeds max_num_documents=2"):
        _adapter(max_num_documents=2).inspect_microbatch(_microbatch())


def test_rejects_unexpected_page_locking() -> None:
    with pytest.raises(ValueError, match="input pinned-memory state"):
        _adapter(expect_pinned_memory=True).inspect_microbatch(_microbatch())
