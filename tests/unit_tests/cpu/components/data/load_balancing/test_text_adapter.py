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

from torchtitan.components.data.collators import TextCollator
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


def _adapter() -> TokenizedTextPackingAdapter:
    return TokenizedTextPackingAdapter.Config().build()


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

    metadata = _adapter().inspect_microbatch(batch)

    assert metadata.segment_lengths == (3, 2, 1)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "padding_mask",
            torch.tensor([False, True, False, False, False, False, True, True]),
            "padding_mask must be one trailing suffix",
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


def test_ignores_unrelated_values_and_counts_model_kwarg_tensor_bytes() -> None:
    batch = replace(
        _microbatch(),
        input=torch.tensor([10, 11, 12, 20, 21, 30, 7, 8], dtype=torch.int32),
        labels=torch.tensor([11, 12, -1, 21, 22, 31, 7, 8], dtype=torch.int32),
        positions=torch.tensor([0, 1, 2, 0, 1, 0, 7, 6], dtype=torch.int32),
        num_valid_tokens=123,
        model_kwargs={"extra": torch.ones(3, dtype=torch.int16)},
    )

    metadata = _adapter().inspect_microbatch(batch)

    assert metadata.segment_lengths == (3, 2, 1)
    assert metadata.payload_bytes == 110
