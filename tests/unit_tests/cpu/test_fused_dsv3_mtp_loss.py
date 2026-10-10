# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.config import apply_overrides, OverrideConfig
from torchtitan.models.deepseek_v3.mtp import MTPLoss
from torchtitan_recipes.overrides._dsv3_mtp_cross_entropy import ops
from torchtitan_recipes.overrides.fused_dsv3_mtp_loss import (
    _cross_entropy_loss,
    fused_dsv3_mtp_loss,
    FusedDSv3MTPLoss,
)


OVERRIDE = "torchtitan_recipes.overrides.fused_dsv3_mtp_loss.fused_dsv3_mtp_loss"


def test_nested_override_preserves_chunking_and_mtp_settings(caplog):
    config = ChunkedLossWrapper.Config(
        num_chunks=8,
        loss_fn=MTPLoss.Config(global_vocab_size=129280, mtp_scale=0.125),
    )
    apply_overrides(OverrideConfig(imports=[OVERRIDE]), config)
    assert type(config) is ChunkedLossWrapper.Config
    assert config.num_chunks == 8
    assert type(config.loss_fn) is FusedDSv3MTPLoss.Config
    assert config.loss_fn.global_vocab_size == 129280
    assert config.loss_fn.mtp_scale == 0.125
    assert ops.ACCEPTED is False
    loss = config.build().loss_fn
    assert loss.mtp_scale == 0.125
    assert "has not met all acceptance gates" in caplog.text


def test_other_loss_configs_and_vocabularies_stay_native(caplog):
    config = ChunkedLossWrapper.Config(loss_fn=CrossEntropyLoss.Config())
    original = config.loss_fn
    apply_overrides(OverrideConfig(imports=[OVERRIDE]), config)
    assert config.loss_fn is original
    unsupported = MTPLoss.Config(global_vocab_size=16)
    assert fused_dsv3_mtp_loss(unsupported) is unsupported
    assert "full 129280 vocabulary" in caplog.text


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("counts", [None, [16, 11, 0]])
def test_cpu_fallback_preserves_mtp_loss_gradients_and_metrics(accepted, counts):
    torch.manual_seed(42)
    logits = tuple(torch.randn(16, 32) for _ in range(3))
    labels = tuple(torch.randint(32, (16,)) for _ in range(3))
    labels[1][11:] = -100
    labels[2].fill_(-100)
    denominators = None if counts is None else torch.tensor(counts)
    results = []
    with patch.object(ops, "ACCEPTED", accepted), patch.object(
        ops, "cross_entropy_sum", side_effect=AssertionError("CPU must use native CE")
    ):
        for cls in (MTPLoss, FusedDSv3MTPLoss):
            loss_fn = cls.Config(mtp_scale=0.125).build()
            inputs = tuple(value.clone().requires_grad_() for value in logits)
            loss, metrics = loss_fn(inputs, labels, denominators)
            gradients = torch.autograd.grad(loss, inputs, torch.tensor(-0.375))
            assert metrics == {}
            results.append((loss, *gradients))
    for expected, actual in zip(*results):
        assert torch.equal(
            expected.detach().reshape(-1).view(torch.uint8),
            actual.detach().reshape(-1).view(torch.uint8),
        )


def test_acceptance_gate_prevents_model_dispatch_even_for_supported_metadata():
    logits = torch.zeros(2, 4)
    labels = torch.zeros(2, dtype=torch.int64)
    with patch.object(ops, "supports", return_value=True), patch.object(
        ops, "cross_entropy_sum", side_effect=AssertionError("gate must stay closed")
    ):
        assert ops.ACCEPTED is False
        actual = _cross_entropy_loss(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("reduction", ["sum", "none"])
def test_unsupported_inputs_keep_native_reduction_and_validation(reduction):
    logits = torch.randn(5, 16, dtype=torch.float64)
    labels = torch.tensor([0, 15, -100, 8, 3])
    with patch.object(ops, "ACCEPTED", True):
        actual = _cross_entropy_loss(
            logits, labels, global_vocab_size=16, reduction=reduction
        )
        with pytest.raises(ValueError, match="Unsupported cross-entropy reduction"):
            _cross_entropy_loss(logits, labels, reduction="mean")
    expected = torch.nn.functional.cross_entropy(
        logits.float(), labels, reduction=reduction
    )
    assert torch.equal(actual, expected)
