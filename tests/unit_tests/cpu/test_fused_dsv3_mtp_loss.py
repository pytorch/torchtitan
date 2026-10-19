# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from unittest.mock import patch

# Original tests by Yidi Wu (@ydwu4), pytorch/torchtitan#5261.
# Source revision: 1acef597590c0a4933ddcf50e7a6769d99aabdfa.

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.config import apply_overrides, OverrideConfig

from torchtitan.config.override import _REGISTRY
from torchtitan.models.deepseek_v3.mtp import MTPLoss
from torchtitan_recipes.overrides import fused_dsv3_mtp_loss as ops
from torchtitan_recipes.overrides.fused_dsv3_mtp_loss import (
    _cross_entropy_loss,
    fused_dsv3_mtp_loss,
    FusedDSv3MTPLoss,
)


OVERRIDE = "torchtitan_recipes.overrides.fused_dsv3_mtp_loss.fused_dsv3_mtp_loss"


_REGISTERED_OVERRIDES = {
    key: value
    for key, value in _REGISTRY.items()
    if key.startswith("torchtitan_recipes.overrides.fused_dsv3_")
}


@pytest.fixture(autouse=True)
def _registered_overrides():
    with patch.dict(_REGISTRY, _REGISTERED_OVERRIDES):
        yield


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
    loss = config.build().loss_fn
    assert loss.mtp_scale == 0.125
    assert all(record.levelno < logging.WARNING for record in caplog.records)


def test_other_loss_configs_stay_native_and_vocabularies_are_generic(caplog):
    config = ChunkedLossWrapper.Config(loss_fn=CrossEntropyLoss.Config())
    original = config.loss_fn
    apply_overrides(OverrideConfig(imports=[OVERRIDE]), config)
    assert config.loss_fn is original
    assert "produced no replacements" in caplog.text
    caplog.clear()
    small = MTPLoss.Config(global_vocab_size=16)
    fused = fused_dsv3_mtp_loss(small, require_fusion=True)
    assert isinstance(fused, FusedDSv3MTPLoss.Config)
    assert fused.global_vocab_size == 16
    assert not caplog.text


@pytest.mark.parametrize("counts", [None, [16, 11, 0]])
def test_cpu_fallback_preserves_mtp_loss_gradients_and_metrics(counts):
    torch.manual_seed(42)
    logits = tuple(torch.randn(16, 32) for _ in range(3))
    labels = tuple(torch.randint(32, (16,)) for _ in range(3))
    labels[1][11:] = -100
    labels[2].fill_(-100)
    denominators = None if counts is None else torch.tensor(counts)
    results = []
    with patch.object(
        ops, "cross_entropy_sum", side_effect=AssertionError("CPU must use native CE")
    ):
        for cls in (MTPLoss, FusedDSv3MTPLoss):
            loss_fn = cls.Config(mtp_scale=0.125).build()
            inputs = tuple(value.clone().requires_grad_() for value in logits)
            loss, metrics = loss_fn(inputs, labels, denominators)
            gradients = torch.autograd.grad(loss, inputs, torch.tensor(-0.375))
            assert set(metrics) == {
                "lm_loss_mean",
                "mtp_loss_mean",
                "combined_loss_mean",
            }
            assert torch.equal(metrics["combined_loss_mean"], loss.detach())
            assert all(not value.requires_grad for value in metrics.values())
            results.append(
                (loss, *gradients, *(metrics[key] for key in sorted(metrics)))
            )
    for expected, actual in zip(*results):
        assert torch.equal(
            expected.detach().reshape(-1).view(torch.uint8),
            actual.detach().reshape(-1).view(torch.uint8),
        )


def test_required_fusion_rejects_unsupported_inputs():
    logits = torch.zeros(2, 4)
    labels = torch.zeros(2, dtype=torch.int64)
    with pytest.raises(ValueError, match="Required DSv3 MTP CE"):
        _cross_entropy_loss(logits, labels, require_fusion=True)


def test_general_fake_ce_metadata_without_storage_reads():
    with FakeTensorMode(), patch.object(
        torch.Tensor, "data_ptr", side_effect=AssertionError("fake CE read storage")
    ):
        logits = torch.empty(8193, 4096, device="cuda", dtype=torch.bfloat16)
        labels = torch.empty(8193, device="cuda", dtype=torch.int64)
        assert ops.supports(logits, labels)
        loss, stats = ops.forward_op(logits, labels, -100)
        assert loss.shape == () and loss.dtype == torch.float32
        assert stats.shape == (2, 8193)
        gradient = ops.backward_op(logits, labels, stats, loss, -100)
        assert gradient.shape == logits.shape and gradient.dtype == logits.dtype


def test_fake_ce_rejects_inconsistent_stats():
    with FakeTensorMode():
        logits = torch.empty(17, 129, device="cuda", dtype=torch.bfloat16)
        labels = torch.empty(17, device="cuda", dtype=torch.int64)
        with pytest.raises(ValueError, match="FP32 stats"):
            ops.backward_op(
                logits,
                labels,
                torch.empty(2, 16, device="cuda"),
                torch.empty((), device="cuda"),
                -100,
            )


@pytest.mark.parametrize("reduction", ["sum", "none"])
def test_unsupported_inputs_keep_native_reduction_and_validation(reduction):
    logits = torch.randn(5, 16, dtype=torch.float64)
    labels = torch.tensor([0, 15, -100, 8, 3])
    actual = _cross_entropy_loss(
        logits, labels, global_vocab_size=16, reduction=reduction
    )
    with pytest.raises(ValueError, match="Unsupported cross-entropy reduction"):
        _cross_entropy_loss(logits, labels, reduction="mean")
    expected = torch.nn.functional.cross_entropy(
        logits.float(), labels, reduction=reduction
    )
    assert torch.equal(actual, expected)
