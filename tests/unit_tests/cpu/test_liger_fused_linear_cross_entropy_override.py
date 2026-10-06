# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss, MSELoss
from torchtitan.models.common.linear import Linear
from torchtitan_recipes.overrides import liger_fused_linear_cross_entropy as override


def _fake_liger_fused_linear_cross_entropy(
    input,
    weight,
    target,
    *,
    bias,
    ignore_index,
    reduction,
    chunk_mem_const,
):
    assert ignore_index == -100
    assert reduction == "sum"
    assert chunk_mem_const > 0
    return F.cross_entropy(
        F.linear(input, weight, bias).float(),
        target,
        ignore_index=ignore_index,
        reduction=reduction,
    )


@pytest.fixture
def fake_liger():
    with (
        patch.object(override, "_LIGER_IMPORT_ERROR", None),
        patch.object(
            override,
            "_liger_fused_linear_cross_entropy",
            _fake_liger_fused_linear_cross_entropy,
        ),
    ):
        yield


def test_factories_build_paired_configs(fake_liger) -> None:
    head_config = override.liger_fused_linear_cross_entropy_head(
        Linear.Config(in_features=8, out_features=16),
        chunk_mem_const=4,
    )
    loss_config = override.liger_fused_linear_cross_entropy_loss(
        ChunkedLossWrapper.Config(
            num_chunks=2,
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=16),
        )
    )

    assert isinstance(head_config, override.LigerFusedLinearCrossEntropyHead.Config)
    assert head_config.chunk_mem_const == 4
    assert isinstance(loss_config, override.LigerFusedLinearCrossEntropyLoss.Config)


def test_loss_factory_rejects_non_cross_entropy(fake_liger) -> None:
    with pytest.raises(ValueError, match="requires.*CrossEntropyLoss"):
        override.liger_fused_linear_cross_entropy_loss(
            ChunkedLossWrapper.Config(loss_fn=MSELoss.Config())
        )


def test_loss_requires_paired_head() -> None:
    loss = override.LigerFusedLinearCrossEntropyLoss.Config().build()
    head = Linear.Config(in_features=8, out_features=16).build()

    with pytest.raises(ValueError, match="paired.*head override"):
        loss.set_lm_head(head)


def test_fused_loss_matches_linear_cross_entropy(fake_liger) -> None:
    torch.manual_seed(42)
    head = override.LigerFusedLinearCrossEntropyHead.Config(
        in_features=8,
        out_features=16,
        chunk_mem_const=4,
    ).build()
    loss_fn = override.LigerFusedLinearCrossEntropyLoss.Config().build()
    loss_fn.set_lm_head(head)

    hidden = torch.randn(12, 8, requires_grad=True)
    labels = torch.randint(0, 16, (12,))
    labels[0] = -100
    global_valid_tokens = torch.tensor(11)

    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_weight = head.weight.detach().clone().requires_grad_(True)
    ref_loss = (
        F.cross_entropy(
            F.linear(ref_hidden, ref_weight).float(),
            labels,
            ignore_index=-100,
            reduction="sum",
        )
        / global_valid_tokens
    )
    loss, metrics = loss_fn(hidden, labels, global_valid_tokens)

    ref_loss.backward()
    loss.backward()

    torch.testing.assert_close(loss, ref_loss)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad)
    torch.testing.assert_close(head.weight.grad, ref_weight.grad)
    assert metrics == {}


def test_head_keeps_standard_logits_forward(fake_liger) -> None:
    head = override.LigerFusedLinearCrossEntropyHead.Config(
        in_features=8,
        out_features=16,
    ).build()
    hidden = torch.randn(12, 8)

    torch.testing.assert_close(head(hidden), F.linear(hidden, head.weight))


def test_head_rejects_tensor_parallel_loss(fake_liger) -> None:
    head = override.LigerFusedLinearCrossEntropyHead.Config(
        in_features=8,
        out_features=16,
    ).build()
    hidden = torch.randn(12, 8)
    labels = torch.randint(0, 16, (12,))

    with (
        patch.object(override, "spmd_mesh_size", return_value=2),
        pytest.raises(ValueError, match="requires TP=1"),
    ):
        head(hidden, target=labels)


def test_chunk_mem_const_must_be_positive() -> None:
    with pytest.raises(ValueError, match="chunk_mem_const must be positive"):
        override.LigerFusedLinearCrossEntropyHead.Config(
            in_features=8,
            out_features=16,
            chunk_mem_const=0,
        )
