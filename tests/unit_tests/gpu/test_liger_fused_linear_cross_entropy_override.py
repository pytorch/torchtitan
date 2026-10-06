# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import pytest
import torch

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common.linear import Linear


pytest.importorskip("liger_kernel")

from torchtitan_recipes.overrides.liger_fused_linear_cross_entropy import (  # noqa: E402
    LigerFusedLinearCrossEntropyHead,
    LigerFusedLinearCrossEntropyLoss,
)


_RunLoss = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


def _run_torchtitan_loss(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    global_valid_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    head = (
        Linear.Config(in_features=hidden.shape[1], out_features=weight.shape[0])
        .build()
        .to(device=hidden.device, dtype=hidden.dtype)
    )
    with torch.no_grad():
        head.weight.copy_(weight)
    loss_fn = ChunkedLossWrapper.Config(
        num_chunks=8,
        loss_fn=CrossEntropyLoss.Config(global_vocab_size=weight.shape[0]),
    ).build()
    loss_fn.set_lm_head(head)

    loss, _ = loss_fn(hidden, labels, global_valid_tokens)
    loss.backward()
    assert hidden.grad is not None
    assert head.weight.grad is not None
    return loss.detach(), hidden.grad.detach(), head.weight.grad.detach()


def _run_liger_loss(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    global_valid_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    head = (
        LigerFusedLinearCrossEntropyHead.Config(
            in_features=hidden.shape[1],
            out_features=weight.shape[0],
            chunk_mem_const=2,
        )
        .build()
        .to(device=hidden.device, dtype=hidden.dtype)
    )
    with torch.no_grad():
        head.weight.copy_(weight)
    loss_fn = LigerFusedLinearCrossEntropyLoss.Config().build()
    loss_fn.set_lm_head(head)

    loss, _ = loss_fn(hidden, labels, global_valid_tokens)
    loss.backward()
    assert hidden.grad is not None
    assert head.weight.grad is not None
    return loss.detach(), hidden.grad.detach(), head.weight.grad.detach()


def _run_with_fresh_hidden(
    run_loss: _RunLoss,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    global_valid_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return run_loss(
        hidden.detach().clone().requires_grad_(True),
        labels,
        weight,
        global_valid_tokens,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_liger_matches_eager_and_local_compiled_cross_entropy() -> None:
    generator = torch.Generator(device="cuda").manual_seed(42)
    hidden = torch.randn(
        256,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        2048,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    labels = torch.randint(0, 2048, (256,), device="cuda", generator=generator)
    labels[::17] = -100
    global_valid_tokens = (labels != -100).sum()

    try:
        apply_local_compile([])
        eager = _run_with_fresh_hidden(
            _run_torchtitan_loss,
            hidden,
            labels,
            weight,
            global_valid_tokens,
        )

        apply_local_compile(["loss"])
        local_compiled = _run_with_fresh_hidden(
            _run_torchtitan_loss,
            hidden,
            labels,
            weight,
            global_valid_tokens,
        )

        apply_local_compile([])
        liger = _run_with_fresh_hidden(
            _run_liger_loss,
            hidden,
            labels,
            weight,
            global_valid_tokens,
        )
    finally:
        apply_local_compile([])
        torch._dynamo.reset()

    for actual, expected in zip(local_compiled, eager, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-5)
    for expected in (eager, local_compiled):
        torch.testing.assert_close(liger[0], expected[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(liger[1], expected[1], rtol=2e-2, atol=1e-4)
        torch.testing.assert_close(liger[2], expected[2], rtol=2e-2, atol=1e-3)
