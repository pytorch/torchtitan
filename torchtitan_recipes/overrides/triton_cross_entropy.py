# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Triton cross-entropy override for full, unsharded logits.

The two-pass kernel design follows Transformer Engine's Triton cross-entropy:
https://github.com/NVIDIA/TransformerEngine/blob/v2.7/transformer_engine/pytorch/triton/cross_entropy.py

The first pass computes softmax statistics for each token. The second pass
computes the token loss and replaces the logits buffer with its gradient, so
backward only needs to apply the upstream scalar.
"""

from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl

from torchtitan.components.loss import CrossEntropyLoss, IGNORE_INDEX, LossFunction
from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import spmd_mesh_size

__all__ = [
    "TritonCrossEntropyLoss",
    "triton_cross_entropy",
    "triton_cross_entropy_loss",
]


_MAX_BLOCK_SIZE = 32768


@triton.jit
def _softmax_stats_kernel(
    logits,
    labels,
    stats,
    num_classes,
    LOGITS_ROW_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    row_logits = logits + row * LOGITS_ROW_STRIDE
    label = tl.load(labels + row)

    label_is_valid = (label >= 0) & (label < num_classes)
    label_logit = tl.load(
        row_logits + label,
        mask=label_is_valid,
        other=float("-inf"),
    ).to(tl.float32)

    row_max = float("-inf")
    row_sumexp = 0.0
    for block_start in range(0, num_classes, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        values = tl.load(
            row_logits + offsets,
            mask=offsets < num_classes,
            other=float("-inf"),
        ).to(tl.float32)
        block_max = tl.max(values)
        new_max = tl.maximum(row_max, block_max)
        row_sumexp = row_sumexp * tl.exp(row_max - new_max) + tl.sum(
            tl.exp(values - new_max)
        )
        row_max = new_max

    row_stats = stats + row * 3
    tl.store(row_stats, row_max)
    tl.store(row_stats + 1, row_sumexp)
    tl.store(row_stats + 2, label_logit)


@triton.jit
def _cross_entropy_forward_kernel(
    logits,
    labels,
    losses,
    stats,
    num_classes,
    ignore_index,
    LOGITS_ROW_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    row_logits = logits + row * LOGITS_ROW_STRIDE
    label = tl.load(labels + row)

    if label == ignore_index:
        for block_start in range(0, num_classes, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            tl.store(
                row_logits + offsets,
                0.0,
                mask=offsets < num_classes,
            )
        return

    row_stats = stats + row * 3
    row_max = tl.load(row_stats)
    row_sumexp = tl.load(row_stats + 1)
    label_logit = tl.load(row_stats + 2)

    for block_start in range(0, num_classes, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        values = tl.load(row_logits + offsets, mask=mask, other=float("-inf"))
        grad_dtype = values.dtype
        gradients = tl.exp(values.to(tl.float32) - row_max) / row_sumexp
        tl.store(row_logits + offsets, gradients.to(grad_dtype), mask=mask)

    tl.debug_barrier()
    if label >= 0:
        if label < num_classes:
            label_gradient = tl.load(row_logits + label)
            tl.store(row_logits + label, label_gradient - 1.0)
    tl.store(losses + row, -(label_logit - row_max - tl.log(row_sumexp)))


@triton.jit
def _scale_gradient_kernel(
    gradients,
    grad_output,
    num_classes,
    GRADIENT_ROW_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    row_gradients = gradients + row * GRADIENT_ROW_STRIDE
    scale = tl.load(grad_output)

    for block_start in range(0, num_classes, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        values = tl.load(row_gradients + offsets, mask=mask)
        tl.store(row_gradients + offsets, values * scale, mask=mask)


def _cross_entropy_forward(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_classes = logits.shape
    block_size = min(_MAX_BLOCK_SIZE, triton.next_power_of_2(num_classes))
    losses = torch.zeros(num_tokens, dtype=torch.float32, device=logits.device)
    stats = torch.empty((num_tokens, 3), dtype=torch.float32, device=logits.device)

    _softmax_stats_kernel[(num_tokens,)](
        logits,
        labels,
        stats,
        num_classes,
        LOGITS_ROW_STRIDE=logits.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=32,
    )
    _cross_entropy_forward_kernel[(num_tokens,)](
        logits,
        labels,
        losses,
        stats,
        num_classes,
        IGNORE_INDEX,
        LOGITS_ROW_STRIDE=logits.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=32,
    )
    return losses.sum(), logits


class _TritonCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        loss, grad_logits = _cross_entropy_forward(logits, labels)
        ctx.save_for_backward(grad_logits.detach())
        return loss

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        (grad_logits,) = ctx.saved_tensors
        num_tokens, num_classes = grad_logits.shape
        block_size = min(_MAX_BLOCK_SIZE, triton.next_power_of_2(num_classes))
        _scale_gradient_kernel[(num_tokens,)](
            grad_logits,
            grad_output,
            num_classes,
            GRADIENT_ROW_STRIDE=grad_logits.stride(0),
            BLOCK_SIZE=block_size,
            num_warps=32,
        )
        return grad_logits, None


def triton_cross_entropy_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
    *,
    global_vocab_size: int | None = None,
    reduction: str = "sum",
) -> torch.Tensor:
    """Return summed cross-entropy while storing ``dlogits`` in ``pred``."""
    del global_vocab_size
    if spmd_mesh_size("tp") > 1:
        raise ValueError("Triton cross-entropy requires TP=1.")
    if reduction != "sum":
        raise ValueError("Triton cross-entropy only supports sum reduction.")
    if pred.ndim != 2 or labels.ndim != 1 or pred.shape[0] != labels.shape[0]:
        raise ValueError("Triton cross-entropy requires logits [T, V] and labels [T].")
    if pred.stride(-1) != 1 or labels.stride(-1) != 1:
        raise ValueError("Triton cross-entropy requires contiguous final dimensions.")
    return _TritonCrossEntropy.apply(pred, labels)


class TritonCrossEntropyLoss(CrossEntropyLoss):
    """Full-logits cross-entropy using the two-pass Triton implementation."""

    @dataclass(kw_only=True, slots=True)
    class Config(CrossEntropyLoss.Config):
        pass

    def __init__(self, config: Config) -> None:
        self.fn: LossFunction = triton_cross_entropy_loss
        self.global_vocab_size = config.global_vocab_size


@override(
    target=CrossEntropyLoss.Config,
    fqns=["loss"],
    exact=True,
    description="Use two-pass Triton cross-entropy for full logits (TP=1).",
)
def triton_cross_entropy(
    cfg: CrossEntropyLoss.Config,
) -> TritonCrossEntropyLoss.Config:
    return derive(cfg, TritonCrossEntropyLoss.Config)
