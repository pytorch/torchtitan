# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.tools.logging import logger

_lib: torch.library.Library | None = None


def _embedding_dense_backward_xpu(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
) -> torch.Tensor:
    embedding_dim = grad_output.shape[-1]
    flat_indices = indices.reshape(-1)
    flat_grad = grad_output.reshape(-1, embedding_dim)

    if scale_grad_by_freq:
        counts = torch.zeros(
            num_weights, dtype=flat_grad.dtype, device=flat_grad.device
        )
        counts.index_add_(
            0, flat_indices, torch.ones_like(flat_indices, dtype=flat_grad.dtype)
        )
        scale = counts.clamp(min=1).reciprocal()
        flat_grad = flat_grad * scale[flat_indices].unsqueeze(-1)

    grad_weight = torch.zeros(
        (num_weights, embedding_dim), dtype=flat_grad.dtype, device=flat_grad.device
    )
    grad_weight.index_add_(0, flat_indices, flat_grad)

    if padding_idx >= 0:
        keep = (
            torch.arange(num_weights, device=grad_weight.device) != padding_idx
        ).to(grad_weight.dtype)
        grad_weight = grad_weight * keep.unsqueeze(-1)

    return grad_weight


def enable_capture_safe_embedding_backward() -> None:
    """Override aten::embedding_dense_backward on XPU with a capture-safe form."""
    global _lib
    if _lib is not None:
        return
    _lib = torch.library.Library("aten", "IMPL")
    _lib.impl("embedding_dense_backward", _embedding_dense_backward_xpu, "XPU")
    logger.warning(
        "Overrode aten::embedding_dense_backward on XPU with a zeros+index_add "
        "implementation for XPU graph capture; embedding-gradient accumulation "
        "order is now atomics-dependent, so embedding grads are no longer "
        "bitwise reproducible run to run."
    )
