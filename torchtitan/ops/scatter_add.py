# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


@torch.library.custom_op("torchtitan::deterministic_scatter_add", mutates_args=())
def deterministic_scatter_add(
    out: torch.Tensor, index: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    prev = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        return out.scatter_add(dim=0, index=index, src=src)
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn_only)


@deterministic_scatter_add.register_fake
def _(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(out)


def _backward(
    ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
) -> tuple[torch.Tensor, None, torch.Tensor]:
    (index,) = ctx.saved_tensors  # pyrefly: ignore[missing-attribute]
    grad_src = torch.gather(grad_output, dim=0, index=index)
    return grad_output, None, grad_src


def _setup_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output: torch.Tensor,
) -> None:
    _out, index, _src = inputs
    ctx.save_for_backward(index)


deterministic_scatter_add.register_autograd(
    _backward,
    setup_context=_setup_context,
)
