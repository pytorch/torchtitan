# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


@torch.library.custom_op("torchtitan::deterministic_topk", mutates_args=())
def deterministic_topk(
    input: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    prev = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        return torch.topk(input, k=k, dim=dim, largest=largest, sorted=sorted)
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn_only)


@deterministic_topk.register_fake
def _(
    input: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = list(input.shape)
    shape[dim % input.dim()] = k
    return input.new_empty(shape), input.new_empty(shape, dtype=torch.long)


def _backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad_values: torch.Tensor | None,
    grad_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, None, None, None, None]:
    (indices,) = ctx.saved_tensors  # pyrefly: ignore[missing-attribute]
    if grad_values is None:
        return (
            torch.zeros(
                ctx.input_shape,  # pyrefly: ignore[missing-attribute]
                dtype=ctx.input_dtype,  # pyrefly: ignore[missing-attribute]
                device=indices.device,
            ),
            None,
            None,
            None,
            None,
        )

    grad_input = torch.zeros(
        ctx.input_shape,  # pyrefly: ignore[missing-attribute]
        dtype=grad_values.dtype,
        device=grad_values.device,
    )
    grad_input.scatter_(
        ctx.dim,  # pyrefly: ignore[missing-attribute]
        indices,
        grad_values,
    )
    return grad_input, None, None, None, None


def _setup_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[torch.Tensor, int, int, bool, bool],
    output: tuple[torch.Tensor, torch.Tensor],
) -> None:
    input, _k, dim, _largest, _sorted = inputs
    _values, indices = output
    ctx.save_for_backward(indices)
    ctx.input_shape = tuple(input.shape)  # pyrefly: ignore[missing-attribute]
    ctx.input_dtype = input.dtype  # pyrefly: ignore[missing-attribute]
    ctx.dim = dim % input.dim()  # pyrefly: ignore[missing-attribute]


deterministic_topk.register_autograd(
    _backward,
    setup_context=_setup_context,
)
