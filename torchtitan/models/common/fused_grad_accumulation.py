# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused WGRAD accumulation into a parameter's running gradient.

Later contributions to a gradient (microbatches with gradient sync disabled,
or ``ChunkedLossWrapper`` chunks) normally cost a separate AccumulateGrad add
and a full-size WGRAD temporary. A custom autograd Function can instead add
its WGRAD into the running ``.grad`` in the GEMM epilogue:

    # forward
    ctx.weight_param = weight if can_fuse_grad_accumulation(weight) else None
    # backward
    running_grad = take_grad_for_fused_accumulation(ctx.weight_param, dtype)
    if running_grad is None:
        grad_weight = mm(...)
    else:
        grad_weight = addmm(running_grad, ..., out=running_grad)
    return ..., grad_weight, ...

Returning the running buffer after taking it off the parameter lets
AccumulateGrad reattach it without a copy, instead of adding it to itself.
"""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode


def can_fuse_grad_accumulation(weight: torch.Tensor) -> bool:
    """Whether a backward may fuse its WGRAD into ``weight.grad``.

    Needs the leaf parameter, whose ``.grad`` autograd populates: a view of it
    (a stacked weight's flattened view) or SimpleFSDP's parametrization output
    is not a leaf. A traced backward cannot represent the read-and-clear of
    ``.grad``. Dynamo sets is_compiling; GraphTrainer's make_fx tracer does
    not, so ask the proxy mode as well.
    """
    is_tracing = torch.compiler.is_compiling() or get_proxy_mode() is not None
    return not is_tracing and weight.is_leaf


def take_grad_for_fused_accumulation(
    weight_param: torch.Tensor | None, wgrad_dtype: torch.dtype
) -> torch.Tensor | None:
    """Take ``weight_param.grad`` off the parameter for a backward to add its WGRAD into.

    Returns None when there is nothing to accumulate into: no parameter (see
    ``can_fuse_grad_accumulation``), no gradient since it was last consumed,
    or a gradient narrower than ``wgrad_dtype``, which would round the WGRAD.
    A wider one is fine: under FSDP an activation checkpoint recompute sees
    grad_dtype cleared and builds a BF16 WGRAD, while the running gradient is
    already in the FP32 reduce dtype. Otherwise clears ``.grad``, and the
    caller must add into the returned buffer and return it as the gradient:
    taking it and returning a fresh gradient would drop the earlier
    contributions.
    """
    if weight_param is None:
        return None
    running_grad = weight_param.grad
    if (
        running_grad is None
        or torch.promote_types(running_grad.dtype, wgrad_dtype) != running_grad.dtype
    ):
        return None
    weight_param.grad = None
    return running_grad
