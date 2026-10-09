# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In-place WGRAD accumulation into a parameter's running gradient.

Later contributions to a gradient (microbatches with gradient sync disabled,
or ``ChunkedLossWrapper`` chunks) normally cost a separate AccumulateGrad add
and a full-size WGRAD temporary. A module with an ``inplace_wgrad_accum``
config option can instead add its WGRAD into the running ``.grad`` in the GEMM
epilogue:

    # forward
    inplace = uses_inplace_wgrad_accum(self.inplace_wgrad_accum, weight, self)
    ctx.weight_param = weight if inplace else None
    # backward
    grad = running_grad(ctx.weight_param, wgrad_dtype)
    if grad is None:
        grad_weight = mm(...)
    else:
        addmm(grad, ..., out=grad)
        ctx.weight_param.grad = None
        grad_weight = grad
    return ..., grad_weight, ...

Clearing ``.grad`` and returning the same buffer lets AccumulateGrad reattach
it without a copy, instead of adding it to itself. Tracing (torch.compile,
GraphTrainer) cannot represent the read-and-clear of ``.grad``, so those paths
set ``inplace_wgrad_accum=False``.
"""

import torch
from torch import nn
from torch.fx.experimental.proxy_tensor import get_proxy_mode


def uses_inplace_wgrad_accum(
    inplace_wgrad_accum: bool, weight: torch.Tensor, module: nn.Module
) -> bool:
    """Whether this forward's backward adds its WGRAD into ``weight.grad`` in place.

    False when ``inplace_wgrad_accum`` is off or no WGRAD will be computed
    (grad mode off, or a frozen weight). Raises when it is on but cannot run,
    rather than silently falling back to an ordinary WGRAD: a traced forward
    (torch.compile, or GraphTrainer's make_fx, which does not set
    is_compiling, hence the proxy mode check), or a weight that is not the leaf
    parameter whose ``.grad`` autograd populates.
    """
    if not (inplace_wgrad_accum and torch.is_grad_enabled() and weight.requires_grad):
        return False
    module_name = type(module).__qualname__
    if torch.compiler.is_compiling() or get_proxy_mode() is not None:
        raise RuntimeError(
            f"{module_name} has inplace_wgrad_accum=True but is being traced "
            "(torch.compile or GraphTrainer). A traced backward cannot add into "
            "weight.grad in place; set inplace_wgrad_accum=False."
        )
    if not weight.is_leaf:
        raise RuntimeError(
            f"{module_name} has inplace_wgrad_accum=True but its weight is not "
            "the leaf parameter (e.g. SimpleFSDP's parametrization output), so "
            "autograd never populates its .grad; set inplace_wgrad_accum=False."
        )
    return True


def running_grad(
    weight_param: torch.Tensor | None, wgrad_dtype: torch.dtype
) -> torch.Tensor | None:
    """``weight_param.grad`` if a WGRAD in ``wgrad_dtype`` can be added into it in place.

    None when there is nothing to accumulate into: no parameter (in-place
    accumulation is off), no gradient since it was last consumed, or a gradient
    narrower than ``wgrad_dtype``, which would round the WGRAD. A wider one is
    fine: under FSDP an activation checkpoint recompute sees grad_dtype cleared
    and builds a BF16 WGRAD, while the running gradient is already in the FP32
    reduce dtype. Read-only: the caller that adds into the gradient clears
    ``weight_param.grad`` and returns the gradient as its grad_weight.
    """
    if weight_param is None or weight_param.grad is None:
        return None
    grad = weight_param.grad
    if torch.promote_types(grad.dtype, wgrad_dtype) != grad.dtype:
        return None
    return grad
