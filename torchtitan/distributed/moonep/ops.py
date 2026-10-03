# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonEP dispatch, expert compute and combine as ``torch.library`` ops."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch_remat as remat
from torch import Tensor


@dataclass
class _Plan:
    plan: object
    compute: Callable[..., Tensor] | None = None


# Ops take only tensors, so a plan crosses them as a CPU id into this table; combine removes it.
_plans: dict[int, _Plan] = {}
_next_plan_id = 0


@torch.library.custom_op("moonep::dispatch", mutates_args=())
def _dispatch(
    x: Tensor, weights: Tensor, ids: Tensor, counts: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    from torchtitan.distributed.moonep.moonep import current_buffer

    global _next_plan_id
    buffer, _ = current_buffer()
    hidden, route_weights, cu_seqlens, plan = buffer.dispatch(x, weights, ids, counts)
    _next_plan_id += 1
    _plans[_next_plan_id] = _Plan(plan)
    return hidden, route_weights, cu_seqlens, torch.tensor([_next_plan_id])


def _dispatch_setup_context(ctx, inputs, output) -> None:
    ctx.entry = _plans.get(int(output[3]))


def _dispatch_backward(ctx, grad_hidden, grad_route_weights, *_):
    from torchtitan.distributed.moonep.moonep import current_buffer

    buffer, _ = current_buffer()
    grad_x, grad_weights, _ = buffer.combine(
        plan=ctx.entry.plan,
        hidden_nvsh=grad_hidden.to(torch.bfloat16).contiguous(),
        route_weights_nvs=grad_route_weights.float().contiguous(),
    )
    return grad_x, grad_weights, None, None


@torch.library.custom_op("moonep::experts", mutates_args=())
def _experts(
    x: Tensor, w13: Tensor, w2: Tensor, cu_seqlens: Tensor, plan_id: Tensor
) -> Tensor:
    from torchtitan.distributed.moonep.moonep import current_buffer, prefetch_rows

    buffer, group = current_buffer()
    entry = _plans[int(plan_id)]
    compute = entry.compute
    assert compute is not None
    rows = prefetch_rows(buffer, entry.plan, group, w13, w2)
    with torch.no_grad():
        return compute(x, rows, cu_seqlens)


def _experts_setup_context(ctx, inputs, output) -> None:
    x, w13, w2, cu_seqlens, plan_id = inputs
    ctx.entry = _plans.get(int(plan_id))
    ctx.save_for_backward(x, w13, w2, cu_seqlens)


def _experts_backward(ctx, grad_out):
    from torchtitan.distributed.moonep.moonep import (
        current_buffer,
        prefetch_rows,
        reduce_rows,
    )

    buffer, group = current_buffer()
    x, w13, w2, cu_seqlens = ctx.saved_tensors
    plan, compute = ctx.entry.plan, ctx.entry.compute
    # The pools are shared by every layer: refill them for this plan and recompute.
    rows = {
        name: row.detach().requires_grad_()
        for name, row in prefetch_rows(buffer, plan, group, w13, w2).items()
    }
    x_leaf = x.detach().requires_grad_()
    with torch.enable_grad():
        out = compute(x_leaf, rows, cu_seqlens)
    inputs: list[Tensor] = [x_leaf, *rows.values()]
    grad_x, *row_grads = torch.autograd.grad(out, inputs, grad_out)
    grad_w13, grad_w2 = reduce_rows(buffer, plan, group, dict(zip(rows, row_grads)))
    return grad_x, grad_w13.to(w13.dtype), grad_w2.to(w2.dtype), None, None


@torch.library.custom_op("moonep::combine", mutates_args=())
def _combine(x: Tensor, plan_id: Tensor, will_backward: bool) -> Tensor:
    from torchtitan.distributed.moonep.moonep import current_buffer

    buffer, _ = current_buffer()
    key = int(plan_id)
    entry = _plans[key] if will_backward else _plans.pop(key)
    out, _, _ = buffer.combine(plan=entry.plan, hidden_nvsh=x)
    return out


def _combine_setup_context(ctx, inputs, output) -> None:
    ctx.entry = _plans.pop(int(inputs[1]), None)


def _combine_backward(ctx, grad_out):
    from torchtitan.distributed.moonep.moonep import current_buffer

    buffer, _ = current_buffer()
    grad_x, _, _, _ = buffer.dispatch(
        grad_out.to(torch.bfloat16).contiguous(), plan=ctx.entry.plan
    )
    return grad_x, None, None


for _op, _backward, _setup_context in (
    (_dispatch, _dispatch_backward, _dispatch_setup_context),
    (_experts, _experts_backward, _experts_setup_context),
    (_combine, _combine_backward, _combine_setup_context),
):
    _op.register_autograd(_backward, setup_context=_setup_context)
    # A recompute must not replay these ops: they share the plan table and the process-global pools.
    _op.register_effect(torch.library.EffectType.ORDERED)


def dispatch_tokens(
    x_SH: Tensor, weights_SK: Tensor, ids_SK: Tensor, counts_E: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Route tokens to expert rows; returns the rows, their weights, ``cu_seqlens`` and the plan id."""
    outputs = remat.region(_dispatch, "moonep_dispatch", recompute=False)(
        x_SH.to(torch.bfloat16),
        weights_SK.float(),
        ids_SK.to(torch.int32),
        counts_E.to(torch.int32),
    )
    remat.recompute_needs_tensor(*outputs)
    return outputs


def routed_experts(
    compute: Callable[..., Tensor],
    x_RD: Tensor,
    w13_e2FD: Tensor,
    w2_eDF: Tensor,
    cu_seqlens: Tensor,
    plan_id: Tensor,
) -> Tensor:
    """Run ``compute`` over this rank's expert rows and the copies prefetched for the plan."""
    entry = _plans.get(int(plan_id))
    # A recompute serves the saved output, and combine has removed the plan by then.
    if entry is not None:
        entry.compute = compute
    out_RD = remat.region(_experts, "moonep_experts", recompute=False)(
        x_RD, w13_e2FD, w2_eDF, cu_seqlens, plan_id
    )
    remat.recompute_needs_tensor(out_RD)
    return out_RD


def combine_tokens(hidden_NH: Tensor, plan_id: Tensor) -> Tensor:
    """Sum each token's expert rows; the backward is a dispatch on the same plan."""
    will_backward = torch.is_grad_enabled() and hidden_NH.requires_grad
    out_SH = remat.region(_combine, "moonep_combine", recompute=False)(
        hidden_NH.to(torch.bfloat16).contiguous(), plan_id, will_backward
    )
    remat.recompute_needs_tensor(out_SH)
    return out_SH
