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
    activation: Callable[..., Tensor] | None = None


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


def _rows_mm(input_RI: Tensor, weight_EOI: Tensor, offsets: Tensor) -> Tensor:
    return torch._grouped_mm(input_RI, weight_EOI.transpose(-2, -1), offs=offsets)


@torch.library.custom_op("moonep::experts", mutates_args=())
def _experts(
    x: Tensor,
    weights: Tensor,
    w13: Tensor,
    w2: Tensor,
    cu_seqlens: Tensor,
    plan_id: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    from torchtitan.distributed.moonep.moonep import current_buffer, prefetch_rows

    buffer, group = current_buffer()
    entry = _plans[int(plan_id)]
    activation = entry.activation
    assert activation is not None
    rows = prefetch_rows(buffer, entry.plan, group, w13, w2)
    gate = _rows_mm(x, rows["gate"], cu_seqlens)
    up = _rows_mm(x, rows["up"], cu_seqlens)
    out = _rows_mm(activation(gate, up, offsets=cu_seqlens), rows["down"], cu_seqlens)
    return out.mul_(weights[:, None]), gate, up


def _experts_setup_context(ctx, inputs, output) -> None:
    x, weights, w13, w2, cu_seqlens, plan_id = inputs
    _, gate, up = output
    ctx.entry = _plans.get(int(plan_id))
    ctx.mark_non_differentiable(gate, up)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(x, weights, gate, up, w13, w2, cu_seqlens)


def _experts_backward(ctx, grad_out, *_):
    from torchtitan.distributed.moonep.moonep import (
        current_buffer,
        prefetch_rows,
        reduce_rows,
    )

    buffer, group = current_buffer()
    x, weights, gate, up, w13, w2, cu_seqlens = ctx.saved_tensors
    plan, activation = ctx.entry.plan, ctx.entry.activation
    # The pools are shared by every layer: refill them for this plan.
    rows = prefetch_rows(buffer, plan, group, w13, w2)
    gate_leaf = gate.detach().requires_grad_()
    up_leaf = up.detach().requires_grad_()
    with torch.enable_grad():
        hidden = activation(gate_leaf, up_leaf, offsets=cu_seqlens)
    # The routing-weight gradient <grad_out, out> equals <grad_out @ W_down, hidden>, so out is not kept.
    unscaled_grad_hidden = torch._grouped_mm(grad_out, rows["down"], offs=cu_seqlens)
    grad_weights = unscaled_grad_hidden.float().mul_(hidden.detach()).sum(-1)
    grad_hidden = unscaled_grad_hidden.mul_(weights[:, None])
    grad_down_out = torch.mul(
        grad_out, weights[:, None], out=torch.empty_like(grad_out)
    )
    grad_gate, grad_up = torch.autograd.grad(hidden, (gate_leaf, up_leaf), grad_hidden)
    row_grads = {
        "gate": torch._grouped_mm(grad_gate.t(), x, offs=cu_seqlens),
        "up": torch._grouped_mm(grad_up.t(), x, offs=cu_seqlens),
        "down": torch._grouped_mm(grad_down_out.t(), hidden.detach(), offs=cu_seqlens),
    }
    reduced, local = reduce_rows(buffer, plan, group, row_grads)
    grad_x = torch._grouped_mm(grad_gate, rows["gate"], offs=cu_seqlens)
    grad_x = grad_x + torch._grouped_mm(grad_up, rows["up"], offs=cu_seqlens)
    torch.cuda.current_stream().wait_event(reduced)
    grad_w13 = torch.stack([local["gate"], local["up"]], dim=1)
    return (
        grad_x,
        grad_weights,
        grad_w13.to(w13.dtype),
        local["down"].to(w2.dtype),
        None,
        None,
    )


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
    activation: Callable[..., Tensor],
    x_RD: Tensor,
    weights_R: Tensor,
    w13_e2FD: Tensor,
    w2_eDF: Tensor,
    cu_seqlens: Tensor,
    plan_id: Tensor,
) -> Tensor:
    """Run the gated experts over this rank's expert rows and the copies prefetched for the plan,
    scaled by each row's routing weight."""
    entry = _plans.get(int(plan_id))
    # A recompute serves the saved output, and combine has removed the plan by then.
    if entry is not None:
        entry.activation = activation
    out_RD, _, _ = remat.region(_experts, "moonep_experts", recompute=False)(
        x_RD, weights_R, w13_e2FD, w2_eDF, cu_seqlens, plan_id
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
