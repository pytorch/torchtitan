# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Partial, Replicate
from torch.distributed.tensor.experimental import local_map

from torchtitan.components.loss import ChunkedCELoss, GradAccumulator


def _reduce_scatter_grad_like(local_grad: torch.Tensor, param: torch.Tensor):
    """Turn an unsharded fp32 local grad into ``param``'s sharded DTensor layout
    with a SINGLE reduce-scatter: the grad is ``Partial`` (sum) on the FSDP/DP
    mesh axes (reduced via Partial->Shard) and already ``Shard`` on the TP axis
    (kept as-is, no reduction). Plain (non-DTensor) params: returned unchanged.
    """
    if not isinstance(param, DTensor):
        return local_grad
    mesh = param.device_mesh
    mesh_axis_names = mesh.mesh_dim_names
    src_placements = [
        pl
        if (mesh_axis_names is not None and mesh_axis_names[i] == "tp")
        else Partial()
        for i, pl in enumerate(param.placements)
    ]
    return DTensor.from_local(local_grad, mesh, src_placements).redistribute(
        placements=param.placements
    )


class ChunkedCELossWithParamGrads(ChunkedCELoss):
    """ChunkedCELoss for graph_trainer that (1) exposes the lm_head parameter
    grads as explicit autograd outputs of the returned loss, and (2) coalesces
    the lm_head data-parallel collectives to a SINGLE all-gather + SINGLE
    reduce-scatter at the chunk-loop boundary.

    Why this overrides ``__call__`` instead of just ``_gradient_backprop``:

    graph_trainer shards lm_head with ``simple_fsdp`` (a DTensor parametrization),
    so ``lm_head`` is NOT an FSDP2 ``FSDPModule`` and the gradient-sync coalescing
    in the base ``ChunkedCELoss.__call__`` (``set_requires_gradient_sync``) is
    inert. With the base loop, accessing ``lm_head.weight`` inside each chunk
    fires ``ReplicateComputation`` once per chunk -> one all-gather AND one
    reduce-scatter per chunk, and the per-chunk sharded grads are summed in fp32.
    Eager FSDP2 instead all-gathers the weight once, accumulates the per-chunk
    grads unsharded in the compute dtype, and reduce-scatters once. That
    difference is a last-bit numerics mismatch isolated to ``lm_head.weight.grad``.

    This loop reproduces eager's semantics: it reads the all-gathered weight ONCE
    (via the simple_fsdp parametrization), runs the per-chunk forward+backward
    against a detached *leaf* copy so no collective fires inside the loop, and
    backprops the accumulated leaf grad through the parametrization ONCE to get a
    single reduce-scatter. Reusing the parametrization keeps the all-gather /
    reduce-scatter dtypes (``param_dtype`` / ``reduce_dtype``) and mesh handling
    bitwise-consistent with the rest of simple_fsdp.

    Also designed for graph_trainer's ``torch.autograd.grad(loss, params)``: the
    chunk loop's ``.grad`` side effects don't survive the captured graph, so the
    grads are returned as explicit autograd outputs via
    ``_ChunkedParamGradBridge``. Compatible with both outer ``loss.backward()``
    and ``torch.autograd.grad`` consumers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedCELoss.Config):
        pass

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: float | None = None,
    ) -> torch.Tensor:
        hidden_states = pred
        num_chunks = self.num_chunks
        lm_head = self.lm_head
        assert lm_head is not None, "Set lm_head before calling ChunkedCELoss"

        # SP: redistribute the TP axis of hidden states to Replicate before
        # chunking so lm_head receives Replicate input on TP (mirrors base).
        if isinstance(hidden_states, DTensor):
            mesh = hidden_states.device_mesh
            mesh_axis_names = mesh.mesh_dim_names
            if mesh_axis_names is not None and "tp" in mesh_axis_names:
                tp_axis = mesh_axis_names.index("tp")
                placements = list(hidden_states.placements)
                if not isinstance(placements[tp_axis], Replicate):
                    placements[tp_axis] = Replicate()
                    hidden_states = hidden_states.redistribute(mesh, tuple(placements))

        requires_grad = hidden_states.requires_grad

        # Chunk on the *local* view (see base ChunkedCELoss for the rationale).
        def _chunk_local(t: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return tuple(c.contiguous() for c in torch.chunk(t, num_chunks, dim=1))

        def _chunk(t: torch.Tensor) -> tuple[torch.Tensor, ...]:
            if not isinstance(t, DTensor):
                return _chunk_local(t)
            p = t.placements
            wrapped = local_map(
                _chunk_local,
                out_placements=(p,) * num_chunks,
                in_placements=(p,),
                device_mesh=t.device_mesh,
            )
            return wrapped(t)

        h_chunks = [
            c.detach().requires_grad_(requires_grad) for c in _chunk(hidden_states)
        ]
        label_chunks = list(_chunk(labels))

        # Read the lm_head weight (and bias) ONCE. For simple_fsdp this fires the
        # parametrization a single time (one all-gather, in param_dtype);
        # ``weight`` is the all-gathered tensor and stays connected to the sharded
        # raw param via the parametrization, so we can backprop through it once
        # below for a single reduce-scatter. For a plain nn.Linear (e.g. CPU
        # tests) ``weight is raw_weight`` and there is no collective.
        raw_weight = lm_head._parameters["weight"]
        raw_bias = lm_head._parameters.get("bias")
        weight = lm_head.weight
        bias = lm_head.bias if raw_bias is not None else None

        if not requires_grad:
            total_loss = hidden_states.new_zeros((), dtype=torch.float32)
            for h_chunk, label_chunk in zip(h_chunks, label_chunks):
                logits = F.linear(h_chunk, weight, bias)
                chunk_loss = self.fn(logits, label_chunk)
                if global_valid_tokens is not None:
                    chunk_loss = chunk_loss / global_valid_tokens
                total_loss = total_loss + chunk_loss.detach()
            return total_loss

        # Detached leaves: per-chunk backward computes grads against the once-
        # all-gathered weight with NO data-parallel collective inside the loop.
        w_leaf = weight.detach().requires_grad_(True)
        b_leaf = bias.detach().requires_grad_(True) if bias is not None else None

        grad_accumulator = GradAccumulator(
            hidden_states, num_chunks=num_chunks, dtype=torch.float32
        )
        total_loss = hidden_states.new_zeros((), dtype=torch.float32)

        # Accumulate the per-chunk lm_head param grads UNSHARDED in fp32 (eager
        # FSDP2 also accumulates the unsharded grad in fp32). Summing chunks here
        # and reduce-scattering once below reproduces eager's chunks-then-ranks
        # summation order -- unlike the default per-chunk reduce-scatter, which
        # sums ranks-then-chunks and drifts in the last bit. We accumulate on the
        # *local* view: with TP the all-gathered weight is still a (tp-sharded)
        # DTensor, so its grad is a DTensor; taking ``to_local`` keeps the buffer
        # a plain tensor that ``_reduce_scatter_grad_like`` re-wraps as Partial.
        def _to_local(t: torch.Tensor) -> torch.Tensor:
            return t.to_local() if isinstance(t, DTensor) else t

        w_grad_buf = torch.zeros_like(_to_local(w_leaf), dtype=torch.float32)
        b_grad_buf = (
            torch.zeros_like(_to_local(b_leaf), dtype=torch.float32)
            if b_leaf is not None
            else None
        )

        for h_chunk, label_chunk in zip(h_chunks, label_chunks):
            logits = F.linear(h_chunk, w_leaf, b_leaf)
            chunk_loss = self.fn(logits, label_chunk)
            if global_valid_tokens is not None:
                chunk_loss = chunk_loss / global_valid_tokens
            total_loss = total_loss + chunk_loss.detach()
            inputs = [h_chunk, w_leaf] + ([b_leaf] if b_leaf is not None else [])
            grads = torch.autograd.grad(chunk_loss, inputs)
            grad_accumulator.add(grads[0])
            w_grad_buf += _to_local(grads[1]).float()
            if b_leaf is not None:
                b_grad_buf += _to_local(grads[2]).float()

        accumulated_grad = grad_accumulator.result().to(hidden_states.dtype)

        # Single reduce-scatter per param: fp32 unsharded grad -> sharded layout.
        params: list[torch.Tensor] = [raw_weight]
        param_grads: list[torch.Tensor] = [
            _reduce_scatter_grad_like(w_grad_buf, raw_weight)
        ]
        if bias is not None:
            params.append(raw_bias)
            param_grads.append(_reduce_scatter_grad_like(b_grad_buf, raw_bias))

        return _ChunkedParamGradBridge.apply(
            hidden_states,
            accumulated_grad,
            total_loss,
            len(params),
            *params,
            *param_grads,
        )


class _ChunkedParamGradBridge(torch.autograd.Function):
    """Returns a detached loss whose backward emits the precomputed grads:
    ``accumulated_h_grad`` for ``hidden_states`` (propagating through the decoder)
    and the precomputed sharded param grads for the lm_head params -- so both
    ``loss.backward()`` and ``torch.autograd.grad(loss, [hidden, *lm_head.params])``
    consumers get correct grads without relying on ``.grad`` side effects.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        accumulated_h_grad: torch.Tensor,
        total_loss: torch.Tensor,
        num_params: int,
        *params_and_grads: torch.Tensor,
    ) -> torch.Tensor:
        param_grads = params_and_grads[num_params:]
        ctx.save_for_backward(accumulated_h_grad, *param_grads)
        ctx.num_params = num_params
        return total_loss.detach().clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # pyrefly: ignore[bad-override]
        saved = ctx.saved_tensors
        h_grad = saved[0]
        param_grads = saved[1:]
        # Grad for each forward input, in order:
        # (hidden_states, accumulated_h_grad, total_loss, num_params,
        #  *params, *param_grads)
        return (
            h_grad,
            None,
            None,
            None,
            *param_grads,
            *([None] * ctx.num_params),
        )
