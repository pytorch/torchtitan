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


def _redistribute_grad_like_param(
    local_grad: torch.Tensor, param: torch.Tensor
) -> torch.Tensor:
    """Redistribute an accumulated local grad to ``param``'s DTensor layout.

    Data-parallel axes start as ``Partial`` so redistribution performs the
    needed reduction; TP sharding is preserved because it is already local.
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
    """ChunkedCELoss variant for graph_trainer.

    The parent implementation can override only FSDP2 gradient sync. Graph
    trainer uses simple_fsdp, so this training path materializes lm_head once,
    accumulates local per-chunk parameter grads, then redistributes them once
    as explicit autograd outputs for graph capture.
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

        # Accessing lm_head.weight fires the simple_fsdp all-gather once.
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

        # Detached leaves keep lm_head collectives out of the chunk loop.
        w_leaf = weight.detach().requires_grad_(True)
        b_leaf = bias.detach().requires_grad_(True) if bias is not None else None

        grad_accumulator = GradAccumulator(
            hidden_states, num_chunks=num_chunks, dtype=torch.float32
        )
        total_loss = hidden_states.new_zeros((), dtype=torch.float32)

        # Accumulate local fp32 lm_head grads across chunks, then redistribute
        # once to the raw parameter layout. This matches eager's chunk summation
        # order and avoids per-chunk data-parallel reductions.
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

        params: list[torch.Tensor] = [raw_weight]
        param_grads: list[torch.Tensor] = [
            _redistribute_grad_like_param(w_grad_buf, raw_weight)
        ]
        if bias is not None:
            params.append(raw_bias)
            param_grads.append(_redistribute_grad_like_param(b_grad_buf, raw_bias))

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
