# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline stage carrying the block attention residual across hops.

Suffixes: K blocks on the wire, N blocks in a stack, T tokens, D model dim.
The wire format of a hop's blocks is [K, T, D], one contiguous row per block.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining._utils import flatten_args
from torch.distributed.pipelining.stage import _make_tensor_from_meta

from .cache import PPRankLocalCache
from .layout import BlockLayoutTables


def _assemble_stack(
    hidden_TD: torch.Tensor,
    delta_KTD: torch.Tensor,
    delta_blocks: list[int],
    store_blocks: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, list[int]]:
    """The block stack as a fresh autograd leaf, with each column's block index."""
    if delta_KTD.shape[0] != len(delta_blocks):
        raise ValueError(
            f"received {delta_KTD.shape[0]} block(s) but the routing expects "
            f"{delta_blocks}"
        )
    order = sorted(set(delta_blocks) | set(store_blocks))
    pieces = [
        store_blocks[b] if b in store_blocks else delta_KTD[delta_blocks.index(b)]
        for b in order
    ]
    if pieces:
        stack_TND = torch.stack(pieces, dim=1)
    else:
        stack_TND = hidden_TD.new_zeros(hidden_TD.shape[0], 0, hidden_TD.shape[-1])
    return stack_TND.detach().requires_grad_(True), order


def _outgoing_delta(
    stack_out_TND: torch.Tensor,
    order_out: list[int],
    out_blocks: list[int],
) -> torch.Tensor:
    """The blocks the next hop carries: a [K, T, D] view of the stack's tail columns."""
    if stack_out_TND.shape[1] != len(order_out):
        raise ValueError(
            f"the model returned {stack_out_TND.shape[1]} block(s); the routing "
            f"expects {len(order_out)} ({order_out})"
        )
    first = len(order_out) - len(out_blocks)
    if order_out[first:] != out_blocks:
        raise ValueError(
            f"the outgoing blocks {out_blocks} are not the tail of the stack {order_out}"
        )
    return stack_out_TND[:, first:].transpose(0, 1)


def _split_stack_grad(
    grad_stack_TND: torch.Tensor | None,
    order: list[int],
    delta_blocks: list[int],
    like_TD: torch.Tensor,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Split a stack gradient into the received rows, in wire order, and the stored blocks' deposits."""
    grad_delta_KTD = like_TD.new_zeros(len(delta_blocks), *like_TD.shape)
    deposits: dict[int, torch.Tensor] = {}
    if grad_stack_TND is None:
        return grad_delta_KTD, deposits
    for col, b in enumerate(order):
        if b in delta_blocks:
            grad_delta_KTD[delta_blocks.index(b)] = grad_stack_TND[:, col]
        else:
            deposits[b] = grad_stack_TND[:, col]
    return grad_delta_KTD, deposits


class AttnResPipelineStage(PipelineStage):
    """``PipelineStage`` whose hops carry the block residual's delta."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._layout: BlockLayoutTables | None = None
        self._store: PPRankLocalCache | None = None
        self._rows_needed = 0
        # per micro-batch: the stack's block order and the blocks the delta carried in
        self._order: dict[int, list[int]] = {}
        self._delta_in: dict[int, list[int]] = {}

    def set_routing(self, layout: BlockLayoutTables, store: PPRankLocalCache) -> None:
        self._layout = layout
        self._store = store
        mine = [s for s, r in layout.stage_to_rank.items() if r == self.group_rank]
        self._rows_needed = max(
            len(layout.cache_at_entry(s))
            + (len(layout.delta_to_send(s - 1)) if s else 0)
            + len(layout.commits_at(s))
            for s in mine
        )

    def layout(self) -> BlockLayoutTables:
        if self._layout is None:
            raise RuntimeError(
                f"stage {self.stage_index}: set_routing() must run first"
            )
        return self._layout

    def store(self) -> PPRankLocalCache:
        if self._store is None:
            raise RuntimeError(
                f"stage {self.stage_index}: set_routing() must run first"
            )
        return self._store

    def _in_place(self) -> bool:
        # Blocks are received into the rank store only when each reaches a rank once.
        return self._layout is not None and self._layout.cache

    def _is_first_on_rank(self) -> bool:
        layout = self.layout()
        mine = [s for s, r in layout.stage_to_rank.items() if r == self.group_rank]
        return self.stage_index == min(mine)

    def _is_last_on_rank(self) -> bool:
        layout = self.layout()
        mine = [s for s, r in layout.stage_to_rank.items() if r == self.group_rank]
        return self.stage_index == max(mine)

    def _placeholder(self, like: torch.Tensor | None) -> torch.Tensor:
        dtype = like.dtype if like is not None else torch.bfloat16
        return torch.empty(0, device=self.device, dtype=dtype)

    # Receive buffers: the delta lands in the rank store, the payload gradient in a buffer
    # allocated when its receive is posted; neither is held per micro-batch across steps.

    def _setup_forward_recv_info(self, num_microbatches: int, has_backward: bool) -> None:
        super()._setup_forward_recv_info(num_microbatches, has_backward)
        if self.is_first or not self._in_place():
            return
        for chunk_id in range(num_microbatches):
            info = self.args_recv_info[chunk_id][1]
            info.buffer = self._placeholder(info.buffer)

    def _setup_backward_recv_info(self, num_microbatches: int) -> None:
        super()._setup_backward_recv_info(num_microbatches)
        if self.is_last:
            return
        for mb_index in range(num_microbatches):
            info = self.grad_recv_info[mb_index][1]
            if info.buffer is not None:
                info.buffer = self._placeholder(info.buffer)

    def get_fwd_recv_ops(self, fwd_chunk_id: int):
        if not self.is_first and self._in_place():
            layout, store = self.layout(), self.store()
            hidden_info, delta_info = self.args_recv_info[fwd_chunk_id]
            store.allocate(fwd_chunk_id, self._rows_needed, hidden_info.buffer)
            delta_blocks = layout.delta_to_send(self.stage_index - 1)
            first = delta_blocks[0] if delta_blocks else 0
            delta_info.buffer = store.rows(fwd_chunk_id, first, len(delta_blocks))
        return super().get_fwd_recv_ops(fwd_chunk_id)

    def get_bwd_recv_ops(self, bwd_chunk_id: int):
        if self.has_backward and not self.is_last:
            info = self.grad_recv_info[bwd_chunk_id][1]
            if info.buffer is not None and info.tensor_meta is not None:
                info.buffer = _make_tensor_from_meta(info.tensor_meta, self.device)
        return super().get_bwd_recv_ops(bwd_chunk_id)

    def _assemble(
        self, mb: int, hidden_TD: torch.Tensor, delta_KTD: torch.Tensor
    ) -> torch.Tensor:
        layout, store = self.layout(), self.store()
        delta_blocks = layout.delta_to_send(self.stage_index - 1)
        expected = layout.cache_at_entry(self.stage_index)
        held = store.blocks(mb)
        if set(held) != set(expected):
            raise RuntimeError(
                f"stage {self.stage_index} micro-batch {mb}: the store holds "
                f"blocks {sorted(held)} but the routing expects {sorted(expected)}"
            )
        if self._in_place():
            # The delta was received into its rows; the stack is a view of rows [0, N).
            store.mark(mb, delta_blocks)
            self.args_recv_info[mb][1].buffer = self._placeholder(delta_KTD)
            order = sorted(set(expected) | set(delta_blocks))
            if order != list(range(len(order))):
                raise RuntimeError(
                    f"stage {self.stage_index} micro-batch {mb}: the stack {order} "
                    "is not the leading blocks"
                )
            stack_TND = store.stack(mb, len(order)).detach().requires_grad_(True)
        else:
            stack_TND, order = _assemble_stack(hidden_TD, delta_KTD, delta_blocks, held)
        self._order[mb] = order
        self._delta_in[mb] = delta_blocks
        return stack_TND

    def _commit_and_route(
        self, mb: int, stack_out_TND: torch.Tensor, order_in: list[int]
    ) -> torch.Tensor:
        layout, store = self.layout(), self.store()
        my_commits = layout.commits_at(self.stage_index)
        order_out = order_in + my_commits
        if layout.cache and my_commits:
            store.allocate(mb, self._rows_needed, stack_out_TND[:, 0])
            for i, b in enumerate(my_commits):
                store.put(mb, b, stack_out_TND[:, len(order_in) + i].detach())
        return _outgoing_delta(
            stack_out_TND, order_out, layout.delta_to_send(self.stage_index)
        )

    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
        save_forward_output: bool = True,
    ):
        store = self.store()
        if self.is_first:
            composite_args: tuple[Any, ...] = args
            order_in: list[int] = []
        else:
            hidden_TD, delta_KTD = self._retrieve_recv_activations(fwd_chunk_id)
            stack_TND = self._assemble(fwd_chunk_id, hidden_TD, delta_KTD)
            composite_args = (hidden_TD, stack_TND)
            order_in = self._order[fwd_chunk_id]
        composite_kwargs = kwargs or {}
        output = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)
        if self.is_last:
            output_tuple = (
                (output,) if isinstance(output, torch.Tensor) else tuple(output)
            )
            if save_forward_output:
                self.output_chunks.append(output)
        else:
            hidden_out_TD, stack_out_TND = output
            payload_KTD = self._commit_and_route(fwd_chunk_id, stack_out_TND, order_in)
            output_tuple = (hidden_out_TD, payload_KTD)
        # flatten_args returns a list with detach=False; lists keep the checker on that overload.
        flatten_input_tensors: list[torch.Tensor] = list(
            flatten_args(composite_args)
        ) + list(flatten_args(composite_kwargs))
        self.fwd_cache[fwd_chunk_id] = (output_tuple, flatten_input_tensors)
        # Stacks are views of the store, so a training step keeps it until the rank's last backward.
        if self._is_last_on_rank() and not (self._in_place() and self.has_backward):
            store.release(fwd_chunk_id)
        return output

    def _retrieve_recv_grads(self, bwd_chunk_id: int):
        grads = super()._retrieve_recv_grads(bwd_chunk_id)
        if self.is_last:
            return grads
        info = self.grad_recv_info[bwd_chunk_id][1]
        if info.buffer is not None:
            info.buffer = self._placeholder(info.buffer)
        layout = self.layout()
        grad_hidden, grad_delta = grads
        mine = set(layout.commits_at(self.stage_index))
        out_blocks = layout.delta_to_send(self.stage_index)
        committed = [j for j, b in enumerate(out_blocks) if b in mine]
        if not committed:
            return (grad_hidden, grad_delta)
        if grad_delta is None:
            outputs_meta = self._stage_meta.outputs
            if outputs_meta is not None and not outputs_meta[1].requires_grad:
                # Nothing upstream of the payload's blocks is trainable (a frozen embedding under LoRA):
                # no gradient channel, nowhere for the deposits to go.
                for j in committed:
                    self._collect_into(None, bwd_chunk_id, out_blocks[j])
                return (grad_hidden, None)
            raise RuntimeError(
                f"stage {self.stage_index} micro-batch {bwd_chunk_id}: no gradient "
                f"arrived for the payload carrying its own blocks "
                f"{[out_blocks[j] for j in committed]}"
            )
        grad_delta = grad_delta.clone()
        for j in committed:
            self._collect_into(grad_delta[j], bwd_chunk_id, out_blocks[j])
        return (grad_hidden, grad_delta)

    def _collect_into(self, grad_row_TD: torch.Tensor | None, mb: int, b: int) -> None:
        # Collect block b's deposits, one per later stage on this rank holding b;
        # each such stage needs a stack gradient: a layer, the aggregation or a sent delta.
        layout, store = self.layout(), self.store()
        deposit, count = store.collect(mb, b)
        expected = layout.deposits_expected(b, self.stage_index)
        if count != expected:
            raise RuntimeError(
                f"stage {self.stage_index} micro-batch {mb} block {b}: "
                f"{count} gradient deposit(s) but {expected} expected; a "
                "later stage on this rank did not run its backward"
            )
        if deposit is not None and grad_row_TD is not None:
            grad_row_TD.add_(deposit)

    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        full_backward: bool = True,
        last_backward=False,
    ):
        super().backward_one_chunk(
            bwd_chunk_id,
            loss=loss,
            full_backward=full_backward,
            last_backward=last_backward,
        )
        if not self.has_backward:
            # Forward-only pass (schedule.eval): no backward ran; drop the forward's bookkeeping.
            self.fwd_cache.pop(bwd_chunk_id, None)
            self._order.pop(bwd_chunk_id, None)
            self._delta_in.pop(bwd_chunk_id, None)
            return
        store = self.store()
        if self.is_first:
            if self._is_first_on_rank():
                store.release(bwd_chunk_id)
            return
        grad_hidden, grad_stack = self.bwd_cache[bwd_chunk_id]
        order = self._order.pop(bwd_chunk_id)
        delta_blocks = self._delta_in.pop(bwd_chunk_id)
        like = grad_hidden
        if like is None and grad_stack is not None:
            like = grad_stack[:, 0]
        if like is None:
            raise RuntimeError(
                f"stage {self.stage_index}: backward produced no gradient for "
                "either input"
            )
        grad_delta, deposits = _split_stack_grad(grad_stack, order, delta_blocks, like)
        for b, grad_TD in deposits.items():
            store.deposit(bwd_chunk_id, b, grad_TD)
        for j, b in enumerate(delta_blocks):
            self._collect_into(grad_delta[j], bwd_chunk_id, b)
        # Whether the previous stage expects a delta gradient comes from the receive metadata:
        # the assembled stack is a detached leaf, so autograd cannot tell.
        inputs_meta = self._stage_meta.inputs
        if (
            inputs_meta is None
            or len(inputs_meta) != 2
            or inputs_meta[1] is None
            or len(inputs_meta[1].shape) != 3
        ):
            raise RuntimeError(
                f"stage {self.stage_index}: the receive metadata should describe "
                f"(hidden, delta) with a [K, T, D] delta; got {inputs_meta}"
            )
        delta_needs_grad = inputs_meta[1].requires_grad
        self.bwd_cache[bwd_chunk_id] = (
            grad_hidden.contiguous() if grad_hidden is not None else None,
            grad_delta if delta_needs_grad else None,
        )
        if self._is_first_on_rank():
            if store.has_deposits(bwd_chunk_id):
                raise RuntimeError(
                    f"stage {self.stage_index} micro-batch {bwd_chunk_id}: gradient "
                    "deposits left uncollected after the rank's last backward"
                )
            store.release(bwd_chunk_id)

    def _compute_outputs(
        self, *args: torch.Tensor, module: torch.nn.Module, **kwargs: Any
    ):
        layout = self.layout()
        if self.is_first:
            output = module(*args, **kwargs)
            order_in: list[int] = []
        else:
            hidden_TD, delta_KTD = args
            delta_blocks = layout.delta_to_send(self.stage_index - 1)
            held = {
                b: hidden_TD.new_zeros(hidden_TD.shape)
                for b in layout.cache_at_entry(self.stage_index)
            }
            stack_TND, order_in = _assemble_stack(
                hidden_TD, delta_KTD, delta_blocks, held
            )
            output = module(hidden_TD, stack_TND, **kwargs)
        if self.is_last:
            return output
        hidden_out_TD, stack_out_TND = output
        order_out = order_in + layout.commits_at(self.stage_index)
        payload_KTD = _outgoing_delta(
            stack_out_TND, order_out, layout.delta_to_send(self.stage_index)
        )
        return hidden_out_TD, payload_KTD

    def _compute_input_grads(self, outputs, all_fwd_inputs, grad_outputs=None):
        grads = super()._compute_input_grads(outputs, all_fwd_inputs, grad_outputs)
        return tuple(
            g.contiguous() if isinstance(g, torch.Tensor) else g for g in grads
        )
