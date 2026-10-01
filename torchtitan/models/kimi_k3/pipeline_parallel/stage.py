# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline stage carrying the block attention residual across hops.

Suffixes: T tokens, D model dim.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining._utils import flatten_args
from torch.distributed.pipelining.schedules import _batch_p2p

from .cache import PPRankLocalCache
from .layout import BlockLayoutTables


def _assemble_stack(
    delta_TD: list[torch.Tensor],
    delta_blocks: list[int],
    store_blocks: dict[int, torch.Tensor],
) -> tuple[list[torch.Tensor], list[int]]:
    """The block stack as one autograd leaf per block, on the block's own memory, with the block indices."""
    if len(delta_TD) != len(delta_blocks):
        raise ValueError(
            f"received {len(delta_TD)} block(s) but the routing expects "
            f"{delta_blocks}"
        )
    order = sorted(set(delta_blocks) | set(store_blocks))
    stack_TD = [
        store_blocks[b].detach().requires_grad_(True)
        if b in store_blocks
        else delta_TD[delta_blocks.index(b)]
        for b in order
    ]
    return stack_TD, order


def _pack_outgoing_delta(
    stack_out_TD: list[torch.Tensor],
    order_out: list[int],
    out_blocks: list[int],
) -> list[torch.Tensor]:
    """The blocks the next hop carries, the model's own tensors."""
    if len(stack_out_TD) != len(order_out):
        raise ValueError(
            f"the model returned {len(stack_out_TD)} block(s); the routing "
            f"expects {len(order_out)} ({order_out})"
        )
    return [stack_out_TD[order_out.index(b)] for b in out_blocks]


def _split_stack_grad(
    grad_stack_TD: list[torch.Tensor | None],
    order: list[int],
    delta_blocks: list[int],
    like_TD: torch.Tensor,
) -> tuple[list[torch.Tensor], dict[int, torch.Tensor]]:
    """Split the blocks' gradients into the received blocks', in wire order, and the stored blocks' deposits."""
    deposits: dict[int, torch.Tensor] = {}
    if all(grad_TD is None for grad_TD in grad_stack_TD):
        return [torch.zeros_like(like_TD) for _ in delta_blocks], deposits
    received: dict[int, torch.Tensor] = {}
    for col, b in enumerate(order):
        grad_TD = grad_stack_TD[col]
        if grad_TD is None:
            grad_TD = torch.zeros_like(like_TD)
        if b in delta_blocks:
            received[b] = grad_TD
        else:
            deposits[b] = grad_TD
    return [received[b] for b in delta_blocks], deposits


class AttnResPipelineStage(PipelineStage):
    """``PipelineStage`` whose hops carry the block residual's delta."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._layout: BlockLayoutTables | None = None
        self._store: PPRankLocalCache | None = None
        # per micro-batch: the stack's block order and the blocks the delta carried in
        self._order: dict[int, list[int]] = {}
        self._delta_in: dict[int, list[int]] = {}
        self._wait_sends_at_backward = False
        self._fwd_send_works: dict[int, list] = {}
        self._input_grads: tuple[torch.Tensor | None, ...] = ()

    def set_routing(
        self,
        layout: BlockLayoutTables,
        store: PPRankLocalCache,
        *,
        wait_sends_at_backward: bool = False,
    ) -> None:
        self._layout = layout
        self._store = store
        self._wait_sends_at_backward = wait_sends_at_backward

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

    def _is_first_on_rank(self) -> bool:
        layout = self.layout()
        mine = [s for s, r in layout.stage_to_rank.items() if r == self.group_rank]
        return self.stage_index == min(mine)

    def _is_last_on_rank(self) -> bool:
        layout = self.layout()
        mine = [s for s, r in layout.stage_to_rank.items() if r == self.group_rank]
        return self.stage_index == max(mine)

    def get_fwd_send_ops(self, fwd_chunk_id: int):
        ops = super().get_fwd_send_ops(fwd_chunk_id)
        if self._wait_sends_at_backward and self.has_backward and ops:
            # The receiver has used the tensors by the time this stage's backward of the micro-batch runs.
            self._fwd_send_works[fwd_chunk_id] = _batch_p2p(ops)
            return []
        return ops

    def _assemble(self, mb: int, delta_TD: list[torch.Tensor]) -> list[torch.Tensor]:
        layout, store = self.layout(), self.store()
        delta_blocks = layout.delta_to_send(self.stage_index - 1)
        expected = layout.cache_at_entry(self.stage_index)
        held = store.blocks(mb)
        if set(held) != set(expected):
            raise RuntimeError(
                f"stage {self.stage_index} micro-batch {mb}: the store holds "
                f"blocks {sorted(held)} but the routing expects {sorted(expected)}"
            )
        stack_TD, order = _assemble_stack(delta_TD, delta_blocks, held)
        if layout.cache:
            # Keep what arrived for the rank's later stages.
            for i, b in enumerate(delta_blocks):
                store.put(mb, b, delta_TD[i].detach())
        self._order[mb] = order
        self._delta_in[mb] = delta_blocks
        return stack_TD

    def _commit_and_route(
        self, mb: int, stack_out_TD: list[torch.Tensor], order_in: list[int]
    ) -> list[torch.Tensor]:
        layout, store = self.layout(), self.store()
        my_commits = layout.commits_at(self.stage_index)
        order_out = order_in + my_commits
        if layout.cache:
            for i, b in enumerate(my_commits):
                store.put(mb, b, stack_out_TD[len(order_in) + i].detach())
        return _pack_outgoing_delta(
            stack_out_TD, order_out, layout.delta_to_send(self.stage_index)
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
            hidden_TD, *delta_TD = self._retrieve_recv_activations(fwd_chunk_id)
            stack_TD = self._assemble(fwd_chunk_id, delta_TD)
            composite_args = (hidden_TD, stack_TD)
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
            hidden_out_TD, stack_out_TD = output
            payload_TD = self._commit_and_route(fwd_chunk_id, stack_out_TD, order_in)
            output_tuple = (hidden_out_TD, *payload_TD)
            output = output_tuple

        # flatten_args returns a list with detach=False; lists keep the checker on that overload.
        flatten_input_tensors: list[torch.Tensor] = list(
            flatten_args(composite_args)
        ) + list(flatten_args(composite_kwargs))
        self.fwd_cache[fwd_chunk_id] = (output_tuple, flatten_input_tensors)

        if self._is_last_on_rank():
            store.release(fwd_chunk_id)
        return output

    def _retrieve_recv_grads(self, bwd_chunk_id: int):
        grads = super()._retrieve_recv_grads(bwd_chunk_id)
        if self.is_last:
            return grads
        layout = self.layout()
        grad_hidden, *grad_delta = grads
        mine = set(layout.commits_at(self.stage_index))
        out_blocks = layout.delta_to_send(self.stage_index)
        committed = [j for j, b in enumerate(out_blocks) if b in mine]
        outputs_meta = self._stage_meta.outputs
        for j in committed:
            grad_TD = grad_delta[j]
            if grad_TD is None:
                if outputs_meta is not None and not outputs_meta[1 + j].requires_grad:
                    # Nothing upstream of the block is trainable (a frozen embedding under LoRA):
                    # no gradient channel, nowhere for the deposits to go.
                    self._collect_into(None, bwd_chunk_id, out_blocks[j])
                    continue
                raise RuntimeError(
                    f"stage {self.stage_index} micro-batch {bwd_chunk_id}: no gradient "
                    f"arrived for its own block {out_blocks[j]}"
                )
            grad_delta[j] = grad_TD.clone()
            self._collect_into(grad_delta[j], bwd_chunk_id, out_blocks[j])
        return (grad_hidden, *grad_delta)

    def _collect_into(self, grad_col_TD: torch.Tensor | None, mb: int, b: int) -> None:
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
        if deposit is not None and grad_col_TD is not None:
            grad_col_TD.add_(deposit)

    def backward_maybe_with_nosync(
        self, backward_type, bwd_kwargs: dict, last_backward: bool = False
    ):
        grads, param_groups = super().backward_maybe_with_nosync(
            backward_type, bwd_kwargs, last_backward=last_backward
        )
        if backward_type in ("full", "input"):
            # Core keeps only the received inputs' gradients; the stored blocks' are taken here.
            self._input_grads = grads
        return grads, param_groups

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
        for work in self._fwd_send_works.pop(bwd_chunk_id, []):
            work.wait()
        grads, self._input_grads = self._input_grads, ()
        if self.is_first:
            return
        store = self.store()
        order = self._order.pop(bwd_chunk_id)
        delta_blocks = self._delta_in.pop(bwd_chunk_id)
        grad_hidden, *grad_stack_TD = grads[: 1 + len(order)]
        like = grad_hidden
        if like is None:
            like = next((g for g in grad_stack_TD if g is not None), None)
        if like is None:
            raise RuntimeError(
                f"stage {self.stage_index}: backward produced no gradient for "
                "any input"
            )
        grad_delta, deposits = _split_stack_grad(
            grad_stack_TD, order, delta_blocks, like
        )
        for b, grad_TD in deposits.items():
            store.deposit(bwd_chunk_id, b, grad_TD)
        for j, b in enumerate(delta_blocks):
            self._collect_into(grad_delta[j], bwd_chunk_id, b)
        inputs_meta = self._stage_meta.inputs
        if inputs_meta is None or len(inputs_meta) != 1 + len(delta_blocks):
            raise RuntimeError(
                f"stage {self.stage_index}: the receive metadata should describe "
                f"the hidden state and {len(delta_blocks)} block(s); got {inputs_meta}"
            )
        self.bwd_cache[bwd_chunk_id] = (
            grad_hidden.contiguous() if grad_hidden is not None else None,
            *(
                grad_TD.contiguous() if meta.requires_grad else None
                for grad_TD, meta in zip(grad_delta, inputs_meta[1:], strict=True)
            ),
        )
        if self._is_first_on_rank() and store.has_deposits(bwd_chunk_id):
            raise RuntimeError(
                f"stage {self.stage_index} micro-batch {bwd_chunk_id}: gradient "
                "deposits left uncollected after the rank's last backward"
            )

    def _compute_outputs(
        self, *args: torch.Tensor, module: torch.nn.Module, **kwargs: Any
    ):
        layout = self.layout()
        if self.is_first:
            output = module(*args, **kwargs)
            order_in: list[int] = []
        else:
            hidden_TD, *delta_TD = args
            delta_blocks = layout.delta_to_send(self.stage_index - 1)
            held = {
                b: hidden_TD.new_zeros(hidden_TD.shape)
                for b in layout.cache_at_entry(self.stage_index)
            }
            stack_TD, order_in = _assemble_stack(delta_TD, delta_blocks, held)
            output = module(hidden_TD, stack_TD, **kwargs)
        if self.is_last:
            return output
        hidden_out_TD, stack_out_TD = output
        order_out = order_in + layout.commits_at(self.stage_index)
        payload_TD = _pack_outgoing_delta(
            stack_out_TD, order_out, layout.delta_to_send(self.stage_index)
        )
        return (hidden_out_TD, *payload_TD)

    def _compute_input_grads(self, outputs, all_fwd_inputs, grad_outputs=None):
        grads = super()._compute_input_grads(outputs, all_fwd_inputs, grad_outputs)
        return tuple(
            g.contiguous() if isinstance(g, torch.Tensor) else g for g in grads
        )
