# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A pipeline stage that carries the block attention residual.

A block committed at one stage is read by every later stage, which the
``PipelineStage`` protocol -- outputs to the next stage, gradients from it --
does not express. This subclass keeps that protocol on the wire and adds the
two things the residual needs:

* **Routing.** The hop between adjacent stages carries ``(hidden, delta)``,
  where ``delta`` holds only the blocks the receiving rank has not seen
  (:class:`BlockLayoutTables`). A rank keeps every block it committed or
  received in a store shared by its stages, and a stage assembles the full
  stack its model expects from that store plus the delta it received. The
  model itself takes and returns the full stack, so it knows nothing of this.

* **Gradients on a rank.** A stage's backward returns the gradient of the
  stack it assembled. The columns it received go back over the wire as the
  gradient of the delta, like any stage input. The columns it took from the
  store are deposited in the store, and the stage that brought that block onto
  the rank -- by committing it, or by receiving it -- adds the deposits to its
  own gradient for the block when its backward runs, which the schedule
  orders after every later stage's backward. The routing tables say how many
  deposits each block must have, so a lost gradient raises instead of
  training quietly.

Across ranks nothing new happens: the gradient of a received delta is sent
to the stage that produced it by the schedule's own backward P2P, and there it
is the gradient of that stage's payload, which autograd carries into the
model's graph or, for a relayed block, into that stage's own input gradient.

Everything a stage learns about the micro-batch comes with the chunk id the
schedule passes to ``forward_one_chunk`` and ``backward_one_chunk``; the
blocks of a micro-batch are released after the rank's last stage has run its
forward for it.

Tensor suffixes: ``T`` tokens, ``N`` blocks, ``D`` model dimension.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining._utils import flatten_args

from torchtitan.models.kimi_k3.layout import BlockLayoutTables


class RankStore:
    """The blocks a rank holds per micro-batch, and the gradient deposits."""

    def __init__(self) -> None:
        self._blocks: dict[int, dict[int, torch.Tensor]] = {}
        self._deposits: dict[tuple[int, int], torch.Tensor] = {}
        self._counts: dict[tuple[int, int], int] = {}

    # blocks
    def put(self, mb: int, block_idx: int, block_TD: torch.Tensor) -> None:
        self._blocks.setdefault(mb, {})[block_idx] = block_TD

    def blocks(self, mb: int) -> dict[int, torch.Tensor]:
        return self._blocks.get(mb, {})

    def release(self, mb: int) -> None:
        """Free the blocks of ``mb``; the deposits stay until collected."""
        self._blocks.pop(mb, None)

    # gradient deposits
    def deposit(self, mb: int, block_idx: int, grad_TD: torch.Tensor) -> None:
        key = (mb, block_idx)
        prior = self._deposits.get(key)
        self._deposits[key] = grad_TD.clone() if prior is None else prior + grad_TD
        self._counts[key] = self._counts.get(key, 0) + 1

    def collect(self, mb: int, block_idx: int) -> tuple[torch.Tensor | None, int]:
        key = (mb, block_idx)
        return self._deposits.pop(key, None), self._counts.pop(key, 0)

    def has_deposits(self, mb: int) -> bool:
        return any(key[0] == mb for key in self._deposits)


def assemble_stack(
    hidden_TD: torch.Tensor,
    delta_TND: torch.Tensor,
    delta_blocks: list[int],
    store_blocks: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, list[int]]:
    """The full block stack a stage's model expects, in block order.

    Returns the stack as a fresh autograd leaf and the block index of each
    column. A leaf, because the stage's backward reads ``.grad`` of its inputs
    and hands out the columns itself: the received ones over the wire, the
    stored ones as deposits.
    """
    if delta_TND.shape[1] != len(delta_blocks):
        raise ValueError(
            f"received {delta_TND.shape[1]} block(s) but the routing expects "
            f"{delta_blocks}"
        )
    order = sorted(set(delta_blocks) | set(store_blocks))
    pieces = [
        store_blocks[b] if b in store_blocks else delta_TND[:, delta_blocks.index(b)]
        for b in order
    ]
    if pieces:
        stack_TND = torch.stack(pieces, dim=1)
    else:
        stack_TND = hidden_TD.new_zeros(hidden_TD.shape[0], 0, hidden_TD.shape[-1])
    return stack_TND.detach().requires_grad_(True), order


def route_payload(
    stack_out_TND: torch.Tensor,
    order_out: list[int],
    out_blocks: list[int],
) -> torch.Tensor:
    """The blocks the next hop carries, as views of the model's stack."""
    if stack_out_TND.shape[1] != len(order_out):
        raise ValueError(
            f"the model returned {stack_out_TND.shape[1]} block(s); the routing "
            f"expects {len(order_out)} ({order_out})"
        )
    pieces = [stack_out_TND[:, order_out.index(b)] for b in out_blocks]
    if pieces:
        return torch.stack(pieces, dim=1)
    num_tokens, _, dim = stack_out_TND.shape
    return stack_out_TND.new_zeros(num_tokens, 0, dim)


def split_stack_grad(
    grad_stack_TND: torch.Tensor | None,
    order: list[int],
    delta_blocks: list[int],
    like_TD: torch.Tensor,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Split the gradient of an assembled stack into the received part, dense
    and in wire order, and the per-block deposits for the stored part."""
    num_tokens, dim = like_TD.shape[0], like_TD.shape[-1]
    grad_delta = like_TD.new_zeros(num_tokens, len(delta_blocks), dim)
    deposits: dict[int, torch.Tensor] = {}
    if grad_stack_TND is None:
        return grad_delta, deposits
    for col, b in enumerate(order):
        if b in delta_blocks:
            grad_delta[:, delta_blocks.index(b)] = grad_stack_TND[:, col]
        else:
            deposits[b] = grad_stack_TND[:, col]
    return grad_delta, deposits


class AttnResPipelineStage(PipelineStage):
    """``PipelineStage`` whose hops carry the block residual's delta."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._layout: BlockLayoutTables | None = None
        self._store: RankStore | None = None
        # Per micro-batch, the block order of the assembled stack and the
        # blocks the delta carried in, for the backward split.
        self._order: dict[int, list[int]] = {}
        self._delta_in: dict[int, list[int]] = {}

    def set_routing(self, layout: BlockLayoutTables, store: RankStore) -> None:
        """Install the routing tables and the rank's store; done once the
        schedule exists, since the tables need the stage-to-rank map."""
        self._layout = layout
        self._store = store

    # ----- routing helpers -------------------------------------------- #
    def _routing(self) -> tuple[BlockLayoutTables, RankStore]:
        if self._layout is None or self._store is None:
            raise RuntimeError(
                f"stage {self.stage_index}: set_routing() must run before the "
                "first step"
            )
        return self._layout, self._store

    def _is_first_on_rank(self) -> bool:
        layout, _ = self._routing()
        mine = [s for s, r in layout.stage_to_rank.items() if r == self.group_rank]
        return self.stage_index == min(mine)

    def _is_last_on_rank(self) -> bool:
        layout, _ = self._routing()
        mine = [s for s, r in layout.stage_to_rank.items() if r == self.group_rank]
        return self.stage_index == max(mine)

    def _assemble(
        self, mb: int, hidden_TD: torch.Tensor, delta_TND: torch.Tensor
    ) -> torch.Tensor:
        layout, store = self._routing()
        delta_blocks = layout.delta_to_send(self.stage_index - 1)
        expected = layout.cache_at_entry(self.stage_index)
        held = store.blocks(mb)
        if set(held) != set(expected):
            raise RuntimeError(
                f"stage {self.stage_index} micro-batch {mb}: the store holds "
                f"blocks {sorted(held)} but the routing expects {sorted(expected)}"
            )
        stack_TND, order = assemble_stack(hidden_TD, delta_TND, delta_blocks, held)
        if layout.cache:
            # Keep what arrived for the rank's later stages.
            for i, b in enumerate(delta_blocks):
                store.put(mb, b, delta_TND[:, i].detach())
        self._order[mb] = order
        self._delta_in[mb] = delta_blocks
        return stack_TND

    def _commit_and_route(
        self, mb: int, stack_out_TND: torch.Tensor, order_in: list[int]
    ) -> torch.Tensor:
        layout, store = self._routing()
        my_commits = layout.commits_at(self.stage_index)
        order_out = order_in + my_commits
        if layout.cache:
            for i, b in enumerate(my_commits):
                store.put(mb, b, stack_out_TND[:, len(order_in) + i].detach())
        return route_payload(
            stack_out_TND, order_out, layout.delta_to_send(self.stage_index)
        )

    # ----- forward ----------------------------------------------------- #
    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
        save_forward_output: bool = True,
    ):
        """``_PipelineStageBase.forward_one_chunk`` with the stack assembled
        on the way in and the delta routed on the way out."""
        layout, store = self._routing()
        if self.is_first:
            composite_args: tuple[Any, ...] = args
            order_in: list[int] = []
        else:
            hidden_TD, delta_TND = self._retrieve_recv_activations(fwd_chunk_id)
            stack_TND = self._assemble(fwd_chunk_id, hidden_TD, delta_TND)
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
            payload_TND = self._commit_and_route(fwd_chunk_id, stack_out_TND, order_in)
            output_tuple = (hidden_out_TD, payload_TND)

        flatten_input_tensors = flatten_args(composite_args) + flatten_args(
            composite_kwargs
        )
        self.fwd_cache[fwd_chunk_id] = (output_tuple, flatten_input_tensors)

        if self._is_last_on_rank():
            # No later stage on this rank reads the micro-batch's blocks.
            store.release(fwd_chunk_id)
        return output

    # ----- backward ---------------------------------------------------- #
    def _retrieve_recv_grads(self, bwd_chunk_id: int):
        """The gradient of the payload, plus the deposits for the blocks this
        stage committed, which later stages on the rank read from the store."""
        grads = super()._retrieve_recv_grads(bwd_chunk_id)
        if self.is_last:
            return grads
        layout, store = self._routing()
        grad_hidden, grad_delta = grads
        mine = set(layout.commits_at(self.stage_index))
        out_blocks = layout.delta_to_send(self.stage_index)
        committed = [j for j, b in enumerate(out_blocks) if b in mine]
        if not committed:
            return (grad_hidden, grad_delta)
        if grad_delta is None:
            raise RuntimeError(
                f"stage {self.stage_index} micro-batch {bwd_chunk_id}: no gradient "
                f"arrived for the payload carrying its own blocks "
                f"{[out_blocks[j] for j in committed]}"
            )
        grad_delta = grad_delta.clone()
        for j in committed:
            self._collect_into(grad_delta[:, j], bwd_chunk_id, out_blocks[j])
        return (grad_hidden, grad_delta)

    def _collect_into(self, grad_col_TD: torch.Tensor, mb: int, b: int) -> None:
        layout, store = self._routing()
        deposit, count = store.collect(mb, b)
        expected = layout.deposits_expected(b, self.stage_index)
        if count != expected:
            raise RuntimeError(
                f"stage {self.stage_index} micro-batch {mb} block {b}: "
                f"{count} gradient deposit(s) but {expected} expected; a "
                "later stage on this rank did not run its backward"
            )
        if deposit is not None:
            grad_col_TD.add_(deposit)

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
        if self.is_first:
            return
        layout, store = self._routing()
        grad_hidden, grad_stack = self.bwd_cache[bwd_chunk_id]
        order = self._order.pop(bwd_chunk_id)
        delta_blocks = self._delta_in.pop(bwd_chunk_id)
        like = grad_hidden if grad_hidden is not None else grad_stack
        if like is None:
            raise RuntimeError(
                f"stage {self.stage_index}: backward produced no gradient for "
                "either input"
            )
        grad_delta, deposits = split_stack_grad(grad_stack, order, delta_blocks, like)
        for b, grad_TD in deposits.items():
            store.deposit(bwd_chunk_id, b, grad_TD)
        # This stage brought the received blocks onto the rank: collect what
        # the rank's later stages deposited for them.
        for j, b in enumerate(delta_blocks):
            self._collect_into(grad_delta[:, j], bwd_chunk_id, b)
        # A delta that needs no gradient has no receive buffer on the previous
        # stage: its gradient is None, like any such stage input.
        inputs_meta = self._stage_meta.inputs
        delta_needs_grad = inputs_meta is not None and inputs_meta[1].requires_grad
        self.bwd_cache[bwd_chunk_id] = (
            grad_hidden.contiguous() if grad_hidden is not None else None,
            grad_delta if delta_needs_grad else None,
        )
        if self._is_first_on_rank() and store.has_deposits(bwd_chunk_id):
            raise RuntimeError(
                f"stage {self.stage_index} micro-batch {bwd_chunk_id}: gradient "
                "deposits left uncollected after the rank's last backward"
            )

    # ----- metadata inference ------------------------------------------ #
    def _compute_outputs(
        self, *args: torch.Tensor, module: torch.nn.Module, **kwargs: Any
    ):
        """Run the module on the placeholders the way ``forward_one_chunk``
        would, so the recorded output metadata is the payload's."""
        layout, _ = self._routing()
        if self.is_first:
            output = module(*args, **kwargs)
            order_in: list[int] = []
        else:
            hidden_TD, delta_TND = args
            delta_blocks = layout.delta_to_send(self.stage_index - 1)
            held = {
                b: hidden_TD.new_zeros(hidden_TD.shape)
                for b in layout.cache_at_entry(self.stage_index)
            }
            stack_TND, order_in = assemble_stack(
                hidden_TD, delta_TND, delta_blocks, held
            )
            output = module(hidden_TD, stack_TND, **kwargs)
        if self.is_last:
            return output
        hidden_out_TD, stack_out_TND = output
        order_out = order_in + layout.commits_at(self.stage_index)
        payload_TND = route_payload(
            stack_out_TND, order_out, layout.delta_to_send(self.stage_index)
        )
        return hidden_out_TD, payload_TND

    def _compute_input_grads(self, outputs, all_fwd_inputs, grad_outputs=None):
        """Dense gradient metadata: the receive buffers of the previous stage
        are sized from these strides, and c10d refuses a buffer that is not."""
        grads = super()._compute_input_grads(outputs, all_fwd_inputs, grad_outputs)
        return tuple(
            g.contiguous() if isinstance(g, torch.Tensor) else g for g in grads
        )
