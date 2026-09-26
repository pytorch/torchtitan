# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline stage carrying the block attention residual across hops.

Suffixes: T tokens, D model dim.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining._utils import flatten_args
from torch.distributed.pipelining.schedules import _batch_p2p, _ComputationType
from torch.distributed.pipelining.stage import _make_tensor_from_meta

from torchtitan.distributed.activation_storage import ActivationStorage

from .cache import PPRankLocalCache
from .layout import BlockLayoutTables


def _outgoing_blocks(
    blocks_TD: list[torch.Tensor], order: list[int], out_blocks: list[int]
) -> list[torch.Tensor]:
    """The blocks the next hop carries, picked from the stage's output list."""
    if len(blocks_TD) != len(order):
        raise ValueError(
            f"the model returned {len(blocks_TD)} block(s); the routing expects "
            f"{len(order)} ({order})"
        )
    missing = [b for b in out_blocks if b not in order]
    if missing:
        raise ValueError(
            f"the outgoing blocks {missing} are not among the stage's blocks {order}"
        )
    return [blocks_TD[order.index(b)] for b in out_blocks]


def _grad_send_wait_points(
    order: dict[int, list[Any]], stage_to_rank: dict[int, int], rank: int
) -> dict[tuple[int, int], tuple[int, int]]:
    """Map each input-gradient send of ``rank`` to the first later forward on ``rank`` whose input the
    receiving rank produced after the backward that consumed the gradient."""
    backward = {_ComputationType.FULL_BACKWARD, _ComputationType.BACKWARD_INPUT}
    pos = {
        r: {
            (a.computation_type, a.stage_index, a.microbatch_index): i
            for i, a in enumerate(actions)
            if a is not None
        }
        for r, actions in order.items()
    }
    points: dict[tuple[int, int], tuple[int, int]] = {}
    for i, a in enumerate(order[rank]):
        if a is None or a.computation_type not in backward or a.stage_index == 0:
            continue
        s, mb = a.stage_index, a.microbatch_index
        receiver = stage_to_rank[s - 1]
        consumed = next(
            (
                pos[receiver][(kind, s - 1, mb)]
                for kind in backward
                if (kind, s - 1, mb) in pos[receiver]
            ),
            None,
        )
        if consumed is None:
            continue
        for b in order[receiver][consumed + 1 :]:
            if b is None or b.computation_type != _ComputationType.FORWARD:
                continue
            fed = b.stage_index + 1
            if stage_to_rank.get(fed) != rank:
                continue
            at = pos[rank].get((_ComputationType.FORWARD, fed, b.microbatch_index))
            if at is not None and at > i:
                points[(s, mb)] = (fed, b.microbatch_index)
                break
    return points


class _GradSendWaits:
    """Input-gradient sends a rank issues itself and the forwards that wait them, shared by its stages."""

    def __init__(self, points: dict[tuple[int, int], tuple[int, int]]) -> None:
        self.points = points
        self.pinned: dict[tuple[int, int], list] = {}

    def issue(self, stage: int, mb: int, ops: list) -> bool:
        point = self.points.get((stage, mb))
        if point is None or not ops:
            return False
        self.pinned.setdefault(point, []).extend(_batch_p2p(ops))
        return True

    def wait(self, stage: int, mb: int) -> None:
        for work in self.pinned.pop((stage, mb), []):
            work.wait()


class AttnResPipelineStage(PipelineStage):
    """``PipelineStage`` whose hops carry the block residual's delta, one tensor per block."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._layout: BlockLayoutTables | None = None
        self._store: PPRankLocalCache | None = None
        self._wait_sends_at_backward = False
        self._grad_send_waits: _GradSendWaits | None = None
        self._delta_in: dict[int, list[int]] = {}
        self._held_in: dict[int, list[int]] = {}
        self._fwd_send_works: dict[int, list] = {}
        self._input_grads: tuple[torch.Tensor | None, ...] = ()
        self._activations: ActivationStorage | None = None

    def set_routing(
        self,
        layout: BlockLayoutTables,
        store: PPRankLocalCache,
        *,
        wait_sends_at_backward: bool = False,
        grad_send_waits: _GradSendWaits | None = None,
    ) -> None:
        self._layout = layout
        self._store = store
        self._wait_sends_at_backward = wait_sends_at_backward
        self._grad_send_waits = grad_send_waits

    def set_activation_storage(self, storage: ActivationStorage) -> None:
        """Route the tensors this stage's forward saves through ``storage``."""
        self._activations = storage

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

    def _brought(self) -> list[int]:
        """The blocks this stage brings onto its rank: its received delta and its commits."""
        layout = self.layout()
        delta = layout.delta_to_send(self.stage_index - 1) if self.stage_index else []
        return delta + layout.commits_at(self.stage_index)

    def _placeholder(self, like: torch.Tensor | None) -> torch.Tensor:
        dtype = like.dtype if like is not None else torch.bfloat16
        return torch.empty(0, device=self.device, dtype=dtype)

    def _setup_forward_recv_info(
        self, num_microbatches: int, has_backward: bool
    ) -> None:
        super()._setup_forward_recv_info(num_microbatches, has_backward)
        if self.is_first:
            return
        for chunk_id in range(num_microbatches):
            for info in self.args_recv_info[chunk_id]:
                info.buffer = self._placeholder(info.buffer)

    def _setup_backward_recv_info(self, num_microbatches: int) -> None:
        super()._setup_backward_recv_info(num_microbatches)
        if self.is_last:
            return
        for mb_index in range(num_microbatches):
            for info in self.grad_recv_info[mb_index]:
                if info.buffer is not None:
                    info.buffer = self._placeholder(info.buffer)

    def get_fwd_recv_ops(self, fwd_chunk_id: int):
        if not self.is_first:
            for info in self.args_recv_info[fwd_chunk_id]:
                info.buffer = _make_tensor_from_meta(info.tensor_meta, self.device)
        return super().get_fwd_recv_ops(fwd_chunk_id)

    def get_fwd_send_ops(self, fwd_chunk_id: int):
        ops = super().get_fwd_send_ops(fwd_chunk_id)
        if self._wait_sends_at_backward and self.has_backward and ops:
            # The receiver has used the tensors once this stage's backward of the micro-batch starts.
            self._fwd_send_works[fwd_chunk_id] = _batch_p2p(ops)
            return []
        return ops

    def get_bwd_send_ops(self, bwd_chunk_id: int):
        ops = super().get_bwd_send_ops(bwd_chunk_id)
        # Pinned until waited: a forward that consumes what the receiver made afterwards proves it arrived.
        if self._grad_send_waits is not None and self._grad_send_waits.issue(
            self.stage_index, bwd_chunk_id, ops
        ):
            return []
        return ops

    def get_bwd_recv_ops(self, bwd_chunk_id: int):
        if self.has_backward and not self.is_last:
            for info in self.grad_recv_info[bwd_chunk_id]:
                if info.buffer is not None and info.tensor_meta is not None:
                    info.buffer = _make_tensor_from_meta(info.tensor_meta, self.device)
        return super().get_bwd_recv_ops(bwd_chunk_id)

    def _assemble(
        self, mb: int, delta_TD: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """The model's block list, in block order, and the stored blocks' leaves."""
        layout, store = self.layout(), self.store()
        delta_blocks = layout.delta_to_send(self.stage_index - 1)
        if len(delta_TD) != len(delta_blocks):
            raise ValueError(
                f"stage {self.stage_index}: received {len(delta_TD)} block(s) but "
                f"the routing expects {delta_blocks}"
            )
        expected = layout.cache_at_entry(self.stage_index)
        held = store.blocks(mb)
        if set(held) != set(expected):
            raise RuntimeError(
                f"stage {self.stage_index} micro-batch {mb}: the store holds "
                f"blocks {sorted(held)} but the routing expects {sorted(expected)}"
            )
        requires_grad = self.has_backward and torch.is_grad_enabled()
        held_blocks = sorted(held)
        held_TD = [held[b].detach().requires_grad_(requires_grad) for b in held_blocks]
        if layout.cache:
            for b, block_TD in zip(delta_blocks, delta_TD, strict=True):
                store.put(mb, b, block_TD.detach())
        leaves = dict(zip(held_blocks + delta_blocks, held_TD + delta_TD, strict=True))
        self._delta_in[mb] = delta_blocks
        self._held_in[mb] = held_blocks
        return [leaves[b] for b in sorted(leaves)], held_TD

    def _commit_and_route(
        self, mb: int, blocks_out_TD: list[torch.Tensor], order_in: list[int]
    ) -> list[torch.Tensor]:
        layout, store = self.layout(), self.store()
        commits = layout.commits_at(self.stage_index)
        order_out = order_in + commits
        payload_TD = _outgoing_blocks(
            blocks_out_TD, order_out, layout.delta_to_send(self.stage_index)
        )
        if layout.cache:
            new_TD = blocks_out_TD[len(order_in) :]
            for b, block_TD in zip(commits, new_TD, strict=True):
                store.put(mb, b, block_TD.detach())
        return payload_TD

    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
        save_forward_output: bool = True,
    ):
        if self._grad_send_waits is not None:
            self._grad_send_waits.wait(self.stage_index, fwd_chunk_id)
        composite_kwargs = kwargs or {}
        if self.is_first:
            composite_args: tuple[Any, ...] = args
            order_in: list[int] = []
            input_tensors = list(flatten_args(composite_args))
        else:
            hidden_TD, *delta_TD = self._retrieve_recv_activations(fwd_chunk_id)
            for info in self.args_recv_info[fwd_chunk_id]:
                info.buffer = self._placeholder(info.buffer)
            blocks_TD, held_TD = self._assemble(fwd_chunk_id, delta_TD)
            composite_args = (hidden_TD, blocks_TD)
            order_in = sorted(
                self._held_in[fwd_chunk_id] + self._delta_in[fwd_chunk_id]
            )
            # The received tensors lead, in wire order, where core reads their gradients.
            input_tensors = [hidden_TD, *delta_TD, *held_TD]
        # flatten_args returns a list with detach=False; lists keep the checker on that overload.
        kwarg_tensors: list[torch.Tensor] = list(flatten_args(composite_kwargs))
        saves = contextlib.nullcontext()
        if self._activations is not None and self.has_backward:
            saves = self._activations.forward(
                self.stage_index, fwd_chunk_id, keep=input_tensors + kwarg_tensors
            )
        with saves:
            output = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)
            if not self.is_last:
                hidden_out_TD, blocks_out_TD = output
                payload_TD = self._commit_and_route(
                    fwd_chunk_id, blocks_out_TD, order_in
                )
                output = (hidden_out_TD, *payload_TD)
        output_tuple = (output,) if isinstance(output, torch.Tensor) else tuple(output)
        if self.is_last and save_forward_output:
            self.output_chunks.append(output)
        self.fwd_cache[fwd_chunk_id] = (output_tuple, input_tensors + kwarg_tensors)
        if not self.has_backward and self._is_last_on_rank():
            self.store().release(fwd_chunk_id)
        return output

    def _retrieve_recv_grads(self, bwd_chunk_id: int):
        grads = super()._retrieve_recv_grads(bwd_chunk_id)
        if self.is_last:
            return grads
        for info in self.grad_recv_info[bwd_chunk_id]:
            if info.buffer is not None:
                info.buffer = self._placeholder(info.buffer)
        layout = self.layout()
        commits = set(layout.commits_at(self.stage_index))
        out_blocks = layout.delta_to_send(self.stage_index)
        _, *grad_blocks = grads
        outputs_meta = self._stage_meta.outputs
        for j, b in enumerate(out_blocks):
            if b not in commits:
                continue
            if grad_blocks[j] is None:
                if outputs_meta is not None and not outputs_meta[1 + j].requires_grad:
                    # Nothing upstream of the block is trainable (a frozen embedding under LoRA).
                    self._collect_into(None, bwd_chunk_id, b)
                    continue
                raise RuntimeError(
                    f"stage {self.stage_index} micro-batch {bwd_chunk_id}: no "
                    f"gradient arrived for block {b}, which this stage committed"
                )
            # The receive buffer was allocated for this micro-batch alone, so the deposits add in place.
            self._collect_into(grad_blocks[j], bwd_chunk_id, b)
        return grads

    def _collect_into(self, grad_TD: torch.Tensor | None, mb: int, b: int) -> None:
        # Collect block b's deposits, one per later stage on this rank holding b;
        # each such stage needs a block gradient: a layer, the aggregation or a sent delta.
        layout, store = self.layout(), self.store()
        deposit, count = store.collect(mb, b)
        expected = layout.deposits_expected(b, self.stage_index)
        if count != expected:
            raise RuntimeError(
                f"stage {self.stage_index} micro-batch {mb} block {b}: "
                f"{count} gradient deposit(s) but {expected} expected; a "
                "later stage on this rank did not run its backward"
            )
        if deposit is not None and grad_TD is not None:
            grad_TD.add_(deposit)

    def backward_maybe_with_nosync(
        self, backward_type, bwd_kwargs: dict, last_backward: bool = False
    ):
        grads, param_groups = super().backward_maybe_with_nosync(
            backward_type, bwd_kwargs, last_backward=last_backward
        )
        if backward_type in ("full", "input"):
            self._input_grads = grads
        return grads, param_groups

    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        full_backward: bool = True,
        last_backward=False,
    ):
        for work in self._fwd_send_works.pop(bwd_chunk_id, []):
            work.wait()
        self._input_grads = ()
        super().backward_one_chunk(
            bwd_chunk_id,
            loss=loss,
            full_backward=full_backward,
            last_backward=last_backward,
        )
        if self._activations is not None and full_backward:
            self._activations.finish(self.stage_index, bwd_chunk_id)
        if not self.has_backward:
            # Forward-only pass (schedule.eval): no backward ran; drop the forward's bookkeeping.
            self.fwd_cache.pop(bwd_chunk_id, None)
            self._delta_in.pop(bwd_chunk_id, None)
            self._held_in.pop(bwd_chunk_id, None)
            return
        store = self.store()
        if not self.is_first:
            self._route_input_grads(bwd_chunk_id)
        # The last reader of a block on this rank, in backward order, is the stage that brought it.
        store.release(bwd_chunk_id, self._brought())
        if self._is_first_on_rank():
            if store.has_deposits(bwd_chunk_id) or store.blocks(bwd_chunk_id):
                raise RuntimeError(
                    f"stage {self.stage_index} micro-batch {bwd_chunk_id}: blocks "
                    f"{sorted(store.blocks(bwd_chunk_id))} or their gradient deposits "
                    "outlived the rank's last backward"
                )

    def backward_weight_one_chunk(self, bwd_chunk_id: int, last_backward=False):
        super().backward_weight_one_chunk(bwd_chunk_id, last_backward=last_backward)
        # The weight half of a split backward reads the saves again.
        if self._activations is not None:
            self._activations.finish(self.stage_index, bwd_chunk_id)

    def _route_input_grads(self, mb: int) -> None:
        """Deposit the stored blocks' gradients and send the received ones back with their deposits."""
        store = self.store()
        delta_blocks = self._delta_in.pop(mb)
        held_blocks = self._held_in.pop(mb)
        num_delta = len(delta_blocks)
        grads, self._input_grads = self._input_grads, ()
        if len(grads) < 1 + num_delta + len(held_blocks):
            raise RuntimeError(
                f"stage {self.stage_index} micro-batch {mb}: the backward returned "
                f"{len(grads)} input gradient(s) for {1 + num_delta} received and "
                f"{len(held_blocks)} stored input(s)"
            )
        grad_hidden = grads[0]
        grad_delta = list(grads[1 : 1 + num_delta])
        for b, grad_TD in zip(held_blocks, grads[1 + num_delta :], strict=False):
            store.deposit(mb, b, grad_TD)
        like = (
            grad_hidden
            if grad_hidden is not None
            else next((g for g in grads[1:] if g is not None), None)
        )
        if like is None:
            raise RuntimeError(
                f"stage {self.stage_index}: backward produced no gradient for any input"
            )
        for j, b in enumerate(delta_blocks):
            if grad_delta[j] is None:
                grad_delta[j] = torch.zeros_like(like)
            self._collect_into(grad_delta[j], mb, b)
        inputs_meta = self._stage_meta.inputs
        if inputs_meta is None or len(inputs_meta) != 1 + num_delta:
            raise RuntimeError(
                f"stage {self.stage_index}: the receive metadata should describe the "
                f"hidden state and {num_delta} block(s); got {inputs_meta}"
            )
        self.bwd_cache[mb] = (
            grad_hidden.contiguous() if grad_hidden is not None else None,
            *(
                g.contiguous() if meta.requires_grad else None
                for g, meta in zip(grad_delta, inputs_meta[1:], strict=True)
            ),
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
            if len(delta_TD) != len(delta_blocks):
                raise ValueError(
                    f"stage {self.stage_index}: {len(delta_TD)} block input(s) but "
                    f"the routing expects {delta_blocks}"
                )
            source = {
                b: hidden_TD.new_zeros(hidden_TD.shape)
                for b in layout.cache_at_entry(self.stage_index)
            }
            source.update(zip(delta_blocks, delta_TD, strict=True))
            order_in = sorted(source)
            output = module(hidden_TD, [source[b] for b in order_in], **kwargs)
        if self.is_last:
            return output
        hidden_out_TD, blocks_out_TD = output
        order_out = order_in + layout.commits_at(self.stage_index)
        payload_TD = _outgoing_blocks(
            blocks_out_TD, order_out, layout.delta_to_send(self.stage_index)
        )
        return (hidden_out_TD, *payload_TD)

    def _compute_input_grads(self, outputs, all_fwd_inputs, grad_outputs=None):
        grads = super()._compute_input_grads(outputs, all_fwd_inputs, grad_outputs)
        return tuple(
            g.contiguous() if isinstance(g, torch.Tensor) else g for g in grads
        )
