# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The rank side of the decoupled encoder process, shared by the rank's stages."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.distributed.spmd_types import spmd_local_context

from ...vision_encoder import KimiK3VisionEncoder
from .plan import Anchor, Hook, START, STEP_END, VisionDepPlan

logger = logging.getLogger(__name__)


def _tp_dim(local: torch.Size, full: torch.Size) -> int | None:
    dims = [d for d, (a, b) in enumerate(zip(local, full, strict=True)) if a != b]
    if len(dims) > 1:
        raise ValueError(
            f"A local shape {tuple(local)} of {tuple(full)} splits {dims}."
        )
    return dims[0] if dims else None


class VisionDep:
    """One rank's part of the decoupled encoder process."""

    def __init__(
        self,
        replica: KimiK3VisionEncoder,
        *,
        tower: torch.nn.Module | None,
        pp_ranks: list[int],
        stage0_rank: int,
        group: dist.ProcessGroup,
        dp_group: dist.ProcessGroup | None,
        tp_group: dist.ProcessGroup | None,
        hidden_dim: int,
        compute_dtype: torch.dtype,
        pipeline_order: dict[int, list[Any]] | None,
        cost_ratio: float,
    ) -> None:
        self._replica = replica
        self._tower = tower
        self._pp_ranks = pp_ranks
        self._me = pp_ranks.index(dist.get_rank())
        self._stage0 = stage0_rank
        self._group = group
        self._dp_group = dp_group
        self._tp_group = tp_group
        self._tp = 1 if tp_group is None else dist.get_world_size(tp_group)
        self._tp_rank = 0 if tp_group is None else dist.get_rank(tp_group)
        self._hidden_dim = hidden_dim
        self._dtype = compute_dtype
        self._pipeline_order = pipeline_order
        self._cost_ratio = cost_ratio
        named = list(replica.named_parameters())
        self._names = [canonical_fqn(name) for name, _ in named]
        self._params = [param for _, param in named]
        self._numel = sum(param.numel() for param in self._params)
        # pyrefly: ignore [read-only]
        self._device = self._params[0].device
        self._kh, self._kw = replica.merge_kernel_size
        self._synced = False
        self._trainable = False
        self._accum: torch.Tensor | None = None
        self.active = False
        self.plan: VisionDepPlan | None = None
        self._reset()

    def _reset(self) -> None:
        self._inputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._tokens: dict[int, int] = {}
        self._pending: dict[Anchor, list[tuple[str, int]]] = {}
        self._posts: dict[Hook, list[tuple[str, int, int]]] = {}
        self._after_send: set[Anchor] = set()
        self._features: dict[int, torch.Tensor] = {}
        self._feature_recvs: dict[int, dist.Work] = {}
        self._gradients: dict[int, torch.Tensor] = {}
        self._gradient_recvs: dict[int, dist.Work] = {}
        self._outgoing: dict[int, torch.Tensor] = {}
        self._captured: dict[int, torch.Tensor] = {}
        self._sends: list[tuple[list[dist.Work], Any]] = []

    def begin_step(self, kwarg_mbs: list[dict[str, Any]]) -> None:
        self._reset()
        loads: dict[int, int] = {}
        for mb, kwargs in enumerate(kwarg_mbs):
            kwargs = kwargs or {}
            pixel_values = kwargs.get("pixel_values")
            grid_thw = kwargs.get("grid_thw")
            if pixel_values is None or grid_thw is None:
                continue
            grids = grid_thw.tolist()
            loads[mb] = sum(t * h * w for t, h, w in grids)
            self._tokens[mb] = sum(
                (h // self._kh) * (w // self._kw) for _, h, w in grids
            )
            self._inputs[mb] = (pixel_values, grid_thw)
        self._sync_weights()
        plan = VisionDepPlan(
            loads,
            num_microbatches=len(kwarg_mbs),
            num_ranks=len(self._pp_ranks),
            stage0_rank=self._stage0,
            trainable=self._trainable,
            pipeline_order=self._pipeline_order,
            cost_ratio=self._cost_ratio,
        )
        if self.plan is None:
            placed = [kind for kind, _ in plan.placed]
            logger.info(
                "vision_dep: %d micro-batch(es) with images; encodes %d before the "
                "schedule, %d in idle slots; backwards %d in idle slots, %d after it",
                len(loads),
                sum(len(mbs) for mbs in plan.prologue.values()),
                placed.count("encode"),
                placed.count("backward"),
                sum(len(mbs) for mbs in plan.epilogue.values()),
            )
        self.plan = plan
        self._pending = {
            anchor: list(work) for anchor, work in plan.anchored[self._me].items()
        }
        self._posts = {hook: list(ts) for hook, ts in plan.posts[self._me].items()}
        if self._accum is not None:
            self._accum.zero_()
        self.active = True
        for mb in plan.prologue[self._me]:
            self._encode(mb)
        self._after(START)

    def _sync_weights(self) -> None:
        if self._synced and not self._trainable:
            return
        flat = torch.empty(self._numel, dtype=self._dtype, device=self._device)
        flags = torch.zeros(len(self._params), dtype=torch.uint8, device=self._device)
        if self._me == self._stage0:
            assert self._tower is not None
            sources = {canonical_fqn(n): p for n, p in self._tower.named_parameters()}
            offset = 0
            for i, (name, param) in enumerate(zip(self._names, self._params)):
                if name not in sources:
                    raise RuntimeError(f"The vision tower on stage 0 has no {name}.")
                source = sources[name]
                full = (
                    source.full_tensor()
                    if isinstance(source, DTensor)
                    else source.detach()
                )
                dim = _tp_dim(param.shape, full.shape)
                if dim is not None:
                    full = full.chunk(self._tp, dim=dim)[self._tp_rank]
                flat[offset : offset + param.numel()].copy_(full.reshape(-1))
                flags[i] = int(source.requires_grad)
                offset += param.numel()
        src = self._pp_ranks[self._stage0]
        dist.broadcast(flat, src=src, group=self._group)
        with torch.no_grad():
            offset = 0
            for param in self._params:
                param.copy_(flat[offset : offset + param.numel()].view_as(param))
                offset += param.numel()
        if not self._synced:
            dist.broadcast(flags, src=src, group=self._group)
            for param, flag in zip(self._params, flags.tolist(), strict=True):
                param.requires_grad_(bool(flag))
            self._trainable = any(param.requires_grad for param in self._params)
            if self._trainable:
                self._accum = torch.zeros(
                    self._numel, dtype=torch.float32, device=self._device
                )
            self._synced = True

    def _new_buffer(self, mb: int) -> torch.Tensor:
        return torch.empty(
            self._tokens[mb], self._hidden_dim, dtype=self._dtype, device=self._device
        )

    def _post(self, hook: Hook) -> None:
        for kind, mb, peer in self._posts.pop(hook, ()):
            peer_rank = self._pp_ranks[peer]
            if kind in ("send_feature", "send_gradient"):
                outbox = self._features if kind == "send_feature" else self._outgoing
                if mb not in outbox:
                    raise RuntimeError(f"{kind} of micro-batch {mb} before it exists.")
                tensor = outbox.pop(mb)
                work = dist.isend(tensor, dst=peer_rank, group=self._group)
                assert work is not None
                self._sends.append(([work], tensor))
            elif kind == "recv_feature":
                self._features[mb] = self._new_buffer(mb)
                work = dist.irecv(self._features[mb], src=peer_rank, group=self._group)
                assert work is not None
                self._feature_recvs[mb] = work
            else:
                self._gradients[mb] = self._new_buffer(mb)
                work = dist.irecv(self._gradients[mb], src=peer_rank, group=self._group)
                assert work is not None
                self._gradient_recvs[mb] = work

    def _tower_forward(self, mb: int) -> torch.Tensor:
        pixel_values, grid_thw = self._inputs[mb]
        with spmd_local_context("dp"):
            return self._replica(pixel_values.to(self._dtype), grid_thw=grid_thw)

    def _encode(self, mb: int) -> None:
        with torch.no_grad():
            features = self._tower_forward(mb)
        if features.shape != (self._tokens[mb], self._hidden_dim):
            raise RuntimeError(
                f"micro-batch {mb}: the tower returned {tuple(features.shape)}, "
                f"expected ({self._tokens[mb]}, {self._hidden_dim})."
            )
        self._features[mb] = features

    def _backward(self, mb: int) -> None:
        work = self._gradient_recvs.pop(mb, None)
        if work is not None:
            work.wait()
        grad = self._gradients.pop(mb)
        with torch.enable_grad():
            torch.autograd.backward(self._tower_forward(mb), grad)
        assert self._accum is not None
        offset = 0
        for param in self._params:
            if param.grad is not None:
                self._accum[offset : offset + param.numel()].add_(
                    param.grad.reshape(-1)
                )
                param.grad = None
            offset += param.numel()

    def _after(self, anchor: Anchor) -> None:
        self._post(("after", anchor))
        for kind, mb in self._pending.pop(anchor, ()):
            if kind == "encode":
                self._encode(mb)
            else:
                self._backward(mb)

    def stage0_kwargs(
        self, mb: int, kwargs: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """The forward kwargs of stage 0 with the features it splices in."""
        if not self.active:
            return kwargs
        work = self._feature_recvs.pop(mb, None)
        if work is not None:
            work.wait()
        features = self._features.pop(mb, None)
        if features is None:
            features = torch.empty(
                0, self._hidden_dim, dtype=self._dtype, device=self._device
            )
        elif self._trainable:
            features.requires_grad_(True)
            features.register_hook(self._capture(mb))
        return {**(kwargs or {}), "vision_embeds": features}

    def _capture(self, mb: int):
        def hook(grad: torch.Tensor) -> None:
            self._captured[mb] = grad

        return hook

    def after_stage0_backward(self, mb: int) -> None:
        if not self.active or not self._trainable or mb not in self._inputs:
            return
        plan = self.plan
        assert plan is not None
        grad = self._captured.pop(mb, None)
        if grad is None:
            grad = self._new_buffer(mb).zero_()
        if plan.backward_rank[mb] == self._me:
            self._gradients[mb] = grad
        else:
            self._outgoing[mb] = grad.contiguous()

    def before_action(self, anchor: Anchor) -> None:
        if self.active:
            self._post(("before", anchor))

    def after_action(self, anchor: Anchor, *, sends_follow: bool) -> None:
        """Run what is planned after ``anchor``, or hold it for the action's sends."""
        if not self.active:
            return
        if anchor not in self._pending and ("after", anchor) not in self._posts:
            return
        if sends_follow:
            self._after_send.add(anchor)
        else:
            self._after(anchor)

    def holds_for_send(self, anchor: Anchor) -> bool:
        return self.active and anchor in self._after_send

    def after_send(self, anchor: Anchor, works: list[dist.Work], ops: Any) -> None:
        self._after_send.discard(anchor)
        self._sends.append((works, ops))
        self._after(anchor)

    def end_step(self) -> None:
        plan = self.plan
        assert plan is not None
        self._post(STEP_END)
        if self._pending or self._posts or self._after_send:
            raise RuntimeError(
                "vision work or transfers planned around actions this rank never "
                f"ran: {self._pending}, {self._posts}, {self._after_send}"
            )
        for mb in plan.epilogue[self._me]:
            self._backward(mb)
        for works, _ in self._sends:
            for work in works:
                work.wait()
        if self._feature_recvs or self._gradient_recvs or self._captured:
            raise RuntimeError(
                "vision features or gradients left unconsumed at step end: "
                f"{sorted(self._feature_recvs)}, {sorted(self._gradient_recvs)}, "
                f"{sorted(self._captured)}"
            )
        if self._trainable:
            self._reduce_gradients()
        self.active = False
        self._reset()

    def abort_step(self) -> None:
        self.active = False
        self._reset()

    def _reduce_gradients(self) -> None:
        assert self._accum is not None
        dist.reduce(self._accum, dst=self._pp_ranks[self._stage0], group=self._group)
        if self._me != self._stage0:
            return
        if self._dp_group is not None:
            dist.all_reduce(self._accum, group=self._dp_group)
        assert self._tower is not None
        sources = {canonical_fqn(n): p for n, p in self._tower.named_parameters()}
        offset = 0
        for name, param in zip(self._names, self._params, strict=True):
            source = sources[name]
            numel = param.numel()
            if source.requires_grad:
                grad = self._accum[offset : offset + numel].view(param.shape)
                dim = _tp_dim(param.shape, source.shape)
                if dim is not None:
                    shards = [torch.empty_like(grad) for _ in range(self._tp)]
                    dist.all_gather(shards, grad.contiguous(), group=self._tp_group)
                    grad = torch.cat(shards, dim=dim)
                grad = grad.to(source.dtype, copy=True)
                if isinstance(source, DTensor):
                    grad = distribute_tensor(
                        grad, source.device_mesh, source.placements, src_data_rank=None
                    )
                source.grad = grad if source.grad is None else source.grad + grad
            offset += numel
