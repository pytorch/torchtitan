# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Decoupled encoder process: every pipeline rank encodes and backpropagates a share
of the step's images with its own copy of the vision tower."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.pipelining.schedules import _batch_p2p, _PipelineSchedule
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.distributed.spmd_types import spmd_local_context

from .dep_plan import Anchor, DepPlan, Hook, plan_dep, START, STEP_END
from .stage import AttnResPipelineStage

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
        replica: torch.nn.Module,
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
        self._device = self._params[0].device
        self._kh, self._kw = replica.merge_kernel_size
        self._synced = False
        self._trainable = False
        self._accum: torch.Tensor | None = None
        self.active = False
        self.plan: DepPlan | None = None
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
        plan = plan_dep(
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
                self._sends.append(([work], tensor))
            elif kind == "recv_feature":
                self._features[mb] = self._new_buffer(mb)
                self._feature_recvs[mb] = dist.irecv(
                    self._features[mb], src=peer_rank, group=self._group
                )
            else:
                self._gradients[mb] = self._new_buffer(mb)
                self._gradient_recvs[mb] = dist.irecv(
                    self._gradients[mb], src=peer_rank, group=self._group
                )

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


class VisionDepPipelineStage(AttnResPipelineStage):
    """AttnRes stage that gives stage 0 the encoded features and runs the vision work
    and transfers planned around each of its actions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._dep: VisionDep | None = None

    def set_vision_dep(self, dep: VisionDep) -> None:
        self._dep = dep

    def _sends_to(self, stage: int) -> bool:
        return (
            0 <= stage < self.num_stages
            and self.stage_index_to_group_rank[stage] != self.group_rank
        )

    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
        save_forward_output: bool = True,
    ):
        mb = int(fwd_chunk_id)
        anchor = ("F", self.stage_index, mb)
        if self._dep is not None:
            self._dep.before_action(anchor)
            if self.is_first:
                kwargs = self._dep.stage0_kwargs(mb, kwargs)
        output = super().forward_one_chunk(
            fwd_chunk_id, args, kwargs, save_forward_output
        )
        if self._dep is not None:
            self._dep.after_action(
                anchor, sends_follow=self._sends_to(self.stage_index + 1)
            )
        return output

    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        full_backward: bool = True,
        last_backward=False,
    ):
        mb = int(bwd_chunk_id)
        anchor = ("B" if full_backward else "I", self.stage_index, mb)
        if self._dep is not None:
            self._dep.before_action(anchor)
        super().backward_one_chunk(
            bwd_chunk_id,
            loss=loss,
            full_backward=full_backward,
            last_backward=last_backward,
        )
        if self._dep is None:
            return
        if self.is_first:
            self._dep.after_stage0_backward(mb)
        self._dep.after_action(
            anchor, sends_follow=self._sends_to(self.stage_index - 1)
        )

    def backward_weight_one_chunk(self, bwd_chunk_id: int, last_backward=False):
        anchor = ("W", self.stage_index, int(bwd_chunk_id))
        if self._dep is not None:
            self._dep.before_action(anchor)
        super().backward_weight_one_chunk(bwd_chunk_id, last_backward=last_backward)
        if self._dep is not None:
            self._dep.after_action(anchor, sends_follow=False)

    def _send_then_run(self, ops: list, anchors: tuple[Anchor, ...]) -> list:
        dep = self._dep
        held = [a for a in anchors if dep is not None and dep.holds_for_send(a)]
        if not held:
            return ops
        assert dep is not None
        # Work queued on the stream ahead of a send would hold the send back.
        works = _batch_p2p(ops)
        for anchor in held:
            dep.after_send(anchor, works, ops)
        return []

    def get_fwd_send_ops(self, fwd_chunk_id: int):
        ops = super().get_fwd_send_ops(fwd_chunk_id)
        return self._send_then_run(ops, (("F", self.stage_index, int(fwd_chunk_id)),))

    def get_bwd_send_ops(self, bwd_chunk_id: int):
        ops = super().get_bwd_send_ops(bwd_chunk_id)
        return self._send_then_run(
            ops,
            (
                ("B", self.stage_index, int(bwd_chunk_id)),
                ("I", self.stage_index, int(bwd_chunk_id)),
            ),
        )


class VisionDepSchedule(_PipelineSchedule):
    """The pipeline schedule with the vision phases around each training step."""

    def __init__(self, schedule: _PipelineSchedule, dep: VisionDep) -> None:
        self._schedule = schedule
        self._dep = dep

    def __getattr__(self, name: str) -> Any:
        if name == "_schedule":
            raise AttributeError(name)
        return getattr(self._schedule, name)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        kwarg_mbs = kwargs.get("kwarg_mbs")
        if kwarg_mbs is None:
            raise ValueError("vision_dep plans a step from its kwarg_mbs.")
        self._dep.begin_step(kwarg_mbs)
        try:
            output = self._schedule.step(*args, **kwargs)
        except BaseException:
            self._dep.abort_step()
            raise
        self._dep.end_step()
        return output

    def eval(self, *args: Any, **kwargs: Any) -> Any:
        return self._schedule.eval(*args, **kwargs)

    def _step_microbatches(self, *args: Any, **kwargs: Any) -> Any:
        return self._schedule._step_microbatches(*args, **kwargs)


def install_vision_dep(
    pp_schedule: _PipelineSchedule,
    stages: list[VisionDepPipelineStage],
    *,
    replica: torch.nn.Module,
    pp_groups: list[list[int]],
    dp_group: dist.ProcessGroup | None,
    tp_group: dist.ProcessGroup | None,
    hidden_dim: int,
    compute_dtype: torch.dtype,
    bubble: bool,
    cost_ratio: float,
) -> VisionDepSchedule:
    """Give every rank's stages the vision runtime and wrap the schedule's step."""
    pipeline_order = None
    if bubble:
        pipeline_order = getattr(pp_schedule, "pipeline_order", None)
        if not pipeline_order:
            raise ValueError(
                "vision_dep.bubble places work in the schedule's action order, which "
                "only the multi-stage schedules expose."
            )
    group, _ = dist.new_subgroups_by_enumeration(pp_groups)
    pp_ranks = next(g for g in pp_groups if dist.get_rank() in g)
    stage_to_rank = dict(stages[0].stage_index_to_group_rank)
    stage0_rank = stage_to_rank[0]
    first = next((s for s in stages if s.is_first), None)
    tower = getattr(first.submod, "vision_encoder", None) if first else None
    if first is not None and tower is None:
        raise ValueError("vision_dep needs the vision tower on stage 0.")
    _connect(pp_ranks, stage0_rank, group, replica)
    dep = VisionDep(
        replica,
        tower=tower,
        pp_ranks=pp_ranks,
        stage0_rank=stage0_rank,
        group=group,
        dp_group=dp_group,
        tp_group=tp_group,
        hidden_dim=hidden_dim,
        compute_dtype=compute_dtype,
        pipeline_order=pipeline_order,
        cost_ratio=cost_ratio,
    )
    for stage in stages:
        stage.set_vision_dep(dep)
    return VisionDepSchedule(pp_schedule, dep)


def _connect(
    pp_ranks: list[int],
    stage0_rank: int,
    group: dist.ProcessGroup,
    replica: torch.nn.Module,
) -> None:
    # A first send between two ranks blocks until both reach it, so connect before the schedule runs.
    device = next(replica.parameters()).device
    probe = torch.zeros(1, device=device)
    me, hub = dist.get_rank(), pp_ranks[stage0_rank]
    dist.barrier(group=group)
    for peer in pp_ranks:
        if me == hub and peer != hub:
            dist.recv(probe, peer, group=group)
        elif me == peer and peer != hub:
            dist.send(probe, hub, group=group)
