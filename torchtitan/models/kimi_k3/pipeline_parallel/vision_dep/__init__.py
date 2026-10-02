# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Decoupled encoder process: every pipeline rank encodes and backpropagates a share
of the step's images with its own copy of the vision tower."""

from __future__ import annotations

import copy

import torch
import torch.distributed as dist
from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.spmd_types import annotate_replicated_parameters
from torchtitan.protocols.model import BaseModel

from ...vision_encoder import KimiK3VisionEncoder
from ..stage import AttnResPipelineStage
from .runtime import VisionDep
from .schedule import VisionDepSchedule
from .stage import VisionDepPipelineStage

__all__ = ["build_vision_replica", "install_vision_dep", "pipeline_groups"]


def build_vision_replica(
    model: BaseModel, *, parallelism_context, training, ac_config, dump_folder, device
):
    """A copy of the vision tower for every pipeline rank, parallelized like the original."""
    tower = getattr(model, "vision_encoder", None)
    if tower is None:
        raise ValueError("vision_dep needs a model with a vision encoder.")
    if training.enable_cpu_offload:
        raise ValueError("vision_dep does not support training.enable_cpu_offload.")
    replica = copy.deepcopy(tower)
    with parallelism_context.activate_spmd():
        annotate_replicated_parameters(replica, parallelism_context)
        replica._parallelize(parallelism_context)
        if ac_config is not None:
            ac_config.build(dump_folder=dump_folder).apply(replica)
    replica.to_empty(device=device)
    # The first step overwrites these weights; drawing them must not move the model's seed.
    forked = [device] if torch.device(device).type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=forked):
        replica.init_states()
    with torch.no_grad():
        dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
        for param in replica.parameters():
            param.data = param.data.to(dtype)
    return replica.train()


def pipeline_groups(parallelism_context) -> list[list[int]]:
    """The global ranks of every pipeline group of the world mesh."""
    pp_mesh = parallelism_context.get_mesh("pp")
    if dist.get_backend(pp_mesh.get_group()) == "fake":
        raise ValueError("vision_dep needs a real pipeline process group.")
    pp = parallelism_context.pp
    stride = dist.get_world_size() // pp
    groups = [[base + k * stride for k in range(pp)] for base in range(stride)]
    if pp_mesh.mesh.tolist() not in groups:
        raise RuntimeError(
            f"The pipeline group {pp_mesh.mesh.tolist()} is not a pp slice of the "
            "world mesh."
        )
    return groups


def install_vision_dep(
    pp_schedule: _PipelineSchedule,
    stages: list[AttnResPipelineStage],
    *,
    replica: KimiK3VisionEncoder,
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
    assert isinstance(group, dist.ProcessGroup)
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
        assert isinstance(stage, VisionDepPipelineStage)
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
