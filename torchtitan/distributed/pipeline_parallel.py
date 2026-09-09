# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
import dataclasses
import math
import os
from collections.abc import Callable
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._mesh_layout import _MeshLayout
from torch.distributed.device_mesh import DeviceMesh
import torch.distributed.pipelining.stage as stage_lib
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining._utils import InferenceMode
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    get_schedule_class,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    ScheduleDualPipeV,
    ScheduleZBVZeroBubble,
)

from torchtitan.components.loss import LossFunction
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ParallelizeFunction
from torchtitan.protocols.module import ModuleDict, ModuleList
from torchtitan.tools.logging import logger

# pipeline_llm and pipeline_vlm are the public entrypoints for model-specific PP
# setup. Helpers in this module are implementation details and stay private.
__all__ = ["pipeline_llm", "pipeline_vlm"]


PIPELINE_NEIGHBOR_P2P_ENV = "TORCHTITAN_PIPELINE_NEIGHBOR_P2P"


@dataclasses.dataclass
class _PipelineTransportGroups:
    """This rank's transport for one PP replica: a CPU control group, and for
    every adjacent stage edge this rank sits on, the two-rank NCCL group and
    the peer's index inside it (keyed by the edge's source stage)."""

    metadata_group: dist.ProcessGroup
    edge_groups: dict[int, dist.ProcessGroup]
    edge_peers: dict[int, int]


def _create_pipeline_transport_groups(
    parallel_dims: ParallelDims,
    pp_group: dist.ProcessGroup,
    *,
    num_stages: int,
    pp_schedule: str,
) -> _PipelineTransportGroups | None:
    """Map the physical edge groups ``ParallelDims`` created to logical stage
    edges. Same-rank edges (a looped schedule's consecutive stages on one rank)
    need no group; every other edge this rank sits on must have one."""
    if os.environ.get(PIPELINE_NEIGHBOR_P2P_ENV) != "1":
        return None
    pp_global_ranks = tuple(dist.get_process_group_ranks(pp_group))
    pp_degree = len(pp_global_ranks)
    if pp_degree <= 1:
        return None
    if dist.get_backend(pp_group) != "nccl":
        raise RuntimeError("Pipeline neighbor P2P requires an NCCL PP group")
    transport = parallel_dims.get_pipeline_neighbor_groups(pp_global_ranks)
    if transport is None:
        raise RuntimeError("Pipeline neighbor P2P groups were not created during mesh setup")
    metadata_group, groups_by_pair = transport
    local_pp_rank = dist.get_rank(pp_group)
    my_global = pp_global_ranks[local_pp_rank]

    stage_to_pp_rank: dict[int, int] = {}
    for pp_rank in range(pp_degree):
        for stage_idx in _get_pp_rank_to_stage_indices_mapping(
            pp_rank, pp_degree, pp_schedule, num_stages
        ):
            stage_to_pp_rank[stage_idx] = pp_rank
    if len(stage_to_pp_rank) != num_stages:
        raise RuntimeError("Could not map every pipeline stage to a PP rank")

    edge_groups: dict[int, dist.ProcessGroup] = {}
    edge_peers: dict[int, int] = {}
    for src_stage in range(num_stages - 1):
        src_rank = stage_to_pp_rank[src_stage]
        dst_rank = stage_to_pp_rank[src_stage + 1]
        if src_rank == dst_rank or local_pp_rank not in (src_rank, dst_rank):
            continue
        other = pp_global_ranks[dst_rank if local_pp_rank == src_rank else src_rank]
        key = (min(my_global, other), max(my_global, other))
        group = groups_by_pair.get(key)
        if group is None:
            raise RuntimeError(f"Missing PP edge group for ranks {key}")
        edge_groups[src_stage] = group
        edge_peers[src_stage] = 0 if other == key[0] else 1
    return _PipelineTransportGroups(metadata_group, edge_groups, edge_peers)


class _NeighborP2PTransportMixin:
    """Tensor P2P on two-rank stage-edge groups, metadata on the CPU group.

    Composed in front of the stage class (``_neighbor_p2p_stage_class``), so a
    stage subclass keeps its own forward and backward while the transport
    changes underneath it. Only adjacent-stage traffic is supported.
    """

    def __init__(self, *args: Any, transport: _PipelineTransportGroups, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._transport = transport

    def _edge(self, src_stage: int) -> tuple[dist.ProcessGroup, int] | None:
        group = self._transport.edge_groups.get(src_stage)
        if group is None:
            if self._is_same_rank(src_stage) or self._is_same_rank(src_stage + 1):
                return None
            raise RuntimeError(
                f"Stage {self.stage_index} has no P2P group for edge {src_stage} -> {src_stage + 1}"
            )
        return group, self._transport.edge_peers[src_stage]

    def _recv_edge_ops(self, recv_infos: tuple, edge, expected_source: int) -> list[dist.P2POp]:
        ops: list[dist.P2POp] = []
        for info in recv_infos:
            if info.is_root_arg:
                continue
            if info.buffer is None:
                if info.tensor_meta is not None:
                    raise AssertionError("missing recv buffer has tensor metadata")
                continue
            if info.source != expected_source:
                raise RuntimeError(
                    "Pipeline neighbor P2P supports only adjacent stage inputs; "
                    f"stage {self.stage_index} receives from {info.source}"
                )
            if edge is None:
                raise RuntimeError("missing adjacent PP receive group")
            group, peer = edge
            ops.append(dist.P2POp(dist.irecv, info.buffer, group_peer=peer, group=group))
        return ops

    def get_fwd_recv_ops(self, fwd_chunk_id: int) -> list[dist.P2POp]:
        if self.is_first:
            return []
        return self._recv_edge_ops(
            self.args_recv_info[fwd_chunk_id], self._edge(self.stage_index - 1), self.stage_index - 1
        )

    def get_bwd_recv_ops(self, bwd_chunk_id: int) -> list[dist.P2POp]:
        if not self.has_backward or self.is_last:
            return []
        return self._recv_edge_ops(
            self.grad_recv_info[bwd_chunk_id], self._edge(self.stage_index), self.stage_index + 1
        )

    def get_fwd_send_ops(self, fwd_chunk_id: int) -> list[dist.P2POp]:
        output_tuple, _ = self.fwd_cache[fwd_chunk_id]
        if self.is_last:
            return []
        edge = self._edge(self.stage_index)
        ops: list[dist.P2POp] = []
        for idx, out in enumerate(output_tuple):
            for dst_stage in self.act_send_info[idx]:
                if dst_stage is None:
                    continue
                if dst_stage != self.stage_index + 1:
                    raise RuntimeError(
                        "Pipeline neighbor P2P supports only adjacent stage outputs; "
                        f"stage {self.stage_index} sends to {dst_stage}"
                    )
                if edge is None:
                    raise RuntimeError("missing adjacent PP forward-send group")
                group, peer = edge
                ops.append(
                    dist.P2POp(
                        dist.isend,
                        stage_lib.to_local_if_dtensor(out, detach=True),
                        group_peer=peer,
                        group=group,
                    )
                )
        return ops

    def get_bwd_send_ops(self, bwd_chunk_id: int) -> list[dist.P2POp]:
        if not self.has_backward or self.is_first:
            return []
        edge = self._edge(self.stage_index - 1)
        self._check_chunk_id(bwd_chunk_id)
        if self.grad_send_info is None:
            self.grad_send_info = self._create_grad_send_info(self.args_recv_info[0])
        ops: list[dist.P2POp] = []
        for idx, (grad, dst_stage) in enumerate(
            zip(self.bwd_cache.pop(bwd_chunk_id), self.grad_send_info, strict=True)
        ):
            if dst_stage is None:
                if grad is not None:
                    raise stage_lib.PipeliningMetadataError(
                        f"Stage {self.stage_index} produced an unsent gradient"
                    )
                continue
            if dst_stage != self.stage_index - 1:
                raise RuntimeError(
                    "Pipeline neighbor P2P supports only adjacent gradient sends; "
                    f"stage {self.stage_index} sends to {dst_stage}"
                )
            grad_meta = self._get_grad_send_meta(idx)
            if grad_meta is None:
                if grad is not None:
                    raise stage_lib.PipeliningMetadataError(
                        f"Stage {self.stage_index} has a gradient without metadata"
                    )
                continue
            if grad is None:
                grad = stage_lib._make_tensor_from_meta(grad_meta, self.device).zero_()
            if not isinstance(grad, torch.Tensor):
                raise stage_lib.PipeliningMetadataError(
                    f"unexpected gradient type {type(grad).__name__}"
                )
            if edge is None:
                raise RuntimeError("missing adjacent PP backward-send group")
            group, peer = edge
            ops.append(
                dist.P2POp(
                    dist.isend, stage_lib.to_local_if_dtensor(grad), group_peer=peer, group=group
                )
            )
        return ops

    def _send_meta(self, meta: Any, dst_stage: int) -> None:
        dist.send_object_list(
            [meta],
            dst=self._resolve_peer_global_rank(dst_stage),
            group=self._transport.metadata_group,
            device=torch.device("cpu"),
            use_batch=False,
        )

    def _recv_meta(self, src_stage: int) -> Any:
        objects: list[Any] = [None]
        dist.recv_object_list(
            objects,
            src=self._resolve_peer_global_rank(src_stage),
            group=self._transport.metadata_group,
            device=torch.device("cpu"),
            use_batch=False,
        )
        return objects[0]


def _neighbor_p2p_stage_class(base: type[PipelineStage]) -> type[PipelineStage]:
    """The stage class with the neighbor transport composed in front of it."""
    return type(f"NeighborP2P{base.__name__}", (_NeighborP2PTransportMixin, base), {})


def _configure_neighbor_p2p_schedule(schedule: _PipelineSchedule) -> None:
    """One collective for the static/dynamic decision, for neighbor-transport stages.

    The stock vote is a serial P2P chain over the full PP group, the transport
    path that races the schedule's tensor traffic. The group-wide minimum is
    the same decision; the override binds to this schedule instance only.
    """
    original_warmup_p2p = schedule._warmup_p2p

    def _warmup_p2p(
        _schedule: _PipelineSchedule,
        stages: list[PipelineStage],
        has_backward: bool,
        p2p_done: bool,
    ) -> None:
        if not stages or not all(
            isinstance(stage, _NeighborP2PTransportMixin) for stage in stages
        ):
            return original_warmup_p2p(stages, has_backward, p2p_done)
        local_vote = int(
            all(
                not InferenceMode.needs_dynamic(stage._user_meta, has_backward)
                for stage in stages
            )
        )
        vote = torch.tensor([local_vote], dtype=torch.int32, device=stages[0].device)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=stages[0].group)
        mode = InferenceMode.STATIC if vote.item() == 1 else InferenceMode.DYNAMIC
        for stage in stages:
            stage._inference_mode = mode

    schedule._warmup_p2p = MethodType(_warmup_p2p, schedule)



def _build_get_mesh_callback(
    parallel_dims: ParallelDims,
) -> Callable[[tuple[str, ...], _MeshLayout | None], DeviceMesh | None]:
    """Build a callback that resolves a DeviceMesh from dimension names.

    Pipeline parallelism requires an SPMD mesh during module split so that
    at runtime the current PP rank can reconstruct a DTensor after receiving
    a plain tensor from the previous PP rank. DTensors are not directly
    serializable across PP stages (because ProcessGroup is not serializable),
    so each stage uses this callback to obtain its local DeviceMesh and
    re-wrap incoming tensors as DTensors with the correct placements.
    """

    def _get_mesh(
        mesh_dim_names: tuple[str, ...], mesh_layout: _MeshLayout | None
    ) -> DeviceMesh | None:
        mesh = parallel_dims.get_mesh(list(mesh_dim_names))
        if mesh_layout is not None and mesh._layout != mesh_layout:
            return None
        return mesh

    return _get_mesh


def pipeline_llm(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    device: torch.device,
    model_config: BaseModel.Config,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
    stage_class: type[PipelineStage] = PipelineStage,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    pp_mesh = parallel_dims.get_mesh("pp")

    (
        num_virtual_stages,
        num_layers,
        input_weight,
        output_weight,
    ) = _get_pipeline_metadata(parallel_dims, parallelism, model_config)

    module_names_per_stage = parallelism.module_fqns_per_model_part
    if module_names_per_stage is None:
        module_names_per_stage = _generate_llm_fqn_per_model_part(
            num_virtual_stages, num_layers, input_weight, output_weight
        )
    for i, stage_ms in enumerate(module_names_per_stage):
        logger.debug(f"Stage {i}: {stage_ms}")

    transport_groups = _create_pipeline_transport_groups(
        parallel_dims,
        pp_mesh.get_group("pp"),
        num_stages=len(module_names_per_stage),
        pp_schedule=parallelism.pipeline_parallel_schedule,
    )
    get_mesh_cb = _build_get_mesh_callback(parallel_dims)
    stages, model_parts = _pipeline_module_split(
        model,
        pp_mesh,
        parallelism.pipeline_parallel_schedule,
        device,
        module_names_per_stage,
        get_mesh=get_mesh_cb,
        stage_class=stage_class,
        transport_groups=transport_groups,
    )

    # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
    # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
    # optimizer, and checkpointing
    for i, m in enumerate(model_parts):
        # apply SPMD-style PT-D techniques
        m = parallelize_fn(
            m,
            parallel_dims=parallel_dims,
            training=training,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
        )
        model_parts[i] = m
        # NOTE: this is to update the model in the stage
        #       in case the model is modified e.g. by torch.compile
        stages[i].submod = m

    pp_schedule = _build_pipeline_schedule(
        parallelism=parallelism,
        num_microbatches=parallelism.num_pp_microbatches,
        stages=stages,
        loss_fn=loss_fn,
    )
    if transport_groups is not None:
        _configure_neighbor_p2p_schedule(pp_schedule)
    else:
        _warmup_pp_edge_communicators(stages)

    # This is used in the train loop to determine whether to pass in the input_ids and labels
    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return pp_schedule, model_parts, has_first_stage, has_last_stage


def pipeline_vlm(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    model_config: BaseModel.Config,
    **kwargs,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    """PP entrypoint for vision-language models: co-locate the vision encoder
    with the first stage, then delegate to ``pipeline_llm``.

    The auto-generated LLM stage split only knows about decoder modules
    (``tok_embeddings``, ``layers.*``, ``norm``, ``lm_head``). For a VLM we inject
    ``vision_encoder`` into the first stage's FQN list so it runs alongside
    ``tok_embeddings`` (vision features are scattered into the embedding sequence
    before the decoder layers). On stages other than the first, ``tok_embeddings``
    and ``vision_encoder`` are pruned to ``None``; each model's ``forward`` must
    guard on ``self.tok_embeddings is not None`` so the multimodal logic is
    skipped there.

    NOTE: This adds load to stage 0 that the auto split does not model
    (``input_weight`` only accounts for ``tok_embeddings``); for a heavy vision
    encoder, bump ``parallelism.pipeline_parallel_first_stage_less_layers`` to
    rebalance.
    """
    if parallelism.module_fqns_per_model_part is None:
        (
            num_virtual_stages,
            num_layers,
            input_weight,
            output_weight,
        ) = _get_pipeline_metadata(parallel_dims, parallelism, model_config)
        fqn_per_part = _generate_llm_fqn_per_model_part(
            num_virtual_stages, num_layers, input_weight, output_weight
        )
        if model.vision_encoder is not None:
            fqn_per_part[0].insert(0, "vision_encoder")
        parallelism = dataclasses.replace(
            parallelism, module_fqns_per_model_part=fqn_per_part
        )

    return pipeline_llm(
        model,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        model_config=model_config,
        **kwargs,
    )


def _get_pipeline_metadata(
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    model_config: BaseModel.Config,
) -> tuple[int, int, int, int]:
    """Determine the number of virtual stages and the number of layers in the model.

    Extracted from ``pipeline_llm`` so that Graph PP can compute stage
    metadata without running the full eager pipeline setup.
    """
    # Determine the number of virtual stages based on schedule type
    schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
    layers_per_stage = parallelism.pipeline_parallel_layers_per_stage
    if hasattr(model_config, "layers"):
        num_layers = len(model_config.layers)
    else:
        raise ValueError("Model does not have n_layers attribute.")

    # You can adjust these weights based on the computational cost of embeddings and output layers
    # Higher weights mean these modules are treated as "heavier" in the distribution
    input_weight = parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = parallelism.pipeline_parallel_last_stage_less_layers

    # Calculate number of virtual stages
    if layers_per_stage is not None:

        # Calculate number of virtual stages needed (using ceiling division)
        # This allows for unequal distribution where stages can differ by at most 1 layer
        num_virtual_stages = math.ceil(
            (num_layers + input_weight + output_weight) / layers_per_stage
        )

        # Validation: check stages per rank based on schedule type
        model_config_info = f"Model has {num_layers} layers with pipeline_parallel_layers_per_stage={layers_per_stage}"
        stage_distribution_info = (
            f"resulting in {num_virtual_stages=} across {parallel_dims.pp} PP ranks"
        )

        if num_virtual_stages % parallel_dims.pp != 0:
            raise ValueError(
                f"Number of virtual stages ({num_virtual_stages}) must be divisible by "
                f"pipeline parallel size ({parallel_dims.pp}). "
                f"{model_config_info}. "
                f"Please adjust pipeline_parallel_layers_per_stage to a value that results in a number of stages "
                f"divisible by {parallel_dims.pp}."
            )

        stages_per_rank = num_virtual_stages // parallel_dims.pp

        if is_single_stage_schedule and stages_per_rank != 1:
            raise ValueError(
                f"Single stage schedule requires exactly 1 stage per rank, but got {stages_per_rank} stages per rank. "
                f"{model_config_info}, {stage_distribution_info}. "
                f"Please increase pipeline_parallel_layers_per_stage to {num_layers // parallel_dims.pp} or higher "
                f"to achieve 1 stage per rank."
            )

        if not is_single_stage_schedule and stages_per_rank < 2:
            raise ValueError(
                f"Multi-stage schedule requires at least 2 stages per rank, but got {stages_per_rank} stages per rank. "
                f"{model_config_info}, {stage_distribution_info}. "
                f"Please decrease pipeline_parallel_layers_per_stage to achieve at least 2 stages per rank."
            )
    else:
        # Fallback to default behavior when layers_per_stage is not provided
        # For multi-stage schedules, default is 2 virtual stages per rank
        # For single-stage schedules, default is 1 virtual stage per rank
        stages_per_rank = 1 if is_single_stage_schedule else 2
        num_virtual_stages = parallel_dims.pp * stages_per_rank
    return num_virtual_stages, num_layers, input_weight, output_weight


def _warmup_pp_edge_communicators(stages: list[PipelineStage]) -> None:
    """Create every pipeline-edge NCCL communicator eagerly, before step one.

    Without this, the communicator behind an edge is created lazily by the
    first steady-state ``_batch_p2p`` that touches it. On one node that first
    touch is cheap; across nodes it is a full NCCL bootstrap, and different
    ranks reach their first touch at different times -- the late edges of an
    8-stage pipeline (5->6, 6->7) then sit in communicator creation until the
    300 s default timeout. The schedules module carries a TODO describing this
    exact gap ("STATIC mode group communicator warm-up gap ... lazily created
    on the first mixed `_batch_p2p` call") with this fix prescribed; it is
    applied here from the torchtitan side because the training repo cannot
    patch torch in place.

    Every rank calls this at the same point (right after schedule build), every
    op has its matching counterpart on the neighbouring rank, and the dummy
    payloads are discarded -- so the only effect is that the communicators
    exist before any rank depends on a neighbour's progress to create them.
    """
    from torch.distributed.pipelining.schedules import _batch_p2p, _wait_batch_p2p

    ops: list[dist.P2POp] = []
    for stage in stages:
        get_ops = getattr(stage, "_get_init_p2p_neighbors_ops", None)
        if get_ops is None:
            # A stage type without the hook predates the lazy-creation hazard's
            # fix surface; nothing to warm.
            continue
        ops.extend(get_ops())
    if ops:
        _wait_batch_p2p(_batch_p2p(ops, desc="pp_edge_warmup"))


def _build_pipeline_schedule(
    *,
    parallelism: ParallelismConfig,
    num_microbatches: int,
    stages: list[PipelineStage],
    loss_fn: Callable,
    # Graph PP runs explicit backward graphs instead of autograd
    backward_requires_autograd: bool = True,
) -> _PipelineSchedule:
    """Builds a pipeline schedule for the given job configuration and stages.

    Also used by Graph PP, which passes ``backward_requires_autograd=False``
    because it runs explicit backward graphs instead of autograd.

    Args:
        parallelism (ParallelismConfig): The parallelism configuration.
        num_microbatches (int): Number of pipeline microbatches.
        stages (list[PipelineStage]): The stages to be scheduled.
        loss_fn (Callable): The loss function.

    Returns:
        _PipelineSchedule: The pipeline schedule for the given stages.
    """
    pp_schedule_csv = parallelism.pipeline_parallel_schedule_csv

    # Validate that pp_schedule_csv is a valid path
    if pp_schedule_csv:
        if not os.path.isfile(pp_schedule_csv):
            raise FileNotFoundError(
                f"The specified path {pp_schedule_csv} does not exist or is not a file."
            )
        schedule_class = _PipelineScheduleRuntime
    else:
        schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)

    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    # We expect that the number of local stages (`len(stages)`) is the same across all ranks
    num_total_stages = parallelism.pipeline_parallel_degree * len(stages)
    if num_microbatches < num_total_stages:
        logger.warning(
            f"Number of microbatches ({num_microbatches}) is less than the total number "
            f"of stages ({num_total_stages}) which may result in a bubble in the pipeline."
        )

    if schedule_class is PipelineScheduleSingle:
        raise ValueError(
            "PipelineScheduleSingle is an abstract base class. "
            "Use a concrete single-stage schedule such as GPipe or 1F1B."
        )

    # Pipeline schedules expect a bare scalar loss tensor.
    def _scalar_loss_fn(*args: object, **kwargs: object) -> torch.Tensor:
        loss, _ = loss_fn(*args, **kwargs)
        return loss

    if looped_schedule:
        schedule = schedule_class(
            stages,  # pyrefly: ignore [bad-argument-type]
            n_microbatches=num_microbatches,
            loss_fn=_scalar_loss_fn,
            scale_grads=False,
            backward_requires_autograd=backward_requires_autograd,
        )
    else:
        schedule = schedule_class(
            stages[0],
            n_microbatches=num_microbatches,
            loss_fn=_scalar_loss_fn,
            scale_grads=False,
        )
    logger.info(
        f"Using pipeline schedule {parallelism.pipeline_parallel_schedule} "
        f"with {num_microbatches} microbatches and {num_total_stages} stages."
    )

    if pp_schedule_csv:
        assert schedule_class in [
            PipelineScheduleSingle,
            PipelineScheduleMulti,
            _PipelineScheduleRuntime,
        ], (
            "Only PipelineScheduleSingle (single stage), PipelineScheduleMulti (multistage), "
            "and _PipelineScheduleRuntime support csv schedules"
        )
        # pyrefly: ignore [missing-attribute]
        schedule._load_csv(pp_schedule_csv)

    return schedule


def _generate_llm_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    input_weight: int = 1,
    output_weight: int = 1,
) -> list[list[str]]:
    """Programmatically generates module names per model part, focused on LLM models.

    Also used by Graph PP to compute per-stage module splits independently
    of the full ``pipeline_llm`` setup.

    Args:
        num_stages: Number of pipeline stages
        num_layers: Total number of transformer layers in the model
        input_weight: Weight for input modules (tok_embeddings) in layer calculation
        output_weight: Weight for output modules (norm + output) in layer calculation

    Returns:
        List of lists containing module names for each model part

    Example:
        _generate_llm_fqn_per_model_part(2, 3, input_weight=2, output_weight=2)
        treats embeddings as 2 layers and norm+output as 2 layers for distribution
    """
    if num_stages < 1:
        raise ValueError("Number of stages must be at least 1")

    if num_stages == 1:
        # Single stage gets everything
        layer_names = [f"layers.{i}" for i in range(num_layers)]
        return [["tok_embeddings"] + layer_names + ["norm", "lm_head"]]

    # Calculate effective layers including weights
    num_effective_layers = num_layers + input_weight + output_weight

    if num_stages > num_effective_layers:
        raise ValueError(
            f"Number of stages ({num_stages}) cannot be greater than effective layers ({num_effective_layers})"
        )

    # Calculate layers per stage (distribute evenly)
    layers_per_stage = num_effective_layers // num_stages
    extra_layers = num_effective_layers % num_stages

    # Feasibility check: Ensure at least 1 layer in each PP stage
    if layers_per_stage == 0:
        raise ValueError(
            f"Configuration would result in empty stages. "
            f"With {num_stages} stages and {num_effective_layers} effective layers "
            f"(num_layers={num_layers} + input_weight={input_weight} + output_weight={output_weight}), "
            f"each stage would get {layers_per_stage} layers on average. "
            f"Reduce num_stages or increase num_layers/weights."
        )

    # Balance check: Ensure weights don't exceed minimum layers per stage
    if input_weight > layers_per_stage:
        raise ValueError(
            f"input_weight ({input_weight}) exceeds minimum layers per stage ({layers_per_stage})."
        )
    if output_weight > layers_per_stage:
        raise ValueError(
            f"output_weight ({output_weight}) exceeds minimum layers per stage ({layers_per_stage})."
        )

    module_names_per_stage = []
    current_layer = 0

    for stage_idx in range(num_stages):
        stage_modules = []

        # Calculate effective layers for this stage
        effective_layers_for_stage = layers_per_stage
        if stage_idx < extra_layers:
            effective_layers_for_stage += 1

        # First stage: handle input modules with weighting
        if stage_idx == 0:
            stage_modules.append("tok_embeddings")
            # Account for input weight in layer distribution
            remaining_layers_for_stage = effective_layers_for_stage - input_weight

            # Add transformer layers
            for _ in range(remaining_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

        # Last stage: handle output modules with weighting
        elif stage_idx == num_stages - 1:
            # Account for output weight in layer distribution
            remaining_layers_for_stage = effective_layers_for_stage - output_weight

            # Add transformer layers
            for _ in range(remaining_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

            # Add output modules
            stage_modules.extend(["norm", "lm_head"])

        # Middle stages: only transformer layers
        else:
            for _ in range(effective_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

        module_names_per_stage.append(stage_modules)

    return module_names_per_stage


def _split_module(
    whole_model: nn.Module,
    module_names: list[str],
) -> nn.Module:
    """
    Splits a whole model into a module based on the specified module names.

    Args:
        whole_model: The complete model to be split
        module_names: List of module names to include in the split

    Returns:
        The split module

    Example usage:
        module_names = ["tok_embeddings", "layers.0", "layers.1", "norm", "output"]
        split_module(whole_model, module_names)
    """
    model = copy.deepcopy(whole_model)
    # Create a set of modules to keep for faster lookup
    modules_to_keep = set(module_names)
    for module_name, module_value in model.named_children():
        # Handle layer-like structures (e.g., "layers.0", "layers.1")
        if isinstance(
            module_value, (nn.ModuleDict, nn.ModuleList, ModuleDict, ModuleList)
        ):
            layers_to_keep = {
                name.split(".", 1)[1]
                for name in modules_to_keep
                if name.startswith(f"{module_name}.")
            }
            if layers_to_keep:
                # Keep only specified layers
                if isinstance(module_value, nn.ModuleDict):
                    for layer_name in list(module_value.keys()):
                        if layer_name not in layers_to_keep:
                            del module_value[layer_name]
                elif isinstance(module_value, nn.ModuleList):
                    indices_to_keep = {
                        int(idx) for idx in layers_to_keep if idx.isdigit()
                    }
                    new_layers = ModuleList(
                        [
                            layer
                            for i, layer in enumerate(module_value)
                            if i in indices_to_keep
                        ]
                    )
                    setattr(model, module_name, new_layers)
            else:
                # No layers from this structure needed, set to empty structure
                if isinstance(module_value, (nn.ModuleDict, ModuleDict)):
                    setattr(model, module_name, ModuleDict())
                elif isinstance(module_value, (nn.ModuleList, ModuleList)):
                    setattr(model, module_name, ModuleList())
        # Handle simple module attributes (e.g., "linear", "norm")
        elif module_name not in modules_to_keep:
            # Replace with None
            setattr(model, module_name, None)
    return model


def _get_pp_rank_to_stage_indices_mapping(
    pp_rank: int,
    pp_degree,
    pp_schedule: str,
    num_stages: int,
) -> tuple[int, ...]:
    """
    Returns a mapping from PP rank to stage indices for the given pipeline schedule.

    Args:
        pp_rank: Pipeline parallel rank
        pp_degree: Number of pipeline parallel ranks
        pp_schedule: Name of pipeline parallelism schedule
        num_stages: Number of pipeline stages

    Returns:
        Mapping from PP rank to stage indices
    """
    schedule_class = get_schedule_class(pp_schedule)
    style = (
        "v" if schedule_class in (ScheduleZBVZeroBubble, ScheduleDualPipeV) else "loop"
    )
    assert (
        num_stages % pp_degree == 0
    ), f"num_stages {num_stages} must be evenly divisible by pp_degree {pp_degree}"
    stages_per_rank = num_stages // pp_degree
    if style == "loop":
        return tuple(pp_rank + s * pp_degree for s in range(stages_per_rank))
    elif style == "v":
        assert (
            stages_per_rank == 2
        ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
        stage_v_pairs = list(
            zip(range(pp_degree), range(num_stages - 1, pp_degree - 1, -1))
        )
        return tuple(stage_v_pairs[pp_rank])
    else:
        raise ValueError(f"Unknown style {style}")


def _pipeline_module_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    pp_schedule: str,
    device: torch.device,
    module_names_per_stage: list[list[str]],
    get_mesh: Callable | None = None,
    stage_class: type[PipelineStage] = PipelineStage,
    transport_groups: _PipelineTransportGroups | None = None,
) -> tuple[list[PipelineStage], list[nn.Module]]:
    """Create pipeline stages based on specified module names for each stage.

    Also used by Graph PP to split the model into per-stage chunks before
    exporting joint forward/backward graphs for each stage.

    Some model restrictions include:
    - forward() method should tolerate deleted layers
    - weight initialization methods should tolerate deleted layers
    - Does not support nested moduledict and modulelist structures

    Args:
        whole_model: The complete model to be split
        pp_mesh: Pipeline parallel device mesh
        pp_schedule: Name of pipeline parallelism schedule
        device: Device
        module_names_per_stage: List of lists, where each inner list contains the module names
                               that should be included in that stage. Module names should be
                               dot-separated paths. Examples:
                               - "tok_embeddings" for token embeddings
                               - "layers.0", "layers.1" for specific transformer layers
                               - "norm" for the final normalization layer
                               - "lm_head" for the output projection layer

    Returns:
        Tuple of (stages, models) where stages are PipelineStage objects and models are the
        corresponding model chunks

    Example usage:
        module_names_per_stage = [
            ["tok_embeddings", "layers.0"],     # Stage 0: embeddings + first layer
            ["layers.1", "layers.2"],           # Stage 1: middle layers
            ["norm", "lm_head"]                  # Stage 2: final norm + output
        ]
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_degree = pp_mesh.size()
    num_stages = len(module_names_per_stage)
    stages = []
    models = []
    pp_rank_to_stage_indices = _get_pp_rank_to_stage_indices_mapping(
        pp_rank, pp_degree, pp_schedule, num_stages
    )
    stage_kwargs: dict[str, Any] = {}
    if transport_groups is not None:
        stage_class = _neighbor_p2p_stage_class(stage_class)
        stage_kwargs["transport"] = transport_groups
    for stage_idx in pp_rank_to_stage_indices:
        module_names = module_names_per_stage[stage_idx]
        model_chunk = _split_module(whole_model, module_names)
        stage = stage_class(
            model_chunk,
            stage_idx,
            num_stages,
            device,
            group=pp_mesh.get_group("pp"),
            get_mesh=get_mesh,
            **stage_kwargs,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx} "
            f"with modules {module_names}"
        )
        stages.append(stage)
        models.append(model_chunk)

    return stages, models
