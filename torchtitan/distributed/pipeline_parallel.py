# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
import dataclasses
import math
import os
from collections.abc import Callable, Iterable
from typing import Any, Protocol

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._mesh_layout import _MeshLayout
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    get_schedule_class,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    ScheduleDualPipeV,
    ScheduleZBVZeroBubble,
)
from torch.distributed.tensor import DTensor

from torchtitan.components.loss import LossFunction
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ParallelizeFunction
from torchtitan.protocols.module import ModuleDict, ModuleList
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_module

# pipeline_llm and pipeline_vlm are the public entrypoints for model-specific PP
# setup. Helpers in this module are implementation details and stay private.
__all__ = [
    "PipelineResult",
    "PipelineRuntime",
    "PipelineSharedParameterSpec",
    "SharedParameterPipelineRuntime",
    "pipeline_llm",
    "pipeline_vlm",
]


@dataclasses.dataclass(frozen=True, slots=True)
class PipelineSharedParameterSpec:
    """Describe one logical parameter with a canonical copy and one
    pipeline-stage replica.

    Args:
        fqn: Parameter FQN shared by the canonical and replica stage modules.
        stage_indices: Stage indices ordered as ``(canonical, replica)``.
    """

    fqn: str
    stage_indices: tuple[int, int]


class PipelineRuntime:
    """Lifecycle hooks for model-specific pipeline behavior.

    The default implementation is a no-op.
    """

    def prepare_microbatch(
        self,
        inputs: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepare stage-local keyword inputs for one pipeline microbatch."""
        del inputs
        return kwargs

    def synchronize_parameters(self) -> None:
        """Synchronize runtime parameter replicas after initialization/load."""

    def finalize_gradients(self) -> None:
        """Finalize runtime-owned gradients before clipping and optimization."""

    def parameters_for_grad_norm(
        self,
        parameters: Iterable[nn.Parameter],
    ) -> tuple[nn.Parameter, ...]:
        """Return parameters counted in the logical model gradient norm."""
        return tuple(parameters)


class _PipelineScheduleLike(Protocol):
    """Common schedule interface used by eager and graph pipelines."""

    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Run one pipeline step."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class PipelineResult:
    """Artifacts produced while constructing a pipeline.

    Args:
        schedule: Pipeline schedule executed by the trainer.
        model_parts: Local model fragments owned by this rank.
        stage_indices: Global virtual-stage index for each local model part, in
            the same order as ``model_parts``.
        has_first_stage: Whether this rank owns the first virtual stage.
        has_last_stage: Whether this rank owns the last virtual stage.
        runtime: Model-owned pipeline lifecycle hooks invoked by the trainer.
            Use the no-op ``PipelineRuntime`` when no specialized behavior is
            required.
    """

    schedule: _PipelineScheduleLike
    model_parts: list[nn.Module]
    stage_indices: tuple[int, ...]
    has_first_stage: bool
    has_last_stage: bool
    runtime: PipelineRuntime


@dataclasses.dataclass(slots=True)
class _RankLocalSharedParameterState:
    """Rank-local modules and communication state for one shared parameter."""

    spec: PipelineSharedParameterSpec
    local_modules: dict[int, nn.Module]
    owner_group: dist.ProcessGroup | None
    owner_group_src: int | None


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local communication buffer for a Tensor or DTensor."""
    return tensor._local_tensor if isinstance(tensor, DTensor) else tensor


class SharedParameterPipelineRuntime(PipelineRuntime):
    """Synchronize parameter values and gradients for PP-stage replicas.

    Parameters are resolved from each stage module by FQN at every lifecycle
    hook, avoiding stale references after FSDP wrapping, ``to_empty``,
    initialization, or checkpoint loading.

    Args:
        model_parts: Parallelized stage modules owned by the current PP rank.
        stage_indices: Global virtual-stage index for each local model part, in
            the same order as ``model_parts``.
        pp_mesh: One-dimensional device mesh for the PP axis.
        pp_schedule: Schedule name used to map global virtual stages to PP ranks.
        num_stages: Total number of virtual stages across all PP ranks.
        shared_parameter_specs: Parameter specs containing the shared FQN and its
            canonical and replica virtual-stage indices.
    """

    def __init__(
        self,
        *,
        model_parts: list[nn.Module],
        stage_indices: tuple[int, ...],
        pp_mesh: DeviceMesh,
        pp_schedule: str,
        num_stages: int,
        shared_parameter_specs: tuple[PipelineSharedParameterSpec, ...],
    ) -> None:
        if len(model_parts) != len(stage_indices):
            raise ValueError(
                "model_parts and stage_indices must have the same length, got "
                f"{len(model_parts)} and {len(stage_indices)}."
            )
        stage_to_module = dict(zip(stage_indices, model_parts, strict=True))
        stage_to_rank: dict[int, int] = {}
        for pp_rank in range(pp_mesh.size()):
            for stage_index in _get_pp_rank_to_stage_indices_mapping(
                pp_rank,
                pp_mesh.size(),
                pp_schedule,
                num_stages,
            ):
                stage_to_rank[stage_index] = pp_rank

        pp_group = pp_mesh.get_group("pp")
        groups: dict[tuple[int, int], dist.ProcessGroup | int] = {}
        pp_group_ready_for_split = False
        shared_parameter_states: list[_RankLocalSharedParameterState] = []
        for spec in shared_parameter_specs:
            canonical_stage, replica_stage = spec.stage_indices
            if canonical_stage == replica_stage:
                raise ValueError(
                    f"Pipeline shared parameter {spec.fqn} needs two distinct stages."
                )
            if (
                canonical_stage not in stage_to_rank
                or replica_stage not in stage_to_rank
            ):
                raise ValueError(
                    f"Pipeline shared parameter {spec.fqn} references stages "
                    f"{spec.stage_indices} outside [0, {num_stages})."
                )

            owner_ranks = (stage_to_rank[canonical_stage], stage_to_rank[replica_stage])
            owner_group: dist.ProcessGroup | None = None
            owner_group_src: int | None = None
            if owner_ranks[0] != owner_ranks[1]:
                # With two PP ranks, distinct owners span the full PP group.
                if pp_mesh.size() == 2:
                    owner_group = pp_group
                    owner_group_src = owner_ranks[0]
                else:
                    # ``split_group`` requires an initialized parent PP
                    # communicator. DeviceMesh may initialize it lazily on the
                    # first device collective.
                    if not pp_group_ready_for_split:
                        dist.barrier(
                            group=pp_group,
                            device_ids=[device_module.current_device()],
                        )
                        pp_group_ready_for_split = True
                    group = groups.get(owner_ranks)
                    if group is None:
                        group = dist.split_group(
                            parent_pg=pp_group,
                            split_ranks=[list(owner_ranks)],
                            group_desc=(
                                "pipeline_shared_parameter_"
                                f"{canonical_stage}_{replica_stage}"
                            ),
                        )
                        groups[owner_ranks] = group
                    if pp_mesh.get_local_rank() in owner_ranks:
                        assert isinstance(group, dist.ProcessGroup)
                        owner_group = group
                        owner_group_src = 0

            local_modules = {
                stage_index: stage_to_module[stage_index]
                for stage_index in spec.stage_indices
                if stage_index in stage_to_module
            }
            if local_modules:
                shared_parameter_states.append(
                    _RankLocalSharedParameterState(
                        spec=spec,
                        local_modules=local_modules,
                        owner_group=owner_group,
                        owner_group_src=owner_group_src,
                    )
                )
        self._shared_parameter_states = tuple(shared_parameter_states)

    @staticmethod
    def _resolve_rank_local_parameters(
        state: _RankLocalSharedParameterState,
    ) -> tuple[nn.Parameter | None, nn.Parameter | None]:
        """Resolve the canonical and replica parameters owned by this rank."""
        canonical_stage, replica_stage = state.spec.stage_indices
        canonical_module = state.local_modules.get(canonical_stage)
        replica_module = state.local_modules.get(replica_stage)
        canonical = (
            canonical_module.get_parameter(state.spec.fqn)
            if canonical_module is not None
            else None
        )
        replica = (
            replica_module.get_parameter(state.spec.fqn)
            if replica_module is not None
            else None
        )
        return canonical, replica

    def synchronize_parameters(self) -> None:
        """Copy canonical parameter shards to PP replicas after init/load.

        Use a local copy for same-rank stages and an owner-group broadcast
        otherwise.
        """
        for state in self._shared_parameter_states:
            canonical, replica = self._resolve_rank_local_parameters(state)
            # Both copies are local; copy the canonical value into the replica.
            if canonical is not None and replica is not None:
                with torch.no_grad():
                    _local_tensor(replica).copy_(_local_tensor(canonical))
            else:
                # Different ranks each own one copy; broadcast the canonical value.
                parameter = canonical if canonical is not None else replica
                assert parameter is not None
                if state.owner_group is None or state.owner_group_src is None:
                    raise RuntimeError(
                        "Missing owner group for pipeline shared parameter "
                        f"{state.spec.fqn}."
                    )
                dist.broadcast(
                    _local_tensor(parameter),
                    group=state.owner_group,
                    group_src=state.owner_group_src,
                )

    def finalize_gradients(self) -> None:
        """Sum canonical and replica gradients before clipping and optimization.

        Use local add/copy for same-rank stages and an owner-group all-reduce
        otherwise.
        """
        for state in self._shared_parameter_states:
            canonical, replica = self._resolve_rank_local_parameters(state)
            if canonical is not None and replica is not None:
                if canonical.grad is None or replica.grad is None:
                    raise RuntimeError(
                        f"Pipeline shared parameter {state.spec.fqn} is missing a gradient."
                    )
                canonical_grad = _local_tensor(canonical.grad)
                replica_grad = _local_tensor(replica.grad)
                canonical_grad.add_(replica_grad)
                replica_grad.copy_(canonical_grad)
            else:
                parameter = canonical if canonical is not None else replica
                assert parameter is not None
                if state.owner_group is None:
                    raise RuntimeError(
                        "Missing owner group for pipeline shared parameter "
                        f"{state.spec.fqn}."
                    )
                if parameter.grad is None:
                    raise RuntimeError(
                        "Pipeline shared parameter "
                        f"{state.spec.fqn} is missing a gradient."
                    )
                dist.all_reduce(_local_tensor(parameter.grad), group=state.owner_group)

    def parameters_for_grad_norm(
        self,
        parameters: Iterable[nn.Parameter],
    ) -> tuple[nn.Parameter, ...]:
        """Exclude non-canonical replicas from the logical gradient norm."""
        replica_ids = set()
        for state in self._shared_parameter_states:
            _, replica = self._resolve_rank_local_parameters(state)
            if replica is not None:
                replica_ids.add(id(replica))
        return tuple(
            parameter for parameter in parameters if id(parameter) not in replica_ids
        )


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
    stage_args_factory: Callable[[int, int], tuple[Any, Any]] | None = None,
) -> PipelineResult:
    """Build a pipeline for a decoder model.

    Args:
        model: Complete decoder model before pipeline splitting.
        parallel_dims: Distributed mesh dimensions.
        training: Training shape and dtype configuration.
        parallelism: Parallelism and pipeline schedule configuration.
        compile_config: Model compilation configuration.
        ac_config: Activation-checkpointing configuration.
        dump_folder: Output directory used by parallelization helpers.
        device: Device used to construct pipeline stages.
        model_config: Decoder model configuration.
        parallelize_fn: Function applying stage-local parallelisms.
        loss_fn: Loss used by the pipeline schedule.
        stage_args_factory: Optional factory that takes a global virtual-stage
            index and total stage count, then returns example input and output
            arguments used to configure ``PipelineStage`` communication metadata.

    Returns:
        The schedule, local model parts, stage ownership, and runtime hooks.
    """
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

    get_mesh_cb = _build_get_mesh_callback(parallel_dims)
    stages, model_parts = _pipeline_module_split(
        model,
        pp_mesh,
        parallelism.pipeline_parallel_schedule,
        device,
        module_names_per_stage,
        get_mesh=get_mesh_cb,
        static_stage_args=stage_args_factory,
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

    # This is used in the train loop to determine whether to pass in the input_ids and labels
    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return PipelineResult(
        schedule=pp_schedule,
        model_parts=model_parts,
        stage_indices=tuple(stage.stage_index for stage in stages),
        has_first_stage=has_first_stage,
        has_last_stage=has_last_stage,
        runtime=PipelineRuntime(),
    )


def pipeline_vlm(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    model_config: BaseModel.Config,
    **kwargs,
) -> PipelineResult:
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
    static_stage_args: Callable[[int, int], tuple[Any, Any]] | None = None,
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
        static_stage_args: Optional factory for fixed input/output metadata.

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
    for stage_idx in pp_rank_to_stage_indices:
        module_names = module_names_per_stage[stage_idx]
        model_chunk = _split_module(whole_model, module_names)
        input_args, output_args = (
            static_stage_args(stage_idx, num_stages)
            if static_stage_args is not None
            else (None, None)
        )
        stage = PipelineStage(
            model_chunk,
            stage_idx,
            num_stages,
            device,
            input_args=input_args,
            output_args=output_args,
            group=pp_mesh.get_group("pp"),
            get_mesh=get_mesh,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx} "
            f"with modules {module_names}"
        )
        stages.append(stage)
        models.append(model_chunk)

    return stages, models
