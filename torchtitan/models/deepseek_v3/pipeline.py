# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V3 MTP stage ownership over the standard eager pipeline."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.pipelining.schedules import _PipelineSchedule
from torch.distributed.tensor import DTensor

from torchtitan.components.optim import GradientClipper
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext, utils as dist_utils
from torchtitan.distributed.pipeline_parallel import (
    _get_pp_rank_to_stage_indices_mapping,
    get_module_fqns_per_model_part,
    pipeline_llm,
)
from torchtitan.models.base.model import BaseModel

if TYPE_CHECKING:
    from .model import DeepSeekV3Model

__all__ = ["MTPGradientClipper", "pipeline_dsv3"]


class MTPGradientClipper(GradientClipper):
    """Finalize shared PP embeddings, then clip every physical parameter copy.

    Both embedding owners keep identical optimizer states and updates. The
    embedding contributes once to the logical norm, but both copies receive
    the same clipping coefficient. Finalization runs after all pipeline/FSDP
    backward work, inside the optimizer update and its optional CUDA graph.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GradientClipper.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._owners: list[DeepSeekV3Model] | None = None
        self._replica_ids: set[int] = set()

    def bind_model_parts(self, model_parts: list[torch.nn.Module]) -> None:
        """Bind physical embedding owners once after model/optimizer construction."""
        self._owners = [
            part
            for part in cast("list[DeepSeekV3Model]", model_parts)
            if part._mtp_pipeline_enabled and part.tok_embeddings is not None
        ]
        self._replica_ids = {
            id(part.tok_embeddings.weight)
            for part in self._owners
            if not part._mtp_pipeline_first_stage
        }

    def clip_grad_norm_(
        self,
        parameters: list[torch.nn.Parameter],
        max_norm: float | int,
        *,
        parallelism_context: ParallelismContext,
    ) -> torch.Tensor:
        """Sum owner gradients once and reuse distributed logical-norm clipping."""
        if self._owners is None:
            raise RuntimeError(
                "MTPGradientClipper requires bind_model_parts before updates."
            )
        _finalize_mtp_embedding_gradients(self._owners)
        return dist_utils.clip_grad_norm_(
            parameters,
            max_norm,
            foreach=True,
            pp_mesh=parallelism_context.get_optional_mesh("pp"),
            norm_parameters=[
                parameter
                for parameter in parameters
                if id(parameter) not in self._replica_ids
            ],
        )


def pipeline_dsv3(
    model: DeepSeekV3Model,
    *,
    parallelism_context: ParallelismContext,
    parallelism: ParallelismConfig,
    model_config: DeepSeekV3Model.Config,
    **kwargs: Any,
) -> tuple[_PipelineSchedule, list[BaseModel], bool, bool]:
    """Place MTP1 and an embedding replica on the final virtual stage.

    Only hidden activations cross stage boundaries. Shifted token IDs and
    masks are prepared on each rank and supplied as root keyword inputs.
    The canonical embedding FQN is retained on both owners so model and
    optimizer checkpoint keys remain unchanged.
    """
    if model.num_mtp_layers != 1:
        raise NotImplementedError("DeepSeek-V3 pipeline parallelism supports MTP1.")
    if parallelism_context.cp_enabled:
        raise NotImplementedError("MTP pipeline context parallelism is a TODO.")
    if kwargs["training"].enable_cpu_offload:
        raise NotImplementedError(
            "MTP pipeline embedding synchronization requires device parameters."
        )
    pp_mesh = parallelism_context.get_mesh("pp")
    pp_backend = dist.get_backend(pp_mesh.get_group())
    if pp_backend == "fake":
        raise NotImplementedError(
            "MTP pipeline parallelism requires a real process group."
        )

    # 1. Extend the core decoder partition without mutating the caller's config.
    parallelism = copy.copy(parallelism)
    configured = parallelism.pipeline_parallel_module_fqns_per_model_part
    if configured is None:
        parallelism.pipeline_parallel_last_stage_less_layers += model.num_mtp_layers
        module_fqns = get_module_fqns_per_model_part(
            model,
            first_stage_module_fqns=(),
            last_stage_module_fqns=(),
            parallelism_context=parallelism_context,
            parallelism=parallelism,
            model_config=model_config,
        )
    else:
        module_fqns = copy.deepcopy(configured)
    if "tok_embeddings" not in module_fqns[0]:
        raise ValueError("The first MTP pipeline stage must own tok_embeddings.")
    if any(
        fqn == "tok_embeddings" or fqn.startswith("mtp_layers.")
        for part in module_fqns[1:-1]
        for fqn in part
    ) or any(fqn.startswith("mtp_layers.") for fqn in module_fqns[0]):
        raise ValueError(
            "MTP layers and the embedding replica belong to the final stage."
        )
    for fqn in ("mtp_layers.0", "tok_embeddings"):
        if fqn not in module_fqns[-1]:
            module_fqns[-1].append(fqn)
    parallelism.pipeline_parallel_module_fqns_per_model_part = module_fqns
    model._mtp_pipeline_enabled = True

    # 2. Reuse core stage construction and record the physical embedding owners.
    schedule, model_parts, has_first, has_last = pipeline_llm(
        model,
        parallelism_context=parallelism_context,
        parallelism=parallelism,
        model_config=model_config,
        **kwargs,
    )
    dsv3_parts = cast("list[DeepSeekV3Model]", model_parts)
    rank_stages = {
        rank: _get_pp_rank_to_stage_indices_mapping(
            rank,
            pp_mesh.size(),
            parallelism.pipeline_parallel_schedule,
            len(module_fqns),
        )
        for rank in range(pp_mesh.size())
    }
    for part, stage_index in zip(
        dsv3_parts, rank_stages[pp_mesh.get_local_rank()], strict=True
    ):
        part._mtp_pipeline_first_stage = stage_index == 0

    owner_ranks = [
        rank
        for rank, stages in rank_stages.items()
        if 0 in stages or len(module_fqns) - 1 in stages
    ]
    # 3. Split collectively over PP, then synchronize only the embedding owners.
    group = pp_mesh.get_group() if len(owner_ranks) == pp_mesh.size() else None
    if len(owner_ranks) > 1 and group is None:
        # Every PP parent rank participates, including non-embedding owners.
        group = dist.split_group(
            parent_pg=pp_mesh.get_group(), split_ranks=[owner_ranks]
        )
    if has_first or has_last:
        source_rank = int(
            pp_mesh.mesh[
                next(rank for rank, stages in rank_stages.items() if 0 in stages)
            ]
        )
        if group is not None and not isinstance(group, dist.ProcessGroup):
            raise RuntimeError(
                "MTP embedding owners must be members of their synchronization group."
            )
        for part in dsv3_parts:
            if part.tok_embeddings is not None:
                part._mtp_embedding_group = group
                part._mtp_embedding_src_rank = source_rank
    return schedule, model_parts, has_first, has_last


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local FSDP/TP shard used by the PP owner collective."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _synchronize_mtp_embeddings(model_parts: list[DeepSeekV3Model]) -> None:
    """Synchronize physical embedding copies after init or checkpoint loading."""
    owners = [
        part
        for part in model_parts
        if part._mtp_pipeline_enabled and part.tok_embeddings is not None
    ]
    if len(owners) == 2:
        first = next(part for part in owners if part._mtp_pipeline_first_stage)
        last = next(part for part in owners if not part._mtp_pipeline_first_stage)
        _local_tensor(last.tok_embeddings.weight).copy_(
            _local_tensor(first.tok_embeddings.weight)
        )
    elif owners and owners[0]._mtp_embedding_group is not None:
        owner = owners[0]
        dist.broadcast(
            _local_tensor(owner.tok_embeddings.weight),
            src=owner._mtp_embedding_src_rank,
            group=owner._mtp_embedding_group,
        )


def _finalize_mtp_embedding_gradients(model_parts: list[DeepSeekV3Model]) -> None:
    """Sum both embedding paths before computing the logical gradient norm."""
    owners = [
        part
        for part in model_parts
        if part._mtp_pipeline_enabled and part.tok_embeddings is not None
    ]
    if any(part.tok_embeddings.weight.grad is None for part in owners):
        raise RuntimeError("MTP embedding gradients must exist before finalization.")
    if len(owners) == 2:
        first_grad = _local_tensor(owners[0].tok_embeddings.weight.grad)
        last_grad = _local_tensor(owners[1].tok_embeddings.weight.grad)
        first_grad.add_(last_grad)
        last_grad.copy_(first_grad)
    elif owners and owners[0]._mtp_embedding_group is not None:
        owner = owners[0]
        dist.all_reduce(
            _local_tensor(owner.tok_embeddings.weight.grad),
            group=owner._mtp_embedding_group,
        )
