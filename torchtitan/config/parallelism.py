# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelism configuration."""

from dataclasses import dataclass, field
from typing import Annotated, get_args, Literal, TypeAlias

import torch
import tyro

from torchtitan.distributed.context_parallel import (
    ContextParallelLoadBalancer,
    HeadTailCPLoadBalancer,
)


FSDPSymmMemScope: TypeAlias = Literal["all", "dense", None]
_FSDP_SYMM_MEM_SCOPES = get_args(FSDPSymmMemScope)


@dataclass(kw_only=True, slots=True)
class ParallelismConfig:
    data_parallel_replicate_degree: int = 1
    """
    The `data_parallel_replicate_degree` argument specifies the degree of
    data parallelism for weight replication. When this value is greater
    than 1, weights will be replicated across `data_parallel_replicate_degree`
    ranks. If `data_parallel_shard_degree` is also greater than 1, the parallelism
    method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
    parallelism method used is DDP (Distributed Data Parallelism).
    1 means disabled.
    """

    data_parallel_shard_degree: int = -1
    """
    The `data_parallel_shard_degree` argument specifies the degree of data
    parallelism for weight sharding. When this value is greater than 1, weights
    will be sharded across `data_parallel_shard_degree` ranks. If
    `data_parallel_replicate_degree` is also greater than 1, the parallelism
    method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
    parallelism method used is FSDP (Fully Sharded Data Parallelism).
    -1 means leftover ranks will be used (After DP_REPLICATE/SP/PP). Note that
    only `data_parallel_shard_degree` can be negative. 1 means disabled.
    """

    fsdp_reshard_after_forward: Literal["default", "always", "never"] = "default"
    """
    `reshard_after_forward` specifies the policy for applying `reshard_after_forward`
    within an FSDP setup. `reshard_after_forward` controls parameter behavior after forward,
    trading off memory and communication. See torch's `fully_shard` API for more documentation
    on `reshard_after_forward`.

    The supported policies include "default", "always" and "never":

    - "default" applies default resharding behavior, implementing "smart defaults" for known optimal
      scenarios.
    - "always" will enable `reshard_after_forward` for all forward passes.
    - "never" will disable `reshard_after_forward` for all forward passes.
    """

    fsdp_symm_mem_scope: Annotated[FSDPSymmMemScope, tyro.conf.Suppress] = None
    """
    Which FSDP modules use symmetric-memory communication. None disables it.
    "dense" skips any module with routed experts. An MoE transformer block is
    one FSDP module, so its attention parameters are skipped along with its
    experts.
    """

    tensor_parallel_degree: int = 1
    """Tensor Parallelism degree. 1 means disabled."""

    enable_sequence_parallel: bool = True
    """Whether to use SequenceParallel as part of tensor parallelism. Enabled by default."""

    pipeline_parallel_degree: int = 1
    """
    Pipeline Parallelism degree, or number of ranks. 1 means disabled.
    If using looped schedules, this still specifies the number of physical ranks, not the number
    of stages. Stages per rank are inferred from split points degree, and schedule.
    """

    module_fqns_per_model_part: list[list[str]] | None = None
    """
    Specify a list of lists containing the FQNs (Fully Qualified Names) of modules for each model chunk.
    Each inner list represents one model chunk and contains the module names that belong to that chunk.
    e.g. [['tok_embeddings', 'layers.0'], ['layers.1', 'layers.2'], ['layers.3', 'layers.4']]
    will create 3 chunks: the first containing tok_embeddings and layers.0,
    the second containing layers.1 and layers.2, and the third containing layers.3 and layers.4.
    This provides more explicit control over which modules belong to each chunk compared to split points.
    """

    pipeline_parallel_first_stage_less_layers: int = 1
    """
    The number of layers to reduce in the first stage of pipeline parallelism. This is because
    the first stage has the extra overhead of the embedding layer, which is not present in the other stages.
    """

    pipeline_parallel_last_stage_less_layers: int = 1
    """
    The number of layers to reduce in the last stage of pipeline parallelism. This is because
    the last stage has the extra overhead of the output layer, which is not present in the other stages.
    """

    pipeline_parallel_layers_per_stage: int | None = None
    """
    The number of layers per (virtual) pipeline stage. If specified, the module_fqns_per_model_part will be
    calculated from the number of layers and pipeline_parallel_degree. If not specified, the
    layers per stage will be inferred from the model, schedule, and pipeline_parallel_degree.
    """

    pipeline_parallel_schedule: str = "1F1B"
    """
    Specify the Pipeline Parallel schedule to use. The supported schedules are:
    https://github.com/pytorch/pytorch/blob/de4c2a3b4e89d96334dc678d1c3f2ae51a6630a0/torch/distributed/pipelining/schedules.py#L2161.
    The schedule must be compatible with the split points and stages_per_rank.
    Looped schedules (e.g. Interleaved1F1B) require specifying pipeline_parallel_degree = number of ranks,
    and split_points = number of stages - 1
    """

    pipeline_parallel_schedule_csv: str | None = ""
    """
    Specify the path to the pipeline parallel schedule csv file to use.
    The pipeline_parallel_schedule argument must be either
    PipelineScheduleSingle, PipelineScheduleMulti, or _PipelineScheduleRuntime.
    """

    num_pp_microbatches: int = 1
    """
    Number of pipeline microbatches per data-parallel rank and gradient
    accumulation iteration. This setting is ignored when pipeline parallelism
    is disabled (`pipeline_parallel_degree = 1`, the default).
    """

    context_parallel_degree: int = 1
    """Context parallelism degree. 1 means disabled."""

    context_parallel_load_balancer: Annotated[
        ContextParallelLoadBalancer.Config | None, tyro.conf.Suppress
    ] = field(default_factory=HeadTailCPLoadBalancer.Config)
    """
    Per-batch load-balancer configuration for context parallelism. Defaults to
    head-tail load balancing. Set to None to disable load balancing and use
    contiguous sharding. Ulysses requires None.
    """

    def __post_init__(self):
        if self.fsdp_symm_mem_scope not in _FSDP_SYMM_MEM_SCOPES:
            raise ValueError(
                "parallelism.fsdp_symm_mem_scope must be one of: "
                f"{list(_FSDP_SYMM_MEM_SCOPES)} "
                f"(got {self.fsdp_symm_mem_scope!r})"
            )
        if self.fsdp_symm_mem_scope is not None and (
            not torch.cuda.is_available()
            or (
                torch.version.hip is None
                and torch.cuda.get_device_capability() < (9, 0)
            )
        ):
            raise ValueError(
                "For NVIDIA GPUs, parallelism.fsdp_symm_mem_scope is only supported "
                "for compute capability 9.0 or newer."
            )
        # Import lazily so loading parallelism.py does not pull in pipelining.
        from torch.distributed.pipelining.schedules import get_schedule_class

        try:
            get_schedule_class(self.pipeline_parallel_schedule)
        except ValueError as e:
            raise ValueError(
                "Invalid parallelism.pipeline_parallel_schedule "
                f"{self.pipeline_parallel_schedule!r}: {e}"
            ) from e

    expert_parallel_degree: int = 1
    """
    Expert parallelism degree. 1 means disabled. No effect for non-MoE models.
    For MoE models, this must be at least tensor_parallel_degree.

    Mesh constraint: the dense region (dp_shard * cp * tp) and sparse region
    (efsdp * ep) cover the same ranks, so dp_shard * cp * tp == efsdp * ep.
    EP borrows ranks from FSDP and TP: efsdp = dp_shard * cp * tp / ep.
    pp and dp_replicate are outer dimensions unaffected by this constraint.
    """
