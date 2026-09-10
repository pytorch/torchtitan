# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.distributed as dist
import torch.nn as nn

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    apply_fsdp_to_vision_encoder,
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
)
from torchtitan.distributed.spmd_types import annotate_replicated_parameters
from .model import KimiK3Model


def parallelize_kimi_k3(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    skip_dp: bool = False,
) -> nn.Module:
    """Apply FSDP2 to the Kimi K3 decoder and vision encoder."""

    unsupported_parallelisms = [
        name
        for name, enabled in (
            ("tensor parallel", parallel_dims.tp_enabled),
            ("pipeline parallel", parallel_dims.pp_enabled),
        )
        if enabled
    ]
    if unsupported_parallelisms:
        raise NotImplementedError(
            f"Kimi K3 does not support {', '.join(unsupported_parallelisms)}."
        )
    if compile_config.enable and "model" in compile_config.components:
        raise NotImplementedError("Kimi K3 does not support model compilation yet.")

    assert isinstance(model, KimiK3Model)
    # Seed replicated layouts for parameters outside the explicit expert
    # declarations. Vision buffers declare their DP layouts separately.
    annotate_replicated_parameters(model, parallel_dims)

    # model_registry's moe_comm_backend picks the dispatcher: standard
    # (default) and deepep run on this model; hybridep
    # needs GB200-class hardware.
    model.parallelize(parallel_dims)

    if ac_config is not None:
        ac_policy = ac_config.build(dump_folder=dump_folder)
        ac_policy.apply(model)
        if model.vision_encoder is not None:
            ac_policy.apply(model.vision_encoder)

    # Skip FSDP wrapper for inference. FSDP's forward hooks
    # are incompatible with torch.inference_mode() used by vLLM.
    # AC and compile are disabled via config (mode="none", enable=False).
    if skip_dp:
        return model

    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)

    if parallel_dims.cp_enabled:
        # Dynamic CP for the tower partitions the large images across sub-CP
        # groups; every layout is built here, once, in the same order on every
        # rank. The build is a world-wide collective (new_group and the rank
        # list exchange), so every rank runs it, including a pipeline stage
        # that holds no tower and never uses the result.
        cp_group = parallel_dims.get_mesh("cp").get_group()
        # The tower encodes inside a dp-local SPMD context, so it takes the cp
        # group from here rather than from the active mesh.
        setattr(model, "_cp_group_all", cp_group)  # noqa: B010
        setattr(model, "_cp_subgroups", _build_cp_subgroups(cp_group))  # noqa: B010

    vision_encoder = model.vision_encoder
    if vision_encoder is not None:
        # TODO: An image batch on one DP rank and a text-only batch on another
        # execute different FSDP collectives, deadlock, and hit a 90-second
        # timeout. A general solution is needed.
        apply_fsdp_to_vision_encoder(
            vision_encoder,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=False,
            cpu_offload=training.enable_cpu_offload,
            dp_mesh_dims=dp_mesh_dims,
        )

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=False,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        symm_mem_scope=parallelism.fsdp_symm_mem_scope,
    )

    return model


def _build_cp_subgroups(cp_group) -> dict[int, dist.ProcessGroup]:
    """Every sub-CP group layout this CP group could use, built once.

    Which layout a step wants depends on how many large images the batch
    holds, and a process group cannot be built per batch: ``new_group`` must be
    called by every process, with the same rank lists, in the same order. So
    the divisors of the CP size are all built here, and the CP rank lists are
    all-gathered first so every rank walks the same global list and keeps the
    group it belongs to. Returns ``{num_subgroups: this rank's group}``.
    """
    if cp_group is None:
        return {}
    cp_ranks = dist.get_process_group_ranks(cp_group)
    cp_size = len(cp_ranks)
    if cp_size <= 1:
        return {}
    world = dist.get_world_size()
    all_cp: list[list[int] | None] = [None] * world
    dist.all_gather_object(all_cp, cp_ranks)
    seen: list[list[int]] = []
    for entry in all_cp:
        if entry and list(entry) not in seen:
            seen.append(list(entry))
    seen.sort()
    my_rank = dist.get_rank()
    out: dict[int, dist.ProcessGroup] = {}
    for n_sub in [d for d in range(1, cp_size + 1) if cp_size % d == 0]:
        g = cp_size // n_sub
        mine: dist.ProcessGroup | None = None
        for ranks in seen:
            for s in range(n_sub):
                members = ranks[s * g : (s + 1) * g]
                pg = dist.new_group(ranks=members)
                if my_rank in members and isinstance(pg, dist.ProcessGroup):
                    mine = pg
        if mine is not None:
            out[n_sub] = mine
    return out
