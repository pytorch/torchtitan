# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from collections.abc import Sequence

import torch.nn as nn
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    get_schedule_class,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torch.distributed.pipelining.stage import _PipelineStageBase, PipelineStage

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
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.distributed.spmd_types import annotate_replicated_parameters
from torchtitan.models.kimi_k3.layout import (
    infer_block_layout_tables_from_stages,
    layer_to_stage_from_split,
)
from torchtitan.models.kimi_k3.pipeline_stage import (
    AttnResPipelineStage,
    PPRankLocalCache,
)
from torchtitan.tools.logging import logger
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
            ("context parallel", parallel_dims.cp_enabled),
        )
        if enabled
    ]
    if unsupported_parallelisms:
        raise NotImplementedError(
            "Kimi K3 currently supports FSDP2 data parallelism "
            f"only; disable {', '.join(unsupported_parallelisms)}."
        )
    if compile_config.enable and "model" in compile_config.components:
        raise NotImplementedError("Kimi K3 does not support model compilation yet.")

    assert isinstance(model, KimiK3Model)
    if parallelism.spmd_backend == "spmd_types":
        # Seed replicated layouts for parameters outside the explicit expert
        # declarations. Vision buffers declare their DP layouts separately.
        annotate_replicated_parameters(model, parallel_dims)

    if parallelism.spmd_backend == "spmd_types" or parallel_dims.ep_enabled:
        # model_registry's moe_comm_backend picks the dispatcher: standard
        # (default), deepep and minimal_async_ep run on this model; hybridep
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

    if parallelism.spmd_backend == "spmd_types":
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
    else:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
        dp_mesh_dims = None
        edp_mesh = None
        edp_mesh_dims = None
        if parallel_dims.ep_enabled:
            edp_mesh_names = (
                ["dp_replicate", "efsdp"]
                if parallel_dims.dp_replicate_enabled
                else ["efsdp"]
            )
            edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

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
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            dp_mesh_dims=dp_mesh_dims,
        )

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    return model


_KIMI_K3_FIRST_STAGE_FQNS = ("vision_encoder",)
_KIMI_K3_LAST_STAGE_FQNS = ("output_res_proj", "output_res_norm")


def kimi_k3_module_fqns_per_model_part(
    num_stages: int,
    num_layers: int,
    input_weight: int = 1,
    output_weight: int = 1,
    *,
    first_stage_modules: Sequence[str] = _KIMI_K3_FIRST_STAGE_FQNS,
    last_stage_modules: Sequence[str] = _KIMI_K3_LAST_STAGE_FQNS,
) -> list[list[str]]:
    """Kimi K3's pipeline split: the layers plus the embedding and head weights spread
    evenly, earlier stages taking the remainder; the vision tower rides with the
    embedding and the AttnRes aggregation with the head."""
    first = [*first_stage_modules, "tok_embeddings"]
    last = ["norm", "lm_head", *last_stage_modules]
    if num_stages == 1:
        return [first + [f"layers.{i}" for i in range(num_layers)] + last]
    units = num_layers + input_weight + output_weight
    if not 1 <= num_stages <= units:
        raise ValueError(f"{num_stages} stages for {units} units")
    per_stage, extra = divmod(units, num_stages)
    if max(input_weight, output_weight) > per_stage:
        raise ValueError(
            f"embedding / head weight ({input_weight} / {output_weight}) exceeds "
            f"the {per_stage} units per stage"
        )
    split, start = [], 0
    for s in range(num_stages):
        is_first, is_last = s == 0, s == num_stages - 1
        take = (
            per_stage + (s < extra) - is_first * input_weight - is_last * output_weight
        )
        layers = [f"layers.{i}" for i in range(start, start + take)]
        start += take
        split.append((first if is_first else []) + layers + (last if is_last else []))
    return split


def _kimi_k3_num_stages(
    parallelism: ParallelismConfig, pp: int, num_layers: int
) -> int:
    """From ``pipeline_parallel_layers_per_stage`` when set, else one stage per rank
    for a single-stage schedule and two for a looped one."""
    schedule = get_schedule_class(parallelism.pipeline_parallel_schedule)
    single = issubclass(schedule, PipelineScheduleSingle)
    layers_per_stage = parallelism.pipeline_parallel_layers_per_stage
    if layers_per_stage is None:
        return pp * (1 if single else 2)
    units = (
        num_layers
        + parallelism.pipeline_parallel_first_stage_less_layers
        + parallelism.pipeline_parallel_last_stage_less_layers
    )
    num_stages = -(-units // layers_per_stage)
    per_rank, rest = divmod(num_stages, pp)
    if rest or (per_rank != 1 if single else per_rank < 2):
        raise ValueError(
            f"layers_per_stage={layers_per_stage} gives {num_stages} stages, which "
            f"{parallelism.pipeline_parallel_schedule} cannot run on {pp} ranks"
        )
    return num_stages


def _kimi_k3_pipeline_split(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    model_config,
) -> tuple[list[list[str]], ParallelismConfig]:
    """The split handed to ``pipeline_llm``, and the config that spells it out."""
    num_layers = len(model_config.layers)
    split = kimi_k3_module_fqns_per_model_part(
        _kimi_k3_num_stages(parallelism, parallel_dims.pp, num_layers),
        num_layers,
        parallelism.pipeline_parallel_first_stage_less_layers,
        parallelism.pipeline_parallel_last_stage_less_layers,
        first_stage_modules=[
            n for n in _KIMI_K3_FIRST_STAGE_FQNS if getattr(model, n, None) is not None
        ],
        last_stage_modules=[n for n in _KIMI_K3_LAST_STAGE_FQNS if hasattr(model, n)],
    )
    return split, dataclasses.replace(
        parallelism,
        module_fqns_per_model_part=split,
        pipeline_parallel_layers_per_stage=None,
    )


def _as_attn_res_stage(stage: _PipelineStageBase) -> AttnResPipelineStage:
    """``stage`` rebuilt as an :class:`AttnResPipelineStage` around the same module."""
    assert isinstance(stage, PipelineStage)
    rebuilt = AttnResPipelineStage(
        stage.submod,
        stage.stage_index,
        stage.num_stages,
        stage.device,
        group=stage.group,
        dw_builder=stage.dw_builder,
        get_mesh=stage._mesh_cache._get_mesh_cb,
    )
    # The schedule wrote its stage-to-rank map onto the stage it was handed.
    rebuilt.stage_index_to_group_rank = stage.stage_index_to_group_rank
    return rebuilt


def _swap_in_attn_res_stages(
    schedule: _PipelineSchedule,
) -> list[AttnResPipelineStage]:
    """Replace the stages a schedule holds on this rank with AttnRes stages."""
    if isinstance(schedule, PipelineScheduleSingle):
        rebuilt = _as_attn_res_stage(schedule._stage)
        schedule._stage = rebuilt
        return [rebuilt]
    if isinstance(schedule, PipelineScheduleMulti):
        rebuilt_stages = [_as_attn_res_stage(s) for s in schedule._stages]
        held: list[_PipelineStageBase] = list(rebuilt_stages)
        schedule._stages = held
        return rebuilt_stages
    raise RuntimeError(f"Unexpected pipeline schedule class {type(schedule).__name__}.")


def pipeline_kimi_k3(model: nn.Module, *, attn_res_cache: bool = True, **kwargs):
    """``pipelining_fn`` for Kimi K3: core's split and schedule, each stage rebuilt
    as an :class:`AttnResPipelineStage` and routed by tables built from that split.

    ``attn_res_cache=False`` sends the whole block stack on every hop instead of
    only the blocks the receiving rank lacks; every rank must pass the same value.
    """
    # The vision tower goes with the embedding, the AttnRes aggregation with the head.
    parallelism = kwargs.pop("parallelism")
    module_fqns_per_model_part = parallelism.module_fqns_per_model_part
    if module_fqns_per_model_part is None:
        module_fqns_per_model_part, parallelism = _kimi_k3_pipeline_split(
            model,
            parallel_dims=kwargs["parallel_dims"],
            parallelism=parallelism,
            model_config=kwargs["model_config"],
        )

    pp_schedule, model_parts, has_first_stage, has_last_stage = pipeline_llm(
        model,
        parallelism=parallelism,
        **kwargs,
    )

    stages = _swap_in_attn_res_stages(pp_schedule)
    model_config = kwargs["model_config"]
    layer_cfgs = model_config.layers
    n_layers = len(layer_cfgs)
    layers_per_block = layer_cfgs[0].attn_res_block_size
    num_blocks = -(-n_layers // layers_per_block)
    # Every rank reads the same layer map off the applied split; no collective.
    layer_to_stage = layer_to_stage_from_split(module_fqns_per_model_part)
    layout = infer_block_layout_tables_from_stages(
        stages,
        stage_to_rank=dict(stages[0].stage_index_to_group_rank),
        num_blocks=num_blocks,
        n_layers=n_layers,
        layers_per_block=layers_per_block,
        layer_to_stage=layer_to_stage,
        cache=attn_res_cache,
    )
    store = PPRankLocalCache()
    for stage in stages:
        stage.set_routing(layout, store)
    logger.info(
        "Kimi K3 pipeline: %d stage(s) on this rank %s, block transport %s",
        len(stages),
        [s.stage_index for s in stages],
        "delta with rank store" if attn_res_cache else "whole stack every hop",
    )
    return pp_schedule, model_parts, has_first_stage, has_last_stage
