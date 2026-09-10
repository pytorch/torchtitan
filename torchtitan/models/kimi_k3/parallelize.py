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
_KIMI_K3_VIT_DEP_STAGE_FQNS = ("tok_embeddings", "vision_encoder")


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


def _kimi_k3_vit_dep_split(
    model: nn.Module, *, parallel_dims, parallelism, model_config
) -> tuple[list[list[str]], ParallelismConfig]:
    """The split with the vision tower and the embedding on the first stage alone
    and the layers over the remaining stages (report sec 5.2.3): the encoder's
    compute leaves the text stages' critical path. The stage count the schedule
    sees is unchanged, so the tower stage comes out of the text stages' budget.
    """
    if getattr(model, "vision_encoder", None) is None:
        raise ValueError(
            "vit_dep gives the vision tower a pipeline stage of its own, "
            "and this model has no vision encoder."
        )
    num_layers = len(model_config.layers)
    num_stages = _kimi_k3_num_stages(parallelism, parallel_dims.pp, num_layers)
    if num_stages < 2:
        raise ValueError(
            "vit_dep needs at least two pipeline stages: one for the tower and "
            "the embedding, one for the layers."
        )
    # The embedding rides with the tower: the splice needs the token ids, which
    # only the first stage receives. The text split carries no embedding
    # (input_weight 0) and drops the one the split places.
    text = kimi_k3_module_fqns_per_model_part(
        num_stages - 1,
        num_layers,
        0,
        parallelism.pipeline_parallel_last_stage_less_layers,
        first_stage_modules=(),
        last_stage_modules=[n for n in _KIMI_K3_LAST_STAGE_FQNS if hasattr(model, n)],
    )
    split = [list(_KIMI_K3_VIT_DEP_STAGE_FQNS)] + [
        [n for n in stage if n != "tok_embeddings"] for stage in text
    ]
    return split, dataclasses.replace(
        parallelism,
        module_fqns_per_model_part=split,
        pipeline_parallel_layers_per_stage=None,
    )


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


def pipeline_kimi_k3(
    model: nn.Module,
    *,
    attn_res_cache: bool = True,
    vit_dep: bool = False,
    vit_prefetch: int = 0,
    vit_bubble: bool = False,
    vit_bubble_cost_ratio: float = 1.0,
    vit_bubble_max_pending: int = 0,
    **kwargs,
):
    """``pipelining_fn`` for Kimi K3: core's split and schedule, each stage rebuilt
    as an :class:`AttnResPipelineStage` and routed by tables built from that split.

    ``attn_res_cache=False`` sends the whole block stack on every hop instead of
    only the blocks the receiving rank lacks; every rank must pass the same value.
    ``vit_dep`` gives the vision tower and the embedding the first stage alone
    (report sec 5.2.3); it changes the split every rank applies, so a recipe
    sets it, the same way. On top of it, ``vit_prefetch`` issues the encode
    for micro-batch m+k while m's forward runs (on a side stream outside
    autograd, inline with it), and ``vit_bubble`` runs the encodes in the
    schedule's idle intervals instead, the first ``pp`` of them upfront as
    the report prescribes, with the tower's backwards deferred to the idle
    intervals after backward actions (``dep_bubble_*``); the two are
    alternatives. ``vit_bubble_cost_ratio`` is one encode in units of a text
    stage's forward, the budget an idle run must cover before an encode is
    placed in it; ``vit_bubble_max_pending`` bounds the deferred backwards
    held at once (0 is unbounded).
    """
    # The vision tower goes with the embedding, the AttnRes aggregation with the head.
    parallelism = kwargs.pop("parallelism")
    module_fqns_per_model_part = parallelism.module_fqns_per_model_part
    if module_fqns_per_model_part is None and vit_dep:
        module_fqns_per_model_part, parallelism = _kimi_k3_vit_dep_split(
            model,
            parallel_dims=kwargs["parallel_dims"],
            parallelism=parallelism,
            model_config=kwargs["model_config"],
        )
    elif module_fqns_per_model_part is None:
        module_fqns_per_model_part, parallelism = _kimi_k3_pipeline_split(
            model,
            parallel_dims=kwargs["parallel_dims"],
            parallelism=parallelism,
            model_config=kwargs["model_config"],
        )
    elif vit_dep and set(module_fqns_per_model_part[0]) != set(
        _KIMI_K3_VIT_DEP_STAGE_FQNS
    ):
        raise ValueError(
            "vit_dep puts the vision tower and the embedding on the first stage "
            f"alone; the configured split starts with {module_fqns_per_model_part[0]}."
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
    if vit_prefetch or vit_bubble:
        if not vit_dep:
            raise ValueError(
                "vit_prefetch and vit_bubble place the tower's encodes around the "
                "pipeline's actions, which needs the tower on its own stage: set "
                "vit_dep."
            )
        if vit_prefetch and vit_bubble:
            raise ValueError(
                "vit_prefetch and vit_bubble are alternatives; set exactly one."
            )
        _install_vision_dep(
            pp_schedule,
            stages,
            rank=kwargs["parallel_dims"].get_mesh("pp").get_local_rank(),
            pp_size=kwargs["parallel_dims"].pp,
            prefetch=int(vit_prefetch),
            bubble=bool(vit_bubble),
            cost_ratio=float(vit_bubble_cost_ratio),
            max_pending=int(vit_bubble_max_pending),
        )
    return pp_schedule, model_parts, has_first_stage, has_last_stage


def _install_vision_dep(
    pp_schedule,
    stages: list[AttnResPipelineStage],
    *,
    rank: int,
    pp_size: int,
    prefetch: int,
    bubble: bool,
    cost_ratio: float,
    max_pending: int,
) -> None:
    """Wire the tower stage's encodes to the schedule (report sec 5.2.3).

    The stage holding the tower serves each micro-batch's forward from a
    per-step feature cache when the encode already ran, and the cache is
    filled either ahead of the consumer (``prefetch`` micro-batches ahead) or
    in the schedule's idle intervals (``bubble``: the plan is read off the
    schedule's own action order, so every rank derives the same placements).
    A rank without the tower stage is left alone.
    """
    from torchtitan.models.kimi_k3.dep_bubble_backward import (
        cut_for_deferred_backward,
        GradQueue,
        install_backward_slots,
    )
    from torchtitan.models.kimi_k3.dep_bubble_plan import plan_for_rank
    from torchtitan.models.kimi_k3.dep_bubble_runtime import install_bubble_runtime
    from torchtitan.models.kimi_k3.vit_prefetch import VisionPrefetcher

    tower_stages = [
        s for s in stages if getattr(s.submod, "vision_encoder", None) is not None
    ]
    if not tower_stages:
        return
    if len(tower_stages) != 1:
        raise RuntimeError(
            f"rank {rank} holds {len(tower_stages)} stages with a vision tower; "
            "vit_dep places it on one."
        )
    tower_stage = tower_stages[0]
    module = tower_stage.submod
    prefetcher = VisionPrefetcher(module)
    queue = GradQueue(max_pending=max_pending) if bubble else None
    # FSDP2 initializes its state lazily in the root module's first forward; an
    # encode issued before that makes the tower a root of its own and the
    # stage's forward then refuses. So the first step runs its encodes inline,
    # and the run-ahead or the plan starts with the second.
    warm = {"done": False}

    inner_forward = tower_stage.forward_one_chunk

    def forward_one_chunk(fwd_chunk_id, args, kwargs=None, *rest, **more):
        feats = prefetcher.take(int(fwd_chunk_id))
        if feats is not None:
            feats = feats[0] if isinstance(feats, (list, tuple)) else feats
            if queue is not None:
                feats = cut_for_deferred_backward(feats, queue, int(fwd_chunk_id))
            kwargs = {**(kwargs or {}), "vision_embeds": feats}
        out = inner_forward(fwd_chunk_id, args, kwargs, *rest, **more)
        warm["done"] = True
        if prefetch:
            prefetcher.advance(int(fwd_chunk_id), prefetch)
        return out

    tower_stage.forward_one_chunk = forward_one_chunk  # type: ignore[method-assign]

    if bubble:
        pipeline_order = getattr(pp_schedule, "pipeline_order", None)
        if not pipeline_order:
            raise ValueError(
                "vit_bubble reads the schedule's action order, which only the looped "
                "schedules (Interleaved1F1B and kin) expose; use vit_prefetch with a "
                "single-stage schedule."
            )

        def plan_for_step():
            n = prefetcher._num_mbs
            if n == 0 or not warm["done"]:
                return None
            return plan_for_rank(
                pipeline_order[rank],
                rank=rank,
                vision_microbatches=n,
                cost_ratio=cost_ratio,
                upfront=min(pp_size, n),
                vision_stage=tower_stage.stage_index,
            )

        def encode_now(microbatches):
            for mb in microbatches:
                prefetcher.ensure_sync(mb)

        # Installed before the step wrapper below, so begin_step runs first
        # and the plan sees this step's micro-batch count.
        install_bubble_runtime(
            pp_schedule,
            plan_for_step=plan_for_step,
            encode_now=encode_now,
            upfront_encode=encode_now,
        )
        assert queue is not None
        install_backward_slots(pp_schedule, queue)

    original_step = pp_schedule.step

    def step(*args, **kwargs):
        prefetcher.begin_step(kwargs.get("kwarg_mbs"))
        if prefetch and warm["done"]:
            for mb in range(prefetch):
                prefetcher.ensure(mb)
        return original_step(*args, **kwargs)

    pp_schedule.step = step  # type: ignore[method-assign]

    if not bubble:
        logger.info(
            "DEP vision prefetch installed: depth=%d on stage %d",
            prefetch,
            tower_stage.stage_index,
        )
        return
    logger.info(
        "DEP bubble runtime installed on stage %d (cost ratio %.2f, max pending %d)",
        tower_stage.stage_index,
        cost_ratio,
        max_pending,
    )
