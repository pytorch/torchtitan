# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import torch.nn as nn

from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)

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
)

from torchtitan.distributed.pipeline_parallel import (
    _generate_llm_fqn_per_model_part,
    get_schedule_class,
    pipeline_llm,
)
from torchtitan.models.kimi_k3.layout import (
    gather_layer_to_stage,
    infer_block_layout_tables_from_stages,
)
from torchtitan.models.kimi_k3.pipeline_stage import AttnResPipelineStage, RankStore
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
    if parallelism.spmd_backend != "partial_dtensor":
        raise NotImplementedError(
            "Kimi K3 FSDP2 currently supports the partial_dtensor SPMD backend "
            "only; the config registry pins it."
        )
    if compile_config.enable and "model" in compile_config.components:
        raise NotImplementedError("Kimi K3 does not support model compilation yet.")

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
    # The routed experts shard on their own data-parallel mesh, which excludes
    # the expert axis; the same shape deepseek_v3 resolves.
    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh = parallel_dims.get_optional_mesh(
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )

    assert isinstance(model, KimiK3Model)
    if parallel_dims.ep_enabled:
        # model_registry's moe_comm_backend picks the dispatcher: standard
        # (default), deepep and minimal_async_ep run on this model; hybridep
        # needs GB200-class hardware.
        model.parallelize(parallel_dims)

    if ac_config is not None:
        ac_policy = ac_config.build(dump_folder=dump_folder)
        ac_policy.apply(model)
        if model.vision_encoder is not None:
            ac_policy.apply(model.vision_encoder)

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
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    return model


_KIMI_ATTN_RES_LAST_STAGE_FQNS = ("output_res_proj", "output_res_norm")


def kimi_k3_module_fqns_per_model_part(
    model: nn.Module,
    *,
    model_config,
    parallelism,
    pp: int,
    vit_dep: bool = False,
) -> list[list[str]] | None:
    """The pipeline split of a Kimi K3 model, built from its config.

    Core's layer distribution (``_generate_llm_fqn_per_model_part``) places the
    embedding, the layers and the head; on top of it this model needs the
    AttnRes aggregation modules (``output_res_proj``, ``output_res_norm``) on
    the stage that holds ``lm_head``, since the final block attention runs
    there, and the vision tower on the stage that holds the embedding, since
    vision features are spliced into the embeddings and nothing vision-side
    crosses a stage boundary. With ``vit_dep`` the tower and the embedding
    take the first stage alone and the layers spread over the remaining
    stages (report sec 5.2.3: the encoder's compute leaves the text stages'
    critical path; the stage count the schedule sees is unchanged, so the
    tower stage comes out of the text stages' budget). Returns None when the
    split does not apply (no pipeline parallelism, or a config without
    layers); the caller keeps whatever split the user configured.
    """
    if pp <= 1 or model_config is None:
        return None
    layers = getattr(model_config, "layers", None)
    if layers is None:
        return None
    num_layers = len(layers)
    input_weight = parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = parallelism.pipeline_parallel_last_stage_less_layers
    layers_per_stage = parallelism.pipeline_parallel_layers_per_stage
    schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
    if issubclass(schedule_class, PipelineScheduleSingle):
        stages_per_rank = 1
    elif layers_per_stage is None:
        stages_per_rank = 2
    else:
        # The multiple of pp nearest to units / layers_per_stage: a layer count
        # no shape divides (the 93-layer model's) still splits, with stages
        # differing by a layer, where core's ceiling would refuse it.
        units = num_layers + input_weight + output_weight
        stages_per_rank = max(2, round(units / layers_per_stage / pp))
    num_virtual_stages = pp * stages_per_rank
    has_tower = getattr(model, "vision_encoder", None) is not None
    if vit_dep:
        if not has_tower:
            raise ValueError(
                "vit_dep gives the vision tower a pipeline stage of its own, "
                "and this model has no vision encoder."
            )
        if num_virtual_stages < 2:
            raise ValueError(
                "vit_dep needs at least two pipeline stages: one for the "
                "tower and the embedding, one for the layers."
            )
        # The embedding rides with the tower: the splice needs the token ids,
        # which only the first stage receives. The text split then carries no
        # embedding (input_weight 0) and loses the one core placed.
        text = _generate_llm_fqn_per_model_part(
            num_virtual_stages - 1, num_layers, 0, output_weight
        )
        text = [[n for n in stage if n != "tok_embeddings"] for stage in text]
        fqns = [["tok_embeddings", "vision_encoder"]] + text
    else:
        fqns = _generate_llm_fqn_per_model_part(
            num_virtual_stages, num_layers, input_weight, output_weight
        )
    # Core spells the head ``output``; this model calls it ``lm_head``. Any
    # FQN matching no child makes core set that child to None on every stage.
    fqns = [["lm_head" if n == "output" else n for n in stage] for stage in fqns]
    tail = [n for n in _KIMI_ATTN_RES_LAST_STAGE_FQNS if hasattr(model, n)]
    fqns[-1].extend(tail)
    if has_tower and not vit_dep:
        embed_stage = next(
            (stage for stage in fqns if "tok_embeddings" in stage), fqns[0]
        )
        embed_stage.append("vision_encoder")
    return fqns


def _schedule_stages(schedule: _PipelineSchedule) -> list[AttnResPipelineStage]:
    """The stages a schedule holds on this rank."""
    if isinstance(schedule, PipelineScheduleSingle):
        stages = [schedule._stage]
    elif isinstance(schedule, PipelineScheduleMulti):
        stages = list(schedule._stages)
    else:
        raise RuntimeError(
            f"Unexpected pipeline schedule class {type(schedule).__name__}."
        )
    assert all(isinstance(s, AttnResPipelineStage) for s in stages)
    return cast(list[AttnResPipelineStage], stages)


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
    """``pipelining_fn`` for Kimi K3.

    Splits the model with this model's names, builds the schedule on
    :class:`AttnResPipelineStage`, then gives every stage the routing tables
    computed from the split the trainer applied: the layer-to-stage map is one
    all-gather over the pipeline group, and the stage-to-rank map is the
    schedule's own.

    ``attn_res_cache`` is a property of the transport, not of the model: with
    it, a hop carries only the blocks the receiving rank has not seen and the
    rank's store serves its later stages; without it, every hop carries the
    whole stack. A recipe turns it off with
    ``functools.partial(pipeline_kimi_k3, attn_res_cache=False)`` as the
    ``pipelining_fn``. The two transports sum the block gradients in a
    different order, so they are not bitwise against each other. Every rank
    must resolve it identically: a rank routing differently from its peers
    hangs the first hop with nothing pointing at the cause.
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
    import dataclasses

    parallelism = kwargs["parallelism"]
    if parallelism.module_fqns_per_model_part is None:
        fqns = kimi_k3_module_fqns_per_model_part(
            model,
            model_config=kwargs.get("model_config"),
            parallelism=parallelism,
            pp=kwargs["parallel_dims"].pp,
            vit_dep=vit_dep,
        )
        if fqns is not None:
            # Core validates layers_per_stage with a ceiling the unit count has
            # to divide; the split above already honoured it, so core gets the
            # split alone.
            kwargs["parallelism"] = dataclasses.replace(
                parallelism,
                module_fqns_per_model_part=fqns,
                pipeline_parallel_layers_per_stage=None,
            )
    pp_schedule, model_parts, has_first_stage, has_last_stage = pipeline_llm(
        model, stage_class=AttnResPipelineStage, **kwargs
    )

    stages = _schedule_stages(pp_schedule)
    model_config = kwargs["model_config"]
    layer_cfgs = model_config.layers
    n_layers = len(layer_cfgs)
    layers_per_block = layer_cfgs[0].attn_res_block_size
    num_blocks = -(-n_layers // layers_per_block)
    # The split is whatever the trainer applied, uneven stages included: a
    # rank sees only its own stages, so the layer-to-stage map is one
    # all-gather over the pipeline group; the schedule owns stage-to-rank.
    layer_to_stage = gather_layer_to_stage(stages, stages[0].group)
    layout = infer_block_layout_tables_from_stages(
        stages,
        stage_to_rank=dict(stages[0].stage_index_to_group_rank),
        num_blocks=num_blocks,
        n_layers=n_layers,
        layers_per_block=layers_per_block,
        layer_to_stage=layer_to_stage,
        cache=attn_res_cache,
    )
    store = RankStore()
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
