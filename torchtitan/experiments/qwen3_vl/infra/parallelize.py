# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parallelization utilities for Qwen3-VL.

This module applies PT-D parallelisms and various training techniques
(activation checkpointing, compile, FSDP) to the Qwen3-VL model.
"""

import math

import torch
import torch._inductor.config
import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    get_schedule_class,
    PipelineScheduleSingle,
)

from torchtitan.components.loss import LossFunction
from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.pipeline_parallel import (
    build_pipeline_schedule,
    generate_llm_fqn_per_model_part,
    pipeline_module_split,
)

from torchtitan.models.llama3.infra.parallelize import apply_ddp
from torchtitan.models.llama4.infra.parallelize import (
    apply_compile,
    apply_fsdp,
    apply_moe_ep_tp,
)
from torchtitan.models.qwen3.infra.parallelize import _op_sac_save_list
from torchtitan.protocols.train_spec import BaseModelArgs, ParallelizeFunction
from torchtitan.tools.logging import logger


def _apply_tp_to_decoder(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool,
):
    """Apply tensor parallelism to the decoder without SequenceParallel.

    Unlike Qwen3's apply_non_moe_tp which uses SequenceParallel (hidden states
    are Shard(1) between blocks), this keeps hidden states as Replicate. This is
    necessary for VLM because vision scatter and DeepStack operate on the full
    sequence with boolean masks that aren't DTensor-aware.

    The trade-off is slightly higher activation memory (full sequence on each
    rank instead of 1/TP), but it avoids costly all-gather/re-shard at every
    vision scatter and DeepStack layer.
    """
    # Parallelize embedding, norm, and output — no SequenceParallel
    # Build plan conditionally for PP stages where tok_embeddings/output may be None
    top_level_plan = {}
    if model.tok_embeddings is not None:
        top_level_plan["tok_embeddings"] = RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        )
    if model.output is not None:
        top_level_plan["output"] = ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
            use_local_output=not loss_parallel,
        )
    if top_level_plan:
        parallelize_module(model, tp_mesh, top_level_plan)

    if enable_float8_tensorwise_tp:
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
        )

        rowwise_parallel, colwise_parallel = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
        )
    else:
        rowwise_parallel, colwise_parallel = (
            RowwiseParallel,
            ColwiseParallel,
        )

    # Apply TP to every transformer block's linear layers.
    # Norms run redundantly on Replicate data (cheap).
    for transformer_block in model.layers.values():
        layer_plan = {
            # Wrap attention inputs so rope_cache becomes a Replicate DTensor,
            # needed because wq/wk/wv outputs are DTensors and apply_rotary_emb
            # multiplies them with cos/sin from rope_cache.
            "attention": PrepareModuleInput(
                input_layouts=(Replicate(), Replicate(), None, None),
                desired_input_layouts=(Replicate(), Replicate(), None, None),
            ),
            "attention.wq": colwise_parallel(use_local_output=False),
            "attention.wk": colwise_parallel(use_local_output=False),
            "attention.wv": colwise_parallel(use_local_output=False),
            "attention.q_norm": SequenceParallel(sequence_dim=2),
            "attention.k_norm": SequenceParallel(sequence_dim=2),
            "attention.wo": rowwise_parallel(output_layouts=Replicate()),
        }

        if not transformer_block.moe_enabled:
            layer_plan.update(
                {
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": rowwise_parallel(output_layouts=Replicate()),
                    "feed_forward.w3": colwise_parallel(),
                }
            )

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    if enable_async_tp:
        torch._inductor.config._micro_pipeline_tp = True

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}{'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the decoder (no SequenceParallel)"
    )


def _apply_tp_to_visual(
    visual: nn.Module,
    tp_mesh: DeviceMesh,
):
    """Apply tensor parallelism to the vision encoder.

    Uses TP without SequenceParallel: data between blocks stays Replicate
    (all ranks hold full hidden_states). This is simpler because norms and
    position embeddings don't need DTensor conversion, and vision encoder
    sequence lengths are short enough that redundant norm computation is cheap.
    Memory savings come from sharding the linear layer weights.
    """
    # TP plan for each vision transformer block
    layer_plan = {
        "attn.qkv": ColwiseParallel(),
        "attn.proj": RowwiseParallel(),
        "mlp.linear_fc1": ColwiseParallel(),
        "mlp.linear_fc2": RowwiseParallel(),
    }

    for transformer_block in visual.layers.values():
        parallelize_module(transformer_block, tp_mesh, layer_plan)

    # TP plan for patch mergers (main + deepstack)
    merger_plan = {
        "linear_fc1": ColwiseParallel(),
        "linear_fc2": RowwiseParallel(),
    }

    parallelize_module(visual.merger, tp_mesh, merger_plan)
    for merger in visual.deepstack_merger_list:
        parallelize_module(merger, tp_mesh, merger_plan)

    logger.info("Applied Tensor Parallelism to the vision encoder")


def _apply_fsdp_to_visual(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply FSDP to the vision encoder components individually.

    This must be called before the llama4 apply_fsdp so that vision encoder
    components are individually sharded before the final fully_shard(model).
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            reshard_after_forward = not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    if not hasattr(model, "visual") or model.visual is None:
        return

    # Shard patch embedding
    if hasattr(model.visual, "patch_embed"):
        fully_shard(
            model.visual.patch_embed,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # Shard each vision transformer layer
    for layer_id, transformer_block in model.visual.layers.items():
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # Shard merger if present
    if hasattr(model.visual, "merger"):
        fully_shard(
            model.visual.merger,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # Shard deepstack mergers
    if hasattr(model.visual, "deepstack_merger_list"):
        for merger in model.visual.deepstack_merger_list:
            fully_shard(
                merger,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

    # Shard pos_embed and any other remaining visual parameters by wrapping
    # the visual module itself. Sub-modules (patch_embed, layers, merger,
    # deepstack_merger_list) are already individually wrapped above.
    # This top-level wrap captures pos_embed.weight and rotary_pos_emb
    # so they are unsharded when visual.forward() is called — needed in PP
    # mode where prepare_vision_inputs() calls visual() outside
    # model.forward().
    fully_shard(
        model.visual,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )

    # Shard projector if present
    if hasattr(model, "projector") and model.projector is not None:
        fully_shard(
            model.projector,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )


def _apply_compile_to_visual(visual: nn.Module, compile_config):
    """Apply torch.compile to vision encoder transformer blocks."""
    for layer_id, transformer_block in visual.layers.named_children():
        transformer_block = torch.compile(
            transformer_block,
            backend=compile_config.backend,
            fullgraph=True,
        )
        visual.layers.register_module(layer_id, transformer_block)
    logger.info("Compiling each visual TransformerBlock with torch.compile")


class _VLMPipelineScheduleWrapper:
    """Wraps a PP schedule to preprocess vision data on stage 0 before microbatching.

    Vision data (pixel_values, grid_thw) has dim 0 = num_images, not batch_size,
    so it cannot be split into microbatches. This wrapper intercepts step() on
    the first-stage rank, runs vision preprocessing on the full batch to produce
    batch-aligned inputs_embeds and dense pp_deepstack_embeds, then passes those
    to the real PP schedule for normal microbatch splitting.
    """

    def __init__(self, pp_schedule, model_parts, has_first_stage):
        self.__dict__["_schedule"] = pp_schedule
        self.__dict__["_model_parts"] = model_parts
        self.__dict__["_has_first_stage"] = has_first_stage

    def __getattr__(self, name):
        return getattr(self._schedule, name)

    def __setattr__(self, name, value):
        setattr(self._schedule, name, value)

    def step(self, *args, **kwargs):
        if not self._has_first_stage or len(args) == 0:
            return self._schedule.step(*args, **kwargs)

        # On first call, trigger FSDP lazy init which is normally done by
        # the root module's forward pre-hook. We need it initialized before
        # prepare_vision_inputs() accesses FSDP-wrapped sub-modules.
        if not self.__dict__.get("_fsdp_initialized", False):
            self.__dict__["_fsdp_initialized"] = True
            from torch.distributed.fsdp._fully_shard._fsdp_state import (
                _get_module_fsdp_state,
            )

            for model_part in self._model_parts:
                state = _get_module_fsdp_state(model_part)
                if state is not None:
                    state._lazy_init()

        tokens = args[0]

        # Pop vision-specific kwargs (consumed here, not forwarded to schedule)
        pixel_values = kwargs.pop("pixel_values", None)
        grid_thw = kwargs.pop("grid_thw", None)
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        grid_thw_videos = kwargs.pop("grid_thw_videos", None)
        special_tokens = kwargs.pop("special_tokens", None)

        # Run vision preprocessing on the full batch
        model = self._model_parts[0]
        inputs_embeds, pp_deepstack_embeds = model.prepare_vision_inputs(
            tokens,
            pixel_values,
            pixel_values_videos,
            grid_thw,
            grid_thw_videos,
            special_tokens,
        )

        # Detach from vision graph so each PP microbatch can backward
        # independently. We retain the original tensors and register a
        # backward hook on the detached version to propagate gradients
        # back through the vision encoder after the PP schedule finishes.
        vision_embeds = inputs_embeds
        inputs_embeds = inputs_embeds.detach().requires_grad_()

        ds_originals = []
        if pp_deepstack_embeds is not None:
            ds_original = pp_deepstack_embeds
            pp_deepstack_embeds = pp_deepstack_embeds.detach().requires_grad_()
            ds_originals.append((ds_original, pp_deepstack_embeds))
            kwargs["pp_deepstack_embeds"] = pp_deepstack_embeds

        result = self._schedule.step(inputs_embeds, *args[1:], **kwargs)

        # Backward through vision encoder graph using accumulated gradients
        # from the PP schedule.
        vision_grads = []
        vision_tensors = []
        if inputs_embeds.grad is not None:
            vision_tensors.append(vision_embeds)
            vision_grads.append(inputs_embeds.grad)
        for ds_original, ds_detached in ds_originals:
            if ds_detached.grad is not None:
                vision_tensors.append(ds_original)
                vision_grads.append(ds_detached.grad)
        if vision_tensors:
            torch.autograd.backward(vision_tensors, vision_grads)

        return result


def pipeline_qwen3_vl(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: torch.device,
    model_args: BaseModelArgs,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    """Pipeline-parallelize Qwen3-VL.

    Reuses the standard LLM stage assignment, then adds ``"visual"`` to the
    first stage so the vision encoder + DeepStack layers (which only touch the
    first few decoder layers) are co-located on stage 0.

    Vision data is preprocessed outside the PP schedule via
    _VLMPipelineScheduleWrapper so that batch-aligned tensors (inputs_embeds,
    pp_deepstack_embeds) can be microbatch-split normally.
    """
    pp_mesh = parallel_dims.get_mesh("pp")

    # Determine number of virtual stages (mirrors pipeline_llm logic)
    schedule_class = get_schedule_class(
        job_config.parallelism.pipeline_parallel_schedule
    )
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
    layers_per_stage = job_config.parallelism.pipeline_parallel_layers_per_stage

    if hasattr(model_args, "n_layers"):
        num_layers = model_args.n_layers
    else:
        raise ValueError("Model does not have n_layers attribute.")

    input_weight = job_config.parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = job_config.parallelism.pipeline_parallel_last_stage_less_layers

    if layers_per_stage is not None:
        num_virtual_stages = math.ceil(
            (num_layers + input_weight + output_weight) / layers_per_stage
        )

        if num_virtual_stages % parallel_dims.pp != 0:
            raise ValueError(
                f"Number of virtual stages ({num_virtual_stages}) must be divisible by "
                f"pipeline parallel size ({parallel_dims.pp})."
            )

        stages_per_rank = num_virtual_stages // parallel_dims.pp

        if is_single_stage_schedule and stages_per_rank != 1:
            raise ValueError(
                f"Single stage schedule requires exactly 1 stage per rank, "
                f"but got {stages_per_rank}."
            )
        if not is_single_stage_schedule and stages_per_rank < 2:
            raise ValueError(
                f"Multi-stage schedule requires at least 2 stages per rank, "
                f"but got {stages_per_rank}."
            )
    else:
        stages_per_rank = 1 if is_single_stage_schedule else 2
        num_virtual_stages = parallel_dims.pp * stages_per_rank

    # Generate standard LLM FQNs, then prepend "visual" to stage 0
    module_names_per_stage = job_config.parallelism.module_fqns_per_model_part
    if module_names_per_stage is None:
        module_names_per_stage = generate_llm_fqn_per_model_part(
            num_virtual_stages, num_layers, input_weight, output_weight
        )
    # Add visual encoder to the first stage
    module_names_per_stage[0].insert(0, "visual")

    for i, stage_ms in enumerate(module_names_per_stage):
        logger.debug(f"Stage {i}: {stage_ms}")

    stages, model_parts = pipeline_module_split(
        model,
        pp_mesh,
        job_config.parallelism.pipeline_parallel_schedule,
        device,
        module_names_per_stage,
    )

    # Apply SPMD parallelism to each model part
    for i, m in enumerate(model_parts):
        m = parallelize_fn(m, parallel_dims, job_config)
        model_parts[i] = m
        stages[i].submod = m

    pp_schedule = build_pipeline_schedule(job_config, stages, loss_fn)

    has_first_stage = any(stage.is_first for stage in stages)
    has_last_stage = any(stage.is_last for stage in stages)

    # Wrap schedule to preprocess vision data outside microbatching
    pp_schedule = _VLMPipelineScheduleWrapper(pp_schedule, model_parts, has_first_stage)

    return pp_schedule, model_parts, has_first_stage, has_last_stage


def parallelize_qwen3_vl(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the Qwen3-VL model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    assert isinstance(model.visual, nn.Module) or model.visual is None, (
        "model.visual must be an nn.Module or None (when PP splits the model)"
    )

    # Validate sequence length divisibility
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    # Check attention type compatibility
    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "Context Parallel is not yet supported for Qwen3-VL."
        )

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )

    if parallel_dims.tp_enabled:
        if (
            job_config.parallelism.enable_async_tensor_parallel
            and not model_compile_enabled
        ):
            raise RuntimeError("Async TP requires torch.compile")

        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.quantize.linear.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        # For now, float8 all-gather with TP is only supported for tensorwise
        # float8 scaling recipes. For rowwise recipes, we use regular TP and
        # all-gather happens in high precision.
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        tp_mesh = parallel_dims.get_mesh("tp")

        # Apply TP to vision encoder
        if model.visual is not None:
            _apply_tp_to_visual(model.visual, tp_mesh)

        # Apply TP to decoder without SequenceParallel.
        # VLM needs full-sequence access between decoder blocks for vision
        # scatter and DeepStack, so hidden states stay Replicate.
        _apply_tp_to_decoder(
            model,
            tp_mesh,
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            enable_async_tp=job_config.parallelism.enable_async_tensor_parallel,
        )

    # Apply MoE expert parallelism to decoder layers
    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
        )

    # Apply activation checkpointing
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            op_sac_save_list=_op_sac_save_list,
        )
        if model.visual is not None:
            apply_ac(model.visual, job_config.activation_checkpoint)

    # Apply torch.compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model, job_config.compile, parallel_dims.ep_enabled)
    if job_config.compile.enable:
        if model.visual is not None:
            _apply_compile_to_visual(model.visual, job_config.compile)

    # Apply FSDP or HSDP
    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

        # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

        # FSDP the vision encoder components individually for memory efficiency
        _apply_fsdp_to_visual(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
        )

        # FSDP the decoder with MoE-aware sharding (reuses llama4 apply_fsdp)
        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the Qwen3-VL model")
        else:
            logger.info("Applied FSDP to the Qwen3-VL model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the Qwen3-VL model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the Qwen3-VL model")

    elif parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.get_mesh("dp_replicate")
        if dp_mesh is not None and dp_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            dp_mesh,
            enable_compile=job_config.compile.enable,
        )

    return model
