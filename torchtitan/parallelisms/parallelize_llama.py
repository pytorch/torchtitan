# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# this file applies the PTD parallelisms and various training techniques to the
# llama model, i.e. activation checkpointing, etc.

import copy
from collections import defaultdict
from typing import Dict, Tuple

import torch

from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.pipelining import pipeline, PipelineStage, SplitPoint
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torch.utils.checkpoint import checkpoint

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.logging_utils import logger
from torchtitan.parallelisms.pipelining_utils import stage_ids_this_rank

# for selective AC
no_recompute_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
}


# Uses PTD FSDP AC wrapper
# currently selective per op and per layer checkpointing are supported
def checkpoint_wrapper(module, config):
    if config.mode == "selective" and config.selective_ac_option == "op":
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        def _get_custom_policy(meta):
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except every second mm
                to_save = func in no_recompute_list and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                )
                return (
                    CheckpointPolicy.MUST_SAVE
                    if to_save
                    else CheckpointPolicy.PREFER_RECOMPUTE
                )

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            context_fn=selective_checkpointing_context_fn,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    elif config.mode == "full":
        return ptd_checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    elif config.mode == "selective" and config.selective_ac_option.isdigit():
        """enables selective checkpointing of candidate layers.
        Usage:
        'selective_ac_option' with a positive 'int' value in config controls which layers to checkpoint.
        1 == checkpointing every one (all).
        2 == checkpoint every 2nd one
        """
        ac_freq = int(config.selective_ac_option)
        assert (
            ac_freq >= 0
        ), f"selective layer AC policy (ac_freq) expects a positive integer, received {ac_freq}"

        checkpoint_wrapper.__dict__.setdefault("_count", 0)

        checkpoint_wrapper._count += 1
        if not ac_freq or checkpoint_wrapper._count % ac_freq == 0:
            return ptd_checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                checkpoint_fn=checkpoint,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        # skip activation checkpointing and store activations for this layer
        else:
            return module

    else:
        raise NotImplementedError(
            "Unknown AC type or AC config. Only selective op and selective layer ac implemented currently."
        )


def get_tp_parallel_strategy(
    job_config: JobConfig,
) -> Tuple[RowwiseParallel, ColwiseParallel, PrepareModuleInput]:
    """Get the parallel strategy for the transformer model.

    This function handles the special case of using float8 with tensor parallelism.
    """
    if job_config.training.fp8_linear == "dynamic":
        from float8_experimental.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        return Float8RowwiseParallel, Float8ColwiseParallel, PrepareFloat8ModuleInput
    return RowwiseParallel, ColwiseParallel, PrepareModuleInput


def pipeline_llama(
    model, world_mesh, parallel_dims, job_config: JobConfig, device, model_config: Dict
):
    if job_config.experimental.pipeline_parallel_split_mode == "manual":
        return pipeline_llama_manual(
            model, world_mesh, parallel_dims, job_config, device, model_config
        )
    elif job_config.experimental.pipeline_parallel_split_mode == "tracer":
        return pipeline_llama_tracer(
            model, world_mesh, parallel_dims, job_config, device, model_config
        )
    else:
        raise NotImplementedError(
            f"{job_config.experimental.pipeline_parallel_split_mode} is not a valid split mode"
        )


def _llama_trace_input(job_config, model_config, device="meta"):
    """Get meta tensors with the right input shapes used for tracing"""
    tokens_shape = (job_config.training.batch_size, job_config.training.seq_len)
    tokens = torch.randint(
        model_config.vocab_size, tokens_shape, dtype=torch.int64, device=device
    )
    return (tokens,)


def _mixed_precision_dtype(
    job_config: JobConfig, parallel_dims, default: torch.dtype = torch.float32
) -> torch.dtype:
    """Get the mixed precision dtype if fsdp is enabled, otherwise return the default"""
    mp_arg = job_config.training.mixed_precision_param
    return TORCH_DTYPE_MAP[mp_arg] if parallel_dims.dp_enabled else default


def pipeline_llama_manual(
    whole_model,
    world_mesh,
    parallel_dims,
    job_config: JobConfig,
    device,
    model_config: Dict,
):
    """
    This API extracts one torch.nn.Module objects for the part of the model configured to run inside this stage.

    It wraps the model chunk in a ManualPipelineStage object and returns both the stage and model objects.

    The stage object is used to create a pipeline schedule, and the model object can be used for applying SPMD
    parallelism.
    """
    pp_mesh = world_mesh["pp"]
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    microbatches = (
        job_config.experimental.pipeline_parallel_microbatches or parallel_dims.pp
    )
    splits = job_config.experimental.pipeline_parallel_split_points

    def _build_stage(stage_idx, start_layer, stop_layer, is_first=False, is_last=False):
        model = copy.deepcopy(whole_model)
        if not is_first:
            model.tok_embeddings = None

        drop_layers = start_layer is not None
        for name in list(model.layers.keys()):
            # we keep layers in a contiguous region between start (inclusive) and stop (exclusive)
            if f"layers.{name}" == start_layer:
                drop_layers = False
            if f"layers.{name}" == stop_layer:
                drop_layers = True
            if drop_layers:
                del model.layers[name]

        if not is_last:
            model.norm = None
            model.output = None

        # TODO(whc) once ManualPipelineStage supports lazy shape inference, we can leave model on meta device longer and
        # get rid of the input shape hardcoded here. For now, it should not be a big deal since we only materialize the
        # layers of the model that map to this stage, not the whole model.
        mp_dtype = _mixed_precision_dtype(job_config, parallel_dims)
        batch_size = job_config.training.batch_size
        local_seq_len = int(job_config.training.seq_len // parallel_dims.tp)
        layers_io_shape = (batch_size, local_seq_len, model_config.dim)
        output_layer_shape = (
            batch_size,
            job_config.training.seq_len,
            model_config.vocab_size,
        )
        if is_first:
            (input,) = _llama_trace_input(job_config, model_config, device=device)
        else:
            # later layers (assume all start w/ a transformer layer)
            input = torch.rand(layers_io_shape, dtype=mp_dtype, device=device)

        if is_last:
            output = torch.rand(output_layer_shape, dtype=torch.float32, device=device)
        else:
            # earlier layers (assume all end in a transformer layer)
            output = torch.rand(layers_io_shape, dtype=mp_dtype, device=device)

        model.to_empty(device=device)
        stage = PipelineStage(
            model,
            stage_idx,
            num_stages,
            device,
            input_args=input.chunk(microbatches)[0],
            output_args=output.chunk(microbatches)[0],
            group=pp_mesh.get_group("pp"),
        )
        return stage, model

    num_stages = len(splits) + 1
    stage_idx = pp_rank

    stages = []
    models = []
    for stage_idx in stage_ids_this_rank(pp_rank, pp_size, num_stages, style="loop"):
        start_layer = splits[stage_idx - 1] if stage_idx > 0 else None
        stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None
        stage, model_chunk = _build_stage(
            stage_idx,
            start_layer,
            stop_layer,
            is_first=stage_idx == 0,
            is_last=stage_idx == num_stages - 1,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx}"
            f" with start_layer {start_layer}, stop_layer {stop_layer}: model chunk \n{model_chunk}"
        )
        stages.append(stage)
        models.append(model_chunk)
    return stages, models


def pipeline_llama_tracer(
    model, world_mesh, parallel_dims, job_config: JobConfig, device, model_config: Dict
):
    if job_config.model.norm_type == "fused_rmsnorm":
        # TODO(whc) - torch._dynamo.exc.Unsupported: Illegal getattr invocation stride in strict mode
        # coming from ` if dy.stride(-1) != 1:` in fused_rmsnorm
        raise NotImplementedError(
            "fused_rmsnorm not yet compatible with Pipeline Tracer (strides error). Please use layernorm or rmsnorm."
        )

    if _mixed_precision_dtype(job_config, parallel_dims) == torch.bfloat16:
        raise NotImplementedError(
            "pipeline tracer doesn't work with fsdp mixed precision currently. "
            "To work around, edit fsdp mixed precision config to use fp32."
        )

    pp_mesh = world_mesh["pp"]
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    microbatches = (
        job_config.experimental.pipeline_parallel_microbatches or parallel_dims.pp
    )
    (input,) = _llama_trace_input(job_config, model_config, device=device)
    stage_idx = pp_rank
    split_spec = {
        layer_name: SplitPoint.BEGINNING
        for layer_name in job_config.experimental.pipeline_parallel_split_points
    }
    num_stages = len(split_spec) + 1
    pipe = pipeline(
        model,
        mb_args=(input.chunk(microbatches)[0],),
        split_spec=split_spec,
    )

    stages = []
    models = []
    for stage_idx in stage_ids_this_rank(pp_rank, pp_size, num_stages, style="loop"):
        models.append(pipe.get_stage_module(stage_idx))
        stages.append(
            pipe.build_stage(
                stage_idx,
                device=device,
                group=pp_mesh.get_group(),
            )
        )
    return (stages, models)


def apply_tp(model, world_mesh, parallel_dims, job_config: JobConfig):
    """
    Apply tensor parallelism.
    """

    tp_mesh = world_mesh["tp"]
    (
        row_parallel_strategy,
        col_parallel_strategy,
        prepare_module_input,
    ) = get_tp_parallel_strategy(job_config)
    loss_parallel = parallel_dims.loss_parallel_enabled

    # 1. Parallelize the first embedding and the last linear proj layer
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Shard the first transformer block's inputs
    model = parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "output": col_parallel_strategy(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
            "norm": SequenceParallel(),
        },
    )

    # Apply tensor + sequence parallelism to every transformer block
    for layer_id, transformer_block in model.layers.items():
        layer_plan = {
            "attention": prepare_module_input(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attention.wq": col_parallel_strategy(),
            "attention.wk": col_parallel_strategy(),
            "attention.wv": col_parallel_strategy(),
            "attention.wo": row_parallel_strategy(output_layouts=Shard(1)),
            "attention_norm": SequenceParallel(),
            "feed_forward": prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": col_parallel_strategy(),
            "feed_forward.w2": row_parallel_strategy(output_layouts=Shard(1)),
            "feed_forward.w3": col_parallel_strategy(),
            "ffn_norm": SequenceParallel(),
        }

        # Adjust attention module to use the local number of heads
        attn_layer = transformer_block.attention
        attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    if job_config.experimental.enable_async_tensor_parallel:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info("Applied Tensor Parallelism to the model")
    return model


def apply_ac(model, job_config: JobConfig):
    """
    Apply activation checkpointing to the model.
    """

    ac_config = job_config.activation_checkpoint

    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = checkpoint_wrapper(transformer_block, ac_config)
        model.layers.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")
    return model


def apply_compile(model, job_config: JobConfig):
    """
    Apply torch.compile to the model.
    """

    if job_config.model.norm_type == "fused_rmsnorm":
        raise NotImplementedError(
            "fused_rmsnorm not yet compatible with torch.compile. Please use layernorm or rmsnorm."
        )

    for layer_id, transformer_block in model.layers.named_children():
        # turn on per-transformer block compile after AC wrapping and before FSDP
        # TODO: dynamic shape have some issues so we turn it off for now.
        # TODO: inline inbuilt nn modules does not work yet, enable it to accelarate
        # compile time.
        # torch._dynamo.config.inline_inbuilt_nn_modules = True
        transformer_block = torch.compile(transformer_block, dynamic=False)
        model.layers.register_module(layer_id, transformer_block)

    ac_config = job_config.activation_checkpoint
    if ac_config.mode == "selective" and ac_config.selective_ac_option == "op":
        # some temp flags for torch.compile enablement + SAC
        torch._dynamo.config._experimental_support_context_fn_in_torch_utils_checkpoint = (
            True
        )

    logger.info("Compiled each TransformerBlock with torch.compile")
    return model


def apply_dp(model, world_mesh, parallel_dims, job_config: JobConfig):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """

    dp_mesh = world_mesh["dp"] if world_mesh.ndim > 1 else world_mesh
    assert dp_mesh.mesh_dim_names == ("dp",), dp_mesh.mesh_dim_names

    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    for layer_id, transformer_block in model.layers.items():
        # As an optimization, do not reshard after forward for the last
        # transformer block since FSDP would prefetch it immediately.
        # When using Pipeline Parallelism, generally zero-2 is best so as to avoid repeated reshardings
        # per microbatch.
        reshard_after_forward = (
            int(layer_id) < len(model.layers) - 1 and not parallel_dims.pp_enabled
        )
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
        model.layers[layer_id] = transformer_block

    model = fully_shard(
        model, **fsdp_config, reshard_after_forward=not parallel_dims.pp_enabled
    )

    logger.info("Applied FSDP to the model")
    return model


def parallelize_llama(model, world_mesh, parallel_dims, job_config: JobConfig):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    if parallel_dims.tp_enabled:
        model = apply_tp(model, world_mesh, parallel_dims, job_config)

    if job_config.activation_checkpoint.mode != "none":
        model = apply_ac(model, job_config)

    if job_config.training.compile:
        model = apply_compile(model, job_config)

    if parallel_dims.dp_enabled:
        model = apply_dp(model, world_mesh, parallel_dims, job_config)

    return model
