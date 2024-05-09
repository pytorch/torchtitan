# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# this file applies the PTD parallelisms and various training techniques to the
# llama model, i.e. activation checkpointing, etc.

from collections import defaultdict
from typing import Tuple

import torch
import torch.nn as nn

from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._composable.replicate import replicate
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torch.utils.checkpoint import _pt2_selective_checkpoint_context_fn_gen, checkpoint

from torchtitan.config_manager import JobConfig
from torchtitan.logging_utils import logger


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

        def _get_custom_policy(meta):
            def _custom_policy(mode, func, *args, **kwargs):
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except every second mm
                return func in no_recompute_list and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                )

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return _pt2_selective_checkpoint_context_fn_gen(_get_custom_policy(meta))

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
        every_x_layer = int(config.selective_ac_option)
        assert (
            every_x_layer >= 0
        ), f"selective layer AC policy (every_x_layer) expects a positive integer, received {every_x_layer}"

        checkpoint_wrapper.__dict__.setdefault("_count", 0)

        checkpoint_wrapper._count += 1
        if not every_x_layer or checkpoint_wrapper._count % every_x_layer == 0:
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
) -> Tuple[RowwiseParallel, ColwiseParallel]:
    """Get the parallel strategy for the transformer model.

    This function handles the special case of using float8 with tensor parallelism.
    """
    if job_config.training.fp8_linear == "dynamic":
        from float8_experimental.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
        )

        return Float8RowwiseParallel, Float8ColwiseParallel
    return RowwiseParallel, ColwiseParallel


def maybe_enable_activation_checkpoint(
    model: nn.Module, job_config: JobConfig
) -> nn.Module:
    config = job_config.activation_checkpoint
    ac_mode = config.mode
    if ac_mode in ("full", "selective"):
        for layer_id, transformer_block in enumerate(model.layers):
            model.layers[layer_id] = checkpoint_wrapper(transformer_block, config)
        logger.info(f"Applied {ac_mode} activation checkpointing to the model")

    return model


def enable_fsdp(model: nn.Module, dp_mesh, job_config: JobConfig) -> nn.Module:
    # TODO: Expose `reduce_dtype` as a config option.
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    for layer_id, transformer_block in enumerate(model.layers):
        # As an optimization, do not reshard after forward for the last
        # transformer block since FSDP would prefetch it immediately
        reshard_after_forward = layer_id < len(model.layers) - 1
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
        model.layers[layer_id] = transformer_block
    model = fully_shard(model, **fsdp_config)
    logger.info("Applied FSDP to the model")

    return model


def enable_ddp(model: nn.Module, dp_mesh, job_config: JobConfig) -> nn.Module:
    if job_config.training.compile:
        if job_config.training.compiled_autograd:
            torch._dynamo.config.optimize_ddp = "python_reducer"
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"
    model = replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)
    logger.info("Applied DDP to the model")

    return model


def parallelize_llama(
    model: nn.Module, world_mesh, parallel_dims, job_config: JobConfig
) -> nn.Module:
    """
    Apply parallelisms and activation checkpointing to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    if parallel_dims.pp_enabled:
        raise NotImplementedError("PP not implemented yet.")

    if parallel_dims.tp_enabled:
        if job_config.model.norm_type == "fused_rmsnorm":
            raise NotImplementedError(
                "fused_rmsnorm not yet compatible with TP. Please use layernorm or rmsnorm."
            )
        if parallel_dims.dp_replicate_enabled:
            raise NotImplementedError("DDP/HSDP + TP are not supported yet.")

        tp_mesh = world_mesh["tp"]
        row_parallel_strategy, col_parallel_strategy = get_tp_parallel_strategy(
            job_config
        )
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
                    output_layouts=(Shard(-1) if loss_parallel else Replicate()),
                    use_local_output=not loss_parallel,
                ),
                "norm": SequenceParallel(),
            },
        )

        # Apply tensor + sequence parallelism to every transformer block
        for layer_id, transformer_block in enumerate(model.layers):
            layer_plan = {
                "attention": PrepareModuleInput(
                    input_layouts=(Shard(1), None),
                    desired_input_layouts=(Replicate(), None),
                ),
                "attention.wq": col_parallel_strategy(),
                "attention.wk": col_parallel_strategy(),
                "attention.wv": col_parallel_strategy(),
                "attention.wo": row_parallel_strategy(output_layouts=Shard(1)),
                "attention_norm": SequenceParallel(),
                "feed_forward": PrepareModuleInput(
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

        logger.info("Applied Tensor Parallelism to the model")

    model = maybe_enable_activation_checkpoint(model, job_config)
    if parallel_dims.dp_enabled:
        if parallel_dims.dp_replicate_enabled:
            raise NotImplementedError("HSDP is not supported yet.")
        dp_mesh = world_mesh["dp"] if world_mesh.ndim > 1 else world_mesh
        assert dp_mesh.mesh_dim_names == ("dp",), dp_mesh.mesh_dim_names
        model = enable_fsdp(model, dp_mesh, job_config)
    elif parallel_dims.dp_replicate_enabled:
        dp_mesh = world_mesh["dp_replicate"] if world_mesh.ndim > 1 else world_mesh
        model = enable_ddp(model, dp_mesh, job_config)

    return model
