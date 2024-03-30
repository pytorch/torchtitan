# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# this file applies the PTD parallelisms and various training techniques to the
# llama model, i.e. activation checkpointing, etc.

from collections import defaultdict
from typing import Tuple

import torch

from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)

from torch.utils.checkpoint import _pt2_selective_checkpoint_context_fn_gen, checkpoint

from torchtrain.config_manager import JobConfig
from torchtrain.logging_utils import logger


# for selective AC
no_recompute_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.c10d_functional.reduce_scatter_tensor.default,
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


def parallelize_llama(model, world_mesh, parallel_dims, job_config: JobConfig):
    """
    Apply parallelisms and activation checkpointing to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    if parallel_dims.pp_enabled:
        raise NotImplementedError("PP not implemented yet.")

    if parallel_dims.tp_enabled:
        tp_mesh = world_mesh["tp"]
        row_parallel_strategy, col_parallel_strategy = get_tp_parallel_strategy(
            job_config
        )

        # 1. Parallelize the first embedding and the last linear proj layer
        # 2. Prepare the freq_cis in rotary embedding as dtensor
        # 3. Parallelize the root norm layer over the sequence dim
        # 4. Shard the first transformer block's inputs
        model = parallelize_module(
            model,
            tp_mesh,
            {
                "embeddings.tok_embeddings": RowwiseParallel(
                    input_layouts=Replicate(),
                ),
                "embeddings": PrepareModuleOutput(
                    output_layouts=(None, Replicate()),
                    desired_output_layouts=(None, Replicate()),
                    use_local_output=False,
                ),
                "output": col_parallel_strategy(
                    input_layouts=Shard(0),
                    output_layouts=(
                        Shard(-1)
                        if parallel_dims.loss_parallel_enabled
                        else Replicate()
                    ),
                    use_local_output=not parallel_dims.loss_parallel_enabled,
                ),
                "norm": SequenceParallel(sequence_dim=0),
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(), None),
                    desired_input_layouts=(Shard(0), None),
                    use_local_output=True,
                ),
            },
        )

        # Apply tensor + sequence parallelism to every transformer block
        for layer_id, transformer_block in enumerate(model.layers):
            layer_plan = {
                "attention": PrepareModuleInput(
                    input_layouts=(Shard(0), None),
                    desired_input_layouts=(Replicate(), None),
                ),
                "attention.wq": col_parallel_strategy(use_local_output=False),
                "attention.wk": col_parallel_strategy(use_local_output=False),
                "attention.wv": col_parallel_strategy(use_local_output=False),
                "attention.wo": row_parallel_strategy(output_layouts=Shard(0)),
                "attention_norm": SequenceParallel(sequence_dim=0),
                "feed_forward": PrepareModuleInput(
                    input_layouts=(Shard(0),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": col_parallel_strategy(),
                "feed_forward.w2": row_parallel_strategy(output_layouts=Shard(0)),
                "feed_forward.w3": col_parallel_strategy(),
                "ffn_norm": SequenceParallel(sequence_dim=0),
            }

            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

        logger.info("Applied Tensor Parallelism to the model")

    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"] if world_mesh.ndim > 1 else world_mesh
        assert dp_mesh.mesh_dim_names == ("dp",), dp_mesh.mesh_dim_names
        # TODO: Expose `reduce_dtype` as a config option.
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        ac_mode = job_config.activation_checkpoint.mode
        fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
        for layer_id, transformer_block in enumerate(model.layers):
            if job_config.activation_checkpoint.mode in ("full", "selective"):
                transformer_block = checkpoint_wrapper(
                    transformer_block, job_config.activation_checkpoint
                )
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
        if ac_mode in ("full", "selective"):
            logger.info(f"Applied {ac_mode} activation checkpointing to the model")
        logger.info("Applied FSDP to the model")

    return model
