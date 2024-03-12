# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# this file applies the PTD parallelisms and various training techniques to the
# llama model, i.e. activation checkpoint, etc.

import logging
from collections import defaultdict

import torch
from torch.distributed._tensor import (
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
)
from torchtrain.config_manager import JobConfig
from torchtrain.logging_utils import rank0_log
from torchtrain.meta_init import meta_to_real_init_fn

logger = logging.getLogger(__name__)


def distribute_rmsnorm(module, device_mesh):
    # temp sharding API until PTD API is added
    def prepare_input_fn(mod, inputs, device_mesh):
        if isinstance(inputs[0], DTensor):
            return inputs
        elif isinstance(inputs[0], torch.Tensor):
            shard_tensor = DTensor.from_local(
                inputs[0], device_mesh, [Shard(0)], run_check=False
            )
            return shard_tensor
        else:
            raise NotImplementedError("!!")

    def partition_fn(name, module, device_mesh):
        for name, param in module.named_parameters():
            dist_param = torch.nn.Parameter(
                distribute_tensor(param, device_mesh, [Replicate()])
            )
            module.register_parameter(name, dist_param)

    return distribute_module(
        module,
        device_mesh,
        partition_fn,
        input_fn=prepare_input_fn,
    )


# for selective AC
no_recompute_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.c10d_functional.reduce_scatter_tensor.default,
}

# Uses PTD FSDP AC wrapper
def checkpoint_wrapper(module, enable_selective_ac):
    if enable_selective_ac:
        from torch.utils.checkpoint import (
            _pt2_selective_checkpoint_context_fn_gen,
            checkpoint,
        )

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
    else:
        return ptd_checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            preserve_rng_state=False,
        )


def parallelize_llama(model, world_mesh, parallel_dims, job_config: JobConfig):
    """
    Apply parallelisms to the model, including PTD parallelisms, and AC.

    NOTE: the model passed in preferrablably shoule be a meta device model,
    otherwise the model needs to be small enough on GPU or can fit into CPU.
    """
    # apply PTD parallelisms
    if parallel_dims.pp_enabled:
        raise NotImplementedError("PP not implemented yet.")

    # First we apply Sequence Parallelism if it's enabled
    if parallel_dims.sp_enabled:
        tp_mesh = world_mesh["sp"]
        sp_degree = job_config.training.sequence_parallel_degree

        # First:
        # 1. parallelize the first embedding and the last linear proj layer
        # 2. shard the first layer of transformer block
        model = parallelize_module(
            model,
            tp_mesh,
            {
                "embeddings.tok_embeddings": RowwiseParallel(
                    input_layouts=Replicate(),
                ),
                "output": ColwiseParallel(
                    input_layouts=Shard(0),
                    output_layouts=Shard(-1)
                    if parallel_dims.loss_parallel_enabled
                    else Replicate(),
                    use_local_output=not parallel_dims.loss_parallel_enabled,
                ),
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(), None),
                    desired_input_layouts=(Shard(0), None),
                    use_local_output=True,
                ),
            },
        )

        # shard the RMSNorm layer before last linear proj layer
        distribute_rmsnorm(model.norm, tp_mesh)

        # apply sequence parallelism to every transformer block
        for layer_id, transformer_block in enumerate(model.layers):
            layer_plan = {
                "attention": PrepareModuleInput(
                    input_layouts=(Shard(0), None),
                    desired_input_layouts=(Replicate(), None),
                ),
                "attention.wq": ColwiseParallel(),
                "attention.wk": ColwiseParallel(),
                "attention.wv": ColwiseParallel(),
                "attention.wo": RowwiseParallel(output_layouts=Shard(0)),
                "feed_forward": PrepareModuleInput(
                    input_layouts=(Shard(0),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": ColwiseParallel(),
                "feed_forward.w2": RowwiseParallel(output_layouts=Shard(0)),
                "feed_forward.w3": ColwiseParallel(),
            }

            # adjust num_heads in attention layer to local heads
            attn_layer = transformer_block.attention
            attn_layer.n_heads = attn_layer.n_heads // sp_degree
            attn_layer.n_kv_heads = attn_layer.n_kv_heads // sp_degree

            # shard RMSNorm layers
            distribute_rmsnorm(transformer_block.attention_norm, tp_mesh)
            distribute_rmsnorm(transformer_block.ffn_norm, tp_mesh)

            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

        rank0_log("Applied Sequence Parallelism to the model...")

    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]

        fsdp_config = {
            "mixed_precision": MixedPrecision(
                param_dtype=torch.bfloat16,
                # TODO: see whether we should expose a option to user
                reduce_dtype=torch.float16,
            ),
            "sharding_strategy": ShardingStrategy.FULL_SHARD,
            "backward_prefetch": None,
            # When torch.compile is active, it requires us to set use_orig_params=True
            "use_orig_params": True,
            "device_mesh": dp_mesh,
            # "cpu_offload": torch.distributed.fsdp.CPUOffload(offload_params=True)
            "param_init_fn": meta_to_real_init_fn,
        }

        with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
            for layer_id, transformer_block in enumerate(model.layers):

                # apply AC/selective AC
                transformer_block = checkpoint_wrapper(
                    transformer_block, job_config.training.enable_selective_ac
                )

                # Wraps each layer with FSDP
                model.layers[layer_id] = wrap(transformer_block)

            # wrap the rest layers with FSDP
            model = wrap(model)

        rank0_log("Applied FSDP to the model...")
    else:
        meta_to_real_init_fn(model)
        model.cuda()

    # we have now moved from meta to device,
    # reset parameters for proper initialization
    model.reset_parameters()
    return model
