# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# this file applies the PTD parallelisms and various training techniques to the
# llama model, i.e. activation checkpoint, etc.

import logging

import torch
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module

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
    RowwiseParallel,
    parallelize_module,
    PrepareModuleInput,
)

from torchtrain.logging_utils import rank0_log

logger = logging.getLogger(__name__)

def distribute_rmsnorm(module, device_mesh):
    # temp sharding API until PTD API is added
    def prepare_input_fn(inputs, device_mesh):
        if isinstance(inputs[0], DTensor):
            return inputs
        elif isinstance(inputs[0], torch.Tensor):
            shard_tensor = DTensor.from_local(inputs[0], device_mesh, [Shard(0)], run_check=False)
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
        input_fn = prepare_input_fn,
    )


# Uses PTD FSDP AC wrapper
def checkpoint_wrapper(module, config):
    return ptd_checkpoint_wrapper(
        module, checkpoint_impl=CheckpointImpl.NO_REENTRANT, preserve_rng_state=False
    )


def parallelize_llama(model, world_mesh, parallel_dims, args):
    """
    Apply parallelisms to the model, including PTD parallelisms, and AC.

    NOTE: the model passed in preferrablably shoule be a meta device model,
    otherwise the model needs to be small enough on GPU or can fit into CPU.
    # TODO: apply SP
    """
    # apply PTD parallelisms
    if parallel_dims.pp_enabled:
        raise NotImplementedError("PP not implemented yet.")
    if parallel_dims.sp_enabled:
        # First we apply Sequence Parallelism if it's enabled
        tp_mesh = world_mesh["sp"] if world_mesh.ndim > 1 else world_mesh
        sp_degree = args.sp_degree
        # First:
        # 1. parallelize the first embedding and the last linear proj layer
        # 2. shard the first layer of transformer block
        # TODO: enable loss parallel once it's ready
        model = parallelize_module(
            model,
            tp_mesh,
            {
                "embeddings.tok_embeddings": RowwiseParallel(
                        input_layouts=Replicate(),
                    ),
                "output": ColwiseParallel(
                    input_layouts=Shard(0),
                    output_layouts=Replicate(),
                ),
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(), None),
                    desired_input_layouts=(Shard(0), None),
                    use_local_output=True,
                )
            }
        )

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
            # if layer_id == 0:
            #     # in first transformer block we need to shard the input
            #     layer_plan[""] = PrepareModuleInput(
            #         input_layouts=(Replicate(), None),
            #         desired_input_layouts=(Shard(0), None),
            #     )

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

        rank0_log(f"Applied Sequence Parallelism to the model...")

    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"] if world_mesh.ndim > 1 else world_mesh
        assert dp_mesh.mesh_dim_names == ["dp"], dp_mesh.mesh_dim_names
        fsdp_config = {
            "mixed_precision": MixedPrecision(
                param_dtype=torch.bfloat16,
                # TODO: see whether we should expose a option to user
                reduce_dtype=torch.float32,
            ),
            "sharding_strategy": ShardingStrategy.FULL_SHARD,
            "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
            # When torch.compile is active, it requires us to set use_orig_params=True
            "use_orig_params": True,
            "device_mesh": dp_mesh,
        }

        with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
            for layer_id, transformer_block in enumerate(model.layers):
                # apply AC to each layer
                # before wrapping with FSDP, we need to make sure the layer is on GPU
                transformer_block = transformer_block.cuda()
                transformer_block = checkpoint_wrapper(transformer_block, args)

                # Wraps each layer with FSDP
                model.layers[layer_id] = wrap(transformer_block)

            # wrap the rest layers with FSDP
            model = wrap(model.cuda())

        rank0_log(f"Applied FSDP to the model...")

    return model
