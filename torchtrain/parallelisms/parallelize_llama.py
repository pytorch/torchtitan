# this file applies the PTD parallelisms and various training techniques to the
# llama model, i.e. activation checkpoint, etc.

import os
import torch

from torch.distributed.device_mesh import init_device_mesh

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

from torchtrain.logging_utils import rank0_log
from torchtrain.meta_init import meta_to_real_init_fn

# Uses PTD FSDP AC wrapper
def checkpoint_wrapper(module, config):
    return ptd_checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT, preserve_rng_state=False)


def parallelize_llama(model, args):
    """
    Apply parallelisms to the model, including PTD parallelisms, and AC.

    NOTE: the model passed in preferrablably shoule be a meta device model,
    otherwise the model needs to be small enough on GPU or can fit into CPU.
    # TODO: apply SP
    """
    # DeviceMesh or multi-dim pg setup
    # only support cuda for now
    device_type = "cuda"
    # distributed init
    world_size = int(os.environ["WORLD_SIZE"])
    if args.enable_sp:
        sp_degree = args.sp_degree
        dp_degree = world_size // sp_degree
        world_mesh = init_device_mesh(
            device_type, (dp_degree, sp_degree), mesh_dim_names=("dp", "tp")
        )
        dp_mesh = world_mesh["dp"]
    else:
        world_mesh = init_device_mesh(device_type, (world_size,))
        dp_mesh = world_mesh

    # apply PTD parallelisms
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
        "param_init_fn":meta_to_real_init_fn,
    }

    with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
        for layer_id, transformer_block in enumerate(model.layers):
            # apply AC to each layer
            # before wrapping with FSDP, we need to make sure the layer is on GPU
            # todo - config this: transformer_block = transformer_block.cuda()
            # todo - transformer_block = checkpoint_wrapper(transformer_block, args)

            # Wraps each layer with FSDP
            model.layers[layer_id]= wrap(transformer_block)

        # wrap the rest layers with FSDP
        model = wrap(model) # todo - was .cuda()

    rank0_log(f"Applied parallelisms to the model...")

    return model
