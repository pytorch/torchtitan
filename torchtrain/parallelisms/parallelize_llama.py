# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# this file applies the PTD parallelisms and various training techniques to the
# llama model, i.e. activation checkpoint, etc.

import logging

import torch

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

logger = logging.getLogger(__name__)

# Uses PTD FSDP AC wrapper
def checkpoint_wrapper(module, config):
    return ptd_checkpoint_wrapper(
        module, checkpoint_impl=CheckpointImpl.NO_REENTRANT, preserve_rng_state=False
    )


def slice_submesh(world_mesh: DeviceMesh, name: str):
    mesh = world_mesh[name] if world_mesh.ndim > 1 else world_mesh
    assert mesh.mesh_dim_names[0] == name and mesh.ndim == 1, mesh.mesh_dim_names
    return mesh


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
        raise NotImplementedError("SP not implemented yet.")
    if parallel_dims.dp_enabled:
        dp_mesh = slice_submesh(world_mesh, "dp")

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

    rank0_log("Applied parallelisms to the model...")

    return model
