# this file applies the PTD parallelisms and various training techniques to the
# llama model, i.e. activation checkpoint, etc.

import os
import torch
import logging

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
from typing import Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Uses PTD FSDP AC wrapper
def checkpoint_wrapper(module, config):
    return ptd_checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT, preserve_rng_state=False)

@dataclass
class ParallelDims:
    dp: int
    sp: int
    pp: int
    world_size: int

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp, sp, pp = self.dp, self.sp, self.pp
        if dp == -1:
            self.dp = dp = self.world_size // (sp * pp)
        assert dp >= 1, dp
        assert sp >= 1, sp
        assert pp >= 1, pp
        assert dp * sp * pp == self.world_size, (
            f"Invalid parallel dims: dp({dp}) * sp({sp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip([self.dp, self.sp, self.pp], ["dp", "sp", "pp"]):
            if d > 1:
                dims.append(d)
                names.append(name)
        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        return init_device_mesh(device_type, dims, mesh_dim_names=names)

    @property
    def dp_enabled(self):
        return self.dp > 1

    @property
    def sp_enabled(self):
        return self.sp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

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
    parallel_dims = ParallelDims(dp=args.dp_degree, sp=args.sp_degree, pp=args.pp_degree, world_size=world_size)
    world_mesh = parallel_dims.build_mesh(device_type)
    # apply PTD parallelisms
    if parallel_dims.pp_enabled:
        raise NotImplementedError("PP not implemented yet.")
    if parallel_dims.sp_enabled:
        raise NotImplementedError("SP not implemented yet.")
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
                model.layers[layer_id]= wrap(transformer_block)

            # wrap the rest layers with FSDP
            model = wrap(model.cuda())

    rank0_log(f"Applied parallelisms to the model...")

    return model
