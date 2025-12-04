# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
DeepEP Expert Parallel integration for DTensor-based weight sharding.

This module provides a ParallelStyle for sharding expert weights across
expert-parallel ranks when using DeepEP for communication.

Key Difference from Standard EP:
- Standard EP: Handles weight sharding + token communication (all-to-all)
- DeepEP EP: Handles weight sharding ONLY (DeepEP handles token communication)
"""

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor, Shard
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor import distribute_module


class DeepEPExpertParallel(ParallelStyle):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def _partition_fn(name, module, device_mesh):
        """
        Partition function to shard expert weights.
        
        This is called by distribute_module to shard parameters along the expert dimension.
        Similar to standard EP's _partition_fn, but simpler since we don't need to handle
        token communication.
        """
        for param_name, param in module.named_parameters(recurse=False):
            if param_name in ("w1", "w2", "w3"):
                dist_param = nn.Parameter(
                    distribute_tensor(param, device_mesh, [Shard(0)])
                )
                module.register_parameter(param_name, dist_param)
    
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """
        Apply the parallelization to the module.
        
        Uses distribute_module (same as standard EP) but WITHOUT input_fn/output_fn
        since DeepEP handles token communication separately in MoEWithDeepEP.
        
        Compare to standard EP:
            return distribute_module(
                module, device_mesh,
                partition_fn=ExpertParallel._partition_fn,
                input_fn=self._token_dispatch,   # ← no need for this
                output_fn=self._token_combine,    # ← no need for this
            )
        
        We only need partition_fn because DeepEP's dispatch/combine are called
        in MoEWithDeepEP.forward(), not here.
        """
        return distribute_module(
            module,
            device_mesh,
            partition_fn=DeepEPExpertParallel._partition_fn,
        )
