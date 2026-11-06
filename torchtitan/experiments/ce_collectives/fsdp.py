# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Install the experimental collectives through FSDP2's communication hooks."""

import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule

from .collectives import CEAllToAllReduceScatter, CEMulticastAllGather
from .configs import CECollectivesConfig


def enable_fsdp_ce_collectives(model: nn.Module, config: CECollectivesConfig) -> int:
    """Install on selected modules before their first forward; return module count.

    Call in the same module order on all ranks. FSDP retains control of stream
    dependencies, gradient scaling, prefetch and HSDP's replica all-reduce.
    """
    all_gathers: dict[dist.ProcessGroup, CEMulticastAllGather] = {}
    reduce_scatters: dict[dist.ProcessGroup, CEAllToAllReduceScatter] = {}
    num_installed = 0
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        if config.scope == "dense":
            from torchtitan.models.common.moe import MoE

            if any(isinstance(child, MoE) for child in module.children()):
                continue
        # TODO: Use public group accessors when FSDP exposes them alongside
        # its communication hooks: https://github.com/pytorch/pytorch/pull/155189.
        groups = module._get_fsdp_state()._fsdp_param_groups
        if not groups:
            continue
        if len(groups) != 1:
            raise ValueError(
                "CE collectives require one parameter group per FSDP module; use dense scope for EP."
            )
        param_group = groups[0]
        gather_group = param_group._all_gather_process_group
        reduce_group = param_group._reduce_scatter_process_group
        if gather_group.size() == 1 and reduce_group.size() == 1:
            continue
        if config.all_gather and gather_group.size() > 1:
            ranks_per_domain = config.ranks_per_domain or gather_group.size()
            if gather_group.size() != ranks_per_domain:
                raise ValueError(
                    "CE all-gather requires one NVLink domain; disable it for hierarchical reduce-scatter."
                )
            if gather_group not in all_gathers:
                all_gathers[gather_group] = CEMulticastAllGather(gather_group)
            module.set_custom_all_gather(all_gathers[gather_group])
        if config.reduce_scatter and reduce_group.size() > 1:
            if reduce_group not in reduce_scatters:
                reduce_scatters[reduce_group] = CEAllToAllReduceScatter(
                    reduce_group,
                    ranks_per_domain=config.ranks_per_domain or reduce_group.size(),
                )
            # Keep arbitrary gradient divide factors and HSDP scaling in FSDP;
            # the custom collective receives SUM instead of NCCL PREMUL_SUM.
            module.set_force_sum_reduction_for_comms(True)
            module.set_custom_reduce_scatter(reduce_scatters[reduce_group])
        num_installed += 1
    return num_installed
