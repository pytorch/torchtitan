# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torchtitan.components.ft import FTManager
from torchtitan.components.optimizer import build_optimizers, OptimizersContainer
from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims


# for MoE auxiliary-loss-free load balancing
def _update_expert_bias(
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
):
    dp_cp_mesh = (
        parallel_dims.world_mesh["dp_cp"] if parallel_dims.dp_cp_enabled else None
    )
    # TODO: Currently this sync is blocking (thus exposed) and happens on the
    # default compute stream. Need to assess if this is OK performance-wise.
    for model_part in model_parts:
        for transformer_block in model_part.layers.values():
            if transformer_block.moe_enabled:
                moe = transformer_block.moe
                if moe.load_balance_coeff is None:
                    return

                if dp_cp_mesh is not None:
                    torch.distributed.all_reduce(
                        moe.tokens_per_expert, group=dp_cp_mesh.get_group()
                    )

                with torch.no_grad():
                    expert_bias_delta = moe.load_balance_coeff * torch.sign(
                        moe.tokens_per_expert.mean() - moe.tokens_per_expert
                    )
                    expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
                    moe.expert_bias.add_(expert_bias_delta)
                    moe.tokens_per_expert.zero_()


def build_llama4_optimizers(
    model_parts: list[nn.Module],
    job_config: JobConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    optimizers = build_optimizers(
        model_parts=model_parts,
        job_config=job_config,
        parallel_dims=parallel_dims,
        ft_manager=ft_manager,
    )

    optimizers.register_step_pre_hook(
        lambda *args, **kwargs: _update_expert_bias(
            model_parts, parallel_dims=parallel_dims
        )
    )

    return optimizers
