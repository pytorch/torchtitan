# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parallelization logic for DeepSeek-V3 with DeepEP.

This module handles:
- Tensor Parallelism (TP) for non-MoE layers
- Expert Parallelism (EP) via DeepEP for MoE layers
- Activation Checkpointing (AC)
- Data Parallelism (FSDP/HSDP)
"""

import os
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.deepseek_v3.infra.parallelize import (
    apply_ac,
    apply_non_moe_tp,
)
from torchtitan.models.moe.moe import MoE, TokenChoiceTopKRouter, GroupedExperts
from torchtitan.tools.logging import logger

from ..moe_deepep import MoEWithDeepEP, get_deepep_buffer
from torch.distributed.tensor.placement_types import Replicate


def replace_moe_with_deepep(
    model: nn.Module,
    ep_group,
) -> None:
    """
    Replace standard MoE layers with MoEWithDeepEP.
    
    This function walks through the model and replaces any MoE instances
    with MoEWithDeepEP instances, copying over the weights and configuration.
    
    Args:
        model: The model containing MoE layers
        ep_group: Expert parallel process group
    """
    for name, module in model.named_children():
        if isinstance(module, MoE):
            dim = module.router.gate.in_features
            hidden_dim = module.experts.w1.shape[1]  # [num_experts, hidden_dim, dim]
            num_experts_total = module.experts.num_experts
            
            ep_size = ep_group.size() if ep_group else 1
            num_experts_local = num_experts_total // ep_size
            
            router = TokenChoiceTopKRouter(
                dim=dim,
                num_experts=num_experts_total,
                top_k=module.router.top_k,
                score_func=module.router.score_func,
                route_norm=module.router.route_norm,
                route_scale=module.router.route_scale,
            )
            
            experts = GroupedExperts(
                dim=dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts_local,
                use_grouped_mm=module.experts.use_grouped_mm,
            )
            
            hidden_bytes = dim * 2 # bfloat16
            buffer = get_deepep_buffer(ep_group, hidden_bytes)
            
            ep_rank = torch.distributed.get_rank(ep_group) if ep_group else 0
            local_expert_start = ep_rank * num_experts_local
            local_expert_end = (ep_rank + 1) * num_experts_local
            
            new_moe = MoEWithDeepEP(
                router=router,
                experts=experts,
                buffer=buffer,
                num_experts=num_experts_total,
                score_before_experts=module.score_before_experts,
                load_balance_coeff=module.load_balance_coeff,
                ep_group=ep_group,
                shared_experts=module.shared_experts,
            )
            
            if module.experts.w1.device.type != 'meta':
                new_moe.experts.w1.data.copy_(module.experts.w1.data[local_expert_start:local_expert_end])
                new_moe.experts.w2.data.copy_(module.experts.w2.data[local_expert_start:local_expert_end])
                new_moe.experts.w3.data.copy_(module.experts.w3.data[local_expert_start:local_expert_end])
                new_moe.router.gate.weight.data.copy_(module.router.gate.weight.data)
            else:
                logger.info(f"  Model on meta device - weights will be initialized via reset_parameters()")
            
            new_moe = new_moe.to(module.experts.w1.device)
            
            setattr(model, name, new_moe)
        else:
            # Recursively replace in child modules
            replace_moe_with_deepep(module, ep_group)


def parallelize_deepseekv3(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply parallelization strategies to DeepSeek-V3 model with DeepEP.
    
    Parallelization order:
    1. Tensor Parallelism (TP) for non-MoE layers (attention, dense FFN)
    2. Expert Parallelism (EP) via DeepEP for MoE layers
    3. Activation Checkpointing (AC)
    4. torch.compile (applied BEFORE FSDP to avoid hook conflicts)
    5. Data Parallelism (FSDP/HSDP)
    
    Args:
        model: The DeepSeek-V3 model to parallelize
        parallel_dims: Parallelization dimensions
        job_config: Job configuration
    
    Returns:
        Parallelized model
    """
    world_mesh = parallel_dims.world_mesh
    
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}), i.e. {parallel_dims.seq_len_divisor}.
        """
    
    if (
        job_config.parallelism.context_parallel_degree > 1
        and model.model_args.use_flex_attn
    ):
        raise NotImplementedError("CP support for FlexAttention is still in progress.")
    
    if parallel_dims.tp_enabled:
        logger.info("Applying Tensor Parallelism to non-MoE layers...")
        
        use_flex_attn = getattr(model.model_args, "use_flex_attn", False)
        apply_non_moe_tp(
            model,
            world_mesh["tp"],
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=False,  # Not tested for DeepSeek-V3
            use_flex_attn=use_flex_attn,
        )
        maybe_enable_async_tp(job_config, world_mesh["tp"])
    
    if parallel_dims.ep_enabled:
        
        ep_mesh = world_mesh["ep"]
        ep_group = ep_mesh.get_group()
        
        dim = model.model_args.dim
        moe_inter_dim = model.model_args.moe_inter_dim
        
        # Check alignment requirements
        dim_valid = (dim % 256) == 0
        moe_dim_valid = (moe_inter_dim % 256) == 0
        

        num_nodes = parallel_dims.world_size // int(os.environ.get('LOCAL_WORLD_SIZE', 8))
        
        replace_moe_with_deepep(model, ep_group)
    
    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )
    
    if job_config.activation_checkpoint.mode != "none":
        # Selective AC op save list (same as baseline)
        _op_sac_save_list = {
            torch.ops.aten.mm.default,
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            torch.ops._c10d_functional.all_to_all_single.default,
            torch.ops.aten.max.default,
            torch._higher_order_ops.flex_attention,
        }
        
        use_flex_attn = getattr(model.model_args, "use_flex_attn", False)
        
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            use_flex_attn=use_flex_attn,
            op_sac_save_list=_op_sac_save_list,
            base_folder=job_config.job.dump_folder,
        )
        logger.info("Activation Checkpointing applied")
    
    if model_compile_enabled:
        for layer_id, transformer_block in model.layers.named_children():
            fullgraph = True
            if transformer_block.moe_enabled:
                fullgraph = False
                logger.info(f"Compiling layer {layer_id} (MoE) with fullgraph=False")
            else:
                logger.info(f"Compiling layer {layer_id} (non-MoE) with fullgraph=True")
            
            transformer_block = torch.compile(
                transformer_block,
                backend=job_config.compile.backend,
                fullgraph=fullgraph,
            )
            model.layers.register_module(layer_id, transformer_block)
        logger.info("✓ torch.compile applied to all TransformerBlocks")
    
    dp_mesh: DeviceMesh | None = None
    if (
        parallel_dims.fsdp_enabled
        or parallel_dims.ep_enabled
        or parallel_dims.dp_replicate_enabled
    ):
        if parallel_dims.dp_replicate_enabled:
            if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
                dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
                dp_mode = "hybrid_shard"
            else:
                dp_mesh_dim_names = ("dp_replicate",)
                dp_mode = "replicate"
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
            dp_mode = "fully_shard"
        
        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]
        
        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
        )
        
        if parallel_dims.ep_enabled:
            dp_mod_ep_mesh_dim_names = []
            if parallel_dims.dp_replicate_enabled:
                dp_mod_ep_mesh_dim_names.append("dp_replicate")
            dp_mod_ep_mesh_dim_names.append("dp_shard_mod_ep")
            dp_mod_ep_mesh = world_mesh[tuple(dp_mod_ep_mesh_dim_names)]
            
            for _, transformer_block in model.layers.items():
                if transformer_block.moe_enabled and not isinstance(transformer_block.moe, MoEWithDeepEP):
                    experts_shard_dim = 0
                    if (
                        dp_mod_ep_mesh.size() * parallel_dims.ep
                        > transformer_block.moe.experts.num_experts
                    ):
                        experts_shard_dim = 1
                    
                    fully_shard(
                        transformer_block.moe.experts,
                        mesh=dp_mod_ep_mesh,
                        mp_policy=mp_policy,
                        reshard_after_forward=(
                            job_config.parallelism.fsdp_reshard_after_forward == "always"
                        ),
                    )
        
        fully_shard(
            model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=(
                job_config.parallelism.fsdp_reshard_after_forward == "always"
            ),
        )
        
    return model

