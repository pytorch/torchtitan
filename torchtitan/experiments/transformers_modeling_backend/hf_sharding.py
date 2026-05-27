# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ShardingConfig and TP setup for HF model modules.

Root-level modules (tok_embeddings, norm, lm_head) use the titan Module
protocol's ``_sharding_config`` so ``model.parallelize()`` handles them.

Per-layer modules (attention, dense MLP, norms) use ``parallelize_module``
because HF decoder layer's residual connections mix outputs from attention
and norm — using ShardingConfig (DTensor) for norms alongside
parallelize_module (local tensors) for attention creates mismatches at
``residual + hidden_states``.

MoE layers are already native Module instances with ShardingConfig and
are handled by ``model.parallelize()`` directly.
"""

import torch.nn as nn
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.tensor_parallel import NoParallel
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
)
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.tools.logging import logger


def set_hf_sharding_configs(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    enable_sp: bool,
    enable_loss_parallel: bool,
) -> None:
    """Set sharding on all HF modules for TP/EP parallelization.

    Root-level modules use ``_sharding_config`` (Module protocol).
    Per-layer modules use ``parallelize_module`` (legacy approach).
    MoE layers are skipped (already have ShardingConfig from native MoE).

    Args:
        model: The HFTransformerModel with Module-converted children.
        parallel_dims: Parallel dimensions for mesh resolution.
        enable_sp: Whether sequence parallelism is enabled (TP > 1).
        enable_loss_parallel: Whether to shard lm_head output for loss parallel.
    """
    tp_mesh = parallel_dims.get_optional_mesh("tp")
    if tp_mesh is None:
        return

    # Root-level modules via parallelize_module
    root_plan = {}
    if hasattr(model, "tok_embeddings"):
        if isinstance(model.tok_embeddings, nn.Identity):
            root_plan["tok_embeddings"] = NoParallel(use_local_output=True)
        else:
            root_plan["tok_embeddings"] = RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            )

    if hasattr(model, "norm"):
        if isinstance(model.norm, nn.Identity):
            root_plan["norm"] = NoParallel(use_local_output=True)
        else:
            root_plan["norm"] = SequenceParallel()

    if hasattr(model, "lm_head"):
        if isinstance(model.lm_head, nn.Identity):
            root_plan["lm_head"] = NoParallel(use_local_output=True)
        else:
            root_plan["lm_head"] = ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
                use_local_output=not enable_loss_parallel,
            )

    if root_plan:
        parallelize_module(model, tp_mesh, root_plan)

    # Per-layer modules via parallelize_module
    for transformer_block in model.layers:
        _apply_layer_tp(transformer_block, tp_mesh, enable_sp=enable_sp)

    logger.info("Set sharding configs on all HF modules")


def _apply_layer_tp(layer: nn.Module, tp_mesh, *, enable_sp: bool) -> None:
    """Apply TP to a decoder layer via parallelize_module.

    Builds a TP plan covering norms, attention, and (for non-MoE layers)
    dense MLP. MoE layers are skipped — they have their own ShardingConfig.
    """
    plan = {
        "input_layernorm": SequenceParallel(),
        "self_attn": PrepareModuleInput(
            input_kwarg_layouts={"hidden_states": Shard(1)},
            desired_input_kwarg_layouts={"hidden_states": Replicate()},
        ),
        "post_attention_layernorm": SequenceParallel(),
    }

    attn = layer.self_attn

    # Q/K/V projections
    if getattr(attn, "q_lora_rank", None) is None:
        plan.update(
            {
                "self_attn.q_proj": ColwiseParallel(),
                "self_attn.k_proj": ColwiseParallel(),
                "self_attn.v_proj": ColwiseParallel(),
            }
        )
    else:
        plan.update(
            {
                "self_attn.q_a_proj": NoParallel(use_local_output=True),
                "self_attn.q_a_layernorm": NoParallel(use_local_output=True),
                "self_attn.q_b_proj": ColwiseParallel(),
                "self_attn.kv_a_proj_with_mqa": NoParallel(use_local_output=True),
                "self_attn.kv_a_layernorm": NoParallel(use_local_output=True),
                "self_attn.kv_b_proj": ColwiseParallel(),
            }
        )

    # O projection
    o_proj_name = "o_proj" if hasattr(attn, "o_proj") else "dense"
    plan[f"self_attn.{o_proj_name}"] = RowwiseParallel(output_layouts=Shard(1))

    # Q/K norms (Qwen3)
    if hasattr(attn, "q_norm") and hasattr(attn, "k_norm"):
        plan["self_attn.q_norm"] = SequenceParallel(
            sequence_dim=2, use_local_output=True
        )
        plan["self_attn.k_norm"] = SequenceParallel(
            sequence_dim=2, use_local_output=True
        )

    # GLM-5 DSA indexer — operates on local tensors (inputs are already
    # local from PrepareModuleInput on self_attn + NoParallel on q_a_*).
    if hasattr(attn, "indexer"):
        plan["self_attn.indexer"] = NoParallel(use_local_output=True)

    # Dense MLP (non-MoE layers only)
    # Most models use ``mlp``; Llama4 uses ``feed_forward``.
    if not getattr(layer, "moe_enabled", False):
        mlp_attr = "mlp" if hasattr(layer, "mlp") else "feed_forward"
        mlp = getattr(layer, mlp_attr)
        mlp_plan = {
            mlp_attr: PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
        }

        gate_name = "gate_proj" if hasattr(mlp, "gate_proj") else "fc1"
        mlp_plan[f"{mlp_attr}.{gate_name}"] = ColwiseParallel()

        if hasattr(mlp, "up_proj"):
            mlp_plan[f"{mlp_attr}.up_proj"] = ColwiseParallel()

        down_name = "down_proj" if hasattr(mlp, "down_proj") else "fc2"
        mlp_plan[f"{mlp_attr}.{down_name}"] = RowwiseParallel(output_layouts=Shard(1))
        plan.update(mlp_plan)

    # Some models don't have post_attention_layernorm
    if not hasattr(layer, "post_attention_layernorm"):
        plan.pop("post_attention_layernorm")

    parallelize_module(
        module=layer,
        device_mesh=tp_mesh,
        parallelize_plan=plan,
    )


# ---------------------------------------------------------------------------
# HF-specific sharding config helpers
# ---------------------------------------------------------------------------


def _hf_norm_config(*, enable_sp: bool) -> ShardingConfig:
    """Norm sharding using HF's ``hidden_states`` arg name."""
    state = {"weight": dense_param_placement(tp=Replicate())}
    if not enable_sp:
        return ShardingConfig(state_shardings=state)
    return ShardingConfig(
        state_shardings=state,
        in_src_shardings={
            "hidden_states": dense_activation_placement(tp=Shard(1)),
        },
        in_dst_shardings={
            "hidden_states": dense_activation_placement(tp=Shard(1)),
        },
        out_dst_shardings=dense_activation_placement(tp=Shard(1)),
    )
