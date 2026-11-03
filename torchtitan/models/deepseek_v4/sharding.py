# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Declarative sharding configs for DeepSeek V4 components.

``flavors.py`` sets these on each component config as it builds it.
"""

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    residual_activation_placement,
    token_id_placement,
)
from torchtitan.models.common.moe.sharding import router_sharding_config

_dense_param_rep = dense_param_placement(tp=spmd.R)
_attn_sink_placement = dense_param_placement(tp=spmd.S(0))
DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def hc_head_input_sequence_parallel_placement():
    return SpmdType(
        {
            DP: spmd.V,
            CP: spmd.V,
            TP: spmd.V,
        },
        partition_spec=spmd.PartitionSpec((DP, CP, TP), None, None),
    )


def hc_mix_sequence_parallel_placement():
    return SpmdType(
        {
            DP: spmd.V,
            CP: spmd.V,
            TP: spmd.V,
        },
        partition_spec=spmd.PartitionSpec((DP, CP, TP), None),
    )


def _hc_branch_placement(*, enable_sp: bool) -> SpmdType:
    """Placement of the ``(tokens, hc_mult, dim)`` hyper-connection branches."""
    if enable_sp:
        return hc_head_input_sequence_parallel_placement()
    return dense_activation_placement(tp=spmd.I, cp=spmd.S(0))


def _hc_mix_placement(*, enable_sp: bool) -> SpmdType:
    """Placement of the ``(tokens, hc_mult)`` hyper-connection mix weights."""
    if enable_sp:
        return hc_mix_sequence_parallel_placement()
    return dense_activation_placement(tp=spmd.I, cp=spmd.S(0))


def _hc_state_shardings() -> dict[str, SpmdType]:
    return {
        "hc_fn": _dense_param_rep,
        "hc_base": _dense_param_rep,
        "hc_scale": _dense_param_rep,
    }


def dsa_inner_attention_sharding_config(*, compress_ratio: int) -> ShardingConfig:
    """Local-SPMD sparse attention kernel with queries sharded on heads."""
    query_states = dense_activation_placement(tp=spmd.S(1), cp=spmd.S(0))
    replicated_activation = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    input_shardings = {
        "q": query_states,
        "swa_k": replicated_activation,
    }
    if compress_ratio == 4:
        input_shardings.update(
            {
                "cmp_k": replicated_activation,
                "idx_q": replicated_activation,
                "idx_k": replicated_activation,
                "idx_w": replicated_activation,
                "attn_sink": _attn_sink_placement,
                "padding_mask_T": token_id_placement(),
            }
        )
    elif compress_ratio > 1:
        input_shardings.update(
            {
                "cmp_k": replicated_activation,
                "attn_sink": _attn_sink_placement,
            }
        )
    else:
        input_shardings["attn_sink"] = _attn_sink_placement

    return ShardingConfig(
        in_src_shardings=input_shardings,
        in_dst_shardings=dict(input_shardings),
        out_src_shardings=query_states,
        out_dst_shardings=query_states,
        local_spmd=True,
    )


def attention_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    """Attention boundary: ``x`` enters in the residual layout."""
    return ShardingConfig(
        in_src_shardings={
            "x": residual_activation_placement(enable_sp=enable_sp),
        },
    )


def wo_a_sharding_config() -> ShardingConfig:
    """``wo_a`` holds a grouped LoRA-A weight used via einsum (not a standard
    matmul). Colwise sharding distributes the weight along dim-0."""
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))}
    )


def attn_sink_sharding_config() -> ShardingConfig:
    """``attn_sink`` holds a ``(n_heads, 1)`` weight used as a head-wise
    vector in sparse attention, so shard it on the head dimension under TP."""
    return ShardingConfig(
        state_shardings={"weight": _attn_sink_placement},
    )


def compressor_sharding_config() -> ShardingConfig:
    """Replicate the compressor's absolute position embedding across TP."""
    return ShardingConfig(
        state_shardings={"ape": _dense_param_rep},
    )


def indexer_sharding_config() -> ShardingConfig:
    """Indexer boundary: gather ``x`` and ``qr`` across TP."""
    replicated_activation = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    return ShardingConfig(
        in_src_shardings={
            "x": replicated_activation,
            "qr": replicated_activation,
        },
        in_dst_shardings={
            "x": replicated_activation,
            "qr": replicated_activation,
        },
    )


def hc_pre_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    """Hyper-connection pre-mix: branches in, ``(x, post, comb)`` out."""
    hc_branch_layout = _hc_branch_placement(enable_sp=enable_sp)
    return ShardingConfig(
        state_shardings=_hc_state_shardings(),
        in_src_shardings={"x": hc_branch_layout},
        out_src_shardings=(
            residual_activation_placement(enable_sp=enable_sp),
            _hc_mix_placement(enable_sp=enable_sp),
            hc_branch_layout,
        ),
    )


def hc_post_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    """Hyper-connection post-mix back into the branches."""
    hc_branch_layout = _hc_branch_placement(enable_sp=enable_sp)
    return ShardingConfig(
        in_src_shardings={
            "x": residual_activation_placement(enable_sp=enable_sp),
            "residual": hc_branch_layout,
            "post": _hc_mix_placement(enable_sp=enable_sp),
            "comb": hc_branch_layout,
        },
        out_src_shardings=hc_branch_layout,
    )


def hc_head_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    """Root hyper-connection head: branches in, residual layout out."""
    return ShardingConfig(
        state_shardings=_hc_state_shardings(),
        in_src_shardings={"x": _hc_branch_placement(enable_sp=enable_sp)},
        out_src_shardings=residual_activation_placement(enable_sp=enable_sp),
    )


def deepseek_v4_router_sharding_config(
    *, layer_id: int, n_hash_layers: int
) -> ShardingConfig:
    """Router sharding; hash-routed layers also replicate ``tid2eid``."""
    config = router_sharding_config()
    if layer_id < n_hash_layers:
        config.state_shardings["tid2eid"] = _dense_param_rep
    return config


def mtp_hc_head_sharding_config() -> ShardingConfig:
    """MTP hyper-connection head over replicated activations."""
    replicated_activation = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    return ShardingConfig(
        state_shardings=_hc_state_shardings(),
        in_src_shardings={"x": replicated_activation},
        out_src_shardings=replicated_activation,
    )


def mtp_block_sharding_config() -> ShardingConfig:
    """MTP block boundary over replicated activations and token IDs."""
    replicated_activation = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    return ShardingConfig(
        in_src_shardings={
            "mtp_input_embed": replicated_activation,
            "prev_hc_hidden": replicated_activation,
            "mtp_input_ids_T": token_id_placement(),
            "mtp_input_valid_mask": token_id_placement(),
        },
        out_src_shardings=replicated_activation,
    )
