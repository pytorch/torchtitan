# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.models.common.attention import FusedQKVLinear, GQAttention, QKVLinear
from torchtitan.protocols.sharding import LocalMapConfig, NamedPlacement, ShardingConfig
from torchtitan.protocols.types import MeshAxisName

DP_REPLICATE = MeshAxisName.DP_REPLICATE
DP_SHARD = MeshAxisName.DP_SHARD
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def dense_param_placement(*, tp: Placement) -> NamedPlacement:
    """Placement for dense-path params/buffers.

    DP/CP axes are ``Replicate`` at ``distribute_tensor`` time; FSDP reshards
    DP_SHARD post-parallelize. TP placement is caller-specified.

    TODO: Ideally placements would be defined on a computation mesh that
    has a single DP axis (no DP_REPLICATE vs DP_SHARD distinction). That
    requires a mesh switch between FSDP storage and computation — likely
    resolved by FlexShard. Revisit once FlexShard lands.
    """
    return {
        DP_REPLICATE: Replicate(),
        DP_SHARD: Replicate(),
        CP: Replicate(),
        TP: tp,
    }


def dense_activation_placement(
    *,
    tp: Placement,
    cp: Placement = Shard(1),
) -> NamedPlacement:
    """Placement for dense-path activations.

    DP axes are batch-sharded (``Shard(0)``). CP defaults to seq-sharded
    (``Shard(1)``); override to ``Replicate()`` for K/V after all-gather.
    TP placement is caller-specified.
    """
    return {
        DP_REPLICATE: Shard(0),
        DP_SHARD: Shard(0),
        CP: cp,
        TP: tp,
    }


def colwise_config() -> ShardingConfig:
    """ColwiseParallel: weight Shard(0), output Shard(-1)."""
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=Shard(0)),
            "bias": dense_param_placement(tp=Shard(0)),
        },
        out_dst_shardings=dense_activation_placement(tp=Shard(-1)),
    )


def rowwise_config(*, output_sp: bool = False) -> ShardingConfig:
    """RowwiseParallel: weight Shard(1), bias Replicate (no-op if bias absent).

    ``output_sp=True``  -> output ``Shard(1)`` (reduce-scatter into SP region).
    ``output_sp=False`` -> output ``Replicate()`` (all-reduce).
    """
    out_tp: Placement = Shard(1) if output_sp else Replicate()
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=Shard(1)),
            "bias": dense_param_placement(tp=Replicate()),
        },
        out_dst_shardings=dense_activation_placement(tp=out_tp),
    )


def norm_config(*, enable_sp: bool) -> ShardingConfig:
    """Norm sharding.

    ``enable_sp=True``: SequenceParallel — weight Replicate, activations
    Shard(1) across boundary.
    ``enable_sp=False``: Plain-TP — weight Replicate, activations pass
    through. Still distributes the norm's weight as a DTensor so it
    composes with DTensor activations (otherwise plain Tensor + DTensor
    would mix inside the op).
    """
    state = {"weight": dense_param_placement(tp=Replicate())}
    if not enable_sp:
        return ShardingConfig(state_shardings=state)
    return ShardingConfig(
        state_shardings=state,
        in_src_shardings={"input": dense_activation_placement(tp=Shard(1))},
        in_dst_shardings={"input": dense_activation_placement(tp=Shard(1))},
        out_dst_shardings=dense_activation_placement(tp=Shard(1)),
    )


def set_qkv_linear_sharding(qkv_linear_cfg) -> None:
    """Colwise-shard each Q/K/V projection of a ``BaseQKVLinear``.

    Handles both ``QKVLinear`` (separate ``wq`` + ``wkv``) and
    ``FusedQKVLinear`` (single ``wqkv``).
    """
    if isinstance(qkv_linear_cfg, FusedQKVLinear.Config):
        qkv_linear_cfg.wqkv.sharding_config = colwise_config()
    elif isinstance(qkv_linear_cfg, QKVLinear.Config):
        qkv_linear_cfg.wq.sharding_config = colwise_config()
        qkv_linear_cfg.wkv.sharding_config = colwise_config()
    else:
        raise TypeError(
            f"set_qkv_linear_sharding requires QKVLinear.Config or "
            f"FusedQKVLinear.Config, got {type(qkv_linear_cfg).__name__}"
        )


def set_gqa_attention_sharding(attention_cfg, *, enable_sp: bool) -> None:
    """Standard GQA attention (``qkv_linear``/``wo``) TP sharding.

    Shared by llama3, qwen3, and llama4 -- all three have a GQA block whose
    ``forward(x, rope_cache, ...)`` takes ``x`` (per-SP layout, gathered to
    Replicate internally) and a plain ``rope_cache`` (annotated Replicate).

    Callers that have additional attention sub-state (e.g. ``qk_norm``,
    ``sinks``) set those after calling this helper.
    """
    assert isinstance(attention_cfg, GQAttention.Config), (
        f"set_gqa_attention_sharding requires GQAttention.Config, "
        f"got {type(attention_cfg).__name__}"
    )
    attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()
    attention_cfg.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": dense_activation_placement(tp=attn_x_placement),
            "rope_cache": dense_param_placement(tp=Replicate()),
        },
        in_dst_shardings={
            "x": dense_activation_placement(tp=Replicate()),
            "rope_cache": dense_param_placement(tp=Replicate()),
        },
    )
    set_qkv_linear_sharding(attention_cfg.qkv_linear)
    attention_cfg.wo.sharding_config = rowwise_config(output_sp=enable_sp)


def set_gqa_inner_attention_local_map(
    inner_attention_cfg, *, return_lse: bool = False
) -> None:
    """Install a ``LocalMapConfig`` on an inner-attention config.

    q/k/v arrive as ``(bs, seq, heads, head_dim)`` DTensors with heads
    TP-sharded (``Shard(2)``), regardless of SP. ``local_map`` converts them
    to local tensors before the kernel runs, then wraps outputs back.

    ``return_lse=True`` is for kernels that return ``(output, lse)`` (e.g.,
    GPT-OSS's flash attention with ``return_lse=True``); both outputs share
    the same heads-sharded placement.
    """
    qkv_placements: NamedPlacement = {TP: Shard(2)}
    num_outputs = 2 if return_lse else 1
    inner_attention_cfg.sharding_config = ShardingConfig(
        local_map=LocalMapConfig(
            in_placements=(qkv_placements, qkv_placements, qkv_placements),
            out_placements=(qkv_placements,) * num_outputs,
            in_grad_placements=(qkv_placements, qkv_placements, qkv_placements),
        ),
    )


def set_dense_ffn_sharding(
    feed_forward_cfg,
    *,
    attn_x_placement: Placement,
    enable_sp: bool,
) -> None:
    """Standard dense FFN (``w1``/``w2``/``w3``) TP sharding.

    Shared by llama3, qwen3, llama4, and deepseek_v3. ``attn_x_placement``
    should match the layout that the layer's attention block emits so the
    FFN's input wrap is a no-op redistribute when placements already agree.
    """
    feed_forward_cfg.sharding_config = ShardingConfig(
        in_src_shardings={"x": dense_activation_placement(tp=attn_x_placement)},
        in_dst_shardings={"x": dense_activation_placement(tp=Replicate())},
    )
    feed_forward_cfg.w1.sharding_config = colwise_config()
    feed_forward_cfg.w3.sharding_config = colwise_config()
    feed_forward_cfg.w2.sharding_config = rowwise_config(output_sp=enable_sp)


def set_decoder_sharding_config(
    config, *, loss_parallel: bool, enable_sp: bool
) -> None:
    """Set sharding on root-level configs only: ``tok_embeddings``, ``norm``,
    ``output``, and the root ``freqs_cis`` buffer.

    Per-layer sharding (attention, feed_forward, per-layer norms) is the
    caller's responsibility — this helper does not walk ``config.layers``.

    ``enable_sp=True``  -> SequenceParallel: activations are ``Shard(1)`` between
    the embedding, norm, and output layers.
    ``enable_sp=False`` -> activations stay ``Replicate``; root norm is left
    unsharded (equivalent to the legacy ``NoParallel`` plan).
    """
    activation_tp: Placement = Shard(1) if enable_sp else Replicate()
    loss_tp: Placement = Shard(-1) if loss_parallel else Replicate()

    # freqs_cis buffer on the decoder root: Replicate on all axes.
    config.sharding_config = ShardingConfig(
        state_shardings={"freqs_cis": dense_param_placement(tp=Replicate())},
    )
    config.tok_embeddings.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=Shard(0))},
        in_src_shardings={"input": dense_activation_placement(tp=Replicate())},
        in_dst_shardings={"input": dense_activation_placement(tp=Replicate())},
        out_dst_shardings=dense_activation_placement(tp=activation_tp),
    )
    config.norm.sharding_config = norm_config(enable_sp=enable_sp)

    config.lm_head.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=Shard(0))},
        in_src_shardings={"input": dense_activation_placement(tp=activation_tp)},
        in_dst_shardings={"input": dense_activation_placement(tp=Replicate())},
        out_dst_shardings=dense_activation_placement(tp=loss_tp),
    )
