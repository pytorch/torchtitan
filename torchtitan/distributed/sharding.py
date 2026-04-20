# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.models.common.attention import FusedQKVLinear, GQAttention, QKVLinear
from torchtitan.protocols.sharding import MeshDimName, ShardingSpec

TP = MeshDimName.TP


def colwise_spec() -> ShardingSpec:
    """ColwiseParallel: weight Shard(0), output Shard(-1)."""
    return ShardingSpec(
        state_shardings={"weight": {TP: Shard(0)}, "bias": {TP: Shard(0)}},
        out_shardings={TP: Shard(-1)},
    )


def rowwise_spec(*, output_sp: bool = False) -> ShardingSpec:
    """RowwiseParallel: weight Shard(1), bias Replicate (no-op if bias absent).

    ``output_sp=True``  -> output ``Shard(1)`` (reduce-scatter into SP region).
    ``output_sp=False`` -> output ``Replicate()`` (all-reduce).
    """
    output: Placement = Shard(1) if output_sp else Replicate()
    return ShardingSpec(
        state_shardings={"weight": {TP: Shard(1)}, "bias": {TP: Replicate()}},
        out_shardings={TP: output},
    )


def sequence_parallel_spec() -> ShardingSpec:
    """SequenceParallel norm: weight Replicate, activations Shard(1)."""
    return ShardingSpec(
        state_shardings={"weight": {TP: Replicate()}},
        input_layouts={"input": {TP: Shard(1)}},
        in_shardings={"input": {TP: Shard(1)}},
        out_shardings={TP: Shard(1)},
    )


def replicate_norm_spec() -> ShardingSpec:
    """Plain-TP norm (no SP): weight Replicate, activations pass through.

    Needed so the norm's weight becomes a DTensor alongside DTensor
    activations; otherwise we'd mix plain Tensor and DTensor inside the op.
    """
    return ShardingSpec(state_shardings={"weight": {TP: Replicate()}})


def set_qkv_linear_sharding(qkv_linear_cfg) -> None:
    """Colwise-shard each Q/K/V projection of a ``BaseQKVLinear``.

    Handles both ``QKVLinear`` (separate ``wq`` + ``wkv``) and
    ``FusedQKVLinear`` (single ``wqkv``).
    """
    if isinstance(qkv_linear_cfg, FusedQKVLinear.Config):
        qkv_linear_cfg.wqkv.sharding_spec = colwise_spec()
    elif isinstance(qkv_linear_cfg, QKVLinear.Config):
        qkv_linear_cfg.wq.sharding_spec = colwise_spec()
        qkv_linear_cfg.wkv.sharding_spec = colwise_spec()
    else:
        raise TypeError(
            f"set_qkv_linear_sharding requires QKVLinear.Config or "
            f"FusedQKVLinear.Config, got {type(qkv_linear_cfg).__name__}"
        )


def set_gqa_attention_sharding(attention_cfg, *, enable_sp: bool) -> None:
    """Standard GQA attention (``qkv_linear``/``wo``) TP sharding.

    Shared by llama3, qwen3, and llama4 — all three have a GQA block whose
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
    attention_cfg.sharding_spec = ShardingSpec(
        input_layouts={
            "x": {TP: attn_x_placement},
            "rope_cache": {TP: Replicate()},
        },
        in_shardings={
            "x": {TP: Replicate()},
            "rope_cache": {TP: Replicate()},
        },
    )
    set_qkv_linear_sharding(attention_cfg.qkv_linear)
    attention_cfg.wo.sharding_spec = rowwise_spec(output_sp=enable_sp)


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
    feed_forward_cfg.sharding_spec = ShardingSpec(
        input_layouts={"x": {TP: attn_x_placement}},
        in_shardings={"x": {TP: Replicate()}},
    )
    feed_forward_cfg.w1.sharding_spec = colwise_spec()
    feed_forward_cfg.w3.sharding_spec = colwise_spec()
    feed_forward_cfg.w2.sharding_spec = rowwise_spec(output_sp=enable_sp)


def set_decoder_sharding_spec(config, *, loss_parallel: bool, enable_sp: bool) -> None:
    """Set sharding on tok_embeddings, norm, output (and root ``freqs_cis`` buffer)
    — shared by all decoders.

    ``enable_sp=True``  -> SequenceParallel: activations are ``Shard(1)`` between
    the embedding, norm, and output layers.
    ``enable_sp=False`` -> activations stay ``Replicate``; root norm is left
    unsharded (equivalent to the legacy ``NoParallel`` plan).
    """
    activation_sharding: Placement = Shard(1) if enable_sp else Replicate()
    loss_sharding: Placement = Shard(-1) if loss_parallel else Replicate()

    # freqs_cis buffer on the decoder root: Replicate on all dims.
    config.sharding_spec = ShardingSpec(
        state_shardings={"freqs_cis": {TP: Replicate()}},
    )
    config.tok_embeddings.sharding_spec = ShardingSpec(
        state_shardings={"weight": {TP: Shard(0)}},
        input_layouts={"input": {TP: Replicate()}},
        in_shardings={"input": {TP: Replicate()}},
        out_shardings={TP: activation_sharding},
    )
    config.norm.sharding_spec = (
        sequence_parallel_spec() if enable_sp else replicate_norm_spec()
    )

    config.output.sharding_spec = ShardingSpec(
        state_shardings={"weight": {TP: Shard(0)}},
        input_layouts={"input": {TP: activation_sharding}},
        in_shardings={"input": {TP: Replicate()}},
        out_shardings={TP: loss_sharding},
    )
