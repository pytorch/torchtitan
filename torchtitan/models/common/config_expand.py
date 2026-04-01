# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config expansion helpers for filling ``field(init=False)`` fields.

These helpers set dimensional fields on sub-configs so that ``build()``
calls require no kwargs.  They are called from each model registry's
``expand_layer_configs()`` after deep-copying layer templates and
resolving ``DeferredCallable`` markers.
"""

from copy import deepcopy

from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe.moe import MoE


def fill_gqa_fields(attn: GQAttention.Config, dim: int) -> None:
    """Fill expanded fields on a GQAttention.Config."""
    n_kv_heads = attn.n_heads if attn.n_kv_heads is None else attn.n_kv_heads
    head_dim = attn.head_dim if attn.head_dim is not None else dim // attn.n_heads
    attn.dim = dim

    # Expand wqkv template into wq/wk/wv with specific dimensions
    attn.wq = deepcopy(attn.wqkv)
    attn.wq.in_features = dim
    attn.wq.out_features = attn.n_heads * head_dim
    attn.wk = deepcopy(attn.wqkv)
    attn.wk.in_features = dim
    attn.wk.out_features = n_kv_heads * head_dim
    attn.wv = deepcopy(attn.wqkv)
    attn.wv.in_features = dim
    attn.wv.out_features = n_kv_heads * head_dim

    # wo
    attn.wo.in_features = attn.n_heads * head_dim
    attn.wo.out_features = dim

    # Optional QK norms
    if attn.q_norm is not None:
        attn.q_norm.normalized_shape = head_dim
    if attn.k_norm is not None:
        attn.k_norm.normalized_shape = head_dim


def fill_ffn_fields(ffn: FeedForward.Config, dim: int) -> None:
    """Fill expanded fields on a FeedForward.Config."""
    ffn.dim = dim

    # w1 (base init)
    ffn.w1.in_features = dim
    ffn.w1.out_features = ffn.hidden_dim

    # Expand w2w3 template into w2/w3 with specific dimensions
    ffn.w2 = deepcopy(ffn.w2w3)
    ffn.w2.in_features = ffn.hidden_dim
    ffn.w2.out_features = dim
    ffn.w3 = deepcopy(ffn.w2w3)
    ffn.w3.in_features = dim
    ffn.w3.out_features = ffn.hidden_dim


def fill_moe_fields(moe: MoE.Config, dim: int) -> None:
    """Fill expanded fields on a MoE.Config (and its sub-configs)."""
    moe.dim = dim

    # experts
    moe.experts.dim = dim
    moe.experts.hidden_dim = moe.hidden_dim
    moe.experts.num_experts = moe.num_experts

    # router
    moe.router.dim = dim
    moe.router.num_experts = moe.num_experts
    moe.router.gate.in_features = dim
    moe.router.gate.out_features = moe.num_experts

    # shared_experts (FeedForward)
    if moe.shared_experts is not None:
        fill_ffn_fields(moe.shared_experts, dim)


def fill_decoder_fields(config) -> None:
    """Fill top-level Decoder fields (tok_embeddings, norm, output).

    Args:
        config: A Decoder.Config (or subclass) with dim, vocab_size set.
    """
    dim = config.dim
    vocab_size = config.vocab_size

    config.tok_embeddings.num_embeddings = vocab_size
    config.tok_embeddings.embedding_dim = dim
    config.norm.normalized_shape = dim
    config.output.in_features = dim
    config.output.out_features = vocab_size
