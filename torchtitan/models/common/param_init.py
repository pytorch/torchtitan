# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parameter initialization helpers for torchtitan models.

Provides composable helper functions that recipe functions (``param_init_fn``)
use to set ``_param_init`` on built model modules via attribute access.

Example usage in a model's ``__init__.py``::

    from torchtitan.models.common.param_init import (
        init_decoder_common,
        init_gq_attention,
        init_feed_forward,
    )

    def setup_llama3_param_init(model: Llama3Model) -> None:
        init_decoder_common(model)
        for i, layer in enumerate(model.layers.values()):
            ...
"""

from functools import partial

import torch.nn as nn

from torchtitan.protocols.module import ParamInitializer, set_param_init


def init_decoder_common(
    model: nn.Module,
    base_std: float = 0.02,
) -> None:
    """Shared init for Decoder top-level modules: tok_embeddings, norm, output.

    Args:
        model: A Decoder (or subclass) instance with ``tok_embeddings``,
            ``norm``, ``output`` attributes and ``config.dim``.
        base_std: Base standard deviation (unused here but kept for API
            consistency with per-layer helpers).
    """
    dim: int = model.config.dim  # pyrefly: ignore [bad-assignment]
    final_out_std = dim**-0.5
    set_param_init(
        model.tok_embeddings,  # pyrefly: ignore [bad-argument-type]
        {"weight": partial(nn.init.normal_, std=1.0)},
    )
    set_param_init(
        model.norm,  # pyrefly: ignore [bad-argument-type]
        {"weight": nn.init.ones_},
    )
    set_param_init(
        model.output,  # pyrefly: ignore [bad-argument-type]
        {
            "weight": partial(
                nn.init.trunc_normal_,
                std=final_out_std,
                a=-3 * final_out_std,
                b=3 * final_out_std,
            )
        },
    )


def init_gq_attention(
    attn: nn.Module,
    *,
    default: ParamInitializer,
    depth: ParamInitializer,
) -> None:
    """Init for GQAttention: wq/wk/wv default, wo depth-scaled.

    Also handles optional q_norm/k_norm (ones) when present.

    Args:
        attn: A GQAttention instance.
        default: Initializer for non-depth-scaled projections (wq, wk, wv).
        depth: Depth-scaled initializer for output projection (wo).
    """
    set_param_init(attn.wq, {"weight": default})  # pyrefly: ignore [bad-argument-type]
    set_param_init(attn.wk, {"weight": default})  # pyrefly: ignore [bad-argument-type]
    set_param_init(attn.wv, {"weight": default})  # pyrefly: ignore [bad-argument-type]
    set_param_init(attn.wo, {"weight": depth})  # pyrefly: ignore [bad-argument-type]
    if getattr(attn, "q_norm", None) is not None:
        set_param_init(
            attn.q_norm,  # pyrefly: ignore [bad-argument-type]
            {"weight": nn.init.ones_},
        )
        set_param_init(
            attn.k_norm,  # pyrefly: ignore [bad-argument-type]
            {"weight": nn.init.ones_},
        )


def init_feed_forward(
    ffn: nn.Module,
    *,
    default: ParamInitializer,
    depth: ParamInitializer,
) -> None:
    """Init for FeedForward: w1 default, w2/w3 depth-scaled.

    Args:
        ffn: A FeedForward instance.
        default: Initializer for gate projection (w1).
        depth: Depth-scaled initializer for down projections (w2, w3).
    """
    set_param_init(ffn.w1, {"weight": default})  # pyrefly: ignore [bad-argument-type]
    set_param_init(ffn.w2, {"weight": depth})  # pyrefly: ignore [bad-argument-type]
    set_param_init(ffn.w3, {"weight": depth})  # pyrefly: ignore [bad-argument-type]


def init_moe(
    moe: nn.Module,
    *,
    default: ParamInitializer,
    depth: ParamInitializer,
) -> None:
    """Init for MoE: experts w1 default, w2/w3 depth-scaled, router depth-scaled.

    Also handles optional shared_experts when present.

    Args:
        moe: A MoE instance.
        default: Initializer for non-depth-scaled expert weights (w1).
        depth: Depth-scaled initializer for expert down weights (w2, w3)
            and router gate.
    """
    set_param_init(
        moe.experts,  # pyrefly: ignore [bad-argument-type]
        {"w1": default, "w2": depth, "w3": depth},
    )
    set_param_init(
        moe.router.gate,  # pyrefly: ignore [missing-attribute]
        {"weight": depth},
    )
    if getattr(moe, "shared_experts", None) is not None:
        init_feed_forward(
            moe.shared_experts,  # pyrefly: ignore [bad-argument-type]
            default=default,
            depth=depth,
        )
