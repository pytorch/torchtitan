# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared model-agnostic ViT building blocks for VLM vision encoders: a
block-diagonal FlexAttention mask helper and the pre-norm transformer block
(attention + MLP) over a padded ``(N, P, D)`` batch.

RoPE differs per model, so each encoder passes it through the block to the
attention as two per-forward args: ``rope_cache`` (a tensor, so config-based
sharding can DTensor-wrap it before it meets the head-sharded q/k) and
``rope_apply`` (a pass-through callable ``(q, k, rope_cache) -> (q, k)``).

Shape suffixes:
- N = num visual items
- P = max patches per item (padded)
- D = vision dim
- H = num heads
- Dh = head dim
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from torchtitan.models.common import Linear
from torchtitan.models.common.attention import FlexAttention, local_head_split
from torchtitan.models.common.nn_modules import GELU, LayerNorm
from torchtitan.observability import tensor_logging
from torchtitan.protocols.module import Module

compiled_create_block_mask = torch.compile(create_block_mask)

# Applies rotary position embedding: (query, key, rope_cache) -> (query, key).
RopeApply = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


def get_vision_block_mask_mod(num_patches: torch.Tensor) -> Callable:
    """Block-diagonal mask: each visual item attends only to its own patches.

    Args:
        num_patches: (N,) real (non-padding) patch count per visual item (N is
            the number of visual items, i.e. images/videos in the batch).
    """

    def mask_mod(b, h, q_idx, kv_idx):
        valid_q = q_idx < num_patches[b]
        valid_kv = kv_idx < num_patches[b]
        return valid_q & valid_kv

    return mask_mod


class VisionMLP(Module):
    """Feed-forward network with GELU activation (fc1 -> act -> fc2)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        fc1: Linear.Config
        fc2: Linear.Config
        act_fn: GELU.Config = field(
            default_factory=lambda: GELU.Config(approximate="tanh")
        )

    def __init__(self, config: Config):
        super().__init__()
        self.linear_fc1 = config.fc1.build()
        self.linear_fc2 = config.fc2.build()
        self.act_fn = config.act_fn.build()
        tensor_logging.register_fwd_bwd(self, ["act_out"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act_out = self.act_fn(self.linear_fc1(x))
        tensor_logging.log_fwd_bwd_stats(self, act_out=act_out)
        return self.linear_fc2(act_out)


class VisionAttention(Module):
    """Multi-head self-attention with FlexAttention over a padded batch.

    Separate q/k/v projections (clean per-head ColwiseParallel under TP). RoPE is
    applied via the injected ``rope_apply`` callable so this class is reused
    across models with different rotary formulations.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_heads: int
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config
        proj: Linear.Config
        inner_attention: Module.Config = field(default_factory=FlexAttention.Config)

    def __init__(self, config: Config):
        super().__init__()
        if config.dim % config.num_heads != 0:
            raise ValueError(
                f"VisionAttention dim ({config.dim}) must be divisible by "
                f"num_heads ({config.num_heads})."
            )
        self.head_dim = config.dim // config.num_heads

        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.proj = config.proj.build()
        self.flex_attention = config.inner_attention.build()
        tensor_logging.register_fwd_bwd(self, ["xq", "xk", "xv", "head_out"])

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        rope_apply: RopeApply,
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        N, P, _ = x.shape

        # -1 infers the head count locally (= num_heads / TP under tensor
        # parallelism, where wq/wk/wv are colwise-sharded).
        q_NPHDh = local_head_split(self.wq(x), self.head_dim)
        k_NPHDh = local_head_split(self.wk(x), self.head_dim)
        v_NPHDh = local_head_split(self.wv(x), self.head_dim)
        tensor_logging.log_fwd_bwd_stats(
            self,
            xq=q_NPHDh,
            xk=k_NPHDh,
            xv=v_NPHDh,
        )

        q_NPHDh, k_NPHDh = rope_apply(q_NPHDh, k_NPHDh, rope_cache)

        out_NPHDh = self.flex_attention(
            q_NPHDh, k_NPHDh, v_NPHDh, attention_masks=attention_mask
        )
        tensor_logging.log_fwd_bwd_stats(self, head_out=out_NPHDh)
        out_NPD = out_NPHDh.reshape(N, P, -1)
        return self.proj(out_NPD)


class VisionTransformerBlock(Module):
    """Pre-norm transformer block: norm -> attn -> residual -> norm -> mlp."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm1: LayerNorm.Config
        norm2: LayerNorm.Config
        attn: VisionAttention.Config
        mlp: VisionMLP.Config

    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = config.norm1.build()
        self.norm2 = config.norm2.build()
        self.attn = config.attn.build()
        self.mlp = config.mlp.build()
        # Record each normalized input, branch output, and post-residual stream.
        # A residual sum's statistics cannot be reconstructed from its operands.
        tensor_logging.register_fwd_bwd(
            self,
            [
                "post_ln1",
                "post_attn",
                "post_attn_residual",
                "post_ln2",
                "post_mlp",
                "post_mlp_residual",
            ],
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        rope_apply: RopeApply,
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        post_ln1 = self.norm1(x)
        post_attn = self.attn(
            post_ln1,
            rope_cache=rope_cache,
            rope_apply=rope_apply,
            attention_mask=attention_mask,
        )
        post_attn_residual = x + post_attn
        post_ln2 = self.norm2(post_attn_residual)
        post_mlp = self.mlp(post_ln2)
        post_mlp_residual = post_attn_residual + post_mlp
        tensor_logging.log_fwd_bwd_stats(
            self,
            post_ln1=post_ln1,
            post_attn=post_attn,
            post_attn_residual=post_attn_residual,
            post_ln2=post_ln2,
            post_mlp=post_mlp,
            post_mlp_residual=post_mlp_residual,
        )
        return post_mlp_residual
