# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared model-agnostic ViT building blocks for VLM vision encoders: a
block-diagonal FlexInnerAttention mask helper and the pre-norm transformer block
(attention + MLP) over token-major visual patches.

RoPE differs per model, so each encoder passes it through the block to the
attention as two per-forward args: ``rope_cache`` (a tensor, so config-based
sharding can annotate it before it meets the head-sharded q/k) and
``rope_apply`` (a pass-through callable ``(q, k, rope_cache) -> (q, k)``).

Shape suffixes:
- T = packed visual tokens
- D = vision dim
- H = num heads
- Dh = head dim
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import torch_remat as remat
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.attention import FlexInnerAttention, local_head_split
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import GELU, LayerNorm, residual_add, RMSNorm
from torchtitan.protocols.module import Module

compiled_create_block_mask = torch.compile(create_block_mask)

# Applies rotary position embedding: (query, key, rope_cache) -> (query, key).
RopeApply = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


class InvariantRowParallelLinear(Linear):
    """Row-parallel vision projection with an invariant TP output.

    Vision residual activations remain invariant even when decoder sequence
    parallelism is enabled, so this boundary always performs ``P -> I``. Like
    ``RowParallelLinear``, the projection and the reduction are the remat
    regions ``<fqn>.linear`` and ``<fqn>.tp_reduce`` under one save policy.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        weight, bias = self._flatten_weight_and_bias()
        linear_fn = remat.region(
            self._linear,
            self.remat_region_name("linear"),
            recompute=self.remat_should_recompute("linear"),
        )
        if bias is not None and tp_group is not None:
            bias = spmd.convert(
                bias,
                tp_group,
                src=spmd.I,
                dst=spmd.P,
                expert_mode=True,
            )
            # The selected local compute may be native, LoRA, or quantized.
            # Its row-sharded operands and bias jointly produce a partial output.
            # TODO: Remove this suppression once spmd_types recognizes the
            # rowwise F.linear type combination [V, V, P] -> P.
            with spmd.no_typecheck():
                output = linear_fn(input, weight, bias)
            if spmd.is_type_checking():
                spmd.assert_local_type_like(
                    output,
                    input,
                    {tp_group: spmd.P},  # pyrefly: ignore [bad-argument-type]
                )
        else:
            output = linear_fn(input, weight, bias)
        if tp_group is not None:
            # tp_reduce shares the linear policy, so a saved pair never replays
            # the reduction and never needs the TP-times larger partial output.
            output = remat.region(
                spmd.redistribute,
                self.remat_region_name("tp_reduce"),
                recompute=self.remat_should_recompute("linear"),
            )(
                output,
                tp_group,
                src=spmd.P,
                dst=spmd.I,
                backward_options={"op_dtype": output.dtype},
            )
        return self._unflatten_output(output)


def create_block_diagonal_mask(
    segment_lengths: torch.Tensor,
    total_tokens: int,
    device: torch.device,
) -> BlockMask:
    """Create a FlexInnerAttention mask over contiguous packed segments."""
    segment_ids = torch.repeat_interleave(
        torch.arange(segment_lengths.shape[0], device=device, dtype=torch.int32),
        segment_lengths.to(device=device, dtype=torch.int32),
        # Avoid reading segment_lengths.sum() back to the host to size the
        # output; the packed token count is already available from its shape.
        output_size=total_tokens,
    )

    def mask_mod(b, h, q_idx, kv_idx):
        return segment_ids[q_idx] == segment_ids[kv_idx]

    return compiled_create_block_mask(
        mask_mod,
        1,
        None,
        total_tokens,
        total_tokens,
        device=device,
    )


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each Linear declares its own remat regions.
        hidden_TF = self.linear_fc1(x)
        hidden_TF = remat.region(
            self.act_fn,
            self.remat_region_name("activation"),
            recompute=self.remat_should_recompute("activation"),
        )(hidden_TF)
        return self.linear_fc2(hidden_TF)


class VisionAttention(Module):
    """Multi-head self-attention with FlexInnerAttention over visual patches.

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
        inner_attention: Module.Config = field(
            default_factory=FlexInnerAttention.Config
        )

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

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        rope_apply: RopeApply,
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        num_tokens = x.shape[0]

        # Each Linear declares its own remat regions.
        q_TD, k_TD, v_TD = self.wq(x), self.wk(x), self.wv(x)
        # -1 infers the head count locally (= num_heads / TP under tensor
        # parallelism, where wq/wk/wv are colwise-sharded).
        q_THDh = local_head_split(q_TD, self.head_dim)
        k_THDh = local_head_split(k_TD, self.head_dim)
        v_THDh = local_head_split(v_TD, self.head_dim)
        q_THDh, k_THDh = remat.region(
            rope_apply,
            self.remat_region_name("rope"),
            recompute=self.remat_should_recompute("rope"),
        )(q_THDh, k_THDh, rope_cache)

        out_THDh = remat.region(
            self.flex_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(q_THDh, k_THDh, v_THDh, attention_masks=attention_mask)
        remat.recompute_needs_tensor(out_THDh)
        out_TD = out_THDh.reshape(num_tokens, -1)
        out_TD = self.proj(out_TD)
        return out_TD


class VisionTransformerBlock(Module):
    """Pre-norm transformer block: norm -> attn -> residual -> norm -> mlp."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        # MoonViT normalizes with RMSNorm; Qwen3.5 and Muse Glimmer use LayerNorm.
        norm1: LayerNorm.Config | RMSNorm.Config
        norm2: LayerNorm.Config | RMSNorm.Config
        attn: VisionAttention.Config
        mlp: VisionMLP.Config

    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = config.norm1.build()
        self.norm2 = config.norm2.build()
        self.attn = config.attn.build()
        self.mlp = config.mlp.build()

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        rope_apply: RopeApply,
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        attn_out = self.attn(
            self.norm1(x),
            rope_cache=rope_cache,
            rope_apply=rope_apply,
            attention_mask=attention_mask,
        )
        x = residual_add(self, x, attn_out, "attention_residual")
        return residual_add(self, x, self.mlp(self.norm2(x)), "ffn_residual")
