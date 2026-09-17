# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V3 multi-head latent attention."""

import math
from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.models.common.attention import (
    BaseAttention,
    FlexAttentionMetadata,
    MLAFlexInnerAttention,
    MLAInnerAttention,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.linear import Linear, maybe_gather_tp_input
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

__all__ = ["DeepSeekV3MLAAttention"]


class DeepSeekV3MLAAttention(BaseAttention):
    """DeepSeek V3 multi-head latent attention."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        dim: int
        wq: Linear.Config | None = None
        wq_a: Linear.Config | None = None
        wq_b: Linear.Config | None = None
        wkv_a: Linear.Config
        wkv_b: Linear.Config
        wo: Linear.Config
        q_lora_rank: int = 0
        kv_lora_rank: int = 512
        q_norm: RMSNorm.Config
        kv_norm: RMSNorm.Config
        qk_nope_head_dim: int = 128
        qk_rope_head_dim: int = 64
        v_head_dim: int = 128
        rope: RoPE.Config
        inner_attention: Module.Config = field(
            default_factory=MLAFlexInnerAttention.Config
        )
        mscale: float = 1.0

        def __post_init__(self) -> None:
            BaseAttention.Config.__post_init__(self)
            if not isinstance(self.inner_attention, MLAInnerAttention.Config):
                raise ValueError(
                    "DeepSeekV3MLAAttention requires an MLAInnerAttention.Config, "
                    f"but got {type(self.inner_attention).__qualname__}."
                )

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        if self.q_lora_rank == 0:
            assert config.wq is not None, "wq is required when q_lora_rank == 0"
            self.wq = config.wq.build()
        else:
            assert (
                config.wq_a is not None and config.wq_b is not None
            ), "wq_a and wq_b are required when q_lora_rank > 0"
            self.wq_a = config.wq_a.build()
            self.q_norm = config.q_norm.build()
            self.wq_b = config.wq_b.build()

        # TODO(fegin): revisit
        # https://github.com/pytorch/torchtitan/pull/2785#discussion_r3034078575
        self.wkv_a = config.wkv_a.build()
        self.kv_norm = config.kv_norm.build()
        self.wkv_b = config.wkv_b.build()
        self.wo = config.wo.build()
        self.softmax_scale = self.qk_head_dim**-0.5

        if config.rope.scaling == "yarn" and config.rope.rope_factor > 1.0:
            mscale = 0.1 * config.mscale * math.log(config.rope.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        inner_attention = config.inner_attention.build()
        assert isinstance(inner_attention, MLAInnerAttention)
        self.inner_attention = inner_attention
        self.rope = config.rope.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata,
        positions: torch.Tensor | None = None,
    ):
        # The MLA projection branches all consume x. Gather once here.
        x = maybe_gather_tp_input(self, x)

        num_tokens = x.shape[0]

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_a(x)
            # q_norm reads the wq_a projection output with bare ops.
            remat.recompute_needs_tensor(q)
            q = self.wq_b(self.q_norm(q))

        # TODO(pianpwk): same QKV:S(1) unflatten case handled by even sharding
        with spmd.local():
            q = q.view(num_tokens, -1, self.qk_head_dim)
            if spmd.is_type_checking():
                spmd.assert_type(
                    q,
                    spmd.V,
                    spmd.PartitionSpec(("dp", "cp"), "tp", None),
                )

        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Key-value projection
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # rope, the query concat, and kv_norm read the query and wkv_a projection
        # outputs with bare ops.
        remat.recompute_needs_tensor(q_nope, q_pe, kv, k_pe)
        q_pe, k_pe = self.rope(q_pe, k_pe.unsqueeze(1), positions)
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = self.wkv_b(self.kv_norm(kv))

        with spmd.local():
            kv_THP = kv.view(num_tokens, -1, self.qk_nope_head_dim + self.v_head_dim)
            k_shared_TR = k_pe.squeeze(1)

        output = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            q,
            kv_THP,
            k_shared_TR,
            attention_metadata=attention_metadata,
            scale=self.softmax_scale,
        )
        # The copy below reads the inner_attention output with bare ops.
        remat.recompute_needs_tensor(output)
        output = output.contiguous().view(num_tokens, -1)
        return self.wo(output)
