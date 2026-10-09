# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V3 multi-head latent attention."""

import math
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.models.common.attention import (
    AttentionBackend,
    BaseAttention,
    FlexAttentionMetadata,
    MLAAttention,
    register_mla_wkv_b_state_dict_hooks,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.linear import Linear, maybe_gather_tp_input
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE

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
        wo: Linear.Config
        q_lora_rank: int = 0
        kv_lora_rank: int = 512
        q_norm: RMSNorm.Config
        kv_norm: RMSNorm.Config
        qk_nope_head_dim: int = 128
        qk_rope_head_dim: int = 64
        v_head_dim: int = 128
        rope: RoPE.Config
        mla_attention: MLAAttention.Config
        mscale: float = 1.0

        @property
        def attention_backend(self) -> MLAAttention.Config:
            return self.mla_attention

        def __post_init__(self) -> None:
            BaseAttention.Config.__post_init__(self)
            if not isinstance(self.mla_attention, MLAAttention.Config):
                raise ValueError(
                    "DeepSeekV3MLAAttention requires an MLAAttention.Config, "
                    f"but got {type(self.mla_attention).__qualname__}."
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
        self.wo = config.wo.build()
        self.softmax_scale = self.qk_head_dim**-0.5

        if config.rope.scaling == "yarn" and config.rope.rope_factor > 1.0:
            mscale = 0.1 * config.mscale * math.log(config.rope.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.mla_attention = config.mla_attention.build()
        self.rope = config.rope.build()
        register_mla_wkv_b_state_dict_hooks(self)

    @property
    def attention_metadata_key(self) -> type[AttentionBackend]:
        return self.mla_attention.attention_metadata_key

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

        kv_c_normed = self.kv_norm(kv)

        with spmd.local():
            k_shared_TR = k_pe.squeeze(1)

        output = remat.region(
            self.mla_attention,
            self.remat_region_name("mla_attention"),
            recompute=self.remat_should_recompute("mla_attention"),
        )(
            q,
            kv_c_normed,
            k_shared_TR,
            attention_metadata=attention_metadata,
            scale=self.softmax_scale,
        )
        # The copy below reads the MLA attention output with bare ops.
        remat.recompute_needs_tensor(output)
        output = output.contiguous().view(num_tokens, -1)
        return self.wo(output)
