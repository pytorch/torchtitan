# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.common.attention import (
    Attention,
    FlexAttentionMetadata,
    InnerAttention,
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)
from torchtitan.models.common.linear import Linear, maybe_gather_tp_input
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE

__all__ = ["DeepSeekMLAAttention"]


class DeepSeekMLAAttention(Attention):
    """
    Multi-head latent attention (MLA) module.

    This is DeepSeek V3-specific and NOT shared with other models.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Attention.Config):
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
        inner_attention: InnerAttention.Config = field(
            default_factory=VarlenInnerAttention.Config
        )
        mscale: float = 1.0

        @property
        def softmax_scale(self) -> float:
            qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            scale = qk_head_dim**-0.5
            if self.rope.scaling == "yarn" and self.rope.rope_factor > 1.0:
                mscale = 0.1 * self.mscale * math.log(self.rope.rope_factor) + 1.0
                scale = scale * mscale * mscale
            return scale

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
        self.softmax_scale = config.softmax_scale

        self.inner_attention = config.inner_attention.build()
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

        # Key-value projection
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # rope, the query concat, and kv_norm read the query and wkv_a projection
        # outputs with bare ops.
        remat.recompute_needs_tensor(q, kv, k_pe)
        q, k_pe = self._apply_rope_to_q_and_k_pe(q, k_pe, positions)

        kv = self.wkv_b(self.kv_norm(kv))

        with (
            spmd.local()
        ):  # QKV even shard unflatten, but the expand is truly local SPMD
            kv = kv.view(num_tokens, -1, self.qk_nope_head_dim + self.v_head_dim)
            # The key concat reads the wkv_b projection output with bare ops.
            remat.recompute_needs_tensor(kv)
            k, v = self._split_kv_and_build_k(kv, k_pe)
            if spmd.is_type_checking() and not torch.compiler.is_compiling():
                for t in [k, v]:
                    spmd.assert_type(
                        t,
                        spmd.V,
                        spmd.PartitionSpec(("dp", "cp"), "tp", None),
                    )

        output = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(q, k, v, attention_metadata=attention_metadata, scale=self.softmax_scale)
        # The copy below reads the inner_attention output with bare ops.
        remat.recompute_needs_tensor(output)
        output = output.contiguous().view(num_tokens, -1)
        return self.wo(output)

    # Cat options: rotate_tail concatenates q's untouched int32 words with the rotated ones;
    # Inductor's default pointwise cat makes that masked loads, slower than a plain cat.
    # No pointwise autotuning, as in ComplexRoPE.forward: RoPE is inlined here, so this region's options apply.
    @local_compile(
        "mla_qk",
        batch_invariant=True,
        options={
            "max_pointwise_cat_inputs": 0,
            "max_complex_pointwise_cat_inputs": 0,
            "triton.autotune_pointwise": False,
        },
    )
    def _apply_rope_to_q_and_k_pe(
        self,
        q: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate the rope dims of ``q`` and ``k_pe``.

        Args:
            q: ``[T, H, qk_nope_head_dim + qk_rope_head_dim]`` query.
            k_pe: ``[T, qk_rope_head_dim]`` key rope dims, shared by all heads.

        Returns:
            The rotated query, same shape as ``q``, and the rotated ``k_pe`` as
            ``[T, 1, qk_rope_head_dim]``.
        """
        return self.rope.rotate_tail(q, k_pe.unsqueeze(1), positions)

    @local_compile("mla_qk", batch_invariant=True)
    def _split_kv_and_build_k(
        self,
        kv: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the key from ``kv``'s nope dims and the shared ``k_pe``.

        Args:
            kv: ``[T, H, qk_nope_head_dim + v_head_dim]`` packed key nope dims
                and value.
            k_pe: ``[T, 1, qk_rope_head_dim]`` rotated key rope dims.

        Returns:
            Key ``[T, H, qk_nope_head_dim + qk_rope_head_dim]`` and value
            ``[T, H, v_head_dim]``.
        """
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1)], dim=-1)
        if torch.compiler.is_compiling():
            # A strided value view gets a gradient from attention in another
            # layout, which the compiled backward copies before using it. A
            # dense value avoids that copy and fuses into the key's kernel.
            v = v.contiguous()
        return k, v
