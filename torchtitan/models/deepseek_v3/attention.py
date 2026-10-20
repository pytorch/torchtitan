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

from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.common.attention import (
    Attention,
    FlexAttentionMetadata,
    MLAAttention,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.linear import Linear, maybe_gather_tp_input
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import _RotatePairs, ComplexRoPE, RoPE

__all__ = ["DeepSeekMLAAttention"]


class DeepSeekMLAAttention(Attention):
    """DeepSeek V3 multi-head latent attention."""

    @dataclass(kw_only=True, slots=True)
    class Config(Attention.Config):
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
        inner_attention: MLAAttention.Config  # pyrefly: ignore[bad-override]
        mscale: float = 1.0

        @property
        def softmax_scale(self) -> float:
            qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            scale = qk_head_dim**-0.5
            if self.rope.scaling == "yarn" and self.rope.rope_factor > 1.0:
                mscale = 0.1 * self.mscale * math.log(self.rope.rope_factor) + 1.0
                scale = scale * mscale * mscale
            return scale

        def __post_init__(self) -> None:
            Attention.Config.__post_init__(self)
            if not isinstance(self.inner_attention, MLAAttention.Config):
                raise ValueError(
                    "DeepSeekMLAAttention requires an MLAAttention.Config, "
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
        self.inner_attention = config.inner_attention.build()
        self.wo = config.wo.build()
        self.softmax_scale = config.softmax_scale

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

        kv_c_normed = self.kv_norm(kv)

        with spmd.local():
            k_shared_TR = k_pe.squeeze(1)

        output = self.inner_attention(
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

    # Cat options: _RotateTail concatenates q's untouched int32 words with the rotated ones;
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
            positions: ``[T]`` position ids, or ``None`` for ``0..T-1``.

        Returns:
            The rotated query, same shape as ``q``, and the rotated ``k_pe`` as
            ``[T, 1, qk_rope_head_dim]``.
        """
        k_pe = k_pe.unsqueeze(1)
        if (
            torch.compiler.is_compiling()
            and type(self.rope) is ComplexRoPE
            and q.dtype == torch.bfloat16
            and q.is_contiguous()
        ):
            # Under compile, ComplexRoPE's cache holds adjacent (cos, sin) pairs, so a contiguous
            # bf16 q can rotate inside its row viewed as int32 pairs (_RotateTail). Eager and
            # ComplexRoPE subclasses, which may replace forward, keep split -> rope -> cat below.
            cache = self.rope._reshape_cache(q, positions).unflatten(-1, (-1, 2))
            cos, sin = cache[..., 0], cache[..., 1]
            return _RotateTail.apply(q, cos, sin), _RotatePairs.apply(k_pe, cos, sin)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe, k_pe = self.rope(q_pe, k_pe, positions)
        return torch.cat([q_nope, q_pe], dim=-1), k_pe


class _RotateTail(torch.autograd.Function):
    """Rotate the last ``2 * cos.shape[-1]`` dims of contiguous bf16 rows; copy the rest.

    The rotation runs on the whole row viewed as int32 pairs: slicing the tail out first gives
    a strided view, which ``_RotatePairs`` can only read as halves. A Function because the int32
    views have no autograd. Needs an even storage offset; q comes straight from its projection.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(cos, sin)
        return _RotateTail.rotate(x, cos, sin)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # pyrefly: ignore[bad-override]
        cos, sin = ctx.saved_tensors
        return _RotateTail.rotate(grad, cos, -sin), None, None

    @staticmethod
    def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        words = x.view(torch.int32)
        num_pairs = cos.shape[-1]
        kept, tail = words.split([words.shape[-1] - num_pairs, num_pairs], dim=-1)
        rotated = _RotatePairs.rotate(tail, cos, sin)
        return torch.cat([kept, rotated], dim=-1).view(torch.bfloat16)
