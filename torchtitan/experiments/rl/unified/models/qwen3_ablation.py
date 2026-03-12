# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3 model copy for kernel ablation studies.

This module provides a Qwen3 model variant where individual ops can be
swapped to vLLM's fused kernels to measure their performance impact.

Usage:
    In config_registry.py, swap model_spec to use this model:

        from torchtitan.experiments.rl.unified.models.qwen3_ablation import ablation_model_registry
        model_spec=ablation_model_registry("1.7B")

Ablation 1: Replace nn.RMSNorm with vLLM's fused RMSNorm kernel.
Ablation 2: Replace F.silu(w1(x)) * w3(x) with vLLM's fused SiluAndMul kernel.
Ablation 3: Replace apply_rotary_emb_cos_sin with vLLM's fused rotary_embedding kernel.
Ablation 4: Merge separate wq/wk/wv into single wqkv projection (3 GEMMs → 1).
"""

from dataclasses import fields

import torch
import torch.nn.functional as F
from torch import nn

import torchtitan.models.common.attention as attention_module
from torchtitan.models.common.attention import (
    apply_rotary_emb_complex,
    AttentionMasksType,
    GQAttention,
)
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.rope import _reshape_for_broadcast_cos_sin
from torchtitan.models.common.utils import trunc_normal_
from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3.model import Qwen3Model, Qwen3TransformerBlock
from torchtitan.protocols.model_spec import ModelSpec

# vLLM's fused kernels
from vllm.model_executor.layers.activation import SiluAndMul as VLLMSiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm as VLLMRMSNorm


# ── Ablation 3: Fused RoPE ──

def apply_rotary_emb_cos_sin_vllm(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for apply_rotary_emb_cos_sin using vLLM's fused kernel.

    TorchTitan layout: (bsz, seqlen, n_heads, head_dim)
    vLLM kernel expects: (num_tokens, n_heads * head_dim) and modifies in-place.
    """
    bsz, seqlen, n_heads, head_dim = xq.shape
    n_kv_heads = xk.shape[2]

    # Flatten batch and seq dims for vLLM kernel
    xq_flat = xq.reshape(bsz * seqlen, n_heads * head_dim)
    xk_flat = xk.reshape(bsz * seqlen, n_kv_heads * head_dim)

    # Build position indices
    if positions is not None:
        pos_flat = positions.reshape(bsz * seqlen)
    else:
        pos_flat = torch.arange(seqlen, device=xq.device).unsqueeze(0).expand(bsz, -1).reshape(-1)

    # vLLM's fused rotary embedding kernel (in-place)
    # cos_sin_cache shape: (max_seqlen, head_dim) with cos and sin concatenated
    # vLLM kernel requires cache to match query dtype
    cache = rope_cache.to(dtype=xq.dtype, device=xq.device)
    torch.ops._C.rotary_embedding(
        pos_flat,
        xq_flat,
        xk_flat,
        head_dim,
        cache,
        True,  # is_neox_style (Qwen3 uses neox-style RoPE)
    )

    # Reshape back to TorchTitan layout
    xq_out = xq_flat.reshape(bsz, seqlen, n_heads, head_dim)
    xk_out = xk_flat.reshape(bsz, seqlen, n_kv_heads, head_dim)
    return xq_out, xk_out


# ── Ablation 4: Merged QKV projection ──

class GQAttentionMergedQKV(GQAttention):
    """GQAttention with merged QKV projection (3 matmuls → 1).

    Keeps separate wq/wk/wv parameters for weight loading compatibility,
    but merges them into a single wqkv linear at the first forward call.
    """

    def __init__(self, config: GQAttention.Config, *, dim: int):
        super().__init__(config, dim=dim)
        # Sizes for splitting the merged output
        self._q_size = self.n_heads * self.head_dim
        self._kv_size = self.n_kv_heads * self.head_dim
        self._merged_size = self._q_size + 2 * self._kv_size
        # Merged linear — will be populated from wq/wk/wv weights
        self.wqkv = nn.Linear(dim, self._merged_size, bias=config.bias)
        self._merged = False

    def _merge_qkv_weights(self):
        """Concatenate wq/wk/wv weights into wqkv once."""
        if self._merged:
            return
        with torch.no_grad():
            self.wqkv.weight.copy_(
                torch.cat([self.wq.weight, self.wk.weight, self.wv.weight], dim=0)
            )
            if self.wqkv.bias is not None:
                self.wqkv.bias.copy_(
                    torch.cat([self.wq.bias, self.wk.bias, self.wv.bias], dim=0)
                )
        # Free the separate parameters to save memory
        del self.wq, self.wk, self.wv
        self._merged = True

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._merge_qkv_weights()

        bs, seqlen, _ = x.shape

        # Single merged QKV projection
        qkv = self.wqkv(x)
        xq, xk, xv = qkv.split([self._q_size, self._kv_size, self._kv_size], dim=-1)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        if self.use_rope:
            if self.rope_backend == "cos_sin":
                xq, xk = apply_rotary_emb_cos_sin_vllm(xq, xk, rope_cache, positions)
            else:
                xq, xk = apply_rotary_emb_complex(
                    xq, xk, freqs_cis=rope_cache, positions=positions
                )

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scale_kwargs = {"scale": self.scaling} if self.scaling is not None else {}

        # VLLMAttention uses (q, k, v, *, scale) while native wrappers
        # use (q, k, v, attention_masks=..., **scale_kwargs).
        if isinstance(self.inner_attention, nn.Module) and hasattr(
            self.inner_attention, "vllm_attn"
        ):
            output = self.inner_attention(xq, xk, xv, **scale_kwargs)
        else:
            output = self.inner_attention(
                xq, xk, xv, attention_masks=attention_masks, **scale_kwargs
            )

        return self.wo(output.reshape(bs, seqlen, -1))


# ── Ablation 2: Fused SiluAndMul ──

class FeedForwardVLLMSiluAndMul(FeedForward):
    """FeedForward with vLLM's fused SiluAndMul kernel.

    Original: return self.w2(F.silu(self.w1(x)) * self.w3(x))
      - 3 kernel launches: silu, mul, (implicit in w2)
    Fused:    return self.w2(silu_and_mul(cat(w1(x), w3(x))))
      - 1 fused kernel launch for silu+mul
    """

    def __init__(self, config: FeedForward.Config, *, dim: int):
        super().__init__(config, dim=dim)
        self.silu_and_mul = VLLMSiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat([self.w1(x), self.w3(x)], dim=-1)
        return self.w2(self.silu_and_mul(gate_up))


# ── Ablation 1: Fused RMSNorm + Ablation 2 + Ablation 3 in TransformerBlock ──

class Qwen3TransformerBlockAblation(Qwen3TransformerBlock):
    """Qwen3TransformerBlock with all vLLM fused ops."""

    def __init__(
        self,
        config: Qwen3TransformerBlock.Config,
        *,
        layer_id: int,
        dim: int,
        n_layers: int,
    ):
        TransformerBlock.__init__(self)

        # ── Ablation 4: use merged QKV attention ──
        self.attention = GQAttentionMergedQKV(config.attention, dim=dim)

        self.moe_enabled = config.moe_enabled
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build(dim=dim)
        else:
            assert config.feed_forward is not None
            self.feed_forward = FeedForwardVLLMSiluAndMul(
                config.feed_forward, dim=dim
            )

        self.attention_norm = VLLMRMSNorm(dim, eps=config.norm_eps)
        self.ffn_norm = VLLMRMSNorm(dim, eps=config.norm_eps)

        if config.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * n_layers) ** 0.5


class Qwen3ModelAblation(Qwen3Model):
    """Qwen3Model with vLLM's fused kernels for ablation studies.

    Fused ops:
    - RMSNorm: vLLM's Triton-based fused kernel (all layers + final norm)
    - SiluAndMul: vLLM's fused activation kernel (all FFN layers)
    - RoPE: vLLM's fused rotary_embedding kernel (monkey-patched)
    """

    def __init__(self, config: Qwen3Model.Config):
        # Monkey-patch RoPE before building attention modules
        attention_module.apply_rotary_emb_cos_sin = apply_rotary_emb_cos_sin_vllm

        super().__init__(config)

        # Replace final norm with vLLM's fused version
        self.norm = VLLMRMSNorm(config.dim, eps=config.norm_eps)

        # Replace all layers with the ablation block
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.n_layers):
            self.layers[str(layer_id)] = Qwen3TransformerBlockAblation(
                config.layer,
                layer_id=layer_id,
                dim=config.dim,
                n_layers=config.n_layers,
            )


def ablation_model_registry(flavor: str) -> ModelSpec:
    """Return a ModelSpec identical to qwen3's but using the ablation model."""
    base_spec = model_registry(flavor)

    base_config = base_spec.model

    class _AblationConfig(type(base_config)):
        _owner = Qwen3ModelAblation

    field_values = {}
    for f in fields(base_config):
        if f.init:
            field_values[f.name] = getattr(base_config, f.name)
    config = _AblationConfig(**field_values)

    return ModelSpec(
        name=base_spec.name,
        flavor=base_spec.flavor,
        model=config,
        parallelize_fn=base_spec.parallelize_fn,
        pipelining_fn=base_spec.pipelining_fn,
        build_loss_fn=base_spec.build_loss_fn,
        post_optimizer_build_fn=base_spec.post_optimizer_build_fn,
        state_dict_adapter=base_spec.state_dict_adapter,
    )
