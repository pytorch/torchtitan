# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Qwen3 model compatible with vLLM's implementation
# Uses merged gate_up projections and vLLM Flash Attention

import torch
from torch import nn

from torchtitan.components.tokenizer import BaseTokenizer

# Import gradient-enabled operations from experiment utilities
from torchtitan.experiments.rl.vllm_compat.batch_invariant_backward import (
    rms_norm_with_gradients,
    silu_and_mul_with_gradients,
)

from torchtitan.models.common import trunc_normal_
from torchtitan.models.common.attention import AttentionMasksType

# Import from main torchtitan
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.protocols.model import BaseModel

# Import from local experiment's models
from ..attention import VLLMCompatibleFlashAttention


# RoPE functions (same as original)
def precompute_rope_cache(
    dim: int, max_seq_len: int, base: float = 1_000_000.0
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)
    idx_theta = torch.outer(t, freqs).float()
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return rope_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reshape_for_broadcast(rope_cache: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape frequency tensor for broadcasting."""
    ndim = x.ndim
    assert ndim > 1
    _, seqlen, _, head_dim = x.shape
    rope_cache = rope_cache[0:seqlen]
    assert rope_cache.shape == (seqlen, head_dim * 2)
    shape = [-1, seqlen, 1, head_dim * 2]
    return rope_cache.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, rope_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = xq.shape[-1]
    rope_cache = reshape_for_broadcast(rope_cache, xq)
    cos = rope_cache[..., :head_dim].to(dtype=xq.dtype, device=xq.device)
    sin = rope_cache[..., head_dim:].to(dtype=xq.dtype, device=xq.device)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class VLLMRMSNorm(nn.Module):
    """
    RMSNorm using vLLM's exact Triton kernel for bitwise determinism.
    Compatible with PyTorch's nn.RMSNorm interface but uses vLLM's implementation.

    Supports gradients through a custom autograd function that uses vLLM's
    kernel for forward and batch-invariant PyTorch ops for backward.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use vLLM's RMSNorm with gradient support for training
        return rms_norm_with_gradients(x, self.weight, self.eps)

    def reset_parameters(self):
        nn.init.ones_(self.weight)


class FeedForwardVLLMCompat(nn.Module):
    """
    FeedForward module compatible with vLLM implementation.
    Uses merged gate_up projection like vLLM.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        # Merged gate and up projections (like vLLM's gate_up_proj)
        self.gate_up_proj = nn.Linear(dim, hidden_dim * 2, bias=False)

        # Down projection (like vLLM's down_proj)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # Project to gate and up in one go
        gate_up = self.gate_up_proj(x)
        # Apply SiluAndMul activation with gradient support
        activated = silu_and_mul_with_gradients(gate_up)
        # Project down
        output = self.down_proj(activated)
        return output

    def init_weights(self, init_std: float):
        # Initialize like vLLM
        trunc_normal_(self.gate_up_proj.weight, mean=0.0, std=0.02)
        trunc_normal_(self.down_proj.weight, mean=0.0, std=init_std)


class Attention(nn.Module):
    """
    Multi-head attention module compatible with vLLM.
    """

    def __init__(self, model_args: Qwen3Model.Config):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim
        self.scaling = self.head_dim**-0.5

        # QK norm (Qwen3 specific) - use vLLM's RMSNorm
        if model_args.qk_norm:
            self.q_norm = VLLMRMSNorm(self.head_dim, eps=model_args.norm_eps)
            self.k_norm = VLLMRMSNorm(self.head_dim, eps=model_args.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # QKV projections
        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )

        # Always use vLLM compatible flash attention
        self.inner_attention = VLLMCompatibleFlashAttention()

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            trunc_normal_(linear.weight, mean=0.0, std=0.02)
        trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
    ):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape to heads
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # Apply QK norm
        if self.q_norm:
            xq = self.q_norm(xq)
        if self.k_norm:
            xk = self.k_norm(xk)

        # Apply rotary embedding
        xq, xk = apply_rotary_emb(xq, xk, rope_cache)

        # Repeat k/v heads if needed
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        # Transpose for attention
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        # Apply flash attention (vLLM compatible, no flex attention)
        assert (
            attention_masks is None
        ), "vLLM compat mode doesn't use flex attention masks"
        output = self.inner_attention(xq, xk, xv, scale=self.scaling)

        # Transpose back
        output = output.transpose(1, 2).contiguous()
        output = output.view(bs, seqlen, -1)

        return self.wo(output)


class TransformerBlock(nn.Module):
    """
    TransformerBlock with vLLM-compatible FFN.
    """

    def __init__(self, layer_id: int, model_args: Qwen3Model.Config):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim

        self.attention = Attention(model_args)

        # Use vLLM-compatible FFN with merged projections
        self.feed_forward = FeedForwardVLLMCompat(
            dim=model_args.dim, hidden_dim=model_args.hidden_dim
        )

        self.attention_norm = VLLMRMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = VLLMRMSNorm(model_args.dim, eps=model_args.norm_eps)

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
    ):
        # Self attention with residual
        attn_norm_out = self.attention_norm(x)
        x = x + self.attention(attn_norm_out, rope_cache, attention_masks)

        # FFN with residual
        ffn_norm_out = self.ffn_norm(x)
        x = x + self.feed_forward(ffn_norm_out)

        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class Qwen3VLLMCompatModel(BaseModel):
    """
    Qwen3 model with vLLM-compatible implementation.
    Uses merged gate_up projections and vLLM Flash Attention.
    """

    def __init__(self, model_args: Qwen3Model.Config):
        super().__init__()
        self.config = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id
        self.head_dim = model_args.head_dim

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.register_buffer(
            "rope_cache", self._precompute_rope_cache(), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = VLLMRMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

        # IMPORTANT: To match vLLM's behavior and Qwen3's config
        # (tie_word_embeddings: true), tie output layer weights to
        # embedding weights. When either weight updates during training,
        # both update together
        self.output.weight = self.tok_embeddings.weight

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):
        buffer_device = buffer_device or self.rope_cache.device
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.config.dim**-0.5
        cutoff_factor = 3

        if self.output is not None:
            trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _precompute_rope_cache(self) -> torch.Tensor:
        return precompute_rope_cache(
            self.config.head_dim,
            self.config.max_seq_len,
            self.config.rope_theta,
        )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType | None:
        # vLLM compat mode: no flex attention masks
        return None

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
    ):
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(h, self.rope_cache, attention_masks)

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h

        return output
