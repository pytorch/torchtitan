# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks, BlockMask
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.attention import (
    create_attention_mask,
    FlexAttentionWrapper,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_sliding_window_mask_mod,
)
from torchtitan.protocols.model import AttentionMasksType
from torchtitan.protocols.train_spec import ModelProtocol

from .args import GptOssModelArgs
from .moe import GptOssMoE


def _find_yarn_correction_dim(
    num_rotations: float, dim: int, base: float, max_seq_len: int
) -> float:
    """YaRN helper: Find correction dimension."""
    return (
        dim
        * math.log(max_seq_len / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def _find_yarn_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int
) -> tuple[int, int]:
    """YaRN helper: Find correction range."""
    low = math.floor(_find_yarn_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(_find_yarn_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def _linear_ramp_factor(min_val: float, max_val: float, dim: int) -> torch.Tensor:
    """YaRN helper: Linear ramp for smooth interpolation."""
    if min_val == max_val:
        # Degenerate range: return zeros (all dimensions use same blend factor)
        return torch.zeros(dim, dtype=torch.float32)
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def compute_yarn_mscale(rope_factor: float) -> float:
    """
    Compute YaRN mscale (attention scaling factor) for extended context.

    YaRN paper recommends scaling attention by this factor to maintain
    perplexity at extended context lengths. Without mscale, model quality
    degrades significantly when extending beyond the original training length.

    Formula: mscale = 0.1 * ln(rope_factor) + 1.0

    Args:
        rope_factor: YaRN rope scaling factor (e.g., 32 for 32x context extension)

    Returns:
        Attention scaling multiplier (1.0 when rope_factor <= 1)
    """
    if rope_factor <= 1.0:
        return 1.0
    return 0.1 * math.log(rope_factor) + 1.0


def precompute_rope_cache(
    dim: int,
    max_seq_len: int,
    base: float = 1_000_000.0,
    rope_factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
    original_seq_len: int = 4096,
) -> torch.Tensor:
    """
    Precompute RoPE cache with optional YaRN scaling for extended context.

    YaRN (Yet another RoPE extensioN) allows extending context length beyond
    the original training length by applying frequency scaling corrections.

    Args:
        dim: Embedding dimension (head_dim)
        max_seq_len: Maximum sequence length to precompute
        base: RoPE theta base (function default: 1M; GPT-OSS config uses 150k via args.py)
        rope_factor: YaRN scaling factor (GPT-OSS uses 32)
        beta_fast: Fast frequency correction threshold (GPT-OSS uses 32)
        beta_slow: Slow frequency correction threshold (GPT-OSS uses 1)
        original_seq_len: Original pretrained context length (GPT-OSS uses 4096)

    Returns:
        Precomputed RoPE cache: [max_seq_len, dim * 2] (cos and sin concatenated)
    """
    # Basic RoPE frequency calculation
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # YaRN scaling for extended context (applied when seqlen > original_seq_len)
    if max_seq_len > original_seq_len and rope_factor > 1.0:
        low, high = _find_yarn_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - _linear_ramp_factor(low, high, dim // 2)
        # Apply YaRN interpolation: blend scaled and unscaled frequencies
        freqs = freqs / rope_factor * (1 - smooth) + freqs * smooth

    # Create position indexes `[0, 1, ..., max_seq_len - 1]`
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)

    # Outer product of theta and position index; output tensor has
    # a shape of [max_seq_len, dim // 2]
    idx_theta = torch.outer(t, freqs).float()

    # We cache the cos and sin embeddings instead of the IDs. This helps
    # ensure we have correct behavior when training with bf16
    # Size: [max_seq_len, (dim * 2)]
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return rope_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reshape_for_broadcast(rope_cache: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor (represented by cos, sin) for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, head_dim * 2),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        rope_cache (torch.Tensor): RoPE tensor (cos and sin) to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    _, seqlen, _, head_dim = x.shape
    rope_cache = rope_cache[0:seqlen]
    # The shape of rope_cache is (seqlen, head_dim * 2) because we concate cos and sin
    assert rope_cache.shape == (seqlen, head_dim * 2)
    shape = [-1, seqlen, 1, head_dim * 2]
    return rope_cache.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, rope_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # input tensor x has shape [bsz, seq_len, n_heads, head_dim]
    head_dim = xq.shape[-1]

    # reshape for broadcast
    rope_cache = reshape_for_broadcast(rope_cache, xq)

    # [bsz, seq_len, 1, head_dim]
    cos = rope_cache[..., :head_dim].to(dtype=xq.dtype, device=xq.device)
    sin = rope_cache[..., head_dim:].to(dtype=xq.dtype, device=xq.device)

    # xq:  [bsz, seq_len, n_heads, head_dim]
    # xk:  [bsz, seq_len, n_kv_heads, head_dim]
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """
    Multi-head attention (MLA) module.
    """

    def __init__(self, model_args: GptOssModelArgs):
        super().__init__()
        self.head_dim = model_args.head_dim
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_kv_heads
        self.enable_gqa = self.n_heads > self.n_kv_heads

        self.n_rep = self.n_heads // self.n_kv_heads

        # Compute YaRN mscale for extended context attention scaling
        mscale = compute_yarn_mscale(model_args.rope_factor)
        self.softmax_scale = mscale / math.sqrt(self.head_dim)

        self.wq = nn.Linear(
            model_args.dim,
            model_args.n_heads * model_args.head_dim,
            bias=True,
        )
        self.wk = nn.Linear(
            model_args.dim,
            model_args.n_kv_heads * model_args.head_dim,
            bias=True,
        )
        self.wv = nn.Linear(
            model_args.dim,
            model_args.n_kv_heads * model_args.head_dim,
            bias=True,
        )
        self.wo = nn.Linear(
            model_args.n_heads * model_args.head_dim,
            model_args.dim,
            bias=True,
        )
        self.sinks = nn.Parameter(torch.empty(model_args.n_heads))
        self.inner_attention = FlexAttentionWrapper()

    def init_weights(self, init_std: float):
        linear_list = [
            self.wq,
            self.wk,
            self.wv,
        ]

        nn.init.trunc_normal_(self.sinks, mean=0.0, std=init_std)
        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
            nn.init.trunc_normal_(linear.bias, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.wo.bias, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            rope_cache (torch.Tensor): Precomputed cosine and sine frequencies for rope embedding.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        hidden_shape = (bsz, seqlen, -1, self.head_dim)

        q = self.wq(x).view(hidden_shape)
        k = self.wk(x).view(hidden_shape)
        v = self.wv(x).view(hidden_shape)

        q, k = apply_rotary_emb(q, k, rope_cache)

        xq = q.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = k.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)
        xv = v.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)

        assert isinstance(attention_masks, BlockMask), attention_masks
        output, lse = self.inner_attention(
            xq,
            xk,
            xv,
            block_mask=attention_masks,
            scale=self.softmax_scale,
            return_lse=True,
            enable_gqa=self.enable_gqa,
        )

        # Apply attention sink rescaling: rescale by σ(lse - w[h])
        # This is mathematically equivalent to concatenating learnable sink weights
        sink_scale = torch.sigmoid(lse - self.sinks.view(1, -1, 1)).unsqueeze(-1)
        output = output * sink_scale.to(output.dtype)

        output = output.transpose(1, 2).contiguous()  # (B, H, T, D) -> (B, T, H, D)

        # Reshape and project output
        output = output.reshape(
            bsz, seqlen, -1
        ).contiguous()  # (bsz, seqlen, n_heads * v_head_dim)
        output = self.wo(output)  # (bsz, seqlen, dim)
        return output


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward layers.
    """

    def __init__(self, layer_id: int, model_args: GptOssModelArgs):

        super().__init__()
        self.use_sliding_attention = layer_id % 2 == 0
        self.attention = Attention(model_args)
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        self.moe = GptOssMoE(
            model_args, dim=model_args.dim, hidden_dim=model_args.moe_inter_dim
        )
        self.moe_enabled = True  # for composability with load balancing

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType,
    ):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            rope_cache (torch.Tensor): Precomputed cosine and sine frequencies.
            attention_masks (AttentionMasksType): a dict of BlockMasks.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        # Extract the appropriate mask for this layer
        if self.use_sliding_attention:
            # pyrefly: ignore [missing-attribute]
            layer_mask = attention_masks.get("sliding_window_mask", None)
        else:
            # pyrefly: ignore [missing-attribute]
            layer_mask = attention_masks.get("basic_mask", None)
        assert layer_mask is not None

        x = x + self.attention(self.attention_norm(x), rope_cache, layer_mask)
        x = x + self.moe(self.ffn_norm(x))
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.moe.init_weights(self.weight_init_std, buffer_device)


class GptOssModel(ModelProtocol):
    """
    GPT-OSS Transformer model with attention and feed-forward layers.
    """

    def __init__(self, model_args: GptOssModelArgs):
        super().__init__(model_args)
        self.model_args = model_args
        self.max_seq_len = model_args.max_seq_len
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer(
            "rope_cache", self._precompute_rope_cache(), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args).to(
                torch.bfloat16
            )

        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(
            model_args.dim,
            model_args.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False,
        )
        self.model_args = model_args
        self.init_weights()

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or self.rope_cache.device
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                # pyrefly: ignore [not-callable]
                layer.init_weights(buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _precompute_rope_cache(self) -> torch.Tensor:
        return precompute_rope_cache(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
            self.model_args.rope_factor,
            self.model_args.beta_fast,
            self.model_args.beta_slow,
            self.model_args.original_seq_len,
        )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:

        basic_mask_mods = []
        sliding_window_mask_mods = [
            get_sliding_window_mask_mod(self.model_args.sliding_window_size)
        ]
        match self.model_args.attn_mask_type:
            case "causal":
                B = 1
                basic_mask_mods.append(get_causal_mask_mod())
            case "block_causal":
                B = input_batch.shape[0]
                basic_mask_mods.append(
                    get_document_mask_mod(input_batch, tokenizer.eos_id)
                )
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.model_args.attn_mask_type}"
                )

        # create basic attention mask: causal or block_causal
        basic_mask = create_attention_mask(
            and_masks(*basic_mask_mods),
            B,
            None,
            input_batch.shape[1],
            input_batch.shape[1],
        )

        # create sliding window mask, has to be created on top of basic attention mask
        sliding_window_mask = create_attention_mask(
            and_masks(*basic_mask_mods, *sliding_window_mask_mods),
            B,
            None,
            input_batch.shape[1],
            input_batch.shape[1],
        )

        return {"basic_mask": basic_mask, "sliding_window_mask": sliding_window_mask}

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType,
    ):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            attention_masks (AttentionMasksType): a dict of BlockMasks.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        h = self.tok_embeddings(tokens)

        for layer in self.layers.values():
            h = layer(h, self.rope_cache, attention_masks)
        h = self.norm(h)
        output = self.output(h)
        return output
