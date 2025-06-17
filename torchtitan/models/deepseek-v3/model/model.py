# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import nn

from torchtitan.models.attention import build_attention

from .args import DeepseekV3ModelArgs


# Adopted from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L294
def precompute_freqs_cis(args: DeepseekV3ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (DeepseekV3ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, args.original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class Attention(nn.Module):
    """
    Multi-head attention (MLA) module.
    """

    def __init__(self, model_args: DeepseekV3ModelArgs):
        super().__init__()
        self.dim = model_args.dim
        self.n_heads = model_args.n_heads
        self.q_lora_rank = model_args.q_lora_rank
        self.kv_lora_rank = model_args.kv_lora_rank
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.qk_head_dim = model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        self.v_head_dim = model_args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=model_args.norm_eps)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=model_args.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)
        )
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim**-0.5

        if model_args.max_seq_len > model_args.original_seq_len:
            mscale = 0.1 * model_args.mscale * math.log(model_args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        if self.q_lora_rank == 0:
            q = self.wq(x)  # q: (bsz, seqlen, n_heads * qk_head_dim)
        else:
            q = self.wq_b(
                self.q_norm(self.wq_a(x))
            )  # q: (bsz, seqlen, n_heads * qk_head_dim)

        q = q.view(
            bsz, seqlen, self.n_heads, self.qk_head_dim
        )  # q: (bsz, seqlen, n_heads, qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        # q_nope: (bsz, seqlen, n_heads, qk_nope_head_dim)
        # q_pe: (bsz, seqlen, n_heads, qk_rope_head_dim)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)  # q: (bsz, seqlen, n_heads, qk_head_dim)

        kv = self.wkv_a(x)  # kv: (bsz, seqlen, kv_lora_rank + qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # kv: (bsz, seqlen, kv_lora_rank)
        # k_pe: (bsz, seqlen, qk_rope_head_dim)
        k_pe = apply_rotary_emb(
            k_pe.unsqueeze(2), freqs_cis
        )  # k_pe: (bsz, seqlen, 1, qk_rope_head_dim)

        kv = self.wkv_b(
            self.kv_norm(kv)
        )  # kv: (bsz, seqlen, n_heads * (qk_nope_head_dim + v_head_dim))
        kv = kv.view(
            bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim
        )  # (bsz, seqlen, n_heads, qk_nope_head_dim + v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        # k_nope: (bsz, seqlen, n_heads, qk_nope_head_dim)
        # v: (bsz, seqlen, n_heads, v_head_dim)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)
        # k: (bsz, seqlen, n_heads, qk_head_dim)

        # TODO: Need to pass softmax_scale to sdpa() interface.
        # For mask, DeepseekV3 uses causal mask, so we can use the default mask in sdpa
        # https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L17
        output = self.sdpa(q, k, v)

        output = output.transpose(1, 2).contiguous()
        output = output.view(bsz, seqlen, -1)  # (bs, seqlen, n_heads * v_head_dim)
        return self.wo(output)  # (bsz, seqlen, dim)
