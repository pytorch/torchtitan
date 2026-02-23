# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import Literal

import torch

from torchtitan.protocols.module import Module


class RoPE(Module):
    """Shared Rotary Position Embedding module.

    Supports multiple formats and scaling methods:
    - backend="complex": Complex exponential (Llama3/4, DeepSeek V3)
    - backend="cos_sin": Cosine/sine concatenation (Qwen3, GPT-OSS)

    - scaling="none": No scaling applied
    - scaling="llama": Llama3/4-style low/high frequency scaling
    - scaling="yarn": YaRN scaling for extended context (DeepSeek V3, GPT-OSS)

    Config fields should be set at config time. ``dim`` and ``max_seq_len``
    are required.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        max_seq_len: int
        theta: float = 10000.0
        backend: Literal["complex", "cos_sin"] = "complex"
        scaling: Literal["none", "llama", "yarn"] = "none"
        # llama scaling params
        scaling_factor: float = 8.0
        low_freq_factor: float = 1.0
        high_freq_factor: float = 4.0
        original_max_position_embeddings: int = 8192
        # yarn scaling params
        rope_factor: float = 1.0
        beta_fast: float = 32.0
        beta_slow: float = 1.0
        original_seq_len: int = 4096
        mscale: float = 0.0

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # Buffer registered later in init_weights
        self.register_buffer("cache", self._precompute(), persistent=False)

    def _precompute(self) -> torch.Tensor:
        cfg = self.config
        if cfg.backend == "complex":
            return self._precompute_complex()
        else:
            return self._precompute_cos_sin()

    def _precompute_complex(self) -> torch.Tensor:
        cfg = self.config
        dim = cfg.dim
        end = cfg.max_seq_len
        theta = cfg.theta

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

        if cfg.scaling == "llama":
            scaling_factor = cfg.scaling_factor
            low_freq_factor = cfg.low_freq_factor
            high_freq_factor = cfg.high_freq_factor
            original_max_position_embeddings = cfg.original_max_position_embeddings
            wavelen = 2 * math.pi / freqs
            high_freq_wavelen = original_max_position_embeddings / high_freq_factor
            low_freq_wavelen = original_max_position_embeddings / low_freq_factor
            freqs = torch.where(
                wavelen > low_freq_wavelen, freqs / scaling_factor, freqs
            )
            smooth_factor = (
                original_max_position_embeddings / wavelen - low_freq_factor
            ) / (high_freq_factor - low_freq_factor)
            smoothed_freqs = (
                1 - smooth_factor
            ) * freqs / scaling_factor + smooth_factor * freqs
            is_medium_freqs = ~(wavelen < high_freq_wavelen) * ~(
                wavelen > low_freq_wavelen
            )
            freqs = torch.where(is_medium_freqs, smoothed_freqs, freqs)
        elif cfg.scaling == "yarn" and end > cfg.original_seq_len:
            # YaRN (DeepSeek V3 style)
            beta_fast = cfg.beta_fast
            beta_slow = cfg.beta_slow
            base = theta
            original_seq_len = cfg.original_seq_len
            factor = cfg.rope_factor

            def find_correction_dim(
                num_rotations: float, dim: int, base: float, max_seq_len: int
            ) -> float:
                return (
                    dim
                    * math.log(max_seq_len / (num_rotations * 2 * math.pi))
                    / (2 * math.log(base))
                )

            def find_correction_range(
                low_rot: float,
                high_rot: float,
                dim: int,
                base: float,
                max_seq_len: int,
            ) -> tuple[int, int]:
                low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
                high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
                return max(low, 0), min(high, dim - 1)

            def linear_ramp_factor(
                min_val: float, max_val: float, dim: int
            ) -> torch.Tensor:
                if min_val == max_val:
                    max_val += 0.001
                linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (
                    max_val - min_val
                )
                return torch.clamp(linear_func, 0, 1)

            low, high = find_correction_range(
                beta_fast, beta_slow, dim, base, original_seq_len
            )
            smooth = 1 - linear_ramp_factor(low, high, dim // 2)
            freqs = freqs / factor * (1 - smooth) + freqs * smooth

        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def _precompute_cos_sin(self) -> torch.Tensor:
        cfg = self.config
        dim = cfg.dim
        max_seq_len = cfg.max_seq_len
        base = cfg.theta

        freq = base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        mscale = 1.0

        if cfg.scaling == "llama":
            raise NotImplementedError("Cos/sin RoPE does not support Llama scaling.")

        if cfg.scaling == "yarn" and cfg.rope_factor > 1.0:
            rope_factor = cfg.rope_factor
            # YaRN mscale for attention magnitude preservation
            mscale = 0.1 * math.log(rope_factor) + 1.0

            # Compute correction range (NTK by parts)
            d_half = dim / 2
            low = (
                d_half
                * math.log(cfg.original_seq_len / (cfg.beta_slow * 2 * math.pi))
                / math.log(base)
            )
            high = (
                d_half
                * math.log(cfg.original_seq_len / (cfg.beta_fast * 2 * math.pi))
                / math.log(base)
            )
            assert (
                0 < low < high < d_half - 1
            ), f"Invalid YaRN params: 0 < {low} < {high} < {d_half - 1}"

            ramp = (torch.arange(d_half, dtype=torch.float32) - low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            interpolation = 1.0 / (rope_factor * freq)
            extrapolation = 1.0 / freq
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            inv_freq = 1.0 / freq

        t = torch.arange(max_seq_len, dtype=inv_freq.dtype, device=inv_freq.device)
        freqs = torch.outer(t, inv_freq).float()
        theta = torch.cat([freqs, freqs], dim=-1)

        cos = theta.cos() * mscale
        sin = theta.sin() * mscale
        return torch.cat([cos, sin], dim=-1)

    def forward(
        self, seq_len: int, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the precomputed cache tensor (slicing is done by apply_rotary_emb)."""
        return self.cache

    def init_weights(self, **kwargs) -> None:
        buffer_device = kwargs.get("buffer_device")
        if buffer_device is not None:
            with torch.device(buffer_device):
                self.cache = self._precompute()
        else:
            self.cache = self._precompute()


def _reshape_for_broadcast_complex(
    freqs_cis: torch.Tensor,
    x: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reshape complex RoPE tensor for broadcasting.

    freqs_cis: (max_seqlen, dim // 2) complex
    x: (bsz, seqlen, n_heads, dim // 2) complex
    """
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    if positions is None:
        freqs_cis = freqs_cis[0:seqlen]
        assert freqs_cis.shape == (seqlen, x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)
    elif positions.size(0) == 1:
        assert positions.shape == (1, seqlen)
        freqs_cis = freqs_cis[positions.squeeze(0)]
        assert freqs_cis.shape == (seqlen, x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)
    else:
        assert positions.shape == (x.shape[0], seqlen)
        freqs_cis_expanded = freqs_cis[None, :, None, :].expand(x.shape[0], -1, -1, -1)
        freqs_cis = torch.gather(
            freqs_cis_expanded,
            dim=1,
            index=positions.view(x.shape[0], seqlen, 1, 1).expand(
                x.shape[0], seqlen, 1, freqs_cis_expanded.shape[-1]
            ),
        )
        return freqs_cis


def _reshape_for_broadcast_cos_sin(
    rope_cache: torch.Tensor,
    x: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reshape cos/sin RoPE tensor for broadcasting.

    rope_cache: (max_seqlen, head_dim * 2) where first half is cos, second is sin
    x: (bsz, seqlen, n_heads, head_dim) real
    """
    ndim = x.ndim
    assert ndim > 1
    bz, seqlen, _, head_dim = x.shape
    if positions is None:
        rope_cache = rope_cache[0:seqlen]
        assert rope_cache.shape == (seqlen, head_dim * 2)
        shape = [-1, seqlen, 1, head_dim * 2]
        return rope_cache.view(*shape)
    elif positions.size(0) == 1:
        assert positions.shape == (1, seqlen)
        rope_cache = rope_cache[positions.squeeze(0)]
        assert rope_cache.shape == (seqlen, head_dim * 2)
        shape = [-1, seqlen, 1, head_dim * 2]
        return rope_cache.view(*shape)
    else:
        assert positions.shape == (bz, seqlen)
        rope_cache_expanded = rope_cache[None, :, None, :].expand(bz, -1, -1, -1)
        rope_cache = torch.gather(
            rope_cache_expanded,
            dim=1,
            index=positions.view(bz, seqlen, 1, 1).expand(bz, seqlen, 1, head_dim * 2),
        )
        assert rope_cache.shape == (bz, seqlen, 1, head_dim * 2)
        return rope_cache


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# TODO: consolidate apply_rotary_emb_complex and apply_rotary_emb_single_complex
def apply_rotary_emb_complex(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply complex-format RoPE to query and key tensors (Llama3/4 style).

    Args:
        xq: (bsz, seqlen, n_heads, head_dim)
        xk: (bsz, seqlen, n_kv_heads, head_dim)
        freqs_cis: (max_seqlen, head_dim // 2) complex
        positions: optional position indices
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast_complex(freqs_cis, xq_, positions)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_rotary_emb_single_complex(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply complex-format RoPE to a single tensor (DeepSeek V3 MLA style).

    Args:
        x: (bsz, seqlen, n_heads, head_dim) or (bsz, seqlen, 1, head_dim)
        freqs_cis: (max_seqlen, head_dim // 2) complex
        positions: optional position indices
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast_complex(freqs_cis, x, positions)
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


def apply_rotary_emb_cos_sin(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply cos/sin-format RoPE to query and key tensors (Qwen3/GPT-OSS style).

    Args:
        xq: (bsz, seqlen, n_heads, head_dim)
        xk: (bsz, seqlen, n_kv_heads, head_dim)
        rope_cache: (max_seqlen, head_dim * 2) with cos and sin concatenated
        positions: optional position indices
    """
    head_dim = xq.shape[-1]
    rope_cache = _reshape_for_broadcast_cos_sin(rope_cache, xq, positions)
    cos = rope_cache[..., :head_dim].to(dtype=xq.dtype, device=xq.device)
    sin = rope_cache[..., head_dim:].to(dtype=xq.dtype, device=xq.device)
    xq_out = (xq * cos) + (_rotate_half(xq) * sin)
    xk_out = (xk * cos) + (_rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)
