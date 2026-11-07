# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shape suffixes: P = cached positions, F = rotary frequency pairs,
# D = rotary width; query/key use the existing public argument names.

import math
from dataclasses import dataclass

import spmd_types as spmd
import torch

from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.base.module import Module

__all__ = [
    "ComplexRoPE",
    "CosSinRoPE",
    "IdentityRoPE",
    "Llama3ScalingConfig",
    "RoPE",
    "YaRNScalingConfig",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class Llama3ScalingConfig:
    """Wavelength-dependent frequency scaling used by Llama 3."""

    factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


@dataclass(frozen=True, kw_only=True, slots=True)
class YaRNScalingConfig:
    """YaRN frequency scaling and its original context length."""

    factor: float = 1.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    original_seq_len: int = 4096
    truncate: bool = True


_ScalingConfig = Llama3ScalingConfig | YaRNScalingConfig | None


@spmd.no_typecheck()
def _maybe_check_max_pos(positions: torch.Tensor, *, max_valid_pos: int) -> None:
    """Async bounds check: verify all position values <= max_valid_pos.

    Uses ``torch._assert_async`` to avoid a device-host sync while still
    catching out-of-bounds positions (the assertion failure surfaces at a
    later kernel launch).  Skipped entirely under ``torch.compile``.
    """
    if torch.compiler.is_compiling():
        return
    torch._assert_async(
        torch.all(positions <= max_valid_pos),
        f"position_ids exceed {max_valid_pos=}",
    )


def _yarn_inv_freq(
    dim: int,
    base: float,
    rope_factor: float,
    beta_fast: float,
    beta_slow: float,
    original_seq_len: int,
    truncate: bool,
) -> torch.Tensor:
    """Shared YaRN ("NTK-by-parts") inverse-frequency computation.

    Single source of truth for both ``ComplexRoPE`` and ``CosSinRoPE`` so the
    two cache formats are guaranteed to agree. Follows the YaRN paper / HF
    convention: ``low <- beta_fast`` (extrapolation boundary), ``high <-
    beta_slow`` (interpolation boundary). ``truncate`` floors/ceils the cutoffs
    (DeepSeek style); ``truncate=False`` keeps fractional cutoffs (gpt-oss
    style). The range is always clamped to ``[0, dim - 1]``. The YaRN
    attention "mscale" is intentionally NOT applied here -- the rope stays a
    pure rotation and the model folds mscale into its softmax scale.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    def find_correction_dim(num_rotations: float) -> float:
        return (dim * math.log(original_seq_len / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = find_correction_dim(beta_fast)
    high = find_correction_dim(beta_slow)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001

    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(
        0, 1
    )
    return inv_freq / rope_factor * ramp + inv_freq * (1 - ramp)


def _compute_inv_freq(dim: int, theta: float, scaling: _ScalingConfig) -> torch.Tensor:
    """Compute FP32 inverse frequencies; F denotes rotary frequency pairs."""
    if isinstance(scaling, YaRNScalingConfig) and scaling.factor > 1.0:
        return _yarn_inv_freq(
            dim,
            theta,
            scaling.factor,
            scaling.beta_fast,
            scaling.beta_slow,
            scaling.original_seq_len,
            scaling.truncate,
        )
    if scaling is not None and not isinstance(
        scaling, (Llama3ScalingConfig, YaRNScalingConfig)
    ):
        raise ValueError(
            "RoPE scaling must be None, Llama3ScalingConfig, or YaRNScalingConfig."
        )
    inv_freq_F = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    if isinstance(scaling, Llama3ScalingConfig):
        wavelength_F = 2 * math.pi / inv_freq_F
        high_freq_wavelength = (
            scaling.original_max_position_embeddings / scaling.high_freq_factor
        )
        low_freq_wavelength = (
            scaling.original_max_position_embeddings / scaling.low_freq_factor
        )
        inv_freq_F = torch.where(
            wavelength_F > low_freq_wavelength,
            inv_freq_F / scaling.factor,
            inv_freq_F,
        )
        smooth_F = (
            scaling.original_max_position_embeddings / wavelength_F
            - scaling.low_freq_factor
        ) / (scaling.high_freq_factor - scaling.low_freq_factor)
        scaled_F = (1 - smooth_F) * inv_freq_F / scaling.factor + smooth_F * inv_freq_F
        medium_F = ~(wavelength_F < high_freq_wavelength) * ~(
            wavelength_F > low_freq_wavelength
        )
        inv_freq_F = torch.where(medium_F, scaled_F, inv_freq_F)
    return inv_freq_F


def _qk_amplitude(scaling: _ScalingConfig, *, mscale: float = 1.0) -> float:
    """Return the per-Q/K magnitude factor applied by attention, not RoPE."""
    if isinstance(scaling, YaRNScalingConfig) and scaling.factor > 1.0:
        return 0.1 * mscale * math.log(scaling.factor) + 1.0
    return 1.0


class RoPE(Module):
    """Configurable query/key positional transform, including the identity.

    Dependencies and overrides can target ``RoPE.Config`` without requiring a
    position table. Implementations that use rotary dimensions or scaling
    inherit the corresponding ``ComplexRoPE.Config`` or ``CosSinRoPE.Config``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        def qk_amplitude(self, *, mscale: float = 1.0) -> float:
            """Return the magnitude factor used to scale attention scores."""
            return 1.0

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def prepare_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Prepare query-independent position data for ``forward``.

        Encoders with a shared block config call this once per batch and pass
        the result to every layer. The representation belongs to the RoPE
        implementation; the default keeps raw positions unchanged. Only data
        determined by the shared config and positions belongs here;
        layer-specific state must be applied in ``forward``.
        """
        return positions

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Transform query and optional key using prepared position data."""
        raise NotImplementedError


class IdentityRoPE(RoPE):
    """Leave query and optional key unchanged without allocating a cache."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoPE.Config):
        pass

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return query if key is None else (query, key)


class ComplexRoPE(RoPE):
    @dataclass(kw_only=True, slots=True)
    class Config(RoPE.Config):
        dim: int
        max_context_length: int
        theta: float = 10000.0
        scaling: _ScalingConfig = None

        def qk_amplitude(self, *, mscale: float = 1.0) -> float:
            return _qk_amplitude(self.scaling, mscale=mscale)

    def __init__(self, config: Config):
        super().__init__(config)
        self.dim = config.dim
        self.max_context_length = config.max_context_length
        self.theta = config.theta
        self.scaling = config.scaling
        self.register_buffer("cache", self._precompute_cache(), persistent=False)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            buffer_device = self.cache.device
        with torch.device(buffer_device):
            self.cache = self._precompute_cache()

    # Cat options: only fp32/fp16 inputs reach the stack (bf16 writes int32 pairs); they write
    # each half with one store, instead of one masked kernel over both.
    # No pointwise autotuning: the rotation's kernel configs can round differently (likely FMA
    # contraction), so picking one by timing could change numerics from run to run.
    # recompile_limit counts graphs per process; one more raises. DeepSeek-V4 calls this from
    # 4 sites, one graph each, so its recipes compile 4. no_grad calls double that to 8. A
    # second token count recompiles each graph with a dynamic size: up to 16. Unpacked batches
    # and calls without positions take other paths through the compressor, and a token count
    # that leaves one compressed entry adds a few more: up to 26 in one process. Hence 32.
    @local_compile(
        "rope",
        batch_invariant=True,
        recompile_limit=32,
        options={
            "max_pointwise_cat_inputs": 0,
            "max_complex_pointwise_cat_inputs": 0,
            "triton.autotune_pointwise": False,
        },
    )
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        reshaped_cache = self._reshape_cache(query, positions)
        return self.apply_rotary_emb(query, key, reshaped_cache, inverse=inverse)

    def _precompute_cache(self) -> torch.Tensor:
        """Precompute complex cis values of shape (max_context_length, dim / 2)."""
        inv_freq_F = _compute_inv_freq(self.dim, self.theta, self.scaling)
        positions_P = torch.arange(self.max_context_length, device=inv_freq_F.device)
        angles_PF = torch.outer(positions_P, inv_freq_F).float()
        return torch.polar(torch.ones_like(angles_PF), angles_PF)

    def _reshape_cache(
        self,
        query: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the cache shaped for query/key broadcast.

        Returns:
            Complex cache of shape ``(T, 1, dim / 2)``; under compile, its real view
            ``(T, 1, dim)`` with each ``(cos, sin)`` pair adjacent.
        """
        if positions is not None:
            _maybe_check_max_pos(positions, max_valid_pos=self.cache.shape[0] - 1)
        if torch.compiler.is_compiling():
            # Gather rows of the real view, (T, 1, dim): Inductor can't load complex, so a complex
            # gather runs as its own kernel, while a float gather fuses into the rotation.
            return _reshape_for_broadcast(
                torch.view_as_real(self.cache).flatten(-2), query.shape, positions
            )
        # Complex RoPE cache has width dim / 2 because each complex value
        # represents a pair of real dimensions.
        complex_query_shape = (*query.shape[:-1], query.shape[-1] // 2)
        return _reshape_for_broadcast(self.cache, complex_query_shape, positions)

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor | None,
        rope_cache: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply complex RoPE using adjacent-dim pairs."""
        if torch.compiler.is_compiling():
            # Inductor can't compile a complex multiply, so rotate in real arithmetic: it fuses
            # with the casts, within 1 bf16 ulp of eager. Eager keeps the complex ops.
            # _reshape_cache hands over the real view, gathered as floats; direct callers may
            # still pass the complex cache.
            if rope_cache.is_complex():
                cache = torch.view_as_real(rope_cache)
            else:
                cache = rope_cache.unflatten(-1, (-1, 2))
            cos, sin = cache[..., 0], cache[..., 1]
            if inverse:
                sin = -sin
            query_out = _RotatePairs.apply(query, cos, sin)
            if key is None:
                return query_out
            return query_out, _RotatePairs.apply(key, cos, sin)

        if inverse:
            rope_cache = rope_cache.conj()

        xq_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
        query_out = torch.view_as_real(xq_ * rope_cache).flatten(-2).type_as(query)
        if key is None:
            return query_out

        xk_ = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
        key_out = torch.view_as_real(xk_ * rope_cache).flatten(-2).type_as(key)
        return query_out, key_out


class _RotatePairs(torch.autograd.Function):
    """Rotate each ``(x[2i], x[2i + 1])`` pair by the angle with ``cos[i]``, ``sin[i]``::

        out[2i]     = x[2i] * cos[i] - x[2i + 1] * sin[i]
        out[2i + 1] = x[2i] * sin[i] + x[2i + 1] * cos[i]

    A Function because the bf16 path's int32 views have no autograd. The backward is the
    same rotation by the opposite angle, one kernel. ``cos`` and ``sin`` get no gradient:
    they come from the RoPE cache buffer.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(cos, sin)
        return _RotatePairs.rotate(x, cos, sin)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # pyrefly: ignore[bad-override]
        cos, sin = ctx.saved_tensors
        return _RotatePairs.rotate(grad, cos, -sin), None, None

    @staticmethod
    def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """``x`` may also hold bf16 pairs already packed as int32 words (MLA's ``_RotateTail``); the
        result is then int32 too.
        """
        packed = x.dtype == torch.int32
        out_dtype = torch.bfloat16 if packed else x.dtype
        if packed or (x.dtype == torch.bfloat16 and x.is_contiguous()):
            # Read each pair as one int32 (little-endian: x[2i] in the low half), so loads stay contiguous.
            # A bf16 shifted into the high 16 bits is its exact fp32. The int32 view also
            # ends the fusion with whatever produced x, which is written out first. It needs an
            # even element offset: every caller slices q/k at even widths.
            pairs = x if packed else x.view(torch.int32)
            x_real = (pairs << 16).view(torch.float32)
            x_imag = (pairs & ~0xFFFF).view(torch.float32)
        else:
            # Split views and other dtypes: read the two halves. An int32 view of a split
            # view would be copied first.
            x_pairs = x.float().unflatten(-1, (-1, 2))
            x_real, x_imag = x_pairs[..., 0], x_pairs[..., 1]
        real = (x_real * cos - x_imag * sin).to(out_dtype)
        imag = (x_real * sin + x_imag * cos).to(out_dtype)
        if out_dtype == torch.bfloat16:
            # Write each rotated pair as one int32 so stores stay contiguous; a store per half
            # writes every other element.
            low = real.view(torch.int16).to(torch.int32) & 0xFFFF
            high = imag.view(torch.int16).to(torch.int32) << 16
            return low | high if packed else (low | high).view(torch.bfloat16)
        return torch.stack((real, imag), dim=-1).flatten(-2)


class CosSinRoPE(RoPE):
    @dataclass(kw_only=True, slots=True)
    class Config(RoPE.Config):
        dim: int
        max_context_length: int
        theta: float = 10000.0
        scaling: _ScalingConfig = None

        def qk_amplitude(self, *, mscale: float = 1.0) -> float:
            return _qk_amplitude(self.scaling, mscale=mscale)

    def __init__(self, config: Config):
        super().__init__(config)
        self.dim = config.dim
        self.max_context_length = config.max_context_length
        self.theta = config.theta
        self.scaling = config.scaling
        self.register_buffer("cache", self._precompute_cache(), persistent=False)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            buffer_device = self.cache.device
        with torch.device(buffer_device):
            self.cache = self._precompute_cache()

    @local_compile("rope", batch_invariant=True)
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        reshaped_cache = self._reshape_cache(query, positions)
        return self.apply_rotary_emb(query, key, reshaped_cache, inverse=inverse)

    def _precompute_cache(self) -> torch.Tensor:
        """Precompute cos/sin values of shape (max_context_length, dim * 2)."""
        if isinstance(self.scaling, Llama3ScalingConfig):
            raise NotImplementedError("Cos/sin RoPE does not support Llama scaling.")
        inv_freq_F = _compute_inv_freq(self.dim, self.theta, self.scaling)
        positions_P = torch.arange(
            self.max_context_length, dtype=inv_freq_F.dtype, device=inv_freq_F.device
        )
        angles_PF = torch.outer(positions_P, inv_freq_F).float()
        angles_PD = torch.cat([angles_PF, angles_PF], dim=-1)
        return torch.cat([angles_PD.cos(), angles_PD.sin()], dim=-1)

    def _reshape_cache(
        self,
        query: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return cos/sin cache shaped for query/key broadcast.

        Returns:
            Cache of shape ``(T, 1, dim * 2)``.
        """
        if positions is not None:
            _maybe_check_max_pos(positions, max_valid_pos=self.cache.shape[0] - 1)
        return _reshape_for_broadcast(self.cache, query.shape, positions)

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor | None,
        rope_cache: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply cos/sin RoPE using the rotate-half convention."""
        if inverse:
            raise NotImplementedError("CosSinRoPE does not support inverse rotation.")

        head_dim = query.shape[-1]
        cos = rope_cache[..., :head_dim]
        sin = rope_cache[..., head_dim:]
        query_f = query.float()
        xq_out = (query_f * cos) + (CosSinRoPE._rotate_half(query_f) * sin)
        if key is None:
            return xq_out.type_as(query)

        key_f = key.float()
        xk_out = (key_f * cos) + (CosSinRoPE._rotate_half(key_f) * sin)
        return xq_out.type_as(query), xk_out.type_as(key)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


@spmd.local_map(
    out_types=(
        {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
        spmd.PartitionSpec(("dp", "cp"), None, None),
    )
)
def _reshape_for_broadcast(
    rope_cache: torch.Tensor,
    query_shape: torch.Size | tuple[int, ...],
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reshape a RoPE cache for broadcasting with query/key tensors."""
    # cache_width: `head_dim * 2` for CosSinRoPE; `head_dim // 2` for ComplexRoPE's complex
    # cache, `head_dim` for its real view under compile.
    cache_width = rope_cache.shape[-1]
    num_tokens = query_shape[0]
    if positions is None:
        rope_cache = rope_cache[:num_tokens]
    else:
        rope_cache = rope_cache[positions]
    return rope_cache.view(num_tokens, 1, cache_width)
