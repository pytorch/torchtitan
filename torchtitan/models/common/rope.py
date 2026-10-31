# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cache-owning RoPE modules and configurable prepared-cache rotation.

Shape suffixes in rotary apply operations:
- T = tokens, N = query heads, K = key heads, H = head dim, P = half head dim
- X = broadcast-compatible leading dimensions of a cache or rotation input
"""

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import spmd_types as spmd
import torch

from torchtitan.config.function import Function
from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.base.module import Module

__all__ = [
    "ComplexRoPE",
    "ComplexRoPEApply",
    "CosSinRoPE",
    "CosSinRoPEApply",
    "IdentityRoPEApply",
    "RoPE",
    "RoPEApply",
]


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


class RoPEApply(Function[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]):
    """Apply a prepared cache without owning parameters or cache buffers.

    Returns query alone when key is None, otherwise the query/key pair.
    Concrete implementations define the cache format and inverse support.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Function.Config):
        pass

    def __init__(self, config: Config) -> None:
        pass

    @abstractmethod
    def __call__(
        self,
        query_TNH: torch.Tensor,
        key_TKH: torch.Tensor | None,
        rope_cache: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        pass


class ComplexRoPEApply(RoPEApply):
    """Rotate adjacent pairs using a complex cache."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoPEApply.Config):
        pass

    def __call__(
        self,
        query_TNH: torch.Tensor,
        key_TKH: torch.Tensor | None,
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
                cache_XP2 = torch.view_as_real(rope_cache)
            else:
                cache_XP2 = rope_cache.unflatten(-1, (-1, 2))
            cos_XP, sin_XP = cache_XP2[..., 0], cache_XP2[..., 1]
            if inverse:
                sin_XP = -sin_XP
            query_out_TNH = _RotatePairs.apply(query_TNH, cos_XP, sin_XP)
            if key_TKH is None:
                return query_out_TNH
            return query_out_TNH, _RotatePairs.apply(key_TKH, cos_XP, sin_XP)

        if inverse:
            rope_cache = rope_cache.conj()

        query_TNP = torch.view_as_complex(
            query_TNH.float().reshape(*query_TNH.shape[:-1], -1, 2)
        )
        query_out_TNH = (
            torch.view_as_real(query_TNP * rope_cache).flatten(-2).type_as(query_TNH)
        )
        if key_TKH is None:
            return query_out_TNH

        key_TKP = torch.view_as_complex(
            key_TKH.float().reshape(*key_TKH.shape[:-1], -1, 2)
        )
        key_out_TKH = (
            torch.view_as_real(key_TKP * rope_cache).flatten(-2).type_as(key_TKH)
        )
        return query_out_TNH, key_out_TKH


class CosSinRoPEApply(RoPEApply):
    """Rotate half-head pairs using a concatenated cos/sin cache."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoPEApply.Config):
        pass

    def __call__(
        self,
        query_TNH: torch.Tensor,
        key_TKH: torch.Tensor | None,
        rope_cache: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply cos/sin RoPE using the rotate-half convention."""
        if inverse:
            raise NotImplementedError("CosSinRoPE does not support inverse rotation.")

        head_dim = query_TNH.shape[-1]
        cos_XH = rope_cache[..., :head_dim]
        sin_XH = rope_cache[..., head_dim:]
        query_float_TNH = query_TNH.float()
        query_out_TNH = (query_float_TNH * cos_XH) + (
            CosSinRoPEApply._rotate_half(query_float_TNH) * sin_XH
        )
        if key_TKH is None:
            return query_out_TNH.type_as(query_TNH)

        key_float_TKH = key_TKH.float()
        key_out_TKH = (key_float_TKH * cos_XH) + (
            CosSinRoPEApply._rotate_half(key_float_TKH) * sin_XH
        )
        return query_out_TNH.type_as(query_TNH), key_out_TKH.type_as(key_TKH)

    @staticmethod
    def _rotate_half(input_XH: torch.Tensor) -> torch.Tensor:
        first_XP = input_XH[..., : input_XH.shape[-1] // 2]
        second_XP = input_XH[..., input_XH.shape[-1] // 2 :]
        return torch.cat((-second_XP, first_XP), dim=-1)


class IdentityRoPEApply(RoPEApply):
    """Leave query and optional key tensors unrotated."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoPEApply.Config):
        pass

    def __call__(
        self,
        query_TNH: torch.Tensor,
        key_TKH: torch.Tensor | None,
        rope_cache: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if key_TKH is None:
            return query_TNH
        return query_TNH, key_TKH


_DEFAULT_COMPLEX_ROPE_APPLY = ComplexRoPEApply.Config().build()
_DEFAULT_COS_SIN_ROPE_APPLY = CosSinRoPEApply.Config().build()


class RoPE(Module):
    """Shared Rotary Position Embedding module.

    Common base for concrete RoPE formats. Use ``ComplexRoPE.Config`` for
    complex exponential caches and ``CosSinRoPE.Config`` for concatenated
    cosine/sine caches.

    ``Config.rope_apply`` selects the computation used by ``forward``. The
    static ``apply_rotary_emb`` entry points retain the default algorithms;
    subclasses overriding those methods must migrate to the apply config.
    Construction rejects the old hook rather than silently ignoring it.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        max_context_length: int
        rope_apply: RoPEApply.Config
        theta: float = 10000.0
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
        truncate: bool = True

    def __init__(self, config: Config):
        super().__init__()
        if type(self).apply_rotary_emb not in (
            RoPE.apply_rotary_emb,
            ComplexRoPE.apply_rotary_emb,
            CosSinRoPE.apply_rotary_emb,
        ):
            raise ValueError(
                "Overriding RoPE.apply_rotary_emb is no longer supported. "
                "Move the rotation implementation to RoPEApply and set "
                "Config.rope_apply instead."
            )
        self.config = config
        self.rope_apply = config.rope_apply.build()
        self.register_buffer("cache", self._precompute_cache(), persistent=False)

    def _precompute_cache(self) -> torch.Tensor:
        """Build the reusable cache for all positions up to ``max_context_length``.

        Returns:
            RoPE cache for all valid positions.
        """
        raise NotImplementedError

    def _reshape_cache(
        self,
        query: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a cache aligned to ``query`` and ``positions``.

        Args:
            query: Query tensor with shape ``[T, N, H]``.
            positions: Optional position IDs with shape ``[T]``.

        Returns:
            Prepared RoPE cache for the concrete RoPE format.
        """
        raise NotImplementedError

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor | None,
        rope_cache: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply a prepared RoPE cache to query and optional key.

        Args:
            query: Query tensor with shape ``[T, N, H]``.
            key: Optional key tensor with the same leading dimensions as
                ``query``. If ``None``, only ``query`` is rotated and returned.
            rope_cache: Prepared cache broadcastable to ``query`` and ``key``
                according to the concrete RoPE format.
            inverse: Whether to apply the inverse rotation.

        Returns:
            Rotated query tensor when ``key`` is ``None``; otherwise rotated
            query and key tensors with the same shapes and dtypes as inputs.
        """
        raise NotImplementedError

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and optional key tensors."""
        reshaped_cache = self._reshape_cache(query, positions)
        return self.rope_apply(query, key, reshaped_cache, inverse=inverse)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        # TODO: In long-term we need to have buffer abstraction in `Module`` class to infer the buffer_device
        if buffer_device is None:
            # After ``to_empty()``, the existing cache records the target device.
            # Recompute there when the caller does not pass an explicit buffer device.
            buffer_device = self.cache.device
        with torch.device(buffer_device):
            self.cache = self._precompute_cache()


class ComplexRoPE(RoPE):
    @dataclass(kw_only=True, slots=True)
    class Config(RoPE.Config):
        rope_apply: RoPEApply.Config = field(default_factory=ComplexRoPEApply.Config)

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
        return super().forward(query, key, positions, inverse=inverse)

    def _precompute_cache(self) -> torch.Tensor:
        """Precompute complex cis values.

        Returns:
            Cache of shape ``(max_context_length, dim / 2)``.
        """
        cfg = self.config
        dim = cfg.dim
        end = cfg.max_context_length
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
        elif cfg.scaling == "yarn" and cfg.rope_factor > 1.0:
            # YaRN (DeepSeek V3 style)
            freqs = _yarn_inv_freq(
                dim,
                theta,
                cfg.rope_factor,
                cfg.beta_fast,
                cfg.beta_slow,
                cfg.original_seq_len,
                cfg.truncate,
            )

        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

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
        """Compatibility entry point for the default rotary implementation."""
        return _DEFAULT_COMPLEX_ROPE_APPLY(query, key, rope_cache, inverse=inverse)


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
        rope_apply: RoPEApply.Config = field(default_factory=CosSinRoPEApply.Config)

    @local_compile("rope", batch_invariant=True)
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return super().forward(query, key, positions, inverse=inverse)

    def _precompute_cache(self) -> torch.Tensor:
        """Precompute cos/sin values.

        Returns:
            Cache of shape ``(max_context_length, dim * 2)``.
        """
        cfg = self.config
        dim = cfg.dim
        max_context_length = cfg.max_context_length
        base = cfg.theta

        if cfg.scaling == "llama":
            raise NotImplementedError("Cos/sin RoPE does not support Llama scaling.")

        if cfg.scaling == "yarn" and cfg.rope_factor > 1.0:
            inv_freq = _yarn_inv_freq(
                dim,
                base,
                cfg.rope_factor,
                cfg.beta_fast,
                cfg.beta_slow,
                cfg.original_seq_len,
                cfg.truncate,
            )
        else:
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )

        t = torch.arange(
            max_context_length, dtype=inv_freq.dtype, device=inv_freq.device
        )
        freqs = torch.outer(t, inv_freq).float()
        theta = torch.cat([freqs, freqs], dim=-1)

        cos = theta.cos()
        sin = theta.sin()
        return torch.cat([cos, sin], dim=-1)

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
        """Compatibility entry point for the default rotary implementation."""
        return _DEFAULT_COS_SIN_ROPE_APPLY(query, key, rope_cache, inverse=inverse)


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
