# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import contextvars
import copy
import math
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Literal

import spmd_types as spmd
import torch
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from torchtitan.distributed.parallel_dims import SpmdLayout
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import resolve_placements

__all__ = [
    "ComplexRoPE",
    "CosSinRoPE",
    "RoPE",
    "RoPECacheReader",
    "register_rope_cache",
]


@dataclass(frozen=True, slots=True)
class _RoPECacheKey:
    """Hashable identity for a cache-producing RoPE configuration.

    ``SpmdLayout`` contains a dictionary and is intentionally not hashable.  It
    is carried on the key for the private registry to configure its canonical
    buffer, while equality/hash use its stable representation.
    """

    value: tuple
    cache_layout: SpmdLayout | None = field(default=None, compare=False, hash=False)

    def __hash__(self) -> int:
        return hash((self.value, repr(self.cache_layout)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _RoPECacheKey):
            return NotImplemented
        return self.value == other.value and repr(self.cache_layout) == repr(
            other.cache_layout
        )


@dataclass(frozen=True, slots=True)
class _RoPECacheEntry:
    slot_name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    tensor_type: type


_CURRENT_ROPE_CACHE_REGISTRY: contextvars.ContextVar[object | None] = (
    contextvars.ContextVar("current_rope_cache_registry", default=None)
)


class RoPECacheReader:
    """Read-only reference to a model-owned canonical RoPE cache.

    The reader resolves the slot on every ``read()`` call instead of retaining
    a tensor object.  Device/dtype transforms and DTensor distribution may
    replace the registered buffer object during the model lifecycle.
    """

    __slots__ = ("_registry", "_slot_name")

    def __init__(self, registry: "_RoPECacheRegistry", slot_name: str) -> None:
        self._registry = registry
        self._slot_name = slot_name

    def read(self) -> torch.Tensor:
        return self._registry._read(self._slot_name)

    def __deepcopy__(self, memo: dict[int, object]):
        # ``copy.deepcopy`` of a full model must bind readers to the copied
        # registry, not leave them pointing at the source model's buffers.
        registry = copy.deepcopy(self._registry, memo)
        reader = type(self)(registry, self._slot_name)
        memo[id(self)] = reader
        return reader


@contextlib.contextmanager
def _rope_cache_registry_context(registry: "_RoPECacheRegistry"):
    token = _CURRENT_ROPE_CACHE_REGISTRY.set(registry)
    try:
        yield registry
    finally:
        _CURRENT_ROPE_CACHE_REGISTRY.reset(token)


def _current_rope_cache_registry() -> "_RoPECacheRegistry | None":
    registry = _CURRENT_ROPE_CACHE_REGISTRY.get()
    return registry if isinstance(registry, _RoPECacheRegistry) else None


def register_rope_cache(
    cache_key: _RoPECacheKey,
    cache_tensor: torch.Tensor,
) -> RoPECacheReader:
    """Register or read a canonical cache in the active model registry.

    This is the only public registration operation.  The private registry
    retains one buffer per key; equivalent later registrations validate their
    tensor metadata and discard the temporary duplicate.
    """

    registry = _current_rope_cache_registry()
    if registry is None:
        raise RuntimeError(
            "register_rope_cache() requires an active model cache registry. "
            "Standalone RoPE construction should use its local-buffer fallback."
        )
    return registry._register(cache_key, cache_tensor)


class _RoPECacheRegistry(Module):
    """Private module that owns canonical, non-persistent RoPE buffers."""

    def __init__(self) -> None:
        super().__init__()
        self._entries: dict[_RoPECacheKey, _RoPECacheEntry] = {}
        self._slot_keys: dict[str, _RoPECacheKey] = {}
        self._next_slot = 0

    @staticmethod
    def _logical_metadata(cache: torch.Tensor) -> tuple[tuple[int, ...], torch.dtype]:
        return tuple(cache.shape), cache.dtype

    def _register(
        self,
        cache_key: _RoPECacheKey,
        cache_tensor: torch.Tensor,
    ) -> RoPECacheReader:
        shape, dtype = self._logical_metadata(cache_tensor)
        entry = self._entries.get(cache_key)
        if entry is not None:
            if (shape, dtype) != (entry.shape, entry.dtype):
                raise ValueError(
                    "Equivalent RoPE cache key produced incompatible tensors: "
                    f"expected shape/dtype {(entry.shape, entry.dtype)}, got "
                    f"{(shape, dtype)}. Include the missing cache-producing "
                    "configuration in the cache key."
                )
            if type(cache_tensor) is not entry.tensor_type:
                raise ValueError(
                    "Equivalent RoPE cache key produced different tensor types: "
                    f"expected {entry.tensor_type.__name__}, got "
                    f"{type(cache_tensor).__name__}."
                )
            return RoPECacheReader(self, entry.slot_name)

        slot_name = f"_cache_{self._next_slot}"
        self._next_slot += 1
        self.register_buffer(slot_name, cache_tensor, persistent=False)
        self._entries[cache_key] = _RoPECacheEntry(
            slot_name=slot_name,
            shape=shape,
            dtype=dtype,
            tensor_type=type(cache_tensor),
        )
        self._slot_keys[slot_name] = cache_key
        return RoPECacheReader(self, slot_name)

    def _read(self, slot_name: str) -> torch.Tensor:
        cache = self._buffers.get(slot_name)
        if cache is None:
            raise RuntimeError(f"RoPE cache slot {slot_name!r} is not materialized")
        return cache

    def _materialize(
        self,
        reader: RoPECacheReader,
        cache_tensor: torch.Tensor,
    ) -> None:
        entry = self._entry_for_reader(reader)
        shape, dtype = self._logical_metadata(cache_tensor)
        if (shape, dtype) != (entry.shape, entry.dtype):
            raise ValueError(
                "RoPE cache materialization changed shape or dtype: "
                f"expected {(entry.shape, entry.dtype)}, got {(shape, dtype)}."
            )

        current = self._read(entry.slot_name)
        if isinstance(current, DTensor) and not isinstance(cache_tensor, DTensor):
            cache_tensor = distribute_tensor(
                cache_tensor,
                current.device_mesh,
                list(current.placements),
            )

        # Preserve the canonical object when possible.  This keeps any backend
        # annotations attached to the registered buffer while accepting the
        # temporary duplicate computed by each RoPE during initialization.
        if (
            type(current) is type(cache_tensor)
            and current.device == cache_tensor.device
            and tuple(current.shape) == tuple(cache_tensor.shape)
            and current.dtype == cache_tensor.dtype
        ):
            with torch.no_grad():
                current.copy_(cache_tensor)
            return

        persistent = entry.slot_name not in self._non_persistent_buffers_set
        self.register_buffer(entry.slot_name, cache_tensor, persistent=persistent)

    def _entry_for_reader(self, reader: RoPECacheReader) -> _RoPECacheEntry:
        if reader._registry is not self:
            raise ValueError("RoPECacheReader belongs to a different registry")
        for entry in self._entries.values():
            if entry.slot_name == reader._slot_name:
                return entry
        raise KeyError(f"Unknown RoPE cache slot {reader._slot_name!r}")

    def parallelize(self, parallel_dims) -> None:
        """Distribute canonical slots using the layout carried by each key."""
        if self._parallelized:
            raise ValueError(
                f"{type(self).__name__} has already been parallelized. "
                "Module.parallelize() must be called at most once per instance."
            )
        self._parallelized = True

        for cache_key, entry in self._entries.items():
            layout = cache_key.cache_layout
            if layout is None:
                continue
            cache = self._read(entry.slot_name)
            if parallel_dims.spmd_backend == "spmd_types":
                self._spmd_distribute_state(
                    parallel_dims,
                    entry.slot_name,
                    cache,
                    layout,
                    is_param=False,
                )
                continue
            mesh = parallel_dims.resolve_mesh(layout.axes())
            if mesh is None:
                continue
            placements = resolve_placements(layout, mesh)
            if isinstance(cache, DTensor):
                if tuple(cache.placements) != tuple(placements):
                    raise ValueError(
                        f"RoPE cache {entry.slot_name} has placements "
                        f"{cache.placements}, expected {placements}."
                    )
                continue
            self.register_buffer(
                entry.slot_name,
                distribute_tensor(cache, mesh, list(placements)),
                persistent=False,
            )

    def _retain_slots(self, slot_names: set[str]) -> None:
        """Drop canonical buffers that are not used by a PP model chunk."""
        for key, entry in list(self._entries.items()):
            if entry.slot_name in slot_names:
                continue
            self._entries.pop(key)
            self._slot_keys.pop(entry.slot_name, None)
            self._buffers.pop(entry.slot_name, None)
            self._non_persistent_buffers_set.discard(entry.slot_name)


# pyrefly: ignore [not-callable]
@spmd.no_typecheck()
def _maybe_check_max_pos(positions: torch.Tensor, *, max_valid_pos: int) -> None:
    """Async bounds check: verify all position values <= max_valid_pos.

    Uses ``torch._assert_async`` to avoid a device-host sync while still
    catching out-of-bounds positions (the assertion failure surfaces at a
    later kernel launch).  Skipped entirely under ``torch.compile``.
    """
    if torch.compiler.is_compiling():
        return
    pos_local = positions.to_local() if isinstance(positions, DTensor) else positions
    torch._assert_async(
        torch.all(pos_local <= max_valid_pos),
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


class RoPE(Module):
    """Shared Rotary Position Embedding module.

    Common base for concrete RoPE formats. Use ``ComplexRoPE.Config`` for
    complex exponential caches and ``CosSinRoPE.Config`` for concatenated
    cosine/sine caches.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        max_context_length: int
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
        self.config = config
        self._cache_reader: RoPECacheReader | None = None
        cache = self._precompute_cache()
        registry = _current_rope_cache_registry()
        if registry is None:
            self.register_buffer("cache", cache, persistent=False)
        else:
            self._cache_reader = register_rope_cache(self._cache_key(), cache)

    @property
    def cache(self) -> torch.Tensor:
        """Return the current cache tensor for this RoPE instance.

        Model-owned RoPE modules resolve a read-only reader into the private
        registry's canonical buffer. Standalone RoPE modules retain the
        historical direct ``cache`` buffer.
        """
        reader = self.__dict__.get("_cache_reader")
        if reader is not None:
            return reader.read()
        if "cache" not in self._buffers:
            # ``Module.register_buffer`` uses ``hasattr`` to reject collisions;
            # report the pre-registration state as a missing attribute.
            raise AttributeError("cache")
        return self._buffers["cache"]

    @cache.setter
    def cache(self, cache: torch.Tensor) -> None:
        reader = self.__dict__.get("_cache_reader")
        if reader is not None:
            registry = reader._registry
            registry._materialize(reader, cache)
            return
        self.register_buffer("cache", cache, persistent=False)

    def _cache_key(self) -> _RoPECacheKey:
        """Identify cache-equivalent RoPE modules without sharing the modules.

        The cache is a derived, read-only buffer.  Config metadata used only by
        parameter initialization metadata is not cache-producing and is
        excluded. The cache's declared state layout is included separately so
        buffers with conflicting sharding requirements are never coalesced. A
        concrete RoPE class is part of the key because subclasses may use a
        different cache representation.
        """

        def freeze(value):
            if is_dataclass(value):
                return (
                    type(value),
                    tuple(
                        (field.name, freeze(getattr(value, field.name)))
                        for field in fields(value)
                        if field.name not in {"param_init", "sharding_config"}
                    ),
                )
            if isinstance(value, dict):
                items = [(freeze(key), freeze(item)) for key, item in value.items()]
                return tuple(sorted(items, key=repr))
            if isinstance(value, (list, tuple)):
                return tuple(freeze(item) for item in value)
            if isinstance(value, set):
                return tuple(sorted((freeze(item) for item in value), key=repr))
            # Tensor-valued config fields are uncommon, but tensor equality is
            # not a scalar operation and therefore cannot be used in a dict key.
            if isinstance(value, torch.Tensor):
                return type(value), id(value)
            try:
                hash(value)
            except TypeError:
                # Unknown mutable values are conservatively treated as unique.
                return type(value), id(value)
            return value

        sharding = getattr(self.config, "sharding_config", None)
        cache_layout = (
            sharding.state_shardings.get("cache")
            if sharding is not None
            else None
        )
        return _RoPECacheKey(
            value=(type(self), freeze(self.config)),
            cache_layout=cache_layout,
        )

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
        key: torch.Tensor,
        rope_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a prepared RoPE cache to query and key.

        Args:
            query: Query tensor with shape ``[T, N, H]``.
            key: Key tensor with the same leading dimensions as ``query``.
            rope_cache: Prepared cache broadcastable to ``query`` and ``key``
                according to the concrete RoPE format.

        Returns:
            Rotated query and key tensors with the same shapes and dtypes as
            ``query`` and ``key``.
        """
        raise NotImplementedError

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors."""
        reshaped_cache = self._reshape_cache(query, positions)
        return self.apply_rotary_emb(query, key, reshaped_cache)

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
        pass

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
        """Return complex cache shaped for query/key broadcast.

        Returns:
            Cache of shape ``(T, 1, dim / 2)``.
        """
        positions = _maybe_wrap_positions(positions, query)
        if positions is not None:
            _maybe_check_max_pos(positions, max_valid_pos=self.cache.shape[0] - 1)
        # Complex RoPE cache has width dim / 2 because each complex value
        # represents a pair of real dimensions.
        complex_query_shape = (*query.shape[:-1], query.shape[-1] // 2)
        return _reshape_for_broadcast(self.cache, complex_query_shape, positions)

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor,
        rope_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply complex RoPE using adjacent-dim pairs."""
        xq_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
        xq_out = torch.view_as_real(xq_ * rope_cache).flatten(-2)
        xk_out = torch.view_as_real(xk_ * rope_cache).flatten(-2)
        return xq_out.type_as(query), xk_out.type_as(key)


class CosSinRoPE(RoPE):
    @dataclass(kw_only=True, slots=True)
    class Config(RoPE.Config):
        pass

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
        positions = _maybe_wrap_positions(positions, query)
        if positions is not None:
            _maybe_check_max_pos(positions, max_valid_pos=self.cache.shape[0] - 1)
        return _reshape_for_broadcast(self.cache, query.shape, positions)

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor,
        rope_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply cos/sin RoPE using the rotate-half convention."""
        head_dim = query.shape[-1]
        cos = rope_cache[..., :head_dim]
        sin = rope_cache[..., head_dim:]
        query_f = query.float()
        key_f = key.float()
        xq_out = (query_f * cos) + (CosSinRoPE._rotate_half(query_f) * sin)
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
    # cache_width is `head_dim * 2` for CosSinRoPE, and `head_dim // 2` for ComplexRoPE
    cache_width = rope_cache.shape[-1]
    num_tokens = query_shape[0]
    if positions is None:
        rope_cache = rope_cache[:num_tokens]
    else:
        rope_cache = rope_cache[positions]
    return rope_cache.view(num_tokens, 1, cache_width)


def _maybe_wrap_positions(
    positions: torch.Tensor | None,
    x: torch.Tensor,
) -> torch.Tensor | None:
    """Wrap positions as a DTensor deriving mesh and placements from x (xq/xk).

    TODO: positions should be wrapped in/right after dataloading, together
    with inputs and labels, so this helper can go away.

    When TP uses use_local_output=False (DeepSeek V3, Qwen3, GPT-OSS),
    x is a DTensor but positions is a plain tensor. The downstream
    torch.gather requires both operands to be the same type.

    Positions (tokens,) has fewer dimensions than x (tokens, n_heads,
    head_dim), so we only preserve Shard placements for shared dimensions.
    Shard dims beyond positions' rank (e.g. Shard(1) for TP
    on heads) become Replicate.
    """
    if (
        positions is not None
        and isinstance(x, DTensor)
        and not isinstance(positions, DTensor)
    ):
        ndim = positions.ndim
        placements = tuple(
            p if not isinstance(p, Shard) or p.dim < ndim else Replicate()
            for p in x.placements
        )
        positions = DTensor.from_local(
            positions,
            x.device_mesh,
            placements,
            run_check=False,
        )
    return positions
