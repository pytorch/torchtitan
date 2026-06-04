# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP8 all-gather on GroupedRaggedShard with block-wise quantization.

``GroupedRaggedShard`` cuts a bucket into byte-balanced ranges. By forcing those
cuts onto whole ``block_size x block_size`` quantization-tile boundaries
(``alignment = block_size * suffix_numel``), every rank owns *complete* tiles, so
it can quantize its shard to fp8 **locally** -- no cross-rank ``amax``. The
all-gather then moves **fp8 + tiny scales** (~half the bytes), and the gathered
fp8 weight is **bit-identical** to "all-gather bf16, then block-quantize".

The master weight stays bf16-sharded (storage is unchanged); fp8 is materialized
only by the unshard collective (the torchao/float8-FSDP model, ported to the
FlexShard placement contract). See
``fp8_allgather_grouped_ragged_shard_plan.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from typing_extensions import override

from ..flex_shard.placement_contract import (
    PlacementPreparedUnshard,
    PlacementUnshardResult,
)
from ..flex_shard.utils import _record_comm_if_eager, _record_function_if_eager
from .ragged_shard import GroupedRaggedShard

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..flex_shard.bucket_storage import ParamInfo, PlacementFn
    from ..flex_shard.placement_contract import Placement


EPS = 1e-12


def _as_block_pair(block_size: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize ``block_size`` to ``(block_m, block_n)``; an int means a square tile."""
    if isinstance(block_size, int):
        return block_size, block_size
    block_m, block_n = block_size
    return block_m, block_n


def blockwise_quant_weight(
    weight: torch.Tensor,
    block_size: int | tuple[int, int],
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise fp8 weight quant (DeepSeek-V3 / torchao recipe), pure PyTorch.

    Quantizes a 2D ``(M, N)`` weight in ``block_m x block_n`` tiles: one ``amax`` per
    tile -> ``scale = fp8_max / amax`` -> ``clamp(x*scale)``. ``block_size`` is an int
    for a **square** tile (the transpose-reusable weight recipe) or a
    ``(block_m, block_n)`` pair for a **rectangular** tile (e.g. ``(1, 128)`` /
    ``(128, 1)`` groupwise). Returns the fp8 data (same shape) and the **reciprocal**
    scales (fp32, shape ``(M//block_m, N//block_n)``), matching torchao's stored
    ``1/scale`` so the fp8 GEMM applies ``dot * a_s * b_s`` directly.
    """
    if weight.ndim != 2:
        raise ValueError(
            f"blockwise_quant_weight expects 2D, got {tuple(weight.shape)}"
        )
    m, n = weight.shape
    block_m, block_n = _as_block_pair(block_size)
    if m % block_m != 0 or n % block_n != 0:
        raise ValueError(
            f"dims {tuple(weight.shape)} must be divisible by block "
            f"({block_m}, {block_n})"
        )
    fp8_max = torch.finfo(fp8_dtype).max
    tiles = (
        weight.reshape(m // block_m, block_m, n // block_n, block_n)
        .permute(0, 2, 1, 3)
        .reshape(-1, block_m * block_n)
    )
    amax = tiles.abs().amax(dim=1, keepdim=True).clamp(min=EPS).to(torch.float64)
    scale = (fp8_max / amax).to(torch.float32)
    quant = (tiles.to(torch.float32) * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    quant = (
        quant.reshape(m // block_m, n // block_n, block_m, block_n)
        .permute(0, 2, 1, 3)
        .reshape(m, n)
    )
    recip_scale = (1.0 / scale).reshape(m // block_m, n // block_n).to(torch.float32)
    return quant.contiguous(), recip_scale.contiguous()


def blockwise_dequant_weight(
    quant: torch.Tensor,
    recip_scale: torch.Tensor,
    block_size: int | tuple[int, int],
) -> torch.Tensor:
    """Inverse of :func:`blockwise_quant_weight` (fp8 + reciprocal scale -> fp32)."""
    m, n = quant.shape
    block_m, block_n = _as_block_pair(block_size)
    tiles = (
        quant.reshape(m // block_m, block_m, n // block_n, block_n)
        .permute(0, 2, 1, 3)
        .reshape(-1, block_m * block_n)
        .to(torch.float32)
    )
    deq = tiles * recip_scale.reshape(-1, 1).to(torch.float32)
    return (
        deq.reshape(m // block_m, n // block_n, block_m, block_n)
        .permute(0, 2, 1, 3)
        .reshape(m, n)
    )


def promote_to_square_block(block_m: int, block_n: int) -> int:
    """Smallest square block that tiles a ``block_m x block_n`` grid into whole tiles.

    Promoting a non-square request (e.g. ``1x128``) to this square block (``128``) makes
    the gathered weight **transpose-reusable** -- one fp8 buffer serves both the forward
    and backward GEMMs -- at the cost of a **coarser** scale (one per square tile instead
    of per non-square tile), i.e. slightly weaker outlier adaptation. Use when the
    single-gather / transpose win outweighs the fidelity loss; otherwise keep the
    non-square tiling and pay two buffers (:class:`Fp8TwoOrientationGroupedRaggedShard`).
    """
    return math.lcm(block_m, block_n)


def blockwise_transpose(
    quant: torch.Tensor,
    recip_scale: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transpose a block-wise fp8 weight to the other matmul orientation.

    For **square** ``block_size x block_size`` tiles, quantization commutes with
    transpose: tile ``(i, j)`` of ``W`` and tile ``(j, i)`` of ``Wᵀ`` are the same
    elements, so ``fp8(Wᵀ) == fp8(W)ᵀ`` and ``scale(Wᵀ) == scale(W)ᵀ`` *exactly*.
    A single gathered ``W`` can therefore serve both the forward (RHS ``Wᵀ``) and the
    backward dgrad (RHS ``W``) with no re-quantization and no second all-gather.

    Returns zero-copy transposed views ``(quant.t(), recip_scale.t())``: a row-major
    ``(M, N)`` weight becomes a **column-major** ``(N, M)`` view -- the layout the fp8
    GEMM RHS expects for the forward pass.

    GUARD: this identity holds **only for square tiles**. ``recip_scale`` must have
    shape ``(M // block_size, N // block_size)``; a non-square tiling (e.g. ``1x128``
    activation scales, or separate row/col block sizes) regroups elements under
    transpose and would silently corrupt numerics, so it is rejected.
    """
    if quant.ndim != 2:
        raise ValueError(
            f"blockwise_transpose expects a 2D weight, got {tuple(quant.shape)}"
        )
    m, n = quant.shape
    expected = (m // block_size, n // block_size)
    if (
        m % block_size != 0
        or n % block_size != 0
        or tuple(recip_scale.shape) != expected
    ):
        raise ValueError(
            "blockwise_transpose requires square block_size x block_size tiling: "
            f"weight {tuple(quant.shape)} with block_size {block_size} expects scale "
            f"shape {expected}, but got {tuple(recip_scale.shape)}. The "
            "fp8(W^T) == fp8(W)^T transpose-invariance holds only for square tiles."
        )
    return quant.t(), recip_scale.t()


class Fp8BlockwiseGroupedRaggedShard(GroupedRaggedShard):
    """``GroupedRaggedShard`` that all-gathers block-wise fp8 instead of bf16.

    The byte cut is aligned to whole ``block_size x block_size`` tiles
    (``alignment = block_size * suffix_numel``), so each rank owns complete tiles
    and quantizes its shard locally; the gathered fp8 + scales are bit-identical to
    gathering bf16 then quantizing. Storage stays bf16 (the optimizer's master
    weight); only the unshard produces fp8. ``finish_prepared_unshard`` returns the
    full fp8 tensor with ``._blockwise_scale`` / ``._block_size`` attached for an
    fp8-aware consumer. Currently requires even ``local_units``.
    """

    @dataclass(frozen=True)
    class _Fp8UnshardState:
        infos: list[ParamInfo]
        pg: Any
        debug_fqn: str | None
        rank_offsets: tuple[int, ...]
        rank_numels: tuple[int, ...]

    def __init__(
        self,
        dims: tuple[int, ...] = (0,),
        local_units: tuple[int, ...] = (1,),
        block_size: int = 128,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> None:
        super().__init__(dims, local_units)
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}.")
        self.block_size = block_size
        self.fp8_dtype = fp8_dtype

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fp8BlockwiseGroupedRaggedShard):
            return False
        return (
            self.dims == other.dims
            and self.local_units == other.local_units
            and self.block_size == other.block_size
            and self.fp8_dtype == other.fp8_dtype
        )

    def __hash__(self) -> int:
        return hash(
            (type(self), self.dims, self.local_units, self.block_size, self.fp8_dtype)
        )

    def __repr__(self) -> str:
        return (
            "Fp8BlockwiseGroupedRaggedShard("
            f"dims={self.dims}, local_units={self.local_units}, "
            f"block_size={self.block_size})"
        )

    def _require_even_units(self) -> None:
        if len(set(self.local_units)) != 1:
            raise ValueError(
                "Fp8BlockwiseGroupedRaggedShard currently requires even local_units "
                f"(all equal) so every rank's shard is the same size; got "
                f"{self.local_units}."
            )

    @override
    def _param_alignment_numel(
        self,
        named_params: list[tuple[str, nn.Parameter]],
    ) -> int:
        # Align the byte cut to whole block x block tiles: every rank then owns
        # complete tiles and can quantize locally. Requires both the sharded
        # (prefix) and the kept (suffix) dims to be multiples of block_size.
        alignment = 1
        for fqn, param in named_params:
            prefix_numel = self._prefix_numel(param.shape)
            suffix_numel = self._suffix_numel(param.shape)
            if (
                prefix_numel % self.block_size != 0
                or suffix_numel % self.block_size != 0
            ):
                raise ValueError(
                    "Fp8BlockwiseGroupedRaggedShard requires both the sharded prefix "
                    f"({prefix_numel}) and the kept suffix ({suffix_numel}) of {fqn!r} "
                    f"to be divisible by block_size ({self.block_size}) so the "
                    f"{self.block_size}x{self.block_size} tiles are whole."
                )
            alignment = math.lcm(alignment, self.block_size * suffix_numel)
        return alignment

    @override
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        """Quantize each local shard to fp8 + scales and pack the send buffers."""
        self._require_even_units()
        bs = self.block_size
        tile_numel = bs * bs
        bucket_layout = self._bucket_layout(infos[0])
        rank = mesh.get_local_rank()
        device = next((t.device for t in tensors if t.numel() > 0), tensors[0].device)
        rank_offset = bucket_layout.rank_offsets[rank]
        rank_numel = bucket_layout.rank_numels[rank]

        with _record_function_if_eager("FlexShard::all_gather_copy_in", debug_fqn):
            local_fp8 = torch.zeros(rank_numel, dtype=self.fp8_dtype, device=device)
            local_scale = torch.zeros(
                rank_numel // tile_numel, dtype=torch.float32, device=device
            )
            for tensor, info in zip(tensors, infos, strict=True):
                if info.local_numel == 0:
                    continue
                quant, recip_scale = blockwise_quant_weight(
                    tensor.reshape(info.local_shape), bs, self.fp8_dtype
                )
                offset = self._param_layout(info).local_global_offset - rank_offset
                local_fp8[offset : offset + info.local_numel].copy_(quant.reshape(-1))
                local_scale[
                    offset // tile_numel : offset // tile_numel
                    + info.local_numel // tile_numel
                ].copy_(recip_scale.reshape(-1))

        gathered_fp8 = torch.empty(
            bucket_layout.global_numel, dtype=self.fp8_dtype, device=device
        )
        gathered_scale = torch.empty(
            bucket_layout.global_numel // tile_numel, dtype=torch.float32, device=device
        )
        return PlacementPreparedUnshard(
            placement=self,
            buffers=[local_fp8, local_scale, gathered_fp8, gathered_scale],
            placement_state=Fp8BlockwiseGroupedRaggedShard._Fp8UnshardState(
                infos=infos,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                rank_offsets=bucket_layout.rank_offsets,
                rank_numels=bucket_layout.rank_numels,
            ),
        )

    @override
    def run_prepared_unshard(self, prepared: PlacementPreparedUnshard) -> None:
        """All-gather the fp8 data (as bytes) and the fp32 scales.

        Uses the list ``dist.all_gather`` (op ``c10d.allgather_``, like the bf16
        ``GroupedRaggedShard``), which reshard-after-forward already tags
        ``MUST_RECOMPUTE`` -- so the fp8 unshard is freed after forward and
        re-quantized + re-gathered in backward. fp8 is viewed as uint8 so the
        collective is dtype-agnostic across NCCL versions (fp8 is 1 byte, 1:1).
        """
        state = prepared.placement_state
        if not isinstance(state, Fp8BlockwiseGroupedRaggedShard._Fp8UnshardState):
            raise AssertionError(
                "Expected Fp8BlockwiseGroupedRaggedShard._Fp8UnshardState, "
                f"got {type(state).__name__}"
            )
        local_fp8, local_scale, gathered_fp8, gathered_scale = prepared.buffers
        tile_numel = self.block_size * self.block_size
        # Per-rank destination views: gathered[rank r range] receives rank r's shard.
        fp8_views = [
            gathered_fp8[offset : offset + numel].view(torch.uint8)
            for offset, numel in zip(state.rank_offsets, state.rank_numels, strict=True)
        ]
        scale_views = [
            gathered_scale[
                offset // tile_numel : offset // tile_numel + numel // tile_numel
            ]
            for offset, numel in zip(state.rank_offsets, state.rank_numels, strict=True)
        ]
        with _record_comm_if_eager("FlexShard::all_gather", state.debug_fqn):
            dist.all_gather(fp8_views, local_fp8.view(torch.uint8), group=state.pg)
            dist.all_gather(scale_views, local_scale, group=state.pg)

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        """Slice each param's full fp8 + scales from the gathered buffers."""
        if not isinstance(
            prepared.placement_state, Fp8BlockwiseGroupedRaggedShard._Fp8UnshardState
        ):
            raise AssertionError(
                "Expected Fp8BlockwiseGroupedRaggedShard._Fp8UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        gathered_fp8 = prepared.buffers[2]
        gathered_scale = prepared.buffers[3]
        bs = self.block_size
        tile_numel = bs * bs
        full_params: list[torch.Tensor] = []
        with _record_function_if_eager(
            "FlexShard::all_gather_copy_out", prepared.placement_state.debug_fqn
        ):
            for info in prepared.placement_state.infos:
                param_offset = self._param_layout(info).param_offset
                out_dim, in_dim = info.global_shape
                data = gathered_fp8[
                    param_offset : param_offset + info.global_numel
                ].view(info.global_shape)
                num_scales = (out_dim // bs) * (in_dim // bs)
                scale = gathered_scale[
                    param_offset // tile_numel : param_offset // tile_numel + num_scales
                ].view(out_dim // bs, in_dim // bs)
                # Attach the block-wise scale so an fp8-aware consumer can use the
                # gathered weight without a separate dequant.
                data._blockwise_scale = scale
                data._block_size = bs
                full_params.append(data)
        return PlacementUnshardResult(full_params, prepared.buffers)


def make_fp8_blockwise_grouped_ragged_placement_fn(
    *,
    block_size: int = 128,
    local_units: tuple[int, ...],
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> PlacementFn:
    """Return a placement_fn assigning one fp8 GroupedRaggedShard per bucket."""

    def fp8_blockwise_placements(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        if len(local_units) != mesh.size():
            raise ValueError(
                "Fp8BlockwiseGroupedRaggedShard local_units length must match mesh "
                f"size: got {len(local_units)} local units for mesh size {mesh.size()}."
            )
        placement = Fp8BlockwiseGroupedRaggedShard(
            dims=(0,),
            local_units=local_units,
            block_size=block_size,
            fp8_dtype=fp8_dtype,
        )
        return {fqn: (placement,) for fqn, _ in named_params}

    return fp8_blockwise_placements


class Fp8TwoOrientationGroupedRaggedShard(GroupedRaggedShard):
    """Non-square block-wise fp8 all-gather: two fp8 buffers, one per matmul orientation.

    When the weight uses a **non-square** tiling, ``fp8(Wᵀ) != fp8(W)ᵀ``, so a single
    gathered buffer cannot serve both passes. This placement gathers **two** fp8 weights
    from one bf16 shard: the **forward** buffer tiled ``(1, block)`` (scale grouped along
    ``in`` -- the forward contraction dim) and the **backward** buffer tiled
    ``(block, 1)`` (scale grouped along ``out`` -- the backward dgrad contraction dim).
    Each is bit-identical to quantizing the full bf16 weight in that tiling. The cut is
    aligned to whole ``block`` rows so the ``(block, 1)`` tiling owns complete tiles.

    Bandwidth is ~2x the square path (two fp8 data buffers) -- see
    ``fp8_allgather_grouped_ragged_shard_plan.md``; prefer the square
    :class:`Fp8BlockwiseGroupedRaggedShard` (one buffer, transpose-reused) unless a
    non-square recipe is required. Storage stays bf16 (master); even ``local_units``.

    ``finish`` returns the forward fp8 weight with the backward weight and both scales
    attached: ``_fp8_backward``, ``_scale_forward`` ``(out, in/block)``,
    ``_scale_backward`` ``(out/block, in)``, ``_block_size``.
    """

    @dataclass(frozen=True)
    class _TwoOrientationState:
        infos: list[ParamInfo]
        pg: Any
        debug_fqn: str | None
        rank_offsets: tuple[int, ...]
        rank_numels: tuple[int, ...]

    def __init__(
        self,
        dims: tuple[int, ...] = (0,),
        local_units: tuple[int, ...] = (1,),
        block_size: int = 128,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> None:
        super().__init__(dims, local_units)
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}.")
        self.block_size = block_size
        self.fp8_dtype = fp8_dtype

    def __eq__(self, other: object) -> bool:
        if type(other) is not Fp8TwoOrientationGroupedRaggedShard:
            return False
        return (
            self.dims == other.dims
            and self.local_units == other.local_units
            and self.block_size == other.block_size
            and self.fp8_dtype == other.fp8_dtype
        )

    def __hash__(self) -> int:
        return hash(
            (type(self), self.dims, self.local_units, self.block_size, self.fp8_dtype)
        )

    def __repr__(self) -> str:
        return (
            "Fp8TwoOrientationGroupedRaggedShard("
            f"dims={self.dims}, local_units={self.local_units}, "
            f"block_size={self.block_size})"
        )

    def _require_even_units(self) -> None:
        if len(set(self.local_units)) != 1:
            raise ValueError(
                "Fp8TwoOrientationGroupedRaggedShard currently requires even "
                f"local_units (all equal); got {self.local_units}."
            )

    @override
    def _param_alignment_numel(
        self,
        named_params: list[tuple[str, nn.Parameter]],
    ) -> int:
        # Align the cut to whole `block` rows so the (block, 1) backward tiling owns
        # complete tiles (the (1, block) forward tiling is row-granular, always local).
        alignment = 1
        for fqn, param in named_params:
            prefix_numel = self._prefix_numel(param.shape)
            suffix_numel = self._suffix_numel(param.shape)
            if (
                prefix_numel % self.block_size != 0
                or suffix_numel % self.block_size != 0
            ):
                raise ValueError(
                    "Fp8TwoOrientationGroupedRaggedShard requires both the sharded "
                    f"prefix ({prefix_numel}) and the kept suffix ({suffix_numel}) of "
                    f"{fqn!r} to be divisible by block_size ({self.block_size})."
                )
            alignment = math.lcm(alignment, self.block_size * suffix_numel)
        return alignment

    @override
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        """Quantize each shard in both orientations and pack the four send buffers."""
        self._require_even_units()
        bs = self.block_size
        bucket_layout = self._bucket_layout(infos[0])
        rank = mesh.get_local_rank()
        device = next((t.device for t in tensors if t.numel() > 0), tensors[0].device)
        rank_offset = bucket_layout.rank_offsets[rank]
        rank_numel = bucket_layout.rank_numels[rank]

        with _record_function_if_eager("FlexShard::all_gather_copy_in", debug_fqn):
            local_fwd = torch.zeros(rank_numel, dtype=self.fp8_dtype, device=device)
            local_bwd = torch.zeros(rank_numel, dtype=self.fp8_dtype, device=device)
            local_scale_fwd = torch.zeros(
                rank_numel // bs, dtype=torch.float32, device=device
            )
            local_scale_bwd = torch.zeros(
                rank_numel // bs, dtype=torch.float32, device=device
            )
            for tensor, info in zip(tensors, infos, strict=True):
                if info.local_numel == 0:
                    continue
                shard = tensor.reshape(info.local_shape)
                # forward: scale grouped along `in` (the forward contraction dim).
                q_fwd, s_fwd = blockwise_quant_weight(shard, (1, bs), self.fp8_dtype)
                # backward: scale grouped along `out` (the dgrad contraction dim).
                q_bwd, s_bwd = blockwise_quant_weight(shard, (bs, 1), self.fp8_dtype)
                offset = self._param_layout(info).local_global_offset - rank_offset
                local_fwd[offset : offset + info.local_numel].copy_(q_fwd.reshape(-1))
                local_bwd[offset : offset + info.local_numel].copy_(q_bwd.reshape(-1))
                scale_offset = offset // bs
                scale_numel = info.local_numel // bs
                local_scale_fwd[scale_offset : scale_offset + scale_numel].copy_(
                    s_fwd.reshape(-1)
                )
                local_scale_bwd[scale_offset : scale_offset + scale_numel].copy_(
                    s_bwd.reshape(-1)
                )

        gathered_fwd = torch.empty(
            bucket_layout.global_numel, dtype=self.fp8_dtype, device=device
        )
        gathered_bwd = torch.empty(
            bucket_layout.global_numel, dtype=self.fp8_dtype, device=device
        )
        gathered_scale_fwd = torch.empty(
            bucket_layout.global_numel // bs, dtype=torch.float32, device=device
        )
        gathered_scale_bwd = torch.empty(
            bucket_layout.global_numel // bs, dtype=torch.float32, device=device
        )
        return PlacementPreparedUnshard(
            placement=self,
            buffers=[
                local_fwd,
                local_bwd,
                local_scale_fwd,
                local_scale_bwd,
                gathered_fwd,
                gathered_bwd,
                gathered_scale_fwd,
                gathered_scale_bwd,
            ],
            placement_state=Fp8TwoOrientationGroupedRaggedShard._TwoOrientationState(
                infos=infos,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                rank_offsets=bucket_layout.rank_offsets,
                rank_numels=bucket_layout.rank_numels,
            ),
        )

    @override
    def run_prepared_unshard(self, prepared: PlacementPreparedUnshard) -> None:
        """All-gather both fp8 orientations (as bytes) and both fp32 scale sets."""
        state = prepared.placement_state
        if not isinstance(
            state, Fp8TwoOrientationGroupedRaggedShard._TwoOrientationState
        ):
            raise AssertionError(
                "Expected Fp8TwoOrientationGroupedRaggedShard._TwoOrientationState, "
                f"got {type(state).__name__}"
            )
        (
            local_fwd,
            local_bwd,
            local_scale_fwd,
            local_scale_bwd,
            gathered_fwd,
            gathered_bwd,
            gathered_scale_fwd,
            gathered_scale_bwd,
        ) = prepared.buffers
        bs = self.block_size

        def _data_views(gathered: torch.Tensor) -> list[torch.Tensor]:
            return [
                gathered[offset : offset + numel].view(torch.uint8)
                for offset, numel in zip(
                    state.rank_offsets, state.rank_numels, strict=True
                )
            ]

        def _scale_views(gathered: torch.Tensor) -> list[torch.Tensor]:
            return [
                gathered[offset // bs : offset // bs + numel // bs]
                for offset, numel in zip(
                    state.rank_offsets, state.rank_numels, strict=True
                )
            ]

        with _record_comm_if_eager("FlexShard::all_gather", state.debug_fqn):
            dist.all_gather(
                _data_views(gathered_fwd), local_fwd.view(torch.uint8), group=state.pg
            )
            dist.all_gather(
                _data_views(gathered_bwd), local_bwd.view(torch.uint8), group=state.pg
            )
            dist.all_gather(
                _scale_views(gathered_scale_fwd), local_scale_fwd, group=state.pg
            )
            dist.all_gather(
                _scale_views(gathered_scale_bwd), local_scale_bwd, group=state.pg
            )

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        """Slice each param's forward + backward fp8 weight and scales."""
        state = prepared.placement_state
        if not isinstance(
            state, Fp8TwoOrientationGroupedRaggedShard._TwoOrientationState
        ):
            raise AssertionError(
                "Expected Fp8TwoOrientationGroupedRaggedShard._TwoOrientationState, "
                f"got {type(state).__name__}"
            )
        gathered_fwd = prepared.buffers[4]
        gathered_bwd = prepared.buffers[5]
        gathered_scale_fwd = prepared.buffers[6]
        gathered_scale_bwd = prepared.buffers[7]
        bs = self.block_size
        full_params: list[torch.Tensor] = []
        with _record_function_if_eager(
            "FlexShard::all_gather_copy_out", state.debug_fqn
        ):
            for info in state.infos:
                param_offset = self._param_layout(info).param_offset
                out_dim, in_dim = info.global_shape
                num = info.global_numel
                data_fwd = gathered_fwd[param_offset : param_offset + num].view(
                    info.global_shape
                )
                data_bwd = gathered_bwd[param_offset : param_offset + num].view(
                    info.global_shape
                )
                scale_start = param_offset // bs
                scale_fwd = gathered_scale_fwd[
                    scale_start : scale_start + num // bs
                ].view(out_dim, in_dim // bs)
                scale_bwd = gathered_scale_bwd[
                    scale_start : scale_start + num // bs
                ].view(out_dim // bs, in_dim)
                # Forward weight is the returned param; backward weight + both scales
                # ride along for the fp8-aware consumer (no transpose reuse here).
                data_fwd._scale_forward = scale_fwd
                data_fwd._fp8_backward = data_bwd
                data_fwd._scale_backward = scale_bwd
                data_fwd._block_size = bs
                full_params.append(data_fwd)
        return PlacementUnshardResult(full_params, prepared.buffers)


def make_fp8_two_orientation_grouped_ragged_placement_fn(
    *,
    block_size: int = 128,
    local_units: tuple[int, ...],
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> PlacementFn:
    """Return a placement_fn assigning one two-orientation fp8 shard per bucket."""

    def fp8_two_orientation_placements(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        if len(local_units) != mesh.size():
            raise ValueError(
                "Fp8TwoOrientationGroupedRaggedShard local_units length must match mesh "
                f"size: got {len(local_units)} local units for mesh size {mesh.size()}."
            )
        placement = Fp8TwoOrientationGroupedRaggedShard(
            dims=(0,),
            local_units=local_units,
            block_size=block_size,
            fp8_dtype=fp8_dtype,
        )
        return {fqn: (placement,) for fqn, _ in named_params}

    return fp8_two_orientation_placements


__all__ = [
    "blockwise_dequant_weight",
    "blockwise_quant_weight",
    "blockwise_transpose",
    "Fp8BlockwiseGroupedRaggedShard",
    "Fp8TwoOrientationGroupedRaggedShard",
    "make_fp8_blockwise_grouped_ragged_placement_fn",
    "make_fp8_two_orientation_grouped_ragged_placement_fn",
    "promote_to_square_block",
]
