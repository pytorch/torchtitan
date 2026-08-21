# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonViT-V2: Kimi K3's vision tower.

    Reconciled against the RELEASED reference implementation and the shipped
    checkpoint's key list, not against the report's prose -- the two disagree, and
    the checkpoint wins.

    See ``phase13_k3like_48b_posttrain/MOONVIT_RECONCILIATION.md``.
    """

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate

# The wrapper and the placement helper live in model.py; model.py does not
# import this file, so there is no cycle.
from torchtitan.models.kimi_k3.model import _tp_replicate, RMSNorm


@dataclass
class CPPatchPlan:
    """Dynamic CP: one large image split along the PATCH dimension (report 5.2.3).

    Report 5.2.3, verbatim in substance: "A single large image is partitioned
    along the patch dimension across multiple devices, and attention is computed
    by gathering key-value pairs (gather-KV) across CP ranks."

    This is the half that the earlier image-level round-robin did NOT provide, and
    it is the load-bearing one -- the report's stated purpose for it is to reduce
    "the encoder latency of large visual samples and the cross-device load
    imbalance, allowing the remaining encoder computation to be hidden in pipeline
    bubbles", so DEP depends on it rather than the other way round.

    Each rank holds ``shard_len`` consecutive patches of the image and computes q
    for those alone; k and v are all-gathered across ``group`` so every rank
    attends over the whole image. The gather is differentiable, so its transpose
    is the reduce-scatter that returns each rank the gradient for the patches it
    owns.

    ``valid_total`` is the image's true patch count. A partition needs an equal
    shard on every rank for a fixed-shape collective, so the tail is padded and
    the padded KEY positions are masked out of attention. Without the mask the
    padding would contribute to every softmax -- silently, since the shapes are
    all correct.

    ``full_grid`` and ``patch_start`` exist because MoonViT carries position
    information TWICE -- the divided_fixed absolute embedding added at the patch
    embed, and 2-D RoPE applied to q/k in every block -- and both are built from
    the grid starting at row 0. Describing a shard as a standalone image therefore
    gives every rank the same positions, so rank 1's patches would be encoded as
    if they were rank 0's. Measured before this was carried: the partitioned path
    differed from the replicated one by 2.3e-03 in step-1 loss, which is far too
    large for a reduction-order effect. The tables are built for the whole image
    and sliced.
    """

    group: dist.ProcessGroup
    valid_total: int
    """The image's true patch count, for the padded-key mask."""

    full_grid: tuple[int, int, int] = (0, 0, 0)
    """(t, h, w) of the WHOLE image, not of this shard."""

    row_start: int = 0
    """First patch-grid ROW this rank owns, in the whole image's coordinates."""

    band: int = 0
    """Rows in this rank's tensor, including padding: the shard is (t, band, w)."""

    real_rows: int = 0
    """How many of ``band`` are real; the rest are padding."""


def _slice_for_shard(table: torch.Tensor, plan: "CPPatchPlan"):
    """Take this rank's ROW BAND out of a table built for the WHOLE image.

    The band is strided once the image is a video: the rank owns rows
    ``[row_start, row_start + real_rows)`` of EVERY frame, because the projector's
    temporal mean spans all frames and splitting by frame would break it. So the
    table is gathered frame by frame and padded to ``band`` rows per frame, exactly
    mirroring how the caller lays out the pixels.

    Padding rows repeat the last real row rather than being zeroed: a zeroed RoPE
    factor is not a rotation. Neither choice changes the result -- padded queries
    are discarded and padded keys are masked -- but staying in range keeps a NaN
    out of the softmax, where it would reach real rows.
    """
    t, h, w = plan.full_grid
    per_frame = []
    for f in range(t):
        base = f * h * w
        lo = base + plan.row_start * w
        hi = lo + plan.real_rows * w
        rows = table[lo:hi]
        pad_rows = plan.band - plan.real_rows
        if pad_rows > 0:
            src = rows[-1:] if rows.size(0) else table[base : base + 1]
            rows = torch.cat([rows, src.expand(pad_rows * w, *table.shape[1:])], dim=0)
        per_frame.append(rows)
    return torch.cat(per_frame, dim=0)


@dataclass(kw_only=True)
class MoonViTConfig:
    """MoonViT-V2 config. Defaults are K3's released ``vision_config``."""

    num_hidden_layers: int = 27
    hidden_size: int = 1024
    num_attention_heads: int = 12
    qkv_hidden_size: int = 1536
    intermediate_size: int = 4096
    patch_size: int = 14
    in_channels: int = 3
    rms_norm_eps: float = 1e-5
    init_pos_emb_time: int = 4
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    pos_emb_interpolation_mode: str = "bilinear"
    merge_kernel_size: tuple[int, int] = (2, 2)
    text_hidden_size: int = 7168
    projector_ln_eps: float = 1e-5
    initializer_range: float = 0.02
    # 2-D RoPE grid bound; the reference builds it at 512 x 512 patches, which
    # covers 7168 x 7168 pixels at patch_size 14.
    rope_max_grid: int = 512

    @property
    def head_dim(self) -> int:
        if self.qkv_hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"qkv_hidden_size {self.qkv_hidden_size} must be divisible by "
                f"num_attention_heads {self.num_attention_heads}"
            )
        return self.qkv_hidden_size // self.num_attention_heads


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """``gelu_pytorch_tanh``: the tanh approximation, not the erf form."""
    return F.gelu(x, approximate="tanh")


def sincos_1d(dim: int, length: int, device=None) -> torch.Tensor:
    """Fixed 1-D sincos table, ``[length, dim]``.

    The time half of ``divided_fixed``. Fixed rather than learned, which is why
    the checkpoint carries no time-embedding key.
    """
    if dim % 2:
        raise ValueError(f"sincos dim must be even, got {dim}")
    pos = torch.arange(length, dtype=torch.float32, device=device)
    omega = torch.arange(dim // 2, dtype=torch.float32, device=device)
    omega = 1.0 / (10000 ** (omega / (dim / 2.0)))
    out = pos[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


class MoonViTPatchEmbed(nn.Module):
    """Patch projection plus the divided_fixed absolute position embedding.

    The spatial table is learned at a fixed 64 x 64 patch grid and interpolated
    to whatever grid an input has; the time table is fixed sincos. A single
    frame (``t == 1``) gets NO time component at all -- matching the reference,
    which returns the 2-D embedding untouched in that case rather than adding
    the t=0 entry.
    """

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.mode = config.pos_emb_interpolation_mode
        self.num_frames = config.init_pos_emb_time
        self.proj = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        # Named to match the checkpoint's vision_tower.patch_embed.pos_emb.weight
        self.pos_emb = nn.Module()
        self.pos_emb.weight = nn.Parameter(
            torch.empty(
                config.init_pos_emb_height,
                config.init_pos_emb_width,
                config.hidden_size,
            )
        )
        self.register_buffer(
            "time_weight",
            sincos_1d(config.hidden_size, config.init_pos_emb_time),
            persistent=False,
        )

    def _spatial(self, h: int, w: int) -> torch.Tensor:
        weight = self.pos_emb.weight
        if (h, w) == weight.shape[:-1]:
            return weight.flatten(0, 1)
        # Interpolate on the LOCAL tensor. Under TP this table is a replicated
        # DTensor, and with --debug.deterministic bilinear interpolate lowers to
        # aten._unsafe_index, which DTensor cannot dispatch (it fails with "got
        # mixed torch.Tensor and DTensor"). Every rank holds the same values, so
        # dropping to local and lifting the result back is exact and involves no
        # communication. Non-deterministic mode does not take that lowering,
        # which is why this only shows up under the numerics flags.
        mesh = weight.device_mesh if isinstance(weight, DTensor) else None
        local_weight = weight.to_local() if mesh is not None else weight
        # [H, W, D] -> [1, D, H, W] for interpolate, back to [h*w, D]
        resized = F.interpolate(
            local_weight.permute(2, 0, 1).unsqueeze(0).float(),
            size=(h, w),
            mode=self.mode,
            align_corners=False,
        )
        out = (
            resized.squeeze(0)
            .permute(1, 2, 0)
            .reshape(h * w, -1)
            .to(local_weight.dtype)
        )
        if mesh is None:
            return out
        return DTensor.from_local(out, mesh, (Replicate(),), run_check=False)

    def add_pos_emb(self, x_LD: torch.Tensor, grid_thws: torch.Tensor):
        embs = []
        for t, h, w in grid_thws.tolist():
            if t > self.num_frames:
                raise ValueError(
                    f"t={t} exceeds init_pos_emb_time={self.num_frames}; the "
                    "time table is fixed sincos and is not interpolated"
                )
            pos_2d = self._spatial(h, w)
            if t == 1:
                embs.append(pos_2d)
            else:
                pos_3d = pos_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[:t].to(
                    pos_2d.dtype
                ).unsqueeze(1)
                embs.append(pos_3d.reshape(-1, pos_3d.shape[-1]))
        table = torch.cat(embs, dim=0)
        # Under vision TP ``_spatial`` returns a Replicate DTensor, while dynamic CP's
        # caller builds its whole-image placeholder as a plain tensor -- adding those
        # raises "got mixed torch.Tensor and DTensor". Taking ``to_local`` is exact here
        # rather than a lossy fallback: the table is REPLICATED, so the local shard IS
        # the full table. TP x dynamic CP had no coverage until the matrix was rerun on
        # this head, which is where it surfaced.
        if isinstance(table, DTensor) and not isinstance(x_LD, DTensor):
            table = table.to_local()
        return x_LD + table

    def forward(
        self,
        patches_LCHW: torch.Tensor,
        grid_thws: torch.Tensor,
        cp_plan: "CPPatchPlan | None" = None,
    ):
        """``[L, C, p, p]`` patch pixels -> ``[L, D]`` tokens.

        ``L`` is the total token count over the batch, i.e.
        ``sum_i t_i * h_i * w_i``.
        """
        x = self.proj(patches_LCHW).view(patches_LCHW.size(0), -1)
        if cp_plan is not None:
            # Dynamic CP: this stream is a SHARD of one image. Build the whole
            # image's table and take our rows, or every rank would be handed the
            # positions of rank 0's patches.
            t, h, w = cp_plan.full_grid
            full = torch.zeros(t * h * w, x.size(-1), dtype=x.dtype, device=x.device)
            full = self.add_pos_emb(
                full, torch.tensor([[t, h, w]], device=grid_thws.device)
            )
            pos = _slice_for_shard(full, cp_plan)
            # Slice on a LOCAL tensor -- _slice_for_shard indexes rows, so it must not run
            # on a DTensor -- then match x's type. Under vision TP x is a DTensor and the
            # position table is REPLICATED across the TP axis, so wrapping the slice as
            # Replicate is exact rather than a coercion. Adding the two without this is
            # the "mixed torch.Tensor and DTensor" failure that TP x dynamic CP hit.
            if isinstance(x, DTensor) and not isinstance(pos, DTensor):
                pos = DTensor.from_local(
                    pos, x.device_mesh, (Replicate(),), run_check=False
                )
            return x + pos
        return self.add_pos_emb(x, grid_thws)


class MoonViTRope2D(nn.Module):
    """2-D RoPE over the patch grid, repeated across frames.

    Applied to q/k in every block, ON TOP of the absolute embedding above. Half
    the head dim encodes the row index and half the column index; a video
    repeats the same 2-D frequencies for every frame, so RoPE carries no
    temporal signal -- that is the fixed sincos table's job.
    """

    def __init__(self, head_dim: int, max_grid: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 4:
            raise ValueError(f"2-D RoPE needs head_dim divisible by 4, got {head_dim}")
        self.head_dim = head_dim
        self.max_grid = max_grid
        self.theta = theta
        self._cache: torch.Tensor | None = None

    def _freqs(self, device) -> torch.Tensor:
        if self._cache is not None and self._cache.device == device:
            return self._cache
        quarter = self.head_dim // 4
        freqs = 1.0 / (
            self.theta
            ** (torch.arange(quarter, dtype=torch.float32, device=device) / quarter)
        )
        pos = torch.arange(self.max_grid, dtype=torch.float32, device=device)
        angles = torch.outer(pos, freqs)  # [max_grid, quarter]
        self._cache = torch.polar(torch.ones_like(angles), angles)
        return self._cache

    def freqs_cis(self, grid_thws: torch.Tensor, device) -> torch.Tensor:
        """``[L, head_dim // 2]`` complex rotations for the packed stream."""
        table = self._freqs(device)
        out = []
        for t, h, w in grid_thws.tolist():
            if max(h, w) > self.max_grid:
                raise ValueError(f"grid {h}x{w} exceeds rope_max_grid={self.max_grid}")
            rows = table[:h].unsqueeze(1).expand(h, w, -1)
            cols = table[:w].unsqueeze(0).expand(h, w, -1)
            frame = torch.cat([rows, cols], dim=-1).reshape(h * w, -1)
            out.append(frame.repeat(t, 1))
        return torch.cat(out, dim=0)

    @staticmethod
    def apply(x_LAK: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Rotate ``[L, A, K]`` by ``[L, K // 2]`` complex factors.

        Under TP the tower is replicated via NoParallel, which makes its
        activations DTensors while this table is built inside forward and stays
        a plain tensor. Multiplying the two raises "got mixed torch.Tensor and
        DTensor", so promote the table to a replicated DTensor on the same mesh.
        """
        L, A, K = x_LAK.shape
        if isinstance(x_LAK, DTensor) and not isinstance(freqs_cis, DTensor):
            freqs_cis = DTensor.from_local(
                freqs_cis, x_LAK.device_mesh, (Replicate(),), run_check=False
            )
        xc = torch.view_as_complex(x_LAK.float().reshape(L, A, K // 2, 2))
        rotated = xc * freqs_cis.unsqueeze(1)
        return torch.view_as_real(rotated).reshape(L, A, K).to(x_LAK.dtype)


class MoonViTMLP(nn.Module):
    """``mlp2``. Named fc0/fc1 to match the checkpoint."""

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.fc0 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc1 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(_gelu_tanh(self.fc0(x)))


class MoonViTEncoderLayer(nn.Module):
    """Pre-norm block: RMSNorm, one varlen attention, RMSNorm, MLP. No biases."""

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        # (lo, hi) head range this rank attends over; None = all heads.
        self._tp_head_slice: tuple[int, int] | None = None
        # Dynamic CP: set when this rank holds a patch shard of one large image.
        self._cp_patch_plan: CPPatchPlan | None = None
        self.norm0 = RMSNorm.Config(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
            sharding_config=_tp_replicate(),
        ).build()
        self.wqkv = nn.Linear(
            config.hidden_size, 3 * config.qkv_hidden_size, bias=False
        )
        self.wo = nn.Linear(config.qkv_hidden_size, config.hidden_size, bias=False)
        self.norm1 = RMSNorm.Config(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
            sharding_config=_tp_replicate(),
        ).build()
        self.mlp = MoonViTMLP(config)

    def _attend_gather_kv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        plan: CPPatchPlan,
    ) -> torch.Tensor:
        """Attention over one image whose patches are split across ``plan.group``.

        q is this rank's patch shard; k and v are gathered so the shard attends
        over the whole image. There is no cu_seqlens loop here because a plan means
        the local stream IS one shard of one image -- a mixed stream (whole small
        images alongside a shard) needs a per-segment plan and is not supported
        yet, which is why the caller asserts the single-image case rather than
        letting a mixed stream silently attend across image boundaries.
        """
        import torch.distributed.nn.functional as dist_nn

        # Differentiable gather: the backward is the reduce-scatter that returns
        # each rank the gradient for the patches it owns. dist.all_gather would
        # detach and the tower would train on gradients missing every other
        # rank's contribution.
        k_full = torch.cat(dist_nn.all_gather(k.contiguous(), group=plan.group), dim=0)
        v_full = torch.cat(dist_nn.all_gather(v.contiguous(), group=plan.group), dim=0)

        total = k_full.size(0)
        # SDPA wants [B, A, L, K].
        q_ = q.transpose(0, 1).unsqueeze(0)
        k_ = k_full.transpose(0, 1).unsqueeze(0)
        v_ = v_full.transpose(0, 1).unsqueeze(0)

        attn_mask = None
        if plan.valid_total < total:
            # Mask the padded KEY positions. Broadcasting over queries and heads is
            # enough: every query attends to the same key set.
            #
            # NOT a prefix. ``_slice_for_shard`` pads PER FRAME -- it takes each frame's
            # band rows, tops that frame up to ``band``, and only then concatenates the
            # frames -- so a deficit rank's stream is
            # [frame0 real, frame0 pad, frame1 real, frame1 pad, ...] and the padding is
            # INTERLEAVED. A prefix mask admits frame 0's padding into the softmax and
            # masks frame 1's real keys instead: silently wrong encoder output whenever
            # t > 1 and some rank is short. (t == 1 has one frame, so a prefix happens to
            # be right, which is why every earlier test passed.)
            #
            # Each rank's real row count follows from the same ceiling split
            # ``row_partition`` performs, so it needs no extra field or collective:
            # rank r holds min(band, max(0, h - r * band)) real rows.
            t, h, _w = plan.full_grid
            group_size = dist.get_world_size(plan.group)
            keep = torch.zeros(total, dtype=torch.bool, device=q.device)
            if t > 0 and plan.band > 0 and total % (group_size * t * plan.band) == 0:
                row_len = total // (group_size * t * plan.band)
                pos = 0
                for r in range(group_size):
                    real_rows = min(plan.band, max(0, h - r * plan.band))
                    for _frame in range(t):
                        keep[pos : pos + real_rows * row_len] = True
                        pos += plan.band * row_len
            else:
                # A plan carrying no grid (``full_grid``/``band`` left at their defaults)
                # describes a flat patch split with no frame structure, which is how the
                # attention-level unit tests build it. There the padding IS a trailing
                # run, so the prefix is exact. Falling back rather than computing from
                # zeros matters: the general branch above would mark nothing valid and
                # mask every key.
                keep[: plan.valid_total] = True
            attn_mask = keep.view(1, 1, 1, total)
        out = F.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=attn_mask, is_causal=False
        )
        return out.squeeze(0).transpose(0, 1)

    def _attend(
        self,
        x_LD: torch.Tensor,
        seq_bounds: list[int],
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        L = x_LD.size(0)
        qkv = self.wqkv(x_LD).view(L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)
        q = MoonViTRope2D.apply(q, freqs_cis)
        k = MoonViTRope2D.apply(k, freqs_cis)

        # Tensor parallel over heads. wqkv stays REPLICATED and every rank
        # projects all heads: its fused output is [3, A, K] with the 3 outermost,
        # so an even column split would give rank 0 all of q plus half of k, and
        # permuting the weight to fix that would change the checkpoint contract.
        # Slicing after the projection costs a redundant qkv matmul and
        # parallelizes attention, which is the part that scales with sequence.
        heads = self._tp_head_slice
        if heads is not None:
            lo, hi = heads
            # Drop to local BEFORE slicing. q/k/v are DTensor(Replicate) and
            # slicing one yields another Replicate with a SMALLER logical shape,
            # which loses the fact that this is a shard -- wo would then see a
            # logical [L, A_local*K] against a weight whose reduction dim is the
            # full [L, A*K]. A plain tensor lets wo's Shard(-1) input layout say
            # it. grad_placements is Partial by construction: each rank's
            # gradient is its own additive contribution to the replicated wqkv.
            from torch.distributed.tensor import DTensor as _DT, Partial as _P

            if isinstance(q, _DT):
                q, k, v = (t.to_local(grad_placements=[_P()]) for t in (q, k, v))
            q, k, v = q[:, lo:hi], k[:, lo:hi], v[:, lo:hi]

        plan = self._cp_patch_plan
        if plan is not None:
            # Dynamic CP gathers KV over the patch group with a plain process-group
            # collective, which needs local tensors. The head-sharded branch above
            # already dropped to local; the REPLICATED-attention branch has not, and
            # that is the only configuration where vision TP and dynamic CP ever met
            # a DTensor here -- it happens when the head count does not divide the TP
            # ranks (parallelize.py warns and leaves attention replicated), so it was
            # invisible on any tower whose heads divide.
            from torch.distributed.tensor import DTensor as _DT, Replicate as _R

            tp_mesh = None
            if isinstance(q, _DT):
                if any(not isinstance(p, _R) for p in q.placements):
                    raise ValueError(
                        "MoonViT dynamic CP expects replicated attention inputs "
                        f"when attention is not head-sharded, got {q.placements}"
                    )
                tp_mesh = q.device_mesh
                # Replicate, so the local shard IS the full tensor and both
                # conversions are exact rather than coercions.
                #
                # grad_placements is Replicate, NOT Partial. Every TP rank runs the
                # same full-head attention and so receives the same full gradient;
                # summing across them would scale it by tp_size. Partial is right in
                # the head-sharded branch above for the opposite reason -- the slices
                # there are disjoint, so each rank's gradient is an additive part.
                q, k, v = (t.to_local(grad_placements=[_R()]) for t in (q, k, v))
            out = self._attend_gather_kv(q, k, v, plan)
            local_heads = out.size(1)
            out = out.reshape(out.size(0), local_heads * self.head_dim)
            if tp_mesh is not None:
                # wo is not in the TP plan in this branch, so distribute_module left
                # it replicated and it takes a DTensor. Re-wrapping restores exactly
                # the structure the non-CP replicated path hands it.
                out = _DT.from_local(out, tp_mesh, [_R()], run_check=False)
            return self.wo(out)

        # Block-diagonal attention over the packed stream: each sample attends
        # only within itself. Done as a per-sample loop over SDPA rather than a
        # flash varlen kernel so this runs anywhere; the segment boundaries are
        # the same either way.
        out = torch.empty_like(q)
        for start, end in zip(seq_bounds[:-1], seq_bounds[1:]):
            seg = slice(start, end)
            out[seg] = (
                F.scaled_dot_product_attention(
                    q[seg].transpose(0, 1).unsqueeze(0),
                    k[seg].transpose(0, 1).unsqueeze(0),
                    v[seg].transpose(0, 1).unsqueeze(0),
                    is_causal=False,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        local_heads = out.size(1)
        return self.wo(out.reshape(L, local_heads * self.head_dim))

    def forward(
        self,
        x_LD: torch.Tensor,
        seq_bounds: list[int],
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        x_LD = x_LD + self._attend(self.norm0(x_LD), seq_bounds, freqs_cis)
        return x_LD + self.mlp(self.norm1(x_LD))


def tpool_patch_merger(
    x_LD: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[torch.Tensor]:
    """``sd2_tpool``: mean over ALL frames, then a 2x2 space-to-depth.

    The temporal axis is collapsed completely -- ``mean(dim=0)`` over every
    frame, not a pairwise pool -- so a video and a single image both leave one
    frame's worth of tokens. The spatial merge is space-to-depth: a 2x2
    neighbourhood becomes ``kh*kw`` channels, so the 4x token reduction discards
    nothing before the projector.

    Returns one ``[h/kh * w/kw, kh*kw, D]`` tensor per sample; lengths differ
    across samples, which is why this is a list.
    """
    d_model = x_LD.size(-1)
    kh, kw = merge_kernel_size
    outputs, offset = [], 0
    for t, h, w in grid_thws.tolist():
        if h % kh or w % kw:
            raise ValueError(
                f"patch grid {h}x{w} must divide the merge kernel {kh}x{kw}"
            )
        seq = x_LD[offset : offset + t * h * w]
        nh, nw = h // kh, w // kw
        seq = seq.view(t, nh, kh, nw, kw, d_model)
        seq = seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        outputs.append(seq.view(nh * nw, kh * kw, d_model))
        offset += t * h * w
    return outputs


class PatchMergerMLPV2(nn.Module):
    """``patchmergerv2``: two bias-free Linears, GELU, RMSNorm AFTER.

    The post-norm placement (and the absence of a pre-norm) is what
    distinguishes v2 from ``PatchMergerMLP``, and the checkpoint's
    ``mm_projector.post_norm.weight`` with no pre_norm key confirms which one
    shipped.
    """

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        kh, kw = config.merge_kernel_size
        merged = config.hidden_size * kh * kw
        self.merged_size = merged
        self.proj = nn.Sequential(
            nn.Linear(merged, merged, bias=False),
            nn.GELU(),
            nn.Linear(merged, config.text_hidden_size, bias=False),
        )
        self.post_norm = RMSNorm.Config(
            normalized_shape=config.text_hidden_size,
            eps=config.projector_ln_eps,
            sharding_config=_tp_replicate(),
        ).build()

    def forward(self, merged: list[torch.Tensor] | torch.Tensor):
        if isinstance(merged, (list, tuple)):
            return [
                self.post_norm(self.proj(item.reshape(item.shape[0], -1)))
                for item in merged
            ]
        return self.post_norm(self.proj(merged.reshape(*merged.shape[:-2], -1)))

    def init_weights(self) -> None:
        # The reference initializes the projector with trunc_normal_ scaled by
        # fan-in rather than the tower's global init_range.
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))
        nn.init.ones_(self.post_norm.weight)


class MoonViTEncoder(nn.Module):
    """The 27 blocks plus the final norm."""

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.rope_2d = MoonViTRope2D(config.head_dim, config.rope_max_grid)
        self.blocks = nn.ModuleList(
            MoonViTEncoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.final_layernorm = RMSNorm.Config(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
            sharding_config=_tp_replicate(),
        ).build()

    def set_cp_patch_plan(self, plan: CPPatchPlan | None) -> None:
        """Apply (or clear) a dynamic-CP patch partition on every block.

        Set per forward, not once at build: which images are large enough to
        partition depends on the batch, so a plan that outlived its batch would
        make the next batch's attention gather across a group for a partition that
        no longer exists.
        """
        for block in self.blocks:
            block._cp_patch_plan = plan

    def block_inputs(
        self,
        x_LD: torch.Tensor,
        grid_thws: torch.Tensor,
        cp_plan: "CPPatchPlan | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The per-forward ``(freqs_cis, cu_seqlens)`` every block needs.

        Split out so a tower spanning several PP stages can recompute it on each
        stage from that stage's own ``grid_thws`` (report 5.2.3 balances vision
        passes across PP stages). Recomputing rather than sending it over the pipe is
        deliberate: PP's metadata inference pushes DUMMY values through pipe tensors,
        and these are used as RoPE indices and segment bounds, where a dummy asserts
        out of bounds -- the same reason ``input_ids`` never leave the vision stage.
        """
        if cp_plan is not None:
            # 2-D RoPE is the SECOND place position enters, so it needs the same
            # whole-image-then-slice treatment as the absolute embedding.
            t, h, w = cp_plan.full_grid
            full = self.rope_2d.freqs_cis(
                torch.tensor([[t, h, w]], device=grid_thws.device), x_LD.device
            )
            freqs_cis = _slice_for_shard(full, cp_plan)
        else:
            freqs_cis = self.rope_2d.freqs_cis(grid_thws, x_LD.device)
        lengths = grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2]
        cu_seqlens = torch.cat(
            [torch.zeros(1, dtype=lengths.dtype, device=lengths.device), lengths]
        ).cumsum(0, dtype=torch.int32)
        return freqs_cis, cu_seqlens

    def run_blocks(
        self,
        x_LD: torch.Tensor,
        grid_thws: torch.Tensor,
        cp_plan: "CPPatchPlan | None" = None,
        *,
        block_slice: slice | None = None,
        apply_final_norm: bool = True,
    ) -> torch.Tensor:
        """Run a contiguous range of blocks, optionally without the final norm.

        ``block_slice`` selects this stage's share when the tower is split across PP
        stages; ``apply_final_norm`` belongs to the last share only. Chaining the
        shares reproduces the whole encoder exactly, which is asserted by a unit test
        rather than assumed.
        """
        freqs_cis, cu_seqlens = self.block_inputs(x_LD, grid_thws, cp_plan)
        # Converted ONCE per forward, not once per block. cu_seqlens lives on the
        # device and .tolist() is a device-to-host sync, so doing it inside the
        # block loop paid 27 syncs per tower forward at K3's depth -- and the
        # boundaries are identical for every block, being a property of the batch.
        # The tensor stays block_inputs' contract because a later DEP share
        # recomputes it locally rather than receiving it over the pipe.
        seq_bounds = cu_seqlens.tolist()
        blocks = self.blocks if block_slice is None else self.blocks[block_slice]
        self.set_cp_patch_plan(cp_plan)
        try:
            for block in blocks:
                x_LD = block(x_LD, seq_bounds, freqs_cis)
        finally:
            # The encoder owns the plan's lifetime so a caller cannot leak one
            # into the next batch, where it would gather for a partition that no
            # longer exists.
            self.set_cp_patch_plan(None)
        return self.final_layernorm(x_LD) if apply_final_norm else x_LD

    def forward(
        self,
        x_LD: torch.Tensor,
        grid_thws: torch.Tensor,
        cp_plan: "CPPatchPlan | None" = None,
    ) -> torch.Tensor:
        return self.run_blocks(x_LD, grid_thws, cp_plan)


class MoonViT(nn.Module):
    """MoonViT-V2 tower + PatchMergerMLPV2 projector.

    Submodule names (``patch_embed``, ``encoder``, and the projector held
    separately as ``mm_projector``) mirror the checkpoint so the state-dict
    adapter is a prefix rename rather than a structural remap.
    """

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = MoonViTPatchEmbed(config)
        self.encoder = MoonViTEncoder(config)
        self.mm_projector = PatchMergerMLPV2(config)

    @staticmethod
    def patchify(pixels_BFCHW: torch.Tensor, patch_size: int):
        """Rectangular ``[B, F, C, H, W]`` video -> packed patches + grid_thws.

        A convenience for uniform batches. Native-resolution training packs
        variable-sized samples itself and calls :meth:`forward` directly.
        """
        if pixels_BFCHW.dim() == 4:
            pixels_BFCHW = pixels_BFCHW.unsqueeze(1)
        B, Fr, C, H, W = pixels_BFCHW.shape
        if H % patch_size or W % patch_size:
            raise ValueError(f"{H}x{W} is not divisible by patch_size {patch_size}")
        h, w = H // patch_size, W // patch_size
        x = pixels_BFCHW.reshape(B * Fr, C, h, patch_size, w, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(
            B * Fr * h * w, C, patch_size, patch_size
        )
        grid = torch.tensor(
            [[Fr, h, w]] * B, dtype=torch.long, device=pixels_BFCHW.device
        )
        return x, grid

    def block_bounds(self, num_shares: int) -> list[tuple[int, int]]:
        """Split the encoder's blocks into ``num_shares`` contiguous ranges.

        Report 5.2.3 balances vision passes across PP stages, so shares are as even as
        possible. A remainder goes to the LAST shares, because share 0 also carries
        ``patch_embed`` and the final share's projector is cheaper than that -- giving
        share 0 an extra block as well would make the least balanced stage worse.
        """
        n = len(self.encoder.blocks)
        if num_shares < 1 or num_shares > n:
            raise ValueError(
                f"cannot split {n} encoder block(s) into {num_shares} share(s)"
            )
        base, extra = divmod(n, num_shares)
        bounds, lo = [], 0
        for i in range(num_shares):
            hi = lo + base + (1 if i >= num_shares - extra else 0)
            bounds.append((lo, hi))
            lo = hi
        return bounds

    def forward_head(
        self,
        patches_LCHW: torch.Tensor,
        grid_thws: torch.Tensor,
        cp_plan: "CPPatchPlan | None" = None,
        *,
        upto_block: int | None = None,
    ) -> torch.Tensor:
        """Patch embed plus blocks ``[0, upto_block)``, WITHOUT the final norm.

        The first share when the tower spans PP stages. Returns patch hidden states,
        not features -- the projector belongs to the last share.
        """
        x = self.patch_embed(patches_LCHW, grid_thws, cp_plan)
        return self.encoder.run_blocks(
            x,
            grid_thws,
            cp_plan,
            block_slice=slice(0, upto_block),
            apply_final_norm=False,
        )

    def forward_body(
        self,
        x_LD: torch.Tensor,
        grid_thws: torch.Tensor,
        cp_plan: "CPPatchPlan | None" = None,
        *,
        lo: int,
        hi: int,
    ) -> torch.Tensor:
        """Blocks ``[lo, hi)`` only -- a middle share, no norm and no projector."""
        return self.encoder.run_blocks(
            x_LD, grid_thws, cp_plan, block_slice=slice(lo, hi), apply_final_norm=False
        )

    def forward_tail(
        self,
        x_LD: torch.Tensor,
        grid_thws: torch.Tensor,
        cp_plan: "CPPatchPlan | None" = None,
        *,
        from_block: int = 0,
    ):
        """Blocks ``[from_block, end)``, the final norm, the merge and the projector.

        The last share, and the only one that produces features.
        """
        x = self.encoder.run_blocks(
            x_LD,
            grid_thws,
            cp_plan,
            block_slice=slice(from_block, len(self.encoder.blocks)),
            apply_final_norm=True,
        )
        merged = tpool_patch_merger(x, grid_thws, self.config.merge_kernel_size)
        return self.mm_projector(merged)

    def forward(
        self,
        patches_LCHW: torch.Tensor,
        grid_thws: torch.Tensor,
        cp_plan: "CPPatchPlan | None" = None,
        *,
        part: str | None = None,
        upto_block: int | None = None,
        lo: int | None = None,
        hi: int | None = None,
        from_block: int | None = None,
    ):
        """Packed patches -> a list of ``[N_i, text_hidden_size]`` per sample.

        ``cp_plan`` marks the input as one rank's patch shard of a single image
        (dynamic CP, report 5.2.3). ``grid_thws`` then describes the SHARD -- the
        merger and the segment bounds want that -- while the plan carries the whole
        image's grid, which is what the two position sources need.

        ``part`` selects one share of a tower that spans PP stages (report 5.2.3
        clause 2): "head", "body" or "tail". The shares have to be reached THROUGH
        this forward rather than by calling forward_head / forward_body /
        forward_tail directly, because FSDP2 registers its all-gather on the
        module's __call__: a direct method call leaves patch_embed.proj.weight a
        sharded DTensor and the conv fails with "got mixed torch.Tensor and
        DTensor". That is why n_vit > 1 ran only at dp_shard=1 before.
        """
        if part is not None:
            if part == "head":
                return self.forward_head(
                    patches_LCHW, grid_thws, cp_plan, upto_block=upto_block
                )
            if part == "body":
                return self.forward_body(patches_LCHW, grid_thws, cp_plan, lo=lo, hi=hi)
            if part == "tail":
                return self.forward_tail(
                    patches_LCHW, grid_thws, cp_plan, from_block=from_block or 0
                )
            raise ValueError(f"unknown tower part {part!r}")
        x = self.patch_embed(patches_LCHW, grid_thws, cp_plan)
        x = self.encoder(x, grid_thws, cp_plan)
        merged = tpool_patch_merger(x, grid_thws, self.config.merge_kernel_size)
        return self.mm_projector(merged)

    def encoder_num_parameters(self) -> int:
        """Parameters in the tower proper, excluding the projector.

        The model card's 401M figure is the encoder; the projector is described
        separately as "a lightweight MLP projector" and at text_hidden_size 7168
        it is not lightweight relative to the tower.
        """
        proj = {id(p) for p in self.mm_projector.parameters()}
        return sum(p.numel() for p in self.parameters() if id(p) not in proj)

    def init_weights(self, init_range: float | None = None) -> None:
        std = init_range if init_range is not None else self.config.initializer_range
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.RMSNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.patch_embed.pos_emb.weight, mean=0.0, std=std)
        self.mm_projector.init_weights()
