# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonViT3d vision encoder used by Kimi K3.

Shape suffixes:
- M = total merged tokens
- F = merged feature dimension
- O = projected text dimension
"""

from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch_remat as remat

from torchtitan.models.common import Linear
from torchtitan.models.common.attention import create_attention_mask
from torchtitan.models.common.nn_modules import GELU, RMSNorm
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.common.vision_encoder import VisionAttention
from torchtitan.models.kimi_k2_7.vision_encoder import (
    _tpool_patch_merger,
    MoonViTEncoder,
)
from torchtitan.protocols.module import Module


@dataclass
class CPPatchPlan:
    """One image partitioned along its patch dimension across the ranks of ``group``.

    A rank computes q for its own patches and attends over the whole image,
    because k and v are gathered. A fixed-shape collective needs an equal shard
    on every rank, so the tail is padded and the padded key positions are masked
    out of the softmax. The tower's two position tables both index from row 0, so
    they are built for the whole image and sliced to the rank's band rather than
    built from the shard, which would give every rank rank 0's positions.
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
    """Take this rank's row band, frame by frame, out of a table built for the whole image.

    Padding rows repeat the last real row rather than being zeroed, because a
    zeroed RoPE factor is not a rotation and a NaN in the softmax would reach the
    real rows; neither choice changes the result, since padded queries are
    discarded and padded keys are masked.
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


def _padded_key_keep(plan: "CPPatchPlan", total: int, device) -> torch.Tensor:
    """Which of the gathered key positions are real patches, per frame and not as a prefix.

    A rank's shard is padded per frame, so on a video the padding is interleaved
    with real keys; a prefix mask would admit frame 0's padding and mask frame 1's
    real keys. Each rank's real row count follows from the same ceiling split
    ``row_partition`` performs, so this needs no extra field and no collective.
    """
    keep = torch.zeros(total, dtype=torch.bool, device=device)
    t, h, _w = plan.full_grid
    group_size = dist.get_world_size(plan.group)
    if t > 0 and plan.band > 0 and total % (group_size * t * plan.band) == 0:
        row_len = total // (group_size * t * plan.band)
        pos = 0
        for r in range(group_size):
            real_rows = min(plan.band, max(0, h - r * plan.band))
            for _frame in range(t):
                keep[pos : pos + real_rows * row_len] = True
                pos += plan.band * row_len
    else:
        # A plan with no grid describes a flat patch split where the padding IS
        # a trailing run. Falling back matters: the branch above would compute
        # from zeros, mark nothing valid, and mask every key.
        keep[: plan.valid_total] = True
    return keep


def enable_vision_cp_typecheck_rules() -> None:
    """Teach the type checker about the differentiable all-gather the tower uses.

    Called when context parallelism is on rather than at import: the rule is
    process-wide, and the function it names is torch's.
    """
    spmd.register_local_autograd_function(dist_nn._AllGather)


class KimiK3VisionCPAttention(VisionAttention):
    """Vision attention over an image whose patches are split across ranks; without a plan, ``VisionAttention``.

    The gather is the differentiable one, so its transpose is the reduce-scatter
    that returns each rank the gradient for the patches it owns;
    ``dist.all_gather`` would detach and the tower would train on gradients
    missing every other rank's contribution.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(VisionAttention.Config):
        """Same fields; a distinct Config so build() returns this class."""

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        rope_apply,
        attention_mask,
        cp_plan: object | None = None,
    ) -> torch.Tensor:
        # An argument rather than module state: activation checkpointing
        # recomputes this forward from its saved arguments, and state set
        # around the call has been cleared by recompute time.
        if cp_plan is None:
            return super().forward(
                x,
                rope_cache=rope_cache,
                rope_apply=rope_apply,
                attention_mask=attention_mask,
            )
        if not isinstance(cp_plan, CPPatchPlan):
            raise TypeError(f"cp_plan must be a CPPatchPlan, got {type(cp_plan)}")
        plan = cp_plan

        num_tokens = x.shape[0]
        q_THDh, k_THDh, v_THDh = remat.region(
            self._qkv,
            self.remat_region_name("qkv"),
            recompute=self.remat_should_recompute("qkv"),
        )(x)
        remat.recompute_needs_tensor(q_THDh, k_THDh)
        q_THDh, k_THDh = rope_apply(q_THDh, k_THDh, rope_cache)

        # The gather sits outside every recompute region: replaying it would
        # issue the collective a second time.
        k_full = torch.cat(
            dist_nn.all_gather(k_THDh.contiguous(), group=plan.group), dim=0
        )
        v_full = torch.cat(
            dist_nn.all_gather(v_THDh.contiguous(), group=plan.group), dim=0
        )
        total = k_full.size(0)

        # Built unconditionally: flex requires a BlockMask, and with no padding
        # the keep vector is all true, which is the dense mask the replicated
        # path would use.
        keep = _padded_key_keep(plan, total, x.device)

        def _mask_mod(b, h, q_idx, kv_idx):
            del b, h, q_idx
            return keep[kv_idx]

        with spmd.no_typecheck():
            mask = create_attention_mask(
                _mask_mod, None, None, q_THDh.size(0), total, device=x.device
            )
        out_THDh = remat.region(
            self.flex_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(q_THDh, k_full, v_full, attention_masks=mask)
        remat.recompute_needs_tensor(out_THDh)
        out_TD = remat.region(
            self.proj,
            self.remat_region_name("wo"),
            recompute=self.remat_should_recompute("wo"),
        )(out_THDh.reshape(num_tokens, -1))
        remat.recompute_needs_tensor(out_TD)
        return out_TD


class KimiK3VisionProjector(Module):
    """PatchMergerMLPV2 projector from merged vision features to text width."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        linear_1: Linear.Config
        linear_2: Linear.Config
        post_norm: RMSNorm.Config
        activation: GELU.Config = field(default_factory=GELU.Config)

    def __init__(self, config: Config):
        super().__init__()
        self.linear_1 = config.linear_1.build()
        self.linear_2 = config.linear_2.build()
        self.post_norm = config.post_norm.build()
        self.activation = config.activation.build()

    def forward(self, merged_MF: torch.Tensor) -> torch.Tensor:
        projected_MO = self.linear_2(self.activation(self.linear_1(merged_MF)))
        return self.post_norm(projected_MO)


class KimiK3VisionEncoder(MoonViTEncoder):
    @dataclass(kw_only=True, slots=True)
    class Config(MoonViTEncoder.Config):
        patch_size: int
        in_channels: int
        merge_kernel_size: tuple[int, int]  # pyrefly: ignore [bad-override]
        max_num_frames: int

        final_norm: RMSNorm.Config  # pyrefly: ignore [bad-override]
        projector: KimiK3VisionProjector.Config  # pyrefly: ignore [bad-override]

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
        cp_plan: "CPPatchPlan | None" = None,
    ) -> torch.Tensor:
        """The shared tower's forward; with a plan, ``pixel_values`` is this rank's row band of one image."""
        if cp_plan is None:
            return super().forward(pixel_values, grid_thw=grid_thw)

        grids = grid_thw.tolist()
        if len(grids) != 1:
            raise ValueError(
                "a CP patch plan describes one image, but grid_thw carries "
                f"{len(grids)}; a mixed stream needs a per-segment plan."
            )
        # Position tables are built for the WHOLE image, then sliced to this
        # rank's band: ``grid_thw`` describes only the local shard, and building
        # from it gives every rank rank 0's positions. The full grid rides the plan.
        x, rope_cache = self._embed_patches(pixel_values, cp_plan)
        x = self._run_blocks(x, rope_cache=rope_cache, cp_plan=cp_plan)
        return self._merge_and_project(self.final_norm(x), cp_plan)

    def _embed_patches(self, pixel_values, cp_plan):
        """Patch embed plus position tables. Returns (x, rope_cache)."""
        full_grid = [list(cp_plan.full_grid)]
        learned_pos, rope_cache = self.compute_position_embeddings(full_grid)
        learned_pos = _slice_for_shard(learned_pos, cp_plan)
        rope_cache = _slice_for_shard(rope_cache, cp_plan)
        return self.patch_embed(pixel_values) + learned_pos, rope_cache

    def _run_blocks(self, x, *, rope_cache, cp_plan):
        for block in self.layers.values():
            x = block(
                x,
                rope_cache=rope_cache,
                rope_apply=ComplexRoPE.apply_rotary_emb,
                attention_mask=None,
                cp_plan=cp_plan,
            )
        return x

    def _merge_and_project(self, x, cp_plan):
        # The merge sees the SHARD's grid: this rank holds a band of rows for
        # every frame, so the (kh, kw) blocking and the temporal mean are over
        # its own (t, band, w). The positions above needed the whole image.
        t, _, w = cp_plan.full_grid
        kh, kw = self.merge_kernel_size
        merged = _tpool_patch_merger(x, [[t, cp_plan.band, w]], (kh, kw))
        return self.projector(merged)
