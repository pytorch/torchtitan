# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonViT tower that encodes large images as row bands over sub-CP groups."""

from dataclasses import dataclass
from typing import NamedTuple

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.common.vision_encoder import compiled_create_block_mask
from torchtitan.models.kimi_k2_7.vision_encoder import (
    _tpool_patch_merger,
    MoonViTEncoder,
)

from .attention import all_gather_parts, VisionCPLayout
from .plan import DynamicCPPlan, ImageShard, key_runs, plan_dynamic_cp, row_partition


def _band(
    table: torch.Tensor, grid: list[int], row_start: int, real: int, band: int
) -> torch.Tensor:
    t, h, w = grid
    frames = []
    for f in range(t):
        base = f * h * w
        rows = table[base + row_start * w : base + (row_start + real) * w]
        if band > real:
            fill = rows[-w:] if real else table[base : base + w]
            rows = torch.cat([rows, fill.repeat(band - real, *[1] * (table.dim() - 1))])
        frames.append(rows)
    return torch.cat(frames)


def _band_rows(shards: list[ImageShard], r: int) -> tuple[int, int, int]:
    first, own = shards[0], shards[r]
    return (
        own.row_start,
        own.row_end - own.row_start,
        first.row_end - first.row_start,
    )


def _split_mask(
    query_runs: list[tuple[int, int]],
    key_runs: list[tuple[int, int]],
    device: torch.device,
) -> BlockMask:
    def doc_ids(runs: list[tuple[int, int]]) -> torch.Tensor:
        ids = torch.tensor([i for i, _ in runs], dtype=torch.int32, device=device)
        lengths = torch.tensor([n for _, n in runs], device=device)
        return torch.repeat_interleave(
            ids, lengths, output_size=sum(n for _, n in runs)
        )

    query_doc, key_doc = doc_ids(query_runs), doc_ids(key_runs)

    def mask_mod(b, h, q_idx, kv_idx):
        return query_doc[q_idx] == key_doc[kv_idx]

    with spmd.no_typecheck():
        return compiled_create_block_mask(
            mask_mod,
            1,
            None,
            query_doc.shape[0],
            key_doc.shape[0],
            device=device,
        )


class _PackedInputs(NamedTuple):
    x: torch.Tensor
    rope_cache: torch.Tensor
    local_grids: list[list[int]]
    query_runs: list[tuple[int, int]]
    num_whole_tokens: int


class MoonViTCPEncoder(MoonViTEncoder):
    """MoonViT tower that splits large images over sub-CP groups under context parallelism."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoonViTEncoder.Config):
        dynamic_cp_min_patches: int = 256
        """Under context parallelism, images with at least this many patches are split over a sub-CP group."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.dynamic_cp_min_patches = config.dynamic_cp_min_patches
        self._cp_subgroups: dict[int, dist.ProcessGroup] = {}

    def set_cp_subgroups(self, subgroups: dict[int, dist.ProcessGroup]) -> None:
        """The sub-CP groups an image may be split over, from ``build_cp_subgroups``."""
        self._cp_subgroups = subgroups

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        cp_group = spmd_mesh_group(MeshAxisName.CP) if self._cp_subgroups else None
        if cp_group is not None:
            grids = grid_thw.tolist()
            plan = plan_dynamic_cp(
                grids,
                cp_size=cp_group.size(),
                kh=self.merge_kernel_size[0],
                min_patches=self.dynamic_cp_min_patches,
            )
            if plan is not None:
                return self._forward_split(pixel_values, grids, plan, cp_group)
        return super().forward(pixel_values, grid_thw=grid_thw)

    def _forward_split(
        self,
        pixel_values: torch.Tensor,
        grids: list[list[int]],
        plan: DynamicCPPlan,
        cp_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        kh = self.merge_kernel_size[0]
        g = plan.subgroup_size
        subgroup, rank_in_subgroup = divmod(cp_group.rank(), g)
        shards = {
            i: row_partition(grids[i][1], kh=kh, group_size=g) for i in plan.split
        }
        packed = self._pack_inputs(
            pixel_values, grids, plan, shards, subgroup, rank_in_subgroup
        )
        layout = VisionCPLayout(
            block_mask=_split_mask(
                packed.query_runs,
                key_runs(
                    grids,
                    list(plan.whole),
                    plan.images_of(subgroup),
                    kh=kh,
                    group_size=g,
                ),
                packed.x.device,
            ),
            num_whole_tokens=packed.num_whole_tokens,
            group=self._cp_subgroups[plan.num_subgroups] if g > 1 else None,
        )
        features = self._encode(packed, layout)
        return self._assemble_bank(features, grids, plan, shards, subgroup, cp_group)

    def _pack_inputs(
        self,
        pixel_values: torch.Tensor,
        grids: list[list[int]],
        plan: DynamicCPPlan,
        shards: dict[int, list[ImageShard]],
        subgroup: int,
        rank_in_subgroup: int,
    ) -> _PackedInputs:
        whole = list(plan.whole)
        offsets = [0]
        for t, h, w in grids:
            offsets.append(offsets[-1] + t * h * w)

        pieces, positions, ropes, local_grids, query_runs = [], [], [], [], []
        if whole:
            whole_grids = [grids[i] for i in whole]
            learned_pos, rope_cache = self.compute_position_embeddings(whole_grids)
            pieces += [pixel_values[offsets[i] : offsets[i + 1]] for i in whole]
            positions.append(learned_pos)
            ropes.append(rope_cache)
            local_grids += whole_grids
            query_runs += [(i, offsets[i + 1] - offsets[i]) for i in whole]
        num_whole_tokens = sum(n for _, n in query_runs)
        for i in plan.images_of(subgroup):
            row_start, real, band = _band_rows(shards[i], rank_in_subgroup)
            learned_pos, rope_cache = self.compute_position_embeddings([grids[i]])
            image = pixel_values[offsets[i] : offsets[i + 1]]
            pieces.append(_band(image, grids[i], row_start, real, band))
            positions.append(_band(learned_pos, grids[i], row_start, real, band))
            ropes.append(_band(rope_cache, grids[i], row_start, real, band))
            t, _, w = grids[i]
            local_grids.append([t, band, w])
            query_runs.append((i, t * band * w))

        x = self.patch_embed(torch.cat(pieces)) + torch.cat(positions)
        return _PackedInputs(
            x, torch.cat(ropes), local_grids, query_runs, num_whole_tokens
        )

    def _encode(self, packed: _PackedInputs, layout: VisionCPLayout) -> torch.Tensor:
        kh, kw = self.merge_kernel_size
        x = packed.x
        for block in self.layers.values():
            x = block(
                x,
                rope_cache=packed.rope_cache,
                rope_apply=ComplexRoPE.apply_rotary_emb,
                attention_mask=layout,
            )
        x = self.final_norm(x)
        return self.projector(_tpool_patch_merger(x, packed.local_grids, (kh, kw)))

    def _assemble_bank(
        self,
        features: torch.Tensor,
        grids: list[list[int]],
        plan: DynamicCPPlan,
        shards: dict[int, list[ImageShard]],
        subgroup: int,
        cp_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        kh, kw = self.merge_kernel_size
        g = plan.subgroup_size
        out: dict[int, torch.Tensor] = {}
        start = 0
        for i in plan.whole:
            merged = (grids[i][1] // kh) * (grids[i][2] // kw)
            out[i] = features[start : start + merged]
            start += merged

        band_merged = {
            i: _band_rows(shards[i], 0)[2] // kh * (grids[i][2] // kw)
            for i in plan.split
        }
        lengths = [
            sum(band_merged[i] for i in plan.images_of(s))
            for s in range(plan.num_subgroups)
        ]
        longest = max(lengths)
        local = F.pad(features[start:], (0, 0, 0, longest - lengths[subgroup]))
        everyone = all_gather_parts(local, cp_group)
        for s in range(plan.num_subgroups):
            offset = 0
            for i in plan.images_of(s):
                rows = []
                for r in range(g):
                    real = _band_rows(shards[i], r)[1] // kh * (grids[i][2] // kw)
                    rows.append(everyone[s * g + r][offset : offset + real])
                out[i] = torch.cat(rows)
                offset += band_merged[i]
        return torch.cat([out[i] for i in range(len(grids))])
