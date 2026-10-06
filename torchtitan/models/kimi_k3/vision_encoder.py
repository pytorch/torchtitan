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
import torch.nn.functional as F
import torch_remat as remat
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import GELU, RMSNorm
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.common.vision_encoder import (
    compiled_create_block_mask,
    VisionAttention,
)
from torchtitan.models.kimi_k2_7.vision_encoder import (
    _tpool_patch_merger,
    MoonViTEncoder,
)
from torchtitan.protocols.module import Module

from .vit_cp_plan import DynamicCPPlan, key_runs, plan_dynamic_cp, row_partition


@dataclass
class VisionCPLayout:
    """Attention over whole images followed by this rank's bands of the images a sub-CP group splits."""

    block_mask: BlockMask
    num_whole_tokens: int
    group: dist.ProcessGroup | None
    """The sub-CP group the bands gather keys and values over; ``None`` for a group of one."""


def _all_gather(x: torch.Tensor, group: dist.ProcessGroup) -> list[torch.Tensor]:
    # The gather runs outside SPMD type checking, so each part takes the input's type.
    with spmd.no_typecheck():
        parts = list(dist_nn.all_gather(x.contiguous(), group=group))
    if spmd.is_type_checking():
        local_type, partition_spec = spmd.get_local_type(x), spmd.get_partition_spec(x)
        for part in parts:
            spmd.assert_type(part, local_type, partition_spec)
    return parts


def _gather_bands(x: torch.Tensor, layout: VisionCPLayout) -> torch.Tensor:
    if layout.group is None:
        return x
    whole, bands = x[: layout.num_whole_tokens], x[layout.num_whole_tokens :]
    return torch.cat([whole, *_all_gather(bands, layout.group)], dim=0)


class KimiK3VisionCPAttention(VisionAttention):
    """Vision attention that gathers keys and values of split images over their sub-CP group."""

    @dataclass(kw_only=True, slots=True)
    class Config(VisionAttention.Config):
        pass

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        rope_apply,
        attention_mask,
    ) -> torch.Tensor:
        if not isinstance(attention_mask, VisionCPLayout):
            return super().forward(
                x,
                rope_cache=rope_cache,
                rope_apply=rope_apply,
                attention_mask=attention_mask,
            )
        num_tokens = x.shape[0]
        q_THDh, k_THDh, v_THDh = self._qkv(x)
        # RoPE and the gather read the projections with bare ops.
        remat.recompute_needs_tensor(q_THDh, k_THDh, v_THDh)
        q_THDh, k_THDh = rope_apply(q_THDh, k_THDh, rope_cache)
        k_THDh = _gather_bands(k_THDh, attention_mask)
        v_THDh = _gather_bands(v_THDh, attention_mask)
        out_THDh = remat.region(
            self.flex_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(q_THDh, k_THDh, v_THDh, attention_metadata=attention_mask.block_mask)
        remat.recompute_needs_tensor(out_THDh)
        return self.proj(out_THDh.reshape(num_tokens, -1))


def build_cp_subgroups(cp_mesh: DeviceMesh) -> dict[int, dist.ProcessGroup]:
    """This rank's group for every equal split of its CP mesh, keyed by the number of sub-groups."""
    cp_size = cp_mesh.size()
    subgroups = {1: cp_mesh.get_group()}
    for num in range(2, cp_size):
        if cp_size % num:
            continue
        index, sub = f"cp_sub{num}_index", f"cp_sub{num}"
        mesh = cp_mesh._unflatten(
            0, (num, cp_size // num), (index, sub), backend_override={index: "fake"}
        )
        subgroups[num] = mesh[sub].get_group()
    return subgroups


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
    """MoonViT3d tower that splits large images over sub-CP groups under context parallelism."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoonViTEncoder.Config):
        patch_size: int
        in_channels: int
        merge_kernel_size: tuple[int, int]  # pyrefly: ignore [bad-override]
        max_num_frames: int

        final_norm: RMSNorm.Config  # pyrefly: ignore [bad-override]
        projector: KimiK3VisionProjector.Config  # pyrefly: ignore [bad-override]

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
        kh, kw = self.merge_kernel_size
        g = plan.subgroup_size
        cp_rank = cp_group.rank()
        subgroup, rank_in_subgroup = divmod(cp_rank, g)
        mine = plan.images_of(subgroup)
        whole = list(plan.whole)
        offsets = [0]
        for t, h, w in grids:
            offsets.append(offsets[-1] + t * h * w)
        shards = {i: row_partition(*grids[i], kh=kh, group_size=g) for i in plan.split}

        def band_rows(i: int, r: int) -> tuple[int, int, int]:
            first, own = shards[i][0], shards[i][r]
            return (
                own.row_start,
                own.row_end - own.row_start,
                first.row_end - first.row_start,
            )

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
        for i in mine:
            row_start, real, band = band_rows(i, rank_in_subgroup)
            learned_pos, rope_cache = self.compute_position_embeddings([grids[i]])
            image = pixel_values[offsets[i] : offsets[i + 1]]
            pieces.append(_band(image, grids[i], row_start, real, band))
            positions.append(_band(learned_pos, grids[i], row_start, real, band))
            ropes.append(_band(rope_cache, grids[i], row_start, real, band))
            t, _, w = grids[i]
            local_grids.append([t, band, w])
            query_runs.append((i, t * band * w))

        x = self.patch_embed(torch.cat(pieces)) + torch.cat(positions)
        rope_cache = torch.cat(ropes)
        layout = VisionCPLayout(
            block_mask=self._split_mask(
                query_runs,
                key_runs(grids, whole, mine, kh=kh, group_size=g),
                x.device,
            ),
            num_whole_tokens=num_whole_tokens,
            group=self._cp_subgroups[plan.num_subgroups] if g > 1 else None,
        )
        for block in self.layers.values():
            x = block(
                x,
                rope_cache=rope_cache,
                rope_apply=ComplexRoPE.apply_rotary_emb,
                attention_mask=layout,
            )
        x = self.final_norm(x)
        features = self.projector(_tpool_patch_merger(x, local_grids, (kh, kw)))

        merged = {
            i: (grids[i][1] // kh) * (grids[i][2] // kw) for i in range(len(grids))
        }
        out: dict[int, torch.Tensor] = {}
        start = 0
        for i in whole:
            out[i] = features[start : start + merged[i]]
            start += merged[i]

        def band_merged(i: int) -> int:
            return band_rows(i, 0)[2] // kh * (grids[i][2] // kw)

        lengths = [
            sum(band_merged(i) for i in plan.images_of(s))
            for s in range(plan.num_subgroups)
        ]
        longest = max(lengths)
        local = F.pad(features[start:], (0, 0, 0, longest - lengths[subgroup]))
        everyone = _all_gather(local, cp_group)
        for s in range(plan.num_subgroups):
            offset = 0
            for i in plan.images_of(s):
                rows = []
                for r in range(g):
                    real = band_rows(i, r)[1] // kh * (grids[i][2] // kw)
                    rows.append(everyone[s * g + r][offset : offset + real])
                out[i] = torch.cat(rows)
                offset += band_merged(i)
        return torch.cat([out[i] for i in range(len(grids))])

    @staticmethod
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
