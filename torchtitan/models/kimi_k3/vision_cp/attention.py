# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Vision attention over the bands of images split across a sub-CP group."""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch_remat as remat
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.common.vision_encoder import VisionAttention


@dataclass
class VisionCPLayout:
    """Attention over whole images followed by this rank's bands of the images a sub-CP group splits."""

    block_mask: BlockMask
    num_whole_tokens: int
    group: dist.ProcessGroup | None
    """The sub-CP group the bands gather keys and values over; ``None`` for a group of one."""


def all_gather_parts(x: torch.Tensor, group: dist.ProcessGroup) -> list[torch.Tensor]:
    """Autograd all-gather over ``group`` outside SPMD type checking; each part takes the input's type."""
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
    return torch.cat([whole, *all_gather_parts(bands, layout.group)], dim=0)


class VisionCPAttention(VisionAttention):
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
