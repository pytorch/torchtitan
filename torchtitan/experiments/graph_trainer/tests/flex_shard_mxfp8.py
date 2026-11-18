# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, fields
from functools import partial
from typing import TYPE_CHECKING

import torch_remat as remat
from flex_shard import Placement
from flex_shard.blockwise_mxfp8 import BlockwiseMxfp8Weight
from flex_shard.blockwise_mxfp8.mxfp8_blockwise_linear import Mxfp8BlockwiseGemms
from flex_shard.blockwise_quant.linear import FlexShardPrequantizedLinear
from flex_shard.custom_placements.mixed_bucket import MixedBucketPlacement
from flex_shard.custom_placements.mxfp8_bucketed_block_shard import (
    make_mxfp8_bucketed_block_placement_fn,
)
from torch import Tensor

from torchtitan.models.common.linear import Linear
from torchtitan.quantization.mxfp8 import MXFP8Linear


if TYPE_CHECKING:
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer


class FlexShardMXFP8Linear(FlexShardPrequantizedLinear, Linear):
    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    weight_type = BlockwiseMxfp8Weight

    def _prequantized_gemms(self) -> Mxfp8BlockwiseGemms:
        return Mxfp8BlockwiseGemms(
            out_dtype=self.weight.dtype,
            wgrad_dtype=self.weight.grad_dtype or self.weight.dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        return remat.region(
            super().forward,
            self.remat_region_name("linear"),
            recompute=self.remat_should_recompute("linear"),
        )(input)


def _mxfp8_block_placements(
    named_params, mesh, *, linear_weight_fqns
) -> dict[str, tuple[Placement]]:
    mixed = MixedBucketPlacement({})
    quantized = make_mxfp8_bucketed_block_placement_fn(
        weight_factory=BlockwiseMxfp8Weight,
    )(named_params, mesh)
    placements = {}
    for fqn, _ in named_params:
        if fqn in linear_weight_fqns:
            placement = mixed.blockwise_quant_bucketed_block_shard(quantized[fqn][0])
        elif ".routed_experts." in fqn:
            placement = mixed.block_shard(blocks_per_rank=(1,) * mesh.size())
        else:
            placement = mixed.shard0
        placements[fqn] = (placement,)
    return placements


def configure_flex_shard_mxfp8_blocks(
    config: "GraphTrainer.Config",
) -> "GraphTrainer.Config":
    linear_weight_fqns = set()
    for fqn, linear_config, parent, attr in config.model.traverse(MXFP8Linear.Config):
        replacement = FlexShardMXFP8Linear.Config(
            **{
                field.name: getattr(linear_config, field.name)
                for field in fields(FlexShardMXFP8Linear.Config)
            }
        )
        setattr(parent, attr, replacement)
        linear_weight_fqns.add(f"{fqn}.weight")
    config.use_flex_shard_instead_simple_fsdp = True
    config.flex_shard_placement_fn = partial(
        _mxfp8_block_placements, linear_weight_fqns=frozenset(linear_weight_fqns)
    )
    return config
