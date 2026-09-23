# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch

import torchtitan.models.common.vision_encoder as vision_encoder_module

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import LayerNorm
from torchtitan.models.common.vision_encoder import (
    validate_vision_sequence_parallel_input,
    VisionAttention,
    VisionMLP,
    VisionTransformerBlock,
)
from torchtitan.models.common.vision_encoder_sharding import (
    set_vision_transformer_block_sharding_config,
    vision_sequence_parallel_input_config,
    vision_sequence_parallel_output_config,
)


TP = MeshAxisName.TP


def _tp_type(layout: spmd.SpmdType) -> spmd.PerMeshAxisSpmdType:
    return layout.local_type[TP]


def _block_config() -> VisionTransformerBlock.Config:
    dim = 8

    def linear() -> Linear.Config:
        return Linear.Config(in_features=dim, out_features=dim)

    return VisionTransformerBlock.Config(
        norm1=LayerNorm.Config(normalized_shape=dim),
        norm2=LayerNorm.Config(normalized_shape=dim),
        attn=VisionAttention.Config(
            dim=dim,
            num_heads=2,
            wq=linear(),
            wk=linear(),
            wv=linear(),
            proj=linear(),
        ),
        mlp=VisionMLP.Config(fc1=linear(), fc2=linear()),
    )


@pytest.mark.parametrize(
    ("enable_sp", "activation_tp"),
    [(False, spmd.I), (True, spmd.S(0))],
)
def test_vision_block_follows_sequence_parallel_layout(
    enable_sp: bool,
    activation_tp: spmd.PerMeshAxisSpmdType,
) -> None:
    block = _block_config()
    set_vision_transformer_block_sharding_config(
        block,
        enable_sp=enable_sp,
        rope_cache_dp=spmd.V,
    )

    assert block.norm1.sharding_config is not None
    assert block.norm1.sharding_config.in_src_shardings is not None
    assert _tp_type(block.norm1.sharding_config.in_src_shardings["input"]) == (
        activation_tp
    )

    assert block.attn.sharding_config is not None
    assert block.attn.sharding_config.in_src_shardings is not None
    assert block.attn.sharding_config.in_dst_shardings is not None
    assert _tp_type(block.attn.sharding_config.in_src_shardings["x"]) == activation_tp
    assert _tp_type(block.attn.sharding_config.in_dst_shardings["x"]) == spmd.R

    for projection in (block.attn.proj, block.mlp.fc2):
        assert projection.sharding_config is not None
        assert projection.sharding_config.out_src_shardings is not None
        assert isinstance(projection.sharding_config.out_src_shardings, spmd.SpmdType)
        assert projection.sharding_config.out_dst_shardings is None
        assert _tp_type(projection.sharding_config.out_src_shardings) == activation_tp

    assert block.mlp.fc1.sharding_config is not None
    assert block.mlp.fc1.sharding_config.in_src_shardings is not None
    assert _tp_type(block.mlp.fc1.sharding_config.in_src_shardings["input"]) == (
        activation_tp
    )


@pytest.mark.parametrize(
    ("enable_sp", "activation_tp"),
    [(False, spmd.I), (True, spmd.S(0))],
)
def test_vision_sequence_parallel_boundaries(
    enable_sp: bool,
    activation_tp: spmd.PerMeshAxisSpmdType,
) -> None:
    input_config = vision_sequence_parallel_input_config(enable_sp=enable_sp)
    output_config = vision_sequence_parallel_output_config(enable_sp=enable_sp)

    assert input_config.in_src_shardings is not None
    assert input_config.in_dst_shardings is not None
    assert _tp_type(input_config.in_src_shardings["input"]) == spmd.I
    assert _tp_type(input_config.in_dst_shardings["input"]) == activation_tp

    assert output_config.in_src_shardings is not None
    assert output_config.in_dst_shardings is not None
    assert _tp_type(output_config.in_src_shardings["input"]) == activation_tp
    assert _tp_type(output_config.in_dst_shardings["input"]) == spmd.I


def test_vision_sequence_parallel_requires_even_token_shards() -> None:
    with (
        patch.object(vision_encoder_module, "spmd_dense_sp_enabled", return_value=True),
        patch.object(vision_encoder_module, "spmd_mesh_size", return_value=2),
    ):
        validate_vision_sequence_parallel_input(torch.randn(4, 8))
        with pytest.raises(ValueError, match="packed patch-token count"):
            validate_vision_sequence_parallel_input(torch.randn(3, 8))
