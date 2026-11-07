# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import spmd_types as spmd
import torch
from spmd_types.checker import typecheck
from torchtitan.models.common.rope import IdentityRoPE
from torchtitan.models.muse_spark.flavors import muse_spark_navit_encoder_config
from torchtitan.models.muse_spark.vision_encoder import VisionRopeFreq


def test_rope_frequency_meta_initialization_matches_hf():
    with torch.device("meta"):
        rope = VisionRopeFreq.Config(
            dim=96, max_context_length=9, theta=10000.0
        ).build()
    assert rope.cache.device.type == "meta"
    rope.to_empty(device="cpu")
    rope.init_states(buffer_device=torch.device("cpu"))
    exponent = torch.arange(0, 48, 2, dtype=torch.float32) / 48
    inv_freq = 1.0 / (10000.0**exponent)
    angles = torch.outer(torch.arange(9, dtype=torch.float32), inv_freq)
    torch.testing.assert_close(rope.cache, angles, rtol=0, atol=0)
    coordinates = torch.tensor([[1, 2], [3, 4], [8, 8]])
    row_angles, col_angles = angles[coordinates].unbind(1)
    packed_angles = torch.cat((col_angles, row_angles), dim=-1)
    expected = torch.complex(packed_angles.cos(), packed_angles.sin()).unsqueeze(1)
    torch.testing.assert_close(
        rope.prepare_positions(coordinates), expected, rtol=0, atol=0
    )
    assert not rope.state_dict()


@pytest.mark.parametrize("grid_order", ["raster", "legacy_spark"])
def test_position_sampling_and_permutation_spmd_types(grid_order):
    encoder_kwargs = dict(
        latent_dim=32,
        num_heads=2,
        num_layers=1,
        patch_size=2,
        downsample_factor=2,
        sparse_attention_factor=2,
        pos_emb_grid_h=3,
        pos_emb_grid_w=3,
    )
    config = muse_spark_navit_encoder_config(**encoder_kwargs)
    config.pos_emb_grid_order = grid_order
    encoder = config.build()
    encoder.init_states()
    attention = encoder.layers["0"].attn
    rope = attention.rope
    device = torch.device("cpu")
    expected = encoder._get_pos_emb(4, 6, device)
    expected.square().sum().backward()
    expected_grad = encoder.positional_embedding_vlm.grad.clone()
    encoder.zero_grad(set_to_none=True)
    # Pure mesh descriptors exercise type checking without communication.
    mesh = {
        "dp": spmd.MeshAxis.of(2, 4),
        "cp": spmd.MeshAxis.of(2, 2),
        "tp": spmd.MeshAxis.of(2, 1),
    }
    activation = {"dp": spmd.V, "cp": spmd.I, "tp": spmd.I}
    with spmd.set_current_mesh(mesh, local_axes=("dp",)), typecheck(local=False):
        spmd.assert_type(
            encoder.positional_embedding_vlm,
            {"dp": spmd.R, "cp": spmd.I, "tp": spmd.I},
        )
        spmd.assert_type(
            rope.cache, config.block.attn.rope.sharding_config.state_shardings["cache"]
        )
        positions = encoder._get_pos_emb(4, 6, device)
        rope_positions = encoder._make_rope_positions(4, 6, device)
        permutation, _ = encoder._get_sparse_perm_and_slens(4, 6, device)
        spmd.assert_type(positions, activation)
        spmd.assert_type(rope_positions[permutation], activation)
        attention._redistribute_inputs(
            (positions.clone(),),
            {"positions": rope.prepare_positions(rope_positions[permutation])},
        )
        restored = positions[permutation][torch.argsort(permutation)]
        spmd.assert_type(restored, activation)
        # Explicit local gradients avoid implying a distributed scalar reduction.
        restored.backward(2 * restored.detach())
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        encoder.positional_embedding_vlm.grad, expected_grad, rtol=0, atol=0
    )

    config.block.attn.rope = muse_spark_navit_encoder_config(
        **encoder_kwargs, use_rope=False
    ).block.attn.rope
    assert isinstance(config.block.attn.rope, IdentityRoPE.Config)
    disabled_encoder = config.build()
    disabled_encoder.init_states()
    _, rope_positions, *_ = disabled_encoder._embed_image(
        torch.randn(24, config.conv1_linear.in_features),
        4,
        6,
        device,
        torch.float32,
    )
    assert rope_positions.shape == (24, 0)
    query, key = torch.randn(2, 24, 2, 16)
    attention = disabled_encoder.layers["0"].attn
    rope = attention.rope
    rotated_query, rotated_key = rope(query, key, positions=rope_positions)
    assert rotated_query is query
    assert rotated_key is key
    with spmd.set_current_mesh(mesh, local_axes=("dp",)), typecheck(local=False):
        inputs = spmd.mutate_type(
            torch.zeros(24, config.latent_dim), src=spmd.R, dst=activation
        )
        attention._redistribute_inputs(
            (inputs,), {"positions": rope.prepare_positions(rope_positions)}
        )
