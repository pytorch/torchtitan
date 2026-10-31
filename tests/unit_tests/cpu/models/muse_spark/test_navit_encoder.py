# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
from spmd_types.checker import typecheck
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.muse_spark.flavors import muse_spark_navit_encoder_config
from torchtitan.models.muse_spark.vision_encoder import VisionRopeFreq


def test_rope_frequency_meta_initialization_matches_hf():
    with torch.device("meta"):
        freq = VisionRopeFreq.Config(head_dim=96, rope_theta=10000.0).build()
    assert freq.inv_freq.device.type == "meta"
    freq.to_empty(device="cpu")
    freq.init_states(buffer_device=torch.device("cpu"))
    exponent = torch.arange(0, 48, 2, dtype=torch.float32) / 48
    expected = 1.0 / (10000.0**exponent)
    torch.testing.assert_close(freq.inv_freq, expected, rtol=0, atol=0)
    assert not freq.state_dict()


@pytest.mark.parametrize("grid_order", ["raster", "legacy_spark"])
def test_position_sampling_and_rope_spmd_types(grid_order):
    config = muse_spark_navit_encoder_config(
        latent_dim=32,
        num_heads=2,
        num_layers=0,
        mlp_hidden_dim=64,
        patch_size=2,
        downsample_factor=2,
        sparse_attention_factor=2,
        pos_emb_grid_h=3,
        pos_emb_grid_w=3,
    )
    config.pos_emb_grid_order = grid_order
    encoder = config.build()
    encoder.init_states()
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
            encoder.rope_freq.inv_freq, {"dp": spmd.R, "cp": spmd.I, "tp": spmd.I}
        )
        positions = encoder._get_pos_emb(4, 6, device)
        freqs = encoder._make_2d_rope(4, 6, device)
        permutation, _ = encoder._get_sparse_perm_and_slens(4, 6, device)
        spmd.assert_type(positions, activation)
        spmd.assert_type(freqs[permutation], activation)
        restored = positions[permutation][torch.argsort(permutation)]
        spmd.assert_type(restored, activation)
        # Explicit local gradients avoid implying a distributed scalar reduction.
        restored.backward(2 * restored.detach())
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        encoder.positional_embedding_vlm.grad, expected_grad, rtol=0, atol=0
    )

    config.num_layers = 1
    for use_rope in (False, True):
        config.use_rope = use_rope
        encoder = config.build()
        encoder.init_states()
        x_TD, freqs_TD, *_ = encoder._embed_image(
            torch.randn(24, 24), 4, 6, device, torch.float32
        )
        rope_cache = freqs_TD.unsqueeze(1)
        attention = encoder.layers["0"].attn
        query_THDh, key_THDh, _ = attention._qkv(x_TD)
        if use_rope:
            query_THDh, _ = ComplexRoPE.apply_rotary_emb(
                query_THDh, key_THDh, rope_cache
            )
        expected = attention.proj(query_THDh.reshape(24, -1))
        with patch.object(
            attention.flex_attention,
            "forward",
            side_effect=lambda q, k, v, **kwargs: q,
        ):
            actual = attention(
                x_TD, rope_cache=rope_cache, attention_mask=None, use_rope=use_rope
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
