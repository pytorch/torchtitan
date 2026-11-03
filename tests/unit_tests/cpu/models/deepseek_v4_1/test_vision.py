# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.deepseek_v4_1 import build_model_config
from torchtitan.models.deepseek_v4_1.state_dict_adapter import (
    DeepSeekV41StateDictAdapter,
)


@pytest.fixture
def model():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = build_model_config(
        "debugmodel",
        parallelism=ParallelismConfig(data_parallel_shard_degree=1),
        seq_len=32,
        attention_backend="reference",
        indexer_loss_coeff=0,
        vision_enabled=True,
        vision_attention_backend="reference",
    )
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cpu")
    model.init_states()
    model.eval()
    yield model
    torch.set_num_threads(previous)


def _vision_reference(vision, patches, height, width):
    # Independent SDPA oracle following inference/vision.py's fused projections.
    x = F.linear(patches, vision.patch_embed.weight, vision.patch_embed.bias)
    half_dim = vision.config.dim // vision.config.num_heads // 2
    frequencies = 1 / 10000 ** (torch.arange(0, half_dim, 2).float() / half_dim)
    row, col = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    angles = (torch.stack((row, col), -1).reshape(-1, 2, 1) * frequencies).flatten(1)[
        :, None
    ]

    def norm(x, weight):
        return x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * weight

    def rope(x):
        left, right = x.chunk(2, -1)
        return torch.cat(
            (
                left * angles.cos() - right * angles.sin(),
                right * angles.cos() + left * angles.sin(),
            ),
            -1,
        )

    for block in vision.layers:
        normalized = norm(x, block.norm1.weight)
        attn = block.attn
        qkv = F.linear(
            normalized,
            torch.cat((attn.wq.weight, attn.wk.weight, attn.wv.weight)),
            torch.cat((attn.wq.bias, attn.wk.bias, attn.wv.bias)),
        )
        query, key, value = [
            projection.view(height * width, vision.config.num_heads, 2 * half_dim)
            for projection in qkv.chunk(3, -1)
        ]
        output = (
            F.scaled_dot_product_attention(
                rope(query).transpose(0, 1),
                rope(key).transpose(0, 1),
                value.transpose(0, 1),
            )
            .transpose(0, 1)
            .flatten(1)
        )
        x = x + F.linear(output, attn.proj.weight, attn.proj.bias)
        gate, up = F.linear(
            norm(x, block.norm2.weight), block.mlp.w13.weight.flatten(0, 1)
        ).chunk(2, -1)
        x = x + F.linear(F.silu(gate) * up, block.mlp.w2.weight)
    return norm(x, vision.norm.weight)


def test_vision_matches_sdpa_oracle_and_isolates_images(model):
    patches = torch.randn(30, 12, requires_grad=True)
    actual = model.vision(patches, grid_hw=((5, 6),))
    expected = _vision_reference(model.vision, patches, 5, 6)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    packed = model.vision(torch.cat((patches, patches + 1)), grid_hw=((5, 6), (5, 6)))
    separate = torch.cat((actual, model.vision(patches + 1, grid_hw=((5, 6),))))
    torch.testing.assert_close(packed, separate, rtol=2e-5, atol=2e-6)
    actual.square().mean().backward()
    assert (
        patches.grad.norm() > 0
        and model.vision.layers[0].attn.wq.weight.grad.norm() > 0
    )


def test_aligner_uses_channel_major_unshuffle_with_padding(model):
    x = torch.arange(5 * 4 * 64, dtype=torch.float32).view(20, 64) / 100
    actual = model.aligner(x, grid_hw=((5, 4),))
    padded = F.pad(x.view(5, 4, 64).permute(2, 0, 1), (0, 2, 0, 1))
    rows = torch.stack(
        [
            padded[:, row : row + 3, col : col + 3].flatten()
            for row in (0, 3)
            for col in (0, 3)
        ]
    )
    expected = F.linear(
        F.gelu(F.linear(rows, model.aligner.w1.weight, model.aligner.w1.bias)),
        model.aligner.w2.weight,
        model.aligner.w2.bias,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_image_spans_have_visual_gradients_and_no_engram_writes(model):
    tokens = torch.arange(18)
    types = torch.tensor([-1, -1, 0, 1, 1, 2, 1, 1, 2, 3] + [-1] * 8)
    pixels = torch.randn(36, 12, requires_grad=True)
    outputs = model(tokens, token_types=types, pixel_values=pixels, grid_hw=((6, 6),))
    outputs.square().mean().backward()
    assert pixels.grad.norm() > 0 and model.image_embeddings.weight.grad.norm() > 0
    engram = model.layers["1"].engram
    hashes = model.engram_hash(tokens, torch.arange(18), token_mask_T=types < 0)[:, 0]
    branches = torch.randn(18, 4, 128)
    fused = engram(branches, hashes, token_mask_T=types < 0)
    torch.testing.assert_close(fused[types >= 0], branches[types >= 0], rtol=0, atol=0)


def test_multimodal_checkpoint_roundtrip(model):
    adapter = DeepSeekV41StateDictAdapter(model.config)
    hf = adapter.to_hf(model.state_dict())
    assert "vision.blocks.0.attn.wqkv.weight" in hf
    assert hf["vision.blocks.0.mlp.w1.weight"].shape == (256, 64)
    restored = adapter.from_hf(hf)
    assert restored.keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored[key], rtol=0, atol=0)


@pytest.mark.parametrize("component", ["vision", "aligner", "image_embeddings"])
def test_config_rejects_incomplete_vision_components(model, component):
    with pytest.raises(ValueError, match="must be configured together"):
        replace(model.config, **{component: None})
