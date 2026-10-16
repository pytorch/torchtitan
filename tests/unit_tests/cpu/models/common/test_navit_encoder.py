# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NaViT contracts and opt-in immutable before/after numerical references."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.muse_glimmer.flavors import (
    muse_glimmer_vision_adapter_config,
    muse_glimmer_vision_encoder_config,
)
from torchtitan.models.muse_spark.flavors import (
    muse_spark_navit_adapter_config,
    muse_spark_navit_encoder_config,
)
from torchtitan.models.muse_spark.vision_encoder import (
    reorder_patch_vector_channel_temporal,
)


def dense_flex_attention(q, k, v, **kwargs):
    """Evaluate the actual FlexAttention mask at every token pair on CPU."""
    block_mask = kwargs["block_mask"]
    indices = torch.arange(q.shape[-2], device=q.device)
    mask = block_mask.mask_mod(
        torch.tensor(0, device=q.device),
        torch.tensor(0, device=q.device),
        indices[:, None],
        indices[None, :],
    )
    return (
        F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            scale=kwargs["scale"], enable_gqa=kwargs["enable_gqa"],
        ),
        SimpleNamespace(),
    )


CASES = [
    (factory, head_dim, factor, True, "bilinear")
    for factory in ("glimmer", "spark")
    for head_dim, factor in ((16, 1), (16, 2), (96, 4))
] + [
    ("spark", 16, 4, False, "bilinear"),
    ("spark", 16, 2, True, "nearest"),
    ("spark", 16, 2, True, "bicubic"),
]


def encoder_config(factory, head_dim=16, factor=2, use_rope=True, mode="bilinear"):
    settings = dict(
        latent_dim=head_dim * 2, num_heads=2, num_layers=3,
        patch_size=2, patch_temporal=2, downsample_factor=2,
        sparse_attention_factor=factor, pos_emb_grid_h=3, pos_emb_grid_w=3,
    )
    if factory == "glimmer":
        return muse_glimmer_vision_encoder_config(**settings, mlp_ratio=2)
    return muse_spark_navit_encoder_config(
        **settings, mlp_hidden_dim=head_dim * 4,
        use_rope=use_rope, grid_sample_mode=mode,
    )


def build_tower(factory, config, device="cpu"):
    adapter_builder = (
        muse_glimmer_vision_adapter_config if factory == "glimmer"
        else muse_spark_navit_adapter_config
    )
    with torch.device(device):
        tower = torch.nn.Sequential(
            config.build(), adapter_builder(input_dim=config.output_dim, output_dim=16).build()
        )
    for module in tower:
        module.init_states()
    return tower


def numerical_reference(case, device="cpu"):
    factory, head_dim, factor, use_rope, mode = case
    torch.manual_seed(42)
    config = encoder_config(*case)
    tower = build_tower(factory, config, device)
    state = {key: value.detach().cpu().clone() for key, value in tower.state_dict().items()}
    # The 4x6 grid contains partially filled 3x3 sparse windows.
    grids = [[1, 4, 6], [1, 2, 4]] if factory == "glimmer" else [[2, 4, 6], [1, 2, 4]]
    grid = torch.tensor(grids, device=device)
    patches = torch.randn(sum(t*h*w for t,h,w in grids), 24, device=device, requires_grad=True)
    encoded = tower[0](patches, grid_thw=grid)
    adapted = tower[1](encoded)
    upstream = torch.randn_like(adapted)
    (adapted * upstream).sum().backward()
    gradients = {}
    for key, parameter in tower.named_parameters():
        assert parameter.grad is not None, key
        assert torch.isfinite(parameter.grad).all(), key
        gradients[key] = parameter.grad.detach().cpu().clone()
    assert patches.grad is not None
    return {
        "manifest": dict(factory=factory, head_dim=head_dim, factor=factor,
                         use_rope=use_rope, mode=mode, grids=grids,
                         dtype=str(patches.dtype), device=device, torch=str(torch.__version__)),
        "state": state, "patches": patches.detach().cpu(),
        "encoded": encoded.detach().cpu(), "adapted": adapted.detach().cpu(),
        "input_grad": patches.grad.detach().cpu(), "gradients": gradients,
    }


def check_reference(case, actual, device="cpu"):
    mode = os.environ.get("TORCHTITAN_NAVIT_REFERENCE_MODE")
    if mode is None:
        return
    assert mode in {"record", "compare"}
    directory = Path(os.environ["TORCHTITAN_NAVIT_REFERENCE_DIR"])
    path = directory / (device + "_" + "_".join(map(str, case)) + ".pt")
    if mode == "record":
        directory.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents replacing a pre-change reference.
        with path.open("xb") as file:
            torch.save(actual, file)
    else:
        expected = torch.load(path, weights_only=True)
        assert actual["manifest"] == expected["manifest"]
        for key in ("state", "gradients"):
            assert actual[key].keys() == expected[key].keys()
            for name, tensor in actual[key].items():
                torch.testing.assert_close(tensor, expected[key][name], rtol=0, atol=0, msg=name)
        for key in ("patches", "encoded", "adapted", "input_grad"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)
        # Strictly loading the immutable pre-change native weights checks paths/shapes.
        config = encoder_config(*case)
        build_tower(case[0], config).load_state_dict(expected["state"], strict=True)


@pytest.mark.parametrize("case", CASES)
def test_encoder_reference(case):
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with patch.object(FlexInnerAttention, "compiled_flex_attn", dense_flex_attention):
            actual = numerical_reference(case)
        check_reference(case, actual)
    finally:
        torch.use_deterministic_algorithms(was_deterministic)


def test_patch_vector_order():
    patches = torch.arange(2 * 24).reshape(2, 24)
    expected = torch.stack([
        patches[n].reshape(3, 2, 2, 2)[c, t, h, w]
        for n in range(2) for t in range(2) for c in range(3)
        for h in range(2) for w in range(2)
    ]).reshape(2, 24)
    torch.testing.assert_close(
        reorder_patch_vector_channel_temporal(patches, patch_size=2, patch_temporal=2), expected
    )


@pytest.mark.parametrize("factory", ["glimmer", "spark"])
@pytest.mark.parametrize("head_dim", [16, 96])
def test_frequency_evaluation_and_initialization(factory, head_dim):
    config = encoder_config(factory, head_dim)
    torch.manual_seed(42)
    encoder = config.build()
    encoder.init_states()
    exponent = torch.arange(0, head_dim // 2, 2, dtype=torch.float32) / (head_dim // 2)
    expected = 1.0 / (10000.0 ** exponent) if factory == "glimmer" else 10000.0 ** (-exponent)
    torch.testing.assert_close(encoder.rope_freq.inv_freq, expected, rtol=0, atol=0)
    assert "rope_freq.inv_freq" not in encoder.state_dict()
    if factory == "glimmer":
        torch.testing.assert_close(encoder.layers["0"].attn.wq.bias, torch.zeros(head_dim*2))
        assert config.conv1.param_init is not None
    else:
        assert config.conv1.param_init is None
        assert config.pos_emb_grid_order == "legacy_spark"


def test_production_builder_geometry():
    for builder in (muse_glimmer_vision_encoder_config, muse_spark_navit_encoder_config):
        config = builder()
        assert (config.latent_dim, config.output_dim, config.num_layers, config.num_heads) == (1536, 6144, 50, 16)
        assert (config.patch_size, config.patch_temporal, config.downsample_factor, config.sparse_attention_factor) == (14, 2, 2, 4)


@pytest.mark.parametrize("factory", ["glimmer", "spark"])
def test_sparse_permutation_and_pixel_shuffle(factory):
    encoder = encoder_config(factory).build()
    permutation, lengths = encoder._get_sparse_perm_and_slens(4, 6, torch.device("cpu"))
    expected = torch.tensor([r*6+c for br in range(0,4,3) for bc in range(0,6,3)
                             for r in range(br,min(br+3,4)) for c in range(bc,min(bc+3,6))])
    torch.testing.assert_close(permutation, expected)
    torch.testing.assert_close(lengths, torch.tensor([9,9,3,3], dtype=torch.int32))
    values = torch.arange(24 * 32).reshape(24, 32)
    torch.testing.assert_close(values[permutation][torch.argsort(permutation)], values)
    expected_output = torch.stack([
        values.reshape(4,6,32)[r:r+2,c:c+2].reshape(4,32).T.reshape(-1)
        for r in range(0,4,2) for c in range(0,6,2)
    ])
    torch.testing.assert_close(encoder._pixel_shuffle_downsample(values, 4, 6), expected_output)


@pytest.mark.parametrize("factory", ["glimmer", "spark"])
def test_packed_frames_are_isolated(factory):
    torch.manual_seed(42)
    encoder = encoder_config(factory).build()
    encoder.init_states()
    first, second = torch.randn(24,24), torch.randn(8,24)
    with patch.object(FlexInnerAttention, "compiled_flex_attn", dense_flex_attention):
        packed = encoder(torch.cat([first,second]), grid_thw=torch.tensor([[1,4,6],[1,2,4]]))
        single = encoder(first, grid_thw=torch.tensor([[1,4,6]]))
    # Different GEMM batch shapes can change rounding; this checks masking semantics.
    torch.testing.assert_close(packed[:6], single, rtol=1e-5, atol=1e-6)
