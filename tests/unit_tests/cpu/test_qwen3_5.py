# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace
from typing import cast

import pytest
import torch
from torch import nn

pytest.importorskip("fla")

from torchtitan.models.qwen3_5 import model_registry, Qwen35Model, qwen3_5_configs
from torchtitan.models.qwen3_5.config_registry import qwen35_0_8b, qwen35_27b
from torchtitan.models.qwen3_8 import model_registry as qwen3_8_model_registry


@pytest.mark.parametrize("enable_ep", [False, True])
@pytest.mark.parametrize("enable_sp", [False, True])
def test_qwen35_shared_expert_uses_explicit_tp_boundaries(
    enable_ep: bool,
    enable_sp: bool,
) -> None:
    import spmd_types as spmd
    from torchtitan.distributed.parallel_dims import MeshAxisName
    from torchtitan.models.common.linear import Linear
    from torchtitan.models.qwen3_5.moe import SigmoidGatedFeedForward
    from torchtitan.models.qwen3_5.sharding import set_qwen35_sharding_config

    config = cast(
        Qwen35Model.Config,
        model_registry("debugmodel_moe", moe_comm_backend="standard"),
    )
    moe = config.layers[0].moe
    assert moe is not None
    shared_experts = moe.shared_experts
    assert isinstance(shared_experts, SigmoidGatedFeedForward.Config)

    assert type(shared_experts.w13) is Linear.Config
    assert shared_experts.w13.num_linears == 2
    assert type(shared_experts.gate) is Linear.Config
    assert type(shared_experts.w2) is Linear.Config

    set_qwen35_sharding_config(config, enable_sp=enable_sp, enable_ep=enable_ep)
    assert shared_experts.sharding_config is not None
    assert shared_experts.sharding_config.in_src_shardings is not None
    assert shared_experts.sharding_config.in_dst_shardings is not None
    assert shared_experts.w13.sharding_config is not None
    assert shared_experts.w13.sharding_config.in_src_shardings is not None
    assert shared_experts.gate.sharding_config is not None
    assert shared_experts.gate.sharding_config.in_src_shardings is not None
    assert shared_experts.w2.sharding_config is not None

    projection_input = shared_experts.w13.sharding_config.in_src_shardings["input"]
    assert shared_experts.gate.sharding_config.in_src_shardings["input"] == (
        projection_input
    )
    assert shared_experts.w2.sharding_config.out_src_shardings is not None
    assert (
        shared_experts.w2.sharding_config.out_src_shardings.local_type[MeshAxisName.TP]
        == spmd.P
    )
    assert (
        shared_experts.w2.sharding_config.out_dst_shardings
        == shared_experts.sharding_config.out_src_shardings
    )


def test_qwen35_vision_projections_are_not_dense_tp_boundaries() -> None:
    import spmd_types as spmd
    from torchtitan.distributed.parallel_dims import MeshAxisName
    from torchtitan.models.common.linear import Linear
    from torchtitan.models.common.vision_encoder import InvariantRowParallelLinear
    from torchtitan.models.qwen3_5.sharding import set_qwen35_sharding_config

    config = cast(Qwen35Model.Config, model_registry("debugmodel"))
    vision_encoder = config.vision_encoder
    assert vision_encoder is not None

    assert type(vision_encoder.block.mlp.fc1) is Linear.Config
    assert type(vision_encoder.block.mlp.fc2) is InvariantRowParallelLinear.Config
    assert type(vision_encoder.block.attn.proj) is InvariantRowParallelLinear.Config
    assert type(vision_encoder.merger.fc1) is Linear.Config
    assert type(vision_encoder.merger.fc2) is InvariantRowParallelLinear.Config

    set_qwen35_sharding_config(config, enable_sp=True, enable_ep=False)
    for projection in (
        vision_encoder.block.mlp.fc2,
        vision_encoder.block.attn.proj,
        vision_encoder.merger.fc2,
    ):
        sharding = projection.sharding_config
        assert sharding is not None
        assert sharding.out_dst_shardings is None
        assert sharding.out_src_shardings is not None
        assert sharding.out_src_shardings.local_type[MeshAxisName.TP] == spmd.I


@pytest.mark.parametrize("enable_sp", [False, True])
def test_qwen35_attention_output_matches_row_parallel_projection(
    enable_sp: bool,
) -> None:
    from torchtitan.models.qwen3_5.sharding import set_qwen35_sharding_config

    config = cast(Qwen35Model.Config, model_registry("debugmodel"))
    set_qwen35_sharding_config(config, enable_sp=enable_sp, enable_ep=False)

    for layer in config.layers:
        if layer.attention is not None:
            attention = layer.attention
            output_projection = attention.wo
        else:
            assert layer.delta_net is not None
            attention = layer.delta_net
            output_projection = attention.out_proj
        assert attention.sharding_config is not None
        assert output_projection.sharding_config is not None
        assert (
            attention.sharding_config.out_src_shardings
            == output_projection.sharding_config.out_src_shardings
        )


class _RecordingVisionEncoder(nn.Module):
    def __init__(self, patch_dim: int, output_dim: int) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(patch_dim, output_dim, bias=False)
        self.spatial_merge_size = 1
        self.spatial_merge_unit = 1
        self.num_calls = 0

    def forward(
        self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        self.num_calls += 1
        return self.patch_embed(pixel_values)


def _small_qwen35_model() -> Qwen35Model:
    config = cast(Qwen35Model.Config, model_registry("debugmodel", seq_len=8))
    config = replace(
        config,
        vocab_size=8,
        tok_embeddings=replace(config.tok_embeddings, num_embeddings=8),
        lm_head=replace(config.lm_head, out_features=8),
        layers=[],
        vision_encoder=None,
    )
    model = config.build()
    model.init_states()
    model.vision_encoder = _RecordingVisionEncoder(  # pyrefly: ignore [bad-assignment]
        4, config.dim
    )
    return model


def test_qwen35_registry_keeps_released_flavors() -> None:
    assert set(qwen3_5_configs) == {
        "debugmodel",
        "debugmodel_moe",
        "0.8B",
        "2B",
        "4B",
        "9B",
        "27B",
        "35B-A3B",
        "122B-A10B",
        "397B-A17B",
    }


@pytest.mark.parametrize("flavor", sorted(qwen3_5_configs))
def test_qwen35_registry_builds_every_flavor(flavor: str) -> None:
    config = model_registry(
        flavor,
        moe_comm_backend=(
            "standard" if flavor == "debugmodel_moe" or "-A" in flavor else None
        ),
    )

    assert isinstance(config, Qwen35Model.Config)


def test_qwen35_is_the_shared_model_implementation() -> None:
    config = cast(Qwen35Model.Config, model_registry("0.8B"))
    qwen38_config = qwen3_8_model_registry("27B")

    assert config.dim == 1024
    assert len(config.layers) == 24
    assert isinstance(qwen38_config, Qwen35Model.Config)


def test_qwen35_keeps_small_dense_and_moe_models() -> None:
    dense_config = cast(Qwen35Model.Config, model_registry("0.8B"))
    moe_config = cast(
        Qwen35Model.Config,
        model_registry("35B-A3B", moe_comm_backend="standard"),
    )

    assert dense_config.dim == 1024
    assert moe_config.dim == 2048
    assert moe_config.layers[0].moe is not None
    assert moe_config.layers[0].moe.router.num_experts == 256
    assert moe_config.layers[0].moe.router.top_k == 8


@pytest.mark.parametrize(
    ("has_image", "has_video"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_qwen35_always_calls_vision_encoder_twice(
    has_image: bool, has_video: bool
) -> None:
    model = _small_qwen35_model()
    image_pixels = torch.ones(1, 4) if has_image else None
    image_grid = torch.ones(1, 3, dtype=torch.int64) if has_image else None
    video_pixels = torch.full((1, 4), 2.0) if has_video else None
    video_grid = torch.ones(1, 3, dtype=torch.int64) if has_video else None
    tokens = torch.tensor([1 if has_image else 3, 2 if has_video else 4])
    expected_TD = model.tok_embeddings(tokens).detach()

    inputs_TD = model._prepare_multimodal_embeds(
        tokens,
        pixel_values=image_pixels,
        pixel_values_videos=video_pixels,
        grid_thw=image_grid,
        grid_thw_videos=video_grid,
        special_tokens={"image_id": 1, "video_id": 2},
    )
    inputs_TD.sum().backward()

    encoder = cast(_RecordingVisionEncoder, model.vision_encoder)
    assert encoder.num_calls == 2
    assert encoder.patch_embed.weight.grad is not None
    if not has_image and not has_video:
        torch.testing.assert_close(inputs_TD, expected_TD, rtol=0, atol=0)
        torch.testing.assert_close(
            encoder.patch_embed.weight.grad,
            torch.zeros_like(encoder.patch_embed.weight),
        )


def test_qwen35_recipes_keep_versioned_hugging_face_paths() -> None:
    small_config = qwen35_0_8b()
    large_config = qwen35_27b()

    assert small_config.hf_assets_path.endswith("Qwen3.5-0.8B")
    assert isinstance(small_config.model, Qwen35Model.Config)
    assert large_config.hf_assets_path.endswith("Qwen3.5-27B")
    assert isinstance(large_config.model, Qwen35Model.Config)
