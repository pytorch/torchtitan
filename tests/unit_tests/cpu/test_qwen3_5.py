# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import pytest

pytest.importorskip("fla")

from torchtitan.models.qwen3_5 import model_registry, Qwen35Model, qwen3_5_configs
from torchtitan.models.qwen3_5.config_registry import qwen35_0_8b, qwen35_27b
from torchtitan.models.qwen3_8 import model_registry as qwen3_8_model_registry


@pytest.mark.parametrize("enable_ep", [False, True])
@pytest.mark.parametrize("enable_sp", [False, True])
def test_qwen35_shared_expert_gathers_once_for_w13_and_gate(
    enable_ep: bool,
    enable_sp: bool,
) -> None:
    from torchtitan.models.common.feed_forward import SigmoidGatedFeedForward
    from torchtitan.models.common.linear import Linear, RowParallelLinear
    from torchtitan.models.qwen3_5.sharding import set_qwen35_sharding_config

    config = cast(
        Qwen35Model.Config,
        model_registry("debugmodel_moe", moe_comm_backend="standard").model,
    )
    moe = config.layers[0].moe
    assert moe is not None
    shared_experts = moe.shared_experts
    assert isinstance(shared_experts, SigmoidGatedFeedForward.Config)

    assert type(shared_experts.w13) is Linear.Config
    assert type(shared_experts.gate) is Linear.Config
    assert isinstance(shared_experts.w2, RowParallelLinear.Config)

    set_qwen35_sharding_config(config, enable_sp=enable_sp, enable_ep=enable_ep)
    assert shared_experts.sharding_config is not None
    assert shared_experts.sharding_config.in_dst_shardings is not None
    assert shared_experts.w13.sharding_config is not None
    assert shared_experts.w13.sharding_config.in_src_shardings is not None
    assert shared_experts.gate.sharding_config is not None
    assert shared_experts.gate.sharding_config.in_src_shardings is not None
    assert shared_experts.w2.sharding_config is not None

    parent_input = shared_experts.sharding_config.in_dst_shardings["x"]
    assert shared_experts.w13.sharding_config.in_src_shardings["input"] == parent_input
    assert shared_experts.gate.sharding_config.in_src_shardings["input"] == parent_input
    assert (
        shared_experts.w2.sharding_config.out_src_shardings
        == shared_experts.sharding_config.out_src_shardings
    )


@pytest.mark.parametrize("enable_sp", [False, True])
def test_qwen35_attention_output_matches_row_parallel_projection(
    enable_sp: bool,
) -> None:
    from torchtitan.models.qwen3_5.sharding import set_qwen35_sharding_config

    config = cast(Qwen35Model.Config, model_registry("debugmodel").model)
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
    model_spec = model_registry(
        flavor,
        moe_comm_backend=(
            "standard" if flavor == "debugmodel_moe" or "-A" in flavor else None
        ),
    )

    assert model_spec.name == "qwen3_5"
    assert model_spec.flavor == flavor


def test_qwen35_is_the_shared_model_implementation() -> None:
    model_spec = model_registry("0.8B")
    config = cast(Qwen35Model.Config, model_spec.model)
    qwen38_config = qwen3_8_model_registry("27B").model

    assert model_spec.name == "qwen3_5"
    assert model_spec.flavor == "0.8B"
    assert config.dim == 1024
    assert len(config.layers) == 24
    assert isinstance(qwen38_config, Qwen35Model.Config)


def test_qwen35_keeps_small_dense_and_moe_models() -> None:
    dense_config = cast(Qwen35Model.Config, model_registry("0.8B").model)
    moe_config = cast(
        Qwen35Model.Config,
        model_registry("35B-A3B", moe_comm_backend="standard").model,
    )

    assert dense_config.dim == 1024
    assert moe_config.dim == 2048
    assert moe_config.layers[0].moe is not None
    assert moe_config.layers[0].moe.router.num_experts == 256
    assert moe_config.layers[0].moe.router.top_k == 8


def test_qwen35_recipes_keep_versioned_hugging_face_paths() -> None:
    small_config = qwen35_0_8b()
    large_config = qwen35_27b()

    assert small_config.hf_assets_path.endswith("Qwen3.5-0.8B")
    assert small_config.model_spec is not None
    assert small_config.model_spec.name == "qwen3_5"
    assert large_config.hf_assets_path.endswith("Qwen3.5-27B")
    assert large_config.model_spec is not None
    assert large_config.model_spec.name == "qwen3_5"
