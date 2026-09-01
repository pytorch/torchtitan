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
