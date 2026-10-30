# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
from torchtitan.models.common.attention.cp_attention import (
    UlyssesSlidingWindowVarlenInnerAttention,
    UlyssesVarlenInnerAttention,
)
from torchtitan.models.muse_spark.tokenizer import MuseSparkTokenizer

from torchtitan_recipes.models import muse_spark as recipes
from torchtitan_recipes.tests.models.muse_spark import (
    muse_spark_debugmodel,
    muse_spark_debugmodel_mm,
)


@pytest.mark.parametrize(
    "recipe",
    [
        recipes.muse_spark_816b_a42b_dist_moe_bf16_spmd_256_fake,
        recipes.muse_spark_816b_a42b_dist_moe_bf16_spmd_512_fake,
    ],
)
def test_fake_scale_recipe_tokenizer(recipe):
    # Keep Dist-MoE kernels and CUDA-graph checks outside this tokenizer test.
    def configure_dist_moe(config, **kwargs):
        config.training.disable_accelerator_graphs = False
        return config

    with (
        patch.object(recipes, "_with_dist_moe", side_effect=configure_dist_moe),
        patch(
            "torchtitan.distributed.accelerator_graph.accelerator_graphs_supported",
            return_value=False,
        ),
    ):
        config = recipe()
    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
    assert tokenizer.encode("Hello, world!")


def test_production_recipe_keeps_multimodal_tokenizer():
    config = recipes._muse_spark_816b_a42b()
    assert isinstance(config.tokenizer, MuseSparkTokenizer.Config)
    for field, token in recipes.MUSE_SPARK_SPECIAL_TOKENS.items():
        assert getattr(config.tokenizer, field) == token


def test_cp16_lora_recipe_uses_ulysses_varlen():
    # Building raises if the Ulysses CP map misses a varlen layer type.
    config = recipes.muse_spark_816b_a42b_ultradata_lora_sft_cp16_128k()
    for layer in config.model.layers:
        assert isinstance(
            layer.attention.inner_attention,
            (
                UlyssesVarlenInnerAttention.Config,
                UlyssesSlidingWindowVarlenInnerAttention.Config,
            ),
        )


@pytest.mark.parametrize("recipe", [muse_spark_debugmodel, muse_spark_debugmodel_mm])
def test_debug_recipe_tokenizer(recipe):
    config = recipe()
    config.__post_init__()
    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
    assert tokenizer.encode("Hello, world!")
