# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch
from torchtitan.config.transform import transform_model_config_
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
        # Building raises if sdpa_backends holds anything but SDPBackend names.
        layer.attention.inner_attention.build()


@pytest.mark.parametrize("recipe", [muse_spark_debugmodel, muse_spark_debugmodel_mm])
def test_debug_recipe_tokenizer(recipe):
    config = recipe()
    config.__post_init__()
    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
    assert tokenizer.encode("Hello, world!")


_VISION_PREFIXES = ("vision_encoder.", "vision_adapter.", "vision_projection.")


@pytest.mark.parametrize("variant", ["lora_decoder", "lora_vision_decoder"])
def test_slake_lora_trainable_split(variant):
    # LoRA only warns on unmatched targets; this catches a renamed module.
    model_config = muse_spark_debugmodel_mm().model
    model_config = transform_model_config_(
        model_config, [recipes._slake_lora_transform(variant)]
    )
    with torch.device("meta"):
        model = model_config.build()
    trainable = {
        name for name, param in model.named_parameters() if param.requires_grad
    }
    vision = {name for name in trainable if name.startswith(_VISION_PREFIXES)}
    lora = {name for name in trainable if ".lora_" in name}
    assert trainable == lora
    expected_vision = {
        "vision_encoder.layers.0.attn.wq",
        "vision_encoder.layers.0.attn.wk",
        "vision_encoder.layers.0.attn.wv",
        "vision_encoder.layers.0.attn.proj",
        "vision_encoder.layers.0.mlp.linear_fc1",
        "vision_encoder.layers.0.mlp.linear_fc2",
        "vision_adapter.c_fc",
        "vision_adapter.c_proj",
        "vision_projection",
    }
    adapted_vision = {name.rsplit(".lora_", 1)[0] for name in vision}
    decoder = {name for name in lora if name.startswith("layers.")}
    assert decoder
    if variant == "lora_decoder":
        assert not vision
    else:
        assert expected_vision <= adapted_vision
        assert not any("conv1" in name for name in adapted_vision)
