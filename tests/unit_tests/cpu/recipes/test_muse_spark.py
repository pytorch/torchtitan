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
from torchtitan.models.muse_spark import MUSE_SPARK_SPECIAL_TOKENS
from torchtitan.models.muse_spark.tokenizer import MuseSparkTokenizer

from torchtitan_recipes.models import muse_spark as recipes
from torchtitan_recipes.tests.models.muse_spark import (
    muse_spark_debugmodel,
    muse_spark_debugmodel_mm,
)


@pytest.mark.parametrize(
    "recipe",
    [
        recipes.muse_spark_816b_a42b_dist_moe_bf16_spmd_64_fake,
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
    config = recipes.muse_spark_816b_a42b()
    assert isinstance(config.tokenizer, MuseSparkTokenizer.Config)
    for field, token in MUSE_SPARK_SPECIAL_TOKENS.items():
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


def test_dist_moe_64_mxfp8_recipe_quantizes_experts_and_projections():
    pytest.importorskip("dist_moe")
    from torchtitan.config.transform.quantization import MXFP8Linear
    from torchtitan.models.common.dist_moe.mxfp8 import MXFP8DistMoeRoutedExperts

    # The MXFP8 transform checks the local GPU while the config is built.
    with patch(
        "torchtitan.config.transform.quantization.has_cuda_capability",
        return_value=True,
    ):
        config = recipes.muse_spark_816b_a42b_dist_moe_mxfp8_spmd_64_fake()
    for layer in config.model.layers:
        feed_forward = layer.feed_forward
        assert isinstance(feed_forward.routed_experts, MXFP8DistMoeRoutedExperts.Config)
        assert feed_forward.routed_experts.block_scaled_config.fast_math
        assert isinstance(feed_forward.pooler, MXFP8Linear.Config)
        assert isinstance(feed_forward.unpooler, MXFP8Linear.Config)
        assert isinstance(layer.attention.wo, MXFP8Linear.Config)


@pytest.mark.parametrize("recipe", [muse_spark_debugmodel, muse_spark_debugmodel_mm])
def test_debug_recipe_tokenizer(recipe):
    config = recipe()
    config.__post_init__()
    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
    assert tokenizer.encode("Hello, world!")


@pytest.mark.parametrize("fuse_softcap_loss", [False, True])
def test_mxfp8_head_override_preserves_config_and_checkpoint(fuse_softcap_loss):
    import torch
    from torchtitan.models.muse_spark.model import MuseSparkLMHead
    from torchtitan_recipes.overrides.muse_spark_mxfp8_head import (
        _MuseSparkSoftcapCrossEntropy,
        _MXFP8MuseSparkLMHead,
        mxfp8_lm_head,
    )

    config = muse_spark_debugmodel()
    original_head = config.model.lm_head
    result = mxfp8_lm_head(config, fuse_softcap_loss=fuse_softcap_loss)
    assert config.model.lm_head is original_head
    assert isinstance(result.model.lm_head, _MXFP8MuseSparkLMHead.Config)
    assert result.model.lm_head.output_dtype == torch.float32
    assert result.model.lm_head.apply_logit_postprocess == (not fuse_softcap_loss)
    assert (
        isinstance(result.loss.loss_fn, _MuseSparkSoftcapCrossEntropy.Config)
        == fuse_softcap_loss
    )
    source = MuseSparkLMHead.Config(in_features=128, out_features=128).build()
    target = _MXFP8MuseSparkLMHead.Config(in_features=128, out_features=128).build()
    target.load_state_dict(source.state_dict(), strict=True)
    assert set(target.state_dict()) == set(source.state_dict())
    assert torch.equal(target.weight._tensor, source.weight)


def test_mxfp8_head_override_rejects_incompatible_model_and_loss():
    from dataclasses import replace

    from torchtitan.components.loss import CrossEntropyLoss
    from torchtitan_recipes.overrides.muse_spark_mxfp8_head import mxfp8_lm_head
    from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel

    with pytest.raises(ValueError, match="requires MuseSparkModel"):
        mxfp8_lm_head(llama3_debugmodel())
    config = muse_spark_debugmodel()
    with pytest.raises(ValueError, match="requires chunked CrossEntropyLoss"):
        mxfp8_lm_head(replace(config, loss=CrossEntropyLoss.Config()))
