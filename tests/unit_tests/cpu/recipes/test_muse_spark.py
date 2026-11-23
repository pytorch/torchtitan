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


@pytest.mark.parametrize("fake_backend", [False, True])
def test_cp1_benchmark_uses_native_geometry_precision_and_activation_budget(
    fake_backend,
):
    import torch
    from torchtitan.config.transform.quantization import MXFP8Linear
    from torchtitan.models.common.decoder_sharding import vocab_sharded_on_cp
    from torchtitan.models.common.dist_moe.mxfp8 import MXFP8DistMoeRoutedExperts
    from torchtitan_recipes.overrides.muse_spark_mxfp8_head import _MXFP8MuseSparkLMHead

    # CPU config coverage only; exact provider and execution gates run on GPU.
    with patch(
        "torchtitan.config.transform.quantization.has_cuda_capability",
        return_value=True,
    ):
        config = recipes._muse_spark_816b_mxfp8_spmd_64_benchmark_cp1(
            fake_backend=fake_backend
        )
    parallelism = config.parallelism
    assert parallelism.data_parallel_shard_degree == 64
    assert parallelism.expert_parallel_degree == 64
    assert parallelism.context_parallel_degree == 1
    assert parallelism.tensor_parallel_degree == 1
    assert parallelism.fsdp_reshard_after_forward == "always"
    assert config.training.num_tokens_per_train_step == 64 * 32768
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 32768
    assert config.precision.dtype == "float32"
    assert config.precision.param_dtype == config.precision.reduce_dtype == "bfloat16"
    for optimizer in config.optim.optimizer.optimizers:
        assert optimizer.fused and optimizer.moment_dtype == "bfloat16"
        assert optimizer.moment_cpu_offload_fraction == 0.0
    assert config.dist_moe_runtime.activation_slot_bytes == 40 * 2**30
    assert config.dist_moe_runtime.activation_slot_capacity_factor is None
    assert config.dist_moe_runtime.scratch_capacity_factor == 1.0
    assert isinstance(config.model.lm_head, _MXFP8MuseSparkLMHead.Config)
    assert config.model.lm_head.output_dtype == torch.float32
    assert not vocab_sharded_on_cp(config.model.lm_head.sharding_config)
    assert {"moe_router_topk", "muse_softcap_ce"} <= set(
        config.model.local_compile.regions
    )
    for layer in config.model.layers:
        assert isinstance(
            layer.feed_forward.routed_experts, MXFP8DistMoeRoutedExperts.Config
        )
        assert (
            layer.feed_forward.routed_experts.block_scaled_config.pipeline == "staged"
        )
        assert isinstance(layer.attention.wo, MXFP8Linear.Config)
