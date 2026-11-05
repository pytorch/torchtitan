# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.flops import (
    active_parameter_flops_per_token,
    get_parameter_counts,
)
from torchtitan.training_engine import TrainingEngine


def _build(family, flavor="debugmodel", **kwargs):
    module = importlib.import_module(f"torchtitan.models.{family}")
    config = module.build_model_config(
        flavor, parallelism=ParallelismConfig(), seq_len=128, **kwargs
    )
    with torch.device("meta"):
        model = config.build()
    return config, model


@pytest.mark.parametrize(
    "family,flavor,expected",
    [
        ("llama3", "debugmodel", 36_195_840),
        ("qwen3", "debugmodel", 217_093_632),
        ("qwen3", "debugmodel_moe", 331_126_272),
        ("gpt_oss", "debugmodel", 298_042_560),
        ("deepseek_v3", "debugmodel", 189_040_128),
        ("deepseek_v4", "debugmodel", 59_824_686),
        ("kimi_k2_7", "debugmodel", 189_040_128),
        ("muse_glimmer", "debugmodel", 47_236_608),
        ("muse_glimmer", "debugmodel_mm", 47_236_608),
        ("muse_spark", "debugmodel_mm", 31_638_528),
    ],
)
def test_text_estimates_preserve_main(family, flavor, expected):
    # Values recorded from main before introducing the shared estimator.
    config, model = _build(family, flavor)
    estimate = config.build_flops_estimator(model, seq_len=128)
    assert estimate({"input": torch.empty(256, dtype=torch.long)}) == expected * 256


def test_parameter_counts_include_embeddings_and_deduplicate_tied_weights():
    model = nn.Module()
    model.embedding = nn.Embedding(8, 4)
    model.lm_head = nn.Linear(4, 8, bias=False)
    assert get_parameter_counts(model) == (64, 64)
    assert active_parameter_flops_per_token(model) == 6 * 32
    model.embedding.weight = model.lm_head.weight
    assert get_parameter_counts(model) == (32, 32)
    assert active_parameter_flops_per_token(model) == 6 * 32


@pytest.mark.parametrize("family", ["deepseek_v3", "kimi_k2_7"])
def test_mtp_counts_attention_and_repeated_output_projection(family):
    base_config, base_model = _build(family)
    config, model = _build("deepseek_v3", num_mtp_layers=2)
    if family == "kimi_k2_7":
        from dataclasses import replace

        config = replace(base_config, mtp_layers=config.mtp_layers)
        with torch.device("meta"):
            model = config.build()
    inputs = {"input": torch.empty(128)}
    base_flops = base_config.build_flops_estimator(base_model, seq_len=128)(inputs)
    flops = config.build_flops_estimator(model, seq_len=128)(inputs)
    mtp_parameter_flops = active_parameter_flops_per_token(model.mtp_layers)
    repeated_head_flops = 2 * active_parameter_flops_per_token(model.lm_head)
    mtp_attention_flops = sum(
        layer.attention.flops_per_token(128) for layer in config.mtp_layers
    )
    assert flops - base_flops == 128 * (
        mtp_parameter_flops + repeated_head_flops + mtp_attention_flops
    )
    if family == "kimi_k2_7":
        image_inputs = {
            **inputs,
            "pixel_values": torch.empty(16, 1),
            "grid_thw": torch.tensor([[1, 4, 4]]),
        }
        base_image_flops = base_config.build_flops_estimator(base_model, seq_len=128)(
            image_inputs
        )
        image_flops = config.build_flops_estimator(model, seq_len=128)(image_inputs)
        assert base_image_flops > base_flops
        assert image_flops - base_image_flops == flops - base_flops


@pytest.mark.parametrize(
    "family,flavor",
    [
        ("kimi_k2_7", "debugmodel"),
        ("muse_glimmer", "debugmodel_mm"),
        ("muse_spark", "debugmodel_mm"),
    ],
)
def test_multimodal_estimate_sums_items_and_modalities(family, flavor):
    config, model = _build(family, flavor)
    estimate = config.build_flops_estimator(model, seq_len=128)
    text = {"input": torch.empty(128, dtype=torch.long)}
    image = {"pixel_values": torch.empty(16, 1), "grid_thw": torch.tensor([[1, 4, 4]])}
    video = {
        "pixel_values_videos": torch.empty(32, 1),
        "grid_thw_videos": torch.tensor([[2, 4, 4]]),
    }
    text_flops = estimate(text)
    image_flops = estimate({**text, **image}) - text_flops
    video_flops = estimate({**text, **video}) - text_flops
    assert image_flops > 0
    assert video_flops > image_flops
    assert (
        estimate({**text, **image, **video}) == text_flops + image_flops + video_flops
    )
    assert (
        estimate(
            {
                **text,
                "pixel_values": torch.empty(32, 1),
                "grid_thw": torch.tensor([[1, 4, 4], [1, 4, 4]]),
            }
        )
        == text_flops + 2 * image_flops
    )


def test_spark_audio_uses_each_stage_token_count():
    config, model = _build("muse_spark", "debugmodel_mm")
    estimate = config.build_flops_estimator(model, seq_len=128)
    text = {"input": torch.empty(128, dtype=torch.long)}
    num_tokens = 32
    expected = 0
    for stage in model.audio_encoder.layers.values():
        num_tokens //= stage.patchifier.patch_size
        expected += num_tokens * active_parameter_flops_per_token(stage)
    expected += num_tokens * sum(
        active_parameter_flops_per_token(module)
        for module in (
            model.audio_encoder.norm,
            model.audio_encoder.o_proj,
            model.audio_encoder.dual_norm,
            model.perception_emb_norm,
        )
    )
    audio = {
        "waveforms": torch.empty(32, 1),
        "waveform_padded_lengths": torch.tensor([12, 20]),
    }
    assert estimate({**text, **audio}) - estimate(text) == expected
    image = {"pixel_values": torch.empty(16, 1), "grid_thw": torch.tensor([[1, 4, 4]])}
    assert (
        estimate({**text, **audio, **image}) == estimate({**text, **image}) + expected
    )


def test_lazy_preprocessing_counts_flops_for_streamed_groups():
    class StreamedGroups:
        def __len__(self):
            return 2

        def __iter__(self):
            for num_tokens in (4, 8):
                yield [
                    SimpleNamespace(
                        as_input_dict=lambda: {"input": torch.empty(num_tokens)},
                        to_input_dict=lambda *args, **kwargs: {},
                    )
                ]

    engine = SimpleNamespace(
        flops_estimator=lambda inputs: inputs["input"].numel(),
        parallelism_context=SimpleNamespace(
            dp_replicate_enabled=False,
            pp_enabled=False,
            cp=1,
            activate_spmd=contextlib.nullcontext,
        ),
        _global_routing_token_counts=torch.tensor([12]),
        _forward_backward_num_flops=0,
        device=torch.device("cpu"),
        max_num_documents=None,
        preprocess_inputs_kwargs={},
        ntokens_seen=0,
        config=SimpleNamespace(
            parallelism=None,
            training=SimpleNamespace(num_tokens_per_microbatch_per_dp_rank=1),
        ),
        model_parts=[
            SimpleNamespace(
                preprocess_inputs=lambda *args, **kwargs: (None, None, {}, {})
            )
        ],
        _non_pp_forward_backward_microbatch=lambda **kwargs: torch.tensor(1.0),
    )
    engine._preprocess_microbatch_group = lambda group, **kwargs: (
        TrainingEngine._preprocess_microbatch_group(engine, group, **kwargs)
    )
    TrainingEngine._forward_backward_body(
        engine,
        StreamedGroups(),
        torch.tensor(12),
        defer_fsdp_gradient_reduction=False,
        preprocess_microbatch_groups_lazily=True,
    )
    assert engine._forward_backward_num_flops == 12


@pytest.mark.parametrize("family", ["muse_glimmer", "muse_spark"])
def test_muse_projection_is_charged_only_per_vision_output_token(family):
    config, model = _build(family, "debugmodel_mm")
    text = {"input": torch.empty(128, dtype=torch.long)}
    image = {
        **text,
        "pixel_values": torch.empty(16, 1),
        "grid_thw": torch.tensor([[1, 4, 4]]),
    }
    before = config.build_flops_estimator(model, seq_len=128)
    model.vision_projection.register_parameter(
        "extra_weight", nn.Parameter(torch.empty(7, device="meta"))
    )
    after = config.build_flops_estimator(model, seq_len=128)
    assert after(text) == before(text)
    # A 4x4 patch grid produces four tokens after 2x2 downsampling.
    assert after(image) - before(image) == 4 * 6 * 7


def test_spark_shared_perception_norm_is_charged_for_both_encoders():
    config, model = _build("muse_spark", "debugmodel_mm")
    text = {"input": torch.empty(128, dtype=torch.long)}
    multimodal = {
        **text,
        "pixel_values": torch.empty(16, 1),
        "grid_thw": torch.tensor([[1, 4, 4]]),
        "waveforms": torch.empty(32, 1),
        "waveform_padded_lengths": torch.tensor([12, 20]),
    }
    before = config.build_flops_estimator(model, seq_len=128)
    model.perception_emb_norm.register_parameter(
        "extra_weight", nn.Parameter(torch.empty(5, device="meta"))
    )
    after = config.build_flops_estimator(model, seq_len=128)
    assert after(text) == before(text)
    # Four visual tokens and eight audio tokens each use the same norm.
    assert after(multimodal) - before(multimodal) == (4 + 8) * 6 * 5


@pytest.mark.parametrize("family", ["qwen3_5", "kimi_k3"])
def test_vision_estimators_accept_raw_microbatches(family):
    config = (
        importlib.import_module(f"torchtitan.models.{family}")
        .build_model_config("debugmodel", parallelism=ParallelismConfig(), seq_len=128)
        .vision_encoder
    )
    with torch.device("meta"):
        encoder = config.build()
    estimate = config.build_flops_estimator(encoder)
    image = {
        "pixel_values": torch.empty(16, 1),
        "grid_thw": torch.tensor([[1, 4, 4]]),
    }
    two_images = {
        "pixel_values": torch.empty(32, 1),
        "grid_thw": torch.tensor([[1, 4, 4], [1, 4, 4]]),
    }
    assert estimate({}) == 0
    assert estimate(image) > 0
    assert estimate(two_images) == 2 * estimate(image)
