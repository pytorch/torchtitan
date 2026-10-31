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
    active_parameter_flops_per_unit,
    get_parameter_counts,
)
from torchtitan.observability.metrics import compute_training_performance_metrics
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
    assert active_parameter_flops_per_unit(model) == 6 * 32
    model.embedding.weight = model.lm_head.weight
    assert get_parameter_counts(model) == (32, 32)
    assert active_parameter_flops_per_unit(model) == 6 * 32


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
    mtp_parameter_flops = active_parameter_flops_per_unit(model.mtp_layers)
    repeated_head_flops = 2 * active_parameter_flops_per_unit(model.lm_head)
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
        expected += num_tokens * active_parameter_flops_per_unit(stage)
    expected += num_tokens * sum(
        active_parameter_flops_per_unit(module)
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


def test_engine_estimates_before_transfer_and_preprocessing():
    calls = []
    raw = {"input": torch.ones(4, dtype=torch.long)}
    microbatch = SimpleNamespace(
        as_input_dict=lambda: raw,
        to_input_dict=lambda *args, **kwargs: calls.append("transfer") or dict(raw),
    )
    model = SimpleNamespace(
        preprocess_inputs=lambda *args, **kwargs: calls.append("preprocess")
        or (1, 2, {}, {}),
    )
    engine = SimpleNamespace(
        step_num_flops=0,
        flops_estimator=lambda inputs: calls.append("estimate") or 123,
        parallelism_context=SimpleNamespace(
            activate_spmd=contextlib.nullcontext, cp=2, pp_enabled=False
        ),
        model_parts=[model],
        device=torch.device("cpu"),
        config=SimpleNamespace(
            parallelism=None,
            training=SimpleNamespace(
                max_context_length=4, num_tokens_per_microbatch_per_dp_rank=4
            ),
        ),
        max_num_documents=None,
        preprocess_inputs_kwargs={},
        ntokens_seen=0,
    )
    TrainingEngine._preprocess_microbatch_group(
        engine, [microbatch], global_routing_token_counts=torch.tensor([4])
    )
    assert calls == ["estimate", "transfer", "preprocess"]
    assert engine.step_num_flops == 123
    assert engine.ntokens_seen == 2


def test_performance_uses_work_independently_of_text_token_count():
    metrics = compute_training_performance_metrics(
        num_tokens=20,
        num_flops=8000,
        elapsed_time=2,
        non_data_parallel_size=2,
        gpu_peak_flops=10000,
        has_quantization=False,
    )
    assert metrics == {"tokens_per_second": 5, "tflops": 2e-9, "mfu_percent": 20}


@pytest.mark.parametrize("family", ["qwen3_5", "kimi_k3"])
def test_linear_attention_configs_own_sequence_independent_work(family):
    from torchtitan.models.common.attention.gdn import GDN
    from torchtitan.models.common.attention.kda import KDA

    config = importlib.import_module(f"torchtitan.models.{family}").build_model_config(
        "debugmodel", parallelism=ParallelismConfig(), seq_len=128
    )
    linear_configs = [
        layer.attention
        for layer in config.layers
        if isinstance(layer.attention, (GDN.Config, KDA.Config))
    ]
    assert linear_configs
    for attention in linear_configs:
        if isinstance(attention, GDN.Config):
            expected = (
                18
                * attention.n_heads
                * attention.key_head_dim
                * attention.value_head_dim
            )
        else:
            expected = 18 * attention.n_heads * attention.head_dim**2
        assert attention.flops_per_token(16) == expected
        assert attention.flops_per_token(128) == expected


def test_lazy_replay_does_not_count_work_twice():
    engine = SimpleNamespace(
        step_num_flops=0,
        parallelism_context=SimpleNamespace(
            dp_replicate_enabled=False, pp_enabled=False
        ),
        _global_routing_token_counts=torch.tensor([4]),
        _non_pp_forward_backward_microbatch=lambda **kwargs: torch.tensor(1.0),
        loss_metrics={},
    )

    def preprocess(group, **kwargs):
        engine.step_num_flops += 123
        return (torch.ones(4), torch.ones(4), {}, {})

    engine._preprocess_microbatch_group = preprocess
    for _ in range(2):
        TrainingEngine._forward_backward_body(
            engine,
            [[object()], [object()]],
            torch.tensor(8),
            defer_fsdp_gradient_reduction=False,
            preprocess_microbatch_groups_lazily=True,
        )
        assert engine.step_num_flops == 246


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


def test_built_estimator_does_not_retain_encoder_modules():
    import gc
    import weakref

    config, model = _build("muse_spark", "debugmodel_mm")
    encoder_ref = weakref.ref(model.vision_encoder)
    projection_ref = weakref.ref(model.vision_projection)
    estimate = config.build_flops_estimator(model, seq_len=128)
    del model
    gc.collect()
    assert encoder_ref() is None
    assert projection_ref() is None
    assert estimate({"input": torch.empty(128, dtype=torch.long)}) > 0


@pytest.mark.parametrize("family", ["muse_glimmer", "muse_spark"])
def test_muse_inherits_gqa_window_accounting(family):
    config, _ = _build(family, "debugmodel_mm")
    seq_len = 4096
    assert any(layer.attention.window_size is not None for layer in config.layers)
    for layer in config.layers:
        attention = layer.attention
        head_dim = attention.head_dim or attention.dim // attention.n_heads
        attended_tokens = (
            seq_len
            if attention.window_size is None
            else min(seq_len, attention.window_size)
        )
        assert attention.flops_per_token(seq_len) == (
            12 * attention.n_heads * head_dim * attended_tokens
        )


@pytest.mark.parametrize(
    "dp,cp,should_log", [(1, 1, True), (1, 2, True), (2, 2, True), (1, 1, False)]
)
def test_flops_logging_reduces_only_across_dp(dp, cp, should_log):
    from unittest.mock import MagicMock, patch

    from torchtitan.components.data.types import TokenizedTrainingMicrobatch
    from torchtitan.trainer import Trainer
    from torchtitan.training_engine import ForwardBackwardResult

    context = SimpleNamespace(
        dp_enabled=dp > 1,
        dp_cp_enabled=dp * cp > 1,
        dp_replicate=dp,
        dp_shard=1,
        cp=cp,
        get_optional_mesh=lambda name: name,
        get_mesh=lambda name: name,
    )
    engine = SimpleNamespace(
        num_completed_steps=0,
        step_num_flops=200,
        device=torch.device("cpu"),
        parallelism_context=context,
        optim=SimpleNamespace(lr_schedulers=SimpleNamespace(get_metrics=lambda: {})),
        optim_step=lambda: torch.tensor(3.0),
        forward_backward=lambda **kwargs: ForwardBackwardResult(torch.tensor(2.0), []),
        ntokens_seen=4,
    )
    metrics = SimpleNamespace(
        should_log=lambda step: should_log,
        step_last_log=0,
        reset=lambda: None,
        log=MagicMock(),
    )
    trainer = SimpleNamespace(
        engine=engine,
        metrics_processor=metrics,
        gradient_accumulation_steps=1,
        num_pp_microbatches=1,
    )
    batch = TokenizedTrainingMicrobatch(
        input=torch.ones(4, dtype=torch.long),
        labels=torch.ones(4, dtype=torch.long),
        positions=torch.arange(4),
        padding_mask=torch.zeros(4, dtype=torch.bool),
        loss_token_counts=torch.tensor(4),
        routing_token_counts=torch.tensor([4]),
    )
    tensor_reductions = []

    def sum_tensor(value, mesh):
        tensor_reductions.append((value.clone(), mesh))
        return value * dp

    with (
        patch("torchtitan.trainer.dist_utils.dist_sum_tensor", side_effect=sum_tensor),
        patch(
            "torchtitan.trainer.dist_utils.dist_sum",
            side_effect=lambda value, mesh: float(value) * dp * cp,
        ),
        patch(
            "torchtitan.trainer.dist_utils.dist_max",
            side_effect=lambda value, mesh: float(value),
        ),
        patch("torchtitan.trainer.collect_aux_loss_metrics", return_value={}),
        patch("torchtitan.trainer.torch.stack", wraps=torch.stack) as stack,
    ):
        Trainer.train_step(trainer, iter([batch]))
    flops_reductions = [
        (value, mesh)
        for value, mesh in tensor_reductions
        if value.dtype == torch.float64
    ]
    assert len(flops_reductions) == int(dp > 1 and should_log)
    if flops_reductions:
        value, mesh = flops_reductions[0]
        assert value.numel() == 1 and mesh == "dp"
        # FLOPs and grad norm share one host readback.
        stack.assert_called_once()
    else:
        stack.assert_not_called()
    if should_log:
        assert metrics.log.call_args.kwargs["num_flops"] == 200
        assert metrics.log.call_args.args[3] == 3.0
    else:
        metrics.log.assert_not_called()


def test_spark_audio_estimation_uses_shapes_without_reading_tensors():
    config, model = _build("muse_spark", "debugmodel_mm")
    estimate = config.audio_encoder.build_flops_estimator(model.audio_encoder)
    cpu_inputs = {
        "waveforms": torch.empty(32, 1),
        "waveform_padded_lengths": torch.tensor([12, 20]),
    }
    meta_inputs = {
        "waveforms": torch.empty(32, 1, device="meta"),
        "waveform_padded_lengths": torch.empty(2, device="meta"),
    }
    assert estimate(meta_inputs) == estimate(cpu_inputs)


def test_multimodal_estimator_dispatches_no_tensor_computation():
    from torch.utils._python_dispatch import TorchDispatchMode

    class RejectTensorComputation(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            raise AssertionError(
                f"Unexpected tensor operation in estimator: {function}"
            )

    config, model = _build("muse_spark", "debugmodel_mm")
    estimate = config.build_flops_estimator(model, seq_len=128)
    inputs = {
        "input": torch.empty(128, dtype=torch.long),
        "pixel_values": torch.empty(16, 1),
        "grid_thw": torch.tensor([[1, 4, 4]]),
        "waveforms": torch.empty(32, 1),
        "waveform_padded_lengths": torch.tensor([12, 20]),
    }
    with RejectTensorComputation():
        assert estimate(inputs) > 0
