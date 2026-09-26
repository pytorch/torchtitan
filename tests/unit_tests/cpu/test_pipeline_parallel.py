# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from torchtitan.config import ParallelismConfig
from torchtitan.distributed import pipeline_parallel
from torchtitan.distributed.pipeline_parallel import (
    _build_decoder_stage_io,
    _generate_llm_fqn_per_model_part,
    _get_pipeline_metadata,
    _get_pp_rank_to_stage_indices_mapping,
    _static_stage_metadata,
    _unsupported_static_split,
)
from torchtitan.models.llama3.config_registry import model_registry


def test_pipeline_with_first_last_stage_modules_prepends_present_modules(monkeypatch):
    model = nn.Module()
    model.vision_encoder = nn.Linear(2, 2)
    model.vision_adapter = nn.Linear(2, 2)
    model.vision_projection = None
    captured = {}
    expected_result = object()

    def capture_pipeline_llm(model, **kwargs):
        captured["parallelism"] = kwargs["parallelism"]
        return expected_result

    monkeypatch.setattr(pipeline_parallel, "pipeline_llm", capture_pipeline_llm)

    parallelism = ParallelismConfig(pipeline_parallel_degree=2)
    result = pipeline_parallel.pipeline_with_first_last_stage_modules(
        model,
        first_stage_module_fqns=(
            "vision_encoder",
            "vision_adapter",
            "vision_projection",
            "missing_module",
        ),
        parallel_dims=SimpleNamespace(pp=2),
        parallelism=parallelism,
        model_config=SimpleNamespace(layers=[None] * 4),
    )

    assert result is expected_result
    assert parallelism.pipeline_parallel_module_fqns_per_model_part is None
    assert captured["parallelism"].pipeline_parallel_module_fqns_per_model_part == [
        ["vision_encoder", "vision_adapter", "tok_embeddings", "layers.0", "layers.1"],
        ["layers.2", "layers.3", "norm", "lm_head"],
    ]


def test_base_model_pipeline_derives_the_split_only_when_none_is_configured(
    monkeypatch,
):
    from torchtitan.protocols.model import BaseModel

    model = nn.Module()
    model.supports_pipeline_parallel = True
    model.pipeline_first_stage_module_fqns = ("vision_encoder",)
    model.pipeline_last_stage_module_fqns = ()
    model.vision_encoder = nn.Linear(2, 2)
    captured = []

    def capture_pipeline_llm(model, **kwargs):
        captured.append(kwargs["parallelism"])
        return object()

    monkeypatch.setattr(pipeline_parallel, "pipeline_llm", capture_pipeline_llm)
    common = dict(
        parallel_dims=SimpleNamespace(pp=2),
        model_config=SimpleNamespace(layers=[None] * 4),
    )

    parallelism = ParallelismConfig(pipeline_parallel_degree=2)
    BaseModel.pipeline(model, parallelism=parallelism, **common)
    assert captured[-1].pipeline_parallel_module_fqns_per_model_part == [
        ["vision_encoder", "tok_embeddings", "layers.0", "layers.1"],
        ["layers.2", "layers.3", "norm", "lm_head"],
    ]

    configured = ParallelismConfig(
        pipeline_parallel_degree=2,
        pipeline_parallel_module_fqns_per_model_part=[["input"], ["output"]],
    )
    BaseModel.pipeline(model, parallelism=configured, **common)
    assert captured[-1] is configured


def test_pipeline_with_first_last_stage_modules_appends_present_last_stage_modules(
    monkeypatch,
):
    model = nn.Module()
    model.vision_encoder = nn.Linear(2, 2)
    model.output_res_proj = nn.Linear(2, 2)
    model.output_res_norm = None
    captured = {}

    def capture_pipeline_llm(model, **kwargs):
        captured["parallelism"] = kwargs["parallelism"]
        return object()

    monkeypatch.setattr(pipeline_parallel, "pipeline_llm", capture_pipeline_llm)

    pipeline_parallel.pipeline_with_first_last_stage_modules(
        model,
        first_stage_module_fqns=("vision_encoder",),
        last_stage_module_fqns=("output_res_proj", "output_res_norm", "missing"),
        parallel_dims=SimpleNamespace(pp=2),
        parallelism=ParallelismConfig(pipeline_parallel_degree=2),
        model_config=SimpleNamespace(layers=[None] * 4),
    )

    assert captured["parallelism"].pipeline_parallel_module_fqns_per_model_part == [
        ["vision_encoder", "tok_embeddings", "layers.0", "layers.1"],
        ["layers.2", "layers.3", "norm", "lm_head", "output_res_proj"],
    ]


def _assert_layer_assignment(module_names_per_stage: list[list[str]], num_layers: int):
    """Layers are assigned in order with no gaps or duplicates."""
    assigned = [
        name
        for stage in module_names_per_stage
        for name in stage
        if name.startswith("layers.")
    ]
    assert assigned == [f"layers.{i}" for i in range(num_layers)]


def test_generate_llm_fqn_rejects_zero_stages():
    with pytest.raises(ValueError, match="Number of stages must be at least 1"):
        _generate_llm_fqn_per_model_part(0, 4)


def test_generate_llm_fqn_single_stage_includes_all_modules():
    # One stage owns embeddings, every layer, and the output modules.
    assert _generate_llm_fqn_per_model_part(1, 3) == [
        [
            "tok_embeddings",
            "layers.0",
            "layers.1",
            "layers.2",
            "norm",
            "lm_head",
        ]
    ]


def test_generate_llm_fqn_two_stages_default_weights():
    # stages=2, layers=4, in=1, out=1: effective=6, 3 per stage.
    # stage0: tok_embeddings + 2 layers (3-1); stage1: 2 layers + norm/lm_head (3-1).
    result = _generate_llm_fqn_per_model_part(2, 4, input_weight=1, output_weight=1)
    assert result == [
        ["tok_embeddings", "layers.0", "layers.1"],
        ["layers.2", "layers.3", "norm", "lm_head"],
    ]
    _assert_layer_assignment(result, num_layers=4)


def test_generate_llm_fqn_docstring_weighted_example():
    # effective = 3 + 2 + 2 = 7; 7 // 2 = 3 layers/stage, 1 leftover on stage 0.
    # stage0: tok_embeddings + (4 - 2) layers; stage1: (3 - 2) layers + output.
    result = _generate_llm_fqn_per_model_part(2, 3, input_weight=2, output_weight=2)
    assert result == [
        ["tok_embeddings", "layers.0", "layers.1"],
        ["layers.2", "norm", "lm_head"],
    ]
    _assert_layer_assignment(result, num_layers=3)


def test_generate_llm_fqn_middle_stage_is_layers_only():
    # stages=3, layers=6, in=1, out=1: effective=8, 2 per stage, 2 leftovers.
    # stage0 (3): tok_embeddings + 2 layers; stage1 (3): layers only; stage2 (2): 1 layer + output.
    result = _generate_llm_fqn_per_model_part(3, 6, input_weight=1, output_weight=1)
    assert result == [
        ["tok_embeddings", "layers.0", "layers.1"],
        ["layers.2", "layers.3", "layers.4"],
        ["layers.5", "norm", "lm_head"],
    ]
    assert result[0][0] == "tok_embeddings"
    assert result[-1][-2:] == ["norm", "lm_head"]
    assert all(name.startswith("layers.") for name in result[1])
    _assert_layer_assignment(result, num_layers=6)


def test_generate_llm_fqn_rejects_more_stages_than_effective_layers():
    with pytest.raises(
        ValueError,
        match=r"Number of stages \(10\) cannot be greater than effective layers \(6\)",
    ):
        _generate_llm_fqn_per_model_part(10, 4, input_weight=1, output_weight=1)


def test_generate_llm_fqn_rejects_input_weight_exceeding_layers_per_stage():
    # stages=4, layers=4, in=2, out=1: effective=7, min layers/stage = 1.
    with pytest.raises(
        ValueError,
        match=r"input_weight \(2\) exceeds minimum layers per stage \(1\)",
    ):
        _generate_llm_fqn_per_model_part(4, 4, input_weight=2, output_weight=1)


def test_generate_llm_fqn_rejects_output_weight_exceeding_layers_per_stage():
    # stages=4, layers=4, in=1, out=2: effective=7, min layers/stage = 1.
    with pytest.raises(
        ValueError,
        match=r"output_weight \(2\) exceeds minimum layers per stage \(1\)",
    ):
        _generate_llm_fqn_per_model_part(4, 4, input_weight=1, output_weight=2)


@pytest.mark.parametrize("schedule", ["1F1B", "Interleaved1F1B"])
def test_loop_schedule_maps_one_stage_per_rank(schedule: str):
    # 1F1B-style loop: stage i lives on rank i when there is one stage per rank.
    for rank in range(4):
        assert _get_pp_rank_to_stage_indices_mapping(rank, 4, schedule, 4) == (rank,)


@pytest.mark.parametrize("schedule", ["1F1B", "Interleaved1F1B"])
def test_loop_schedule_maps_interleaved_stages(schedule: str):
    # stages_per_rank=2, pp_degree=2: rank r owns (r, r+2).
    assert _get_pp_rank_to_stage_indices_mapping(0, 2, schedule, 4) == (0, 2)
    assert _get_pp_rank_to_stage_indices_mapping(1, 2, schedule, 4) == (1, 3)


@pytest.mark.parametrize("schedule", ["1F1B", "Interleaved1F1B"])
def test_loop_schedule_maps_two_virtual_stages_across_four_ranks(schedule: str):
    # stages_per_rank=2, pp_degree=4: rank r owns (r, r+4).
    expected = {
        0: (0, 4),
        1: (1, 5),
        2: (2, 6),
        3: (3, 7),
    }
    for rank, stages in expected.items():
        assert _get_pp_rank_to_stage_indices_mapping(rank, 4, schedule, 8) == stages


def test_v_schedule_maps_mirrored_stage_pairs():
    # ZBVZeroBubble is a v schedule: rank i owns (i, num_stages-1-i).
    expected = {
        0: (0, 7),
        1: (1, 6),
        2: (2, 5),
        3: (3, 4),
    }
    for rank, stages in expected.items():
        assert (
            _get_pp_rank_to_stage_indices_mapping(rank, 4, "ZBVZeroBubble", 8) == stages
        )


def test_pp_rank_to_stage_mapping_requires_even_division():
    with pytest.raises(AssertionError, match="must be evenly divisible"):
        _get_pp_rank_to_stage_indices_mapping(0, 3, "1F1B", 4)


def test_get_pipeline_metadata_requires_layers_attribute():
    with pytest.raises(ValueError, match="Model does not have layers attribute."):
        _get_pipeline_metadata(object(), ParallelismConfig(), object())


@pytest.mark.parametrize(
    (
        "cp",
        "tp",
        "tp_enabled",
        "enable_sequence_parallel",
        "load_balancer",
        "expected_decoder_tokens",
        "expected_hidden_tokens",
    ),
    [
        (1, 1, False, False, None, 128, 128),
        (2, 1, False, False, None, 64, 64),
        (2, 1, False, False, "ptrr", 64, 64),
        (2, 1, False, False, "headtail", 32, 32),
        (2, 2, True, True, "ptrr", 64, 32),
    ],
)
def test_static_decoder_stage_metadata_is_complete(
    cp,
    tp,
    tp_enabled,
    enable_sequence_parallel,
    load_balancer,
    expected_decoder_tokens,
    expected_hidden_tokens,
):
    model_config = model_registry("debugmodel")
    parallel_dims = SimpleNamespace(cp=cp, tp=tp, tp_enabled=tp_enabled)
    stage_io = _build_decoder_stage_io(
        parallel_dims=parallel_dims,
        parallelism=ParallelismConfig(
            enable_sequence_parallel=enable_sequence_parallel,
            context_parallel_load_balancer=load_balancer,
        ),
        training=pipeline_parallel.TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=128
        ),
        model_config=model_config,
        lm_head_in_loss=True,
    )

    first = _static_stage_metadata(stage_io, 0, 3)
    middle = _static_stage_metadata(stage_io, 1, 3)
    last = _static_stage_metadata(stage_io, 2, 3)

    assert first["input_args"][0].shape == torch.Size([expected_decoder_tokens])
    assert first["input_args"][0].dtype == torch.int64
    assert first["input_grads"] == (None,)
    assert middle["input_args"][0].shape == torch.Size(
        [expected_hidden_tokens, model_config.dim]
    )
    assert middle["input_args"][0].requires_grad
    assert not middle["input_grads"][0].requires_grad
    assert last["output_args"][0].shape == torch.Size(
        [expected_decoder_tokens, model_config.dim]
    )
    assert last["output_grads"] == (None,)


def test_static_decoder_stage_metadata_describes_logits_output():
    model_config = model_registry("debugmodel")
    stage_io = _build_decoder_stage_io(
        parallel_dims=SimpleNamespace(cp=1, tp=1, tp_enabled=False),
        parallelism=ParallelismConfig(),
        training=pipeline_parallel.TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=128
        ),
        model_config=model_config,
        lm_head_in_loss=False,
    )

    last = _static_stage_metadata(stage_io, 2, 3)
    assert last["output_args"][0].shape == torch.Size([128, model_config.vocab_size])


def test_static_decoder_metadata_accepts_standard_decoder_boundaries():
    assert (
        _unsupported_static_split(
            [["tok_embeddings"], ["layers.0", "norm"], ["lm_head"]]
        )
        is None
    )


def test_static_decoder_metadata_rejects_custom_boundary():
    assert (
        _unsupported_static_split(
            [["vision_encoder"], ["tok_embeddings", "layers.0", "norm", "lm_head"]]
        )
        == "stage 0 does not produce decoder hidden states"
    )


@pytest.mark.parametrize(
    ("configured_limit", "expected_limit"),
    [(None, 2), (1, 1)],
)
def test_pipeline_param_residency_limit(monkeypatch, configured_limit, expected_limit):
    class CapturingSchedule(pipeline_parallel.PipelineScheduleMulti):
        __slots__ = ("kwargs",)

        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        pipeline_parallel, "get_schedule_class", lambda _: CapturingSchedule
    )
    parallelism = ParallelismConfig(
        pipeline_parallel_degree=2,
        pipeline_parallel_schedule="Interleaved1F1B",
        pp_max_unsharded_active_stages=configured_limit,
    )

    schedule = pipeline_parallel._build_pipeline_schedule(
        parallelism=parallelism,
        num_microbatches=4,
        stages=[object(), object()],
        loss_fn=lambda: None,
    )

    assert schedule.kwargs["max_active_stages"] == expected_limit
    assert schedule.kwargs["unshard_lookahead"] == "auto"
    assert "reuse_recv_buffers" not in schedule.kwargs


@pytest.mark.parametrize("limit", [0, -1])
def test_pipeline_param_residency_limit_must_be_positive(limit):
    with pytest.raises(ValueError, match="pp_max_unsharded_active_stages"):
        ParallelismConfig(pp_max_unsharded_active_stages=limit)


@pytest.mark.parametrize(
    "lookahead",
    [None, True, 2, "default", "adaptive", [1, 2], (1,), (1, 4), (1, False)],
)
def test_unshard_lookahead_rejects_invalid_values(lookahead):
    with pytest.raises(ValueError, match="pp_num_unshard_lookahead_factor"):
        ParallelismConfig(
            pipeline_parallel_degree=2,
            pp_max_unsharded_active_stages=3,
            pp_num_unshard_lookahead_factor=lookahead,
        )


def test_unshard_lookahead_defaults_to_auto():
    assert ParallelismConfig().pp_num_unshard_lookahead_factor == "auto"


@pytest.mark.parametrize(
    ("lookahead", "expected"),
    [("full", "full"), ("auto", "auto"), ((1, 3), (1, 3))],
)
def test_unshard_lookahead_is_forwarded_to_multistage_schedule(
    monkeypatch, lookahead, expected
):
    class CapturingSchedule(pipeline_parallel.PipelineScheduleMulti):
        __slots__ = ("kwargs",)

        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        pipeline_parallel, "get_schedule_class", lambda _: CapturingSchedule
    )
    parallelism = ParallelismConfig(
        pipeline_parallel_degree=2,
        pipeline_parallel_schedule="Interleaved1F1B",
        pp_max_unsharded_active_stages=3,
        pp_num_unshard_lookahead_factor=lookahead,
    )

    schedule = pipeline_parallel._build_pipeline_schedule(
        parallelism=parallelism,
        num_microbatches=4,
        stages=[object(), object()],
        loss_fn=lambda: None,
    )

    assert schedule.kwargs["max_active_stages"] == 3
    assert schedule.kwargs["unshard_lookahead"] == expected


def test_per_rank_unshard_lookahead_rejects_single_stage_schedule(monkeypatch):
    class CapturingSchedule(pipeline_parallel.PipelineScheduleSingle):
        pass

    monkeypatch.setattr(
        pipeline_parallel, "get_schedule_class", lambda _: CapturingSchedule
    )
    parallelism = ParallelismConfig(
        pipeline_parallel_degree=2,
        pipeline_parallel_schedule="1F1B",
        pp_num_unshard_lookahead_factor=(1, 2),
    )

    with pytest.raises(ValueError, match="only by multi-stage"):
        pipeline_parallel._build_pipeline_schedule(
            parallelism=parallelism,
            num_microbatches=2,
            stages=[object()],
            loss_fn=lambda: None,
        )


def test_get_module_fqns_per_model_part_is_the_split_the_pipeline_uses(monkeypatch):
    model = nn.Module()
    model.vision_encoder = nn.Linear(2, 2)
    model.output_res_norm = nn.Linear(2, 2)
    captured = {}

    def capture_pipeline_llm(model, **kwargs):
        captured["parallelism"] = kwargs["parallelism"]
        return object()

    monkeypatch.setattr(pipeline_parallel, "pipeline_llm", capture_pipeline_llm)
    common = dict(
        first_stage_module_fqns=("vision_encoder",),
        last_stage_module_fqns=("output_res_norm",),
        parallel_dims=SimpleNamespace(pp=2),
        model_config=SimpleNamespace(layers=[None] * 4),
    )
    parallelism = ParallelismConfig(pipeline_parallel_degree=2)

    split = pipeline_parallel.get_module_fqns_per_model_part(
        model, parallelism=parallelism, **common
    )
    pipeline_parallel.pipeline_with_first_last_stage_modules(
        model, parallelism=parallelism, **common
    )
    assert split == [
        ["vision_encoder", "tok_embeddings", "layers.0", "layers.1"],
        ["layers.2", "layers.3", "norm", "lm_head", "output_res_norm"],
    ]
    assert split == captured["parallelism"].pipeline_parallel_module_fqns_per_model_part

    explicit = [["tok_embeddings", "layers.0"], ["layers.1", "norm", "lm_head"]]
    with pytest.raises(ValueError, match="derives the split"):
        pipeline_parallel.get_module_fqns_per_model_part(
            model,
            parallelism=ParallelismConfig(
                pipeline_parallel_degree=2,
                pipeline_parallel_module_fqns_per_model_part=explicit,
            ),
            **common,
        )


def test_layers_per_stage_sizes_the_derived_split(monkeypatch):
    model = nn.Module()
    model.vision_encoder = nn.Linear(2, 2)
    captured = {}

    def capture_pipeline_llm(model, **kwargs):
        captured["parallelism"] = kwargs["parallelism"]
        return object()

    monkeypatch.setattr(pipeline_parallel, "pipeline_llm", capture_pipeline_llm)
    parallel_dims = SimpleNamespace(pp=2)
    model_config = SimpleNamespace(layers=[None] * 10)
    parallelism = ParallelismConfig(
        pipeline_parallel_degree=2,
        pipeline_parallel_schedule="Interleaved1F1B",
        pipeline_parallel_layers_per_stage=2,
    )
    pipeline_parallel.pipeline_with_first_last_stage_modules(
        model,
        first_stage_module_fqns=("vision_encoder",),
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        model_config=model_config,
    )

    handed = captured["parallelism"]
    split = handed.pipeline_parallel_module_fqns_per_model_part
    assert handed.pipeline_parallel_layers_per_stage == 2
    assert parallelism.pipeline_parallel_module_fqns_per_model_part is None
    assert split[0][0] == "vision_encoder"
    num_stages = _get_pipeline_metadata(parallel_dims, handed, model_config)[0]
    unsized = dataclasses.replace(parallelism, pipeline_parallel_layers_per_stage=None)
    assert len(split) == num_stages
    assert num_stages != _get_pipeline_metadata(parallel_dims, unsized, model_config)[0]


def test_parallelism_config_refuses_a_split_with_layers_per_stage():
    with pytest.raises(ValueError, match="set only one of them"):
        ParallelismConfig(
            pipeline_parallel_layers_per_stage=2,
            pipeline_parallel_module_fqns_per_model_part=[["tok_embeddings"], ["norm"]],
        )
