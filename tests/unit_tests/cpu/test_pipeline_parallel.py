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
from torchtitan.components.checkpointer.base import ModelWrapper
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import pipeline_parallel
from torchtitan.distributed.context_parallel import (
    HeadTailCPLoadBalancer,
    PTRRFlexAttentionCPLoadBalancer,
)
from torchtitan.distributed.pipeline_parallel import (
    _build_decoder_stage_io,
    _build_pipeline_schedule,
    _generate_llm_fqn_per_model_part,
    _get_pipeline_metadata,
    _get_pp_rank_to_stage_indices_mapping,
    _static_stage_metadata,
    _unsupported_static_split,
)
from torchtitan.models.deepseek_v3 import mtp
from torchtitan_recipes.tests.models.llama3 import build_model_config


class _FakePPMesh:
    def __init__(self, size, local_rank, group):
        self._size = size
        self._local_rank = local_rank
        self._group = group

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._local_rank

    def get_group(self, axis):
        assert axis == "pp"
        return self._group


def _shared_parameter_stage(value, *, requires_grad=True):
    stage = nn.Module()
    stage.tok_embeddings = nn.Embedding(2, 3)
    stage.tok_embeddings.weight.requires_grad_(requires_grad)
    with torch.no_grad():
        stage.tok_embeddings.weight.fill_(value)
    return stage


def _shared_parameter_runtime(
    model_parts,
    stage_indices,
    *,
    pp_size=1,
    pp_rank=0,
    num_stages=2,
    shared_stages=(0, 1),
    group=None,
):
    return pipeline_parallel.SharedParameterPipelineRuntime(
        model_parts=model_parts,
        stage_indices=stage_indices,
        pp_mesh=_FakePPMesh(pp_size, pp_rank, group or object()),
        pp_schedule="Interleaved1F1B" if num_stages > pp_size else "1F1B",
        num_stages=num_stages,
        shared_parameters=(
            pipeline_parallel.PipelineSharedParameter(
                fqn="tok_embeddings.weight",
                stage_indices=shared_stages,
            ),
        ),
    )


def test_same_rank_pipeline_shared_parameter_lifecycle_and_checkpoint():
    canonical = _shared_parameter_stage(2.0)
    replica = _shared_parameter_stage(7.0)
    runtime = _shared_parameter_runtime([canonical, replica], (0, 1))

    runtime.synchronize_parameters()
    torch.testing.assert_close(
        replica.tok_embeddings.weight,
        canonical.tok_embeddings.weight,
    )

    canonical.tok_embeddings.weight.grad = torch.full_like(
        canonical.tok_embeddings.weight, 3.0
    )
    replica.tok_embeddings.weight.grad = torch.full_like(
        replica.tok_embeddings.weight, 5.0
    )
    runtime.finalize_gradients()
    torch.testing.assert_close(
        canonical.tok_embeddings.weight.grad,
        torch.full_like(canonical.tok_embeddings.weight, 8.0),
    )
    torch.testing.assert_close(
        replica.tok_embeddings.weight.grad,
        canonical.tok_embeddings.weight.grad,
    )

    norm_parameters = runtime.parameters_for_grad_norm(
        (canonical.tok_embeddings.weight, replica.tok_embeddings.weight)
    )
    assert len(norm_parameters) == 1
    assert norm_parameters[0] is canonical.tok_embeddings.weight

    wrapper = ModelWrapper([canonical, replica])
    checkpoint = {
        key: value.detach().clone() for key, value in wrapper.state_dict().items()
    }
    with torch.no_grad():
        canonical.tok_embeddings.weight.fill_(11.0)
        replica.tok_embeddings.weight.fill_(13.0)
    wrapper.load_state_dict(checkpoint)
    torch.testing.assert_close(
        canonical.tok_embeddings.weight,
        replica.tok_embeddings.weight,
    )


def test_same_rank_pipeline_shared_parameter_handles_frozen_and_missing_gradients():
    canonical = _shared_parameter_stage(2.0, requires_grad=False)
    replica = _shared_parameter_stage(7.0, requires_grad=False)
    runtime = _shared_parameter_runtime([canonical, replica], (0, 1))
    runtime.synchronize_parameters()
    runtime.finalize_gradients()

    canonical.tok_embeddings.weight.requires_grad_(True)
    replica.tok_embeddings.weight.requires_grad_(True)
    canonical.tok_embeddings.weight.grad = torch.ones_like(
        canonical.tok_embeddings.weight
    )
    with pytest.raises(RuntimeError, match="missing a gradient"):
        runtime.finalize_gradients()


def test_same_rank_pipeline_shared_parameter_honors_reversed_canonical_stage():
    replica = _shared_parameter_stage(2.0)
    canonical = _shared_parameter_stage(7.0)
    runtime = _shared_parameter_runtime(
        [replica, canonical],
        (0, 1),
        shared_stages=(1, 0),
    )

    runtime.synchronize_parameters()
    torch.testing.assert_close(
        replica.tok_embeddings.weight,
        canonical.tok_embeddings.weight,
    )
    norm_parameters = runtime.parameters_for_grad_norm(
        (replica.tok_embeddings.weight, canonical.tok_embeddings.weight)
    )
    assert len(norm_parameters) == 1
    assert norm_parameters[0] is canonical.tok_embeddings.weight


def test_cross_rank_pipeline_shared_parameter_uses_canonical_owner(monkeypatch):
    class FakeProcessGroup:
        pass

    group = FakeProcessGroup()
    stage = _shared_parameter_stage(7.0)
    broadcast_sources = []
    collective_mode = {"value": "synchronize"}

    def all_reduce(tensor, *, group):
        assert isinstance(group, FakeProcessGroup)
        if tensor.numel() == 1:
            tensor.fill_(2 if collective_mode["value"] != "frozen" else 0)

    def broadcast(tensor, *, group, group_src):
        del tensor
        assert isinstance(group, FakeProcessGroup)
        broadcast_sources.append(group_src)

    monkeypatch.setattr(pipeline_parallel.dist, "ProcessGroup", FakeProcessGroup)
    monkeypatch.setattr(pipeline_parallel.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(pipeline_parallel.dist, "broadcast", broadcast)
    runtime = _shared_parameter_runtime(
        [stage],
        (1,),
        pp_size=2,
        pp_rank=1,
        shared_stages=(1, 0),
        group=group,
    )

    runtime.synchronize_parameters()
    assert broadcast_sources == [1]

    collective_mode["value"] = "frozen"
    stage.tok_embeddings.weight.requires_grad_(False)
    runtime.synchronize_parameters()
    runtime.finalize_gradients()


def test_pp_greater_than_two_splits_shared_parameter_owners(monkeypatch):
    class FakeProcessGroup:
        pass

    parent_group = FakeProcessGroup()
    owner_group = FakeProcessGroup()
    stage = _shared_parameter_stage(7.0)
    split_ranks = []
    broadcast_sources = []

    monkeypatch.setattr(pipeline_parallel.dist, "ProcessGroup", FakeProcessGroup)
    monkeypatch.setattr(pipeline_parallel.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(pipeline_parallel.device_module, "current_device", lambda: 0)

    def split_group(*, parent_pg, split_ranks: list[list[int]], group_desc):
        del group_desc
        assert parent_pg is parent_group
        split_ranks_copy = [list(ranks) for ranks in split_ranks]
        split_ranks_seen.append(split_ranks_copy)
        return owner_group

    split_ranks_seen = split_ranks
    monkeypatch.setattr(pipeline_parallel.dist, "split_group", split_group)
    monkeypatch.setattr(
        pipeline_parallel.dist,
        "all_reduce",
        lambda tensor, **kwargs: tensor.fill_(2),
    )
    monkeypatch.setattr(
        pipeline_parallel.dist,
        "broadcast",
        lambda tensor, *, group, group_src: broadcast_sources.append(group_src),
    )

    runtime = _shared_parameter_runtime(
        [stage],
        (3,),
        pp_size=4,
        pp_rank=3,
        num_stages=4,
        shared_stages=(3, 0),
        group=parent_group,
    )
    runtime.synchronize_parameters()

    assert split_ranks == [[[3, 0]]]
    assert broadcast_sources == [0]


def test_mtp_pipeline_layout_places_mtp_and_embedding_replica_on_final_stage(
    monkeypatch,
):
    monkeypatch.setattr(
        mtp,
        "_get_pipeline_metadata",
        lambda *args, **kwargs: (4, 6, 1, 1),
    )
    parallelism = ParallelismConfig(
        pipeline_parallel_degree=2,
        pipeline_parallel_schedule="Interleaved1F1B",
    )

    resolved, num_stages = mtp._mtp_pipeline_parallelism(
        SimpleNamespace(num_mtp_layers=2),
        parallelism_context=SimpleNamespace(),
        parallelism=parallelism,
        model_config=SimpleNamespace(),
    )

    assert num_stages == 4
    assert resolved is not parallelism
    assert parallelism.pipeline_parallel_module_fqns_per_model_part is None
    layout = resolved.pipeline_parallel_module_fqns_per_model_part
    assert layout is not None
    assert "tok_embeddings" in layout[0]
    assert layout[-1][-3:] == ["mtp_layers.0", "mtp_layers.1", "tok_embeddings"]


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        (
            [["tok_embeddings", "layers.0"], ["mtp_layers.0", "norm", "lm_head"]],
            "tok_embeddings on exactly the first and last",
        ),
        (
            [
                ["tok_embeddings", "layers.0"],
                ["tok_embeddings", "norm", "lm_head"],
            ],
            "mtp_layers.0 on exactly the final",
        ),
        (
            [
                ["tok_embeddings", "layers.0", "norm"],
                ["mtp_layers.0", "tok_embeddings", "lm_head"],
            ],
            "norm on exactly the final",
        ),
        (
            [
                ["tok_embeddings", "layers.0", "lm_head"],
                ["mtp_layers.0", "tok_embeddings", "norm"],
            ],
            "lm_head on exactly the final",
        ),
    ],
)
def test_mtp_pipeline_layout_rejects_invalid_shared_ownership(layout, message):
    parallelism = ParallelismConfig(
        pipeline_parallel_degree=2,
        pipeline_parallel_schedule="1F1B",
        pipeline_parallel_module_fqns_per_model_part=layout,
    )
    with pytest.raises(ValueError, match=message):
        mtp._mtp_pipeline_parallelism(
            SimpleNamespace(num_mtp_layers=1),
            parallelism_context=SimpleNamespace(),
            parallelism=parallelism,
            model_config=SimpleNamespace(),
        )


def test_mtp_eager_pipeline_attaches_shared_parameter_runtime(monkeypatch):
    from torchtitan.models.deepseek_v3 import model as deepseek_model

    schedule = SimpleNamespace()
    model_parts = [object()]
    pipeline = pipeline_parallel.PipelineResult(
        schedule=schedule,
        model_parts=model_parts,
        stage_indices=(0,),
        has_first_stage=True,
        has_last_stage=False,
    )
    expected_result = object()
    resolved_parallelism = ParallelismConfig()
    attached = {}

    monkeypatch.setattr(
        deepseek_model,
        "_mtp_pipeline_parallelism",
        lambda *args, **kwargs: (resolved_parallelism, 2),
    )

    def pipeline_llm(model, **kwargs):
        assert kwargs["parallelism"] is resolved_parallelism
        return pipeline

    def with_runtime(pipeline_arg, **kwargs):
        attached.update(pipeline=pipeline_arg, **kwargs)
        return expected_result

    monkeypatch.setattr(pipeline_parallel, "pipeline_llm", pipeline_llm)
    monkeypatch.setattr(deepseek_model, "_with_mtp_pipeline_runtime", with_runtime)

    result = deepseek_model.DeepSeekV3Model.pipeline(
        SimpleNamespace(num_mtp_layers=1),
        parallelism=ParallelismConfig(),
        parallelism_context=SimpleNamespace(),
        model_config=SimpleNamespace(),
    )

    assert result is expected_result
    assert attached["pipeline"] is pipeline
    assert attached["parallelism"] is resolved_parallelism
    assert attached["num_stages"] == 2


def test_hf_pipeline_returns_pipeline_result(monkeypatch):
    from torchtitan.experiments.transformers_modeling_backend import pipeline as hf_pp

    class ModelPart:
        def parallelize(self, **kwargs):
            return self

    stages = [
        SimpleNamespace(stage_index=0, is_first=True, is_last=False, submod=None),
        SimpleNamespace(stage_index=1, is_first=False, is_last=True, submod=None),
    ]
    model_parts = [ModelPart(), ModelPart()]
    schedule = object()
    monkeypatch.setattr(
        hf_pp,
        "pipeline_module_split",
        lambda *args, **kwargs: (stages, model_parts),
    )
    monkeypatch.setattr(hf_pp, "_build_pipeline_schedule", lambda **kwargs: schedule)

    result = hf_pp.pipeline_hf_transformers(
        SimpleNamespace(),
        SimpleNamespace(get_mesh=lambda axis: object(), pp=2),
        training=SimpleNamespace(),
        parallelism=ParallelismConfig(
            pipeline_parallel_degree=2,
            pipeline_parallel_schedule="1F1B",
        ),
        local_compile_regions=[],
        ac_config=SimpleNamespace(),
        dump_folder="",
        device=torch.device("cpu"),
        model_config=SimpleNamespace(layers=[None, None]),
        loss_fn=object(),
    )

    assert isinstance(result, pipeline_parallel.PipelineResult)
    assert result.schedule is schedule
    assert result.model_parts == model_parts
    assert result.stage_indices == (0, 1)
    assert result.has_first_stage
    assert result.has_last_stage


def test_mtp_graph_pipeline_attaches_shared_parameter_runtime(monkeypatch):
    import sys
    from types import ModuleType

    from torchtitan.experiments.graph_trainer.deepseek_v3 import model as gt_model

    schedule = SimpleNamespace()
    model_parts = [object()]
    pipeline = pipeline_parallel.PipelineResult(
        schedule=schedule,
        model_parts=model_parts,
        stage_indices=(0,),
        has_first_stage=True,
        has_last_stage=False,
    )
    expected_result = object()
    resolved_parallelism = ParallelismConfig()
    attached = {}

    monkeypatch.setattr(
        gt_model,
        "_mtp_pipeline_parallelism",
        lambda *args, **kwargs: (resolved_parallelism, 2),
    )

    def graph_pipeline_llm(model, **kwargs):
        assert kwargs["parallelism"] is resolved_parallelism
        return pipeline

    def with_runtime(pipeline_arg, **kwargs):
        attached.update(pipeline=pipeline_arg, **kwargs)
        return expected_result

    pipeline_module = ModuleType(
        "torchtitan.experiments.graph_trainer.graph_pp.pipeline"
    )
    pipeline_module.graph_pipeline_llm = graph_pipeline_llm
    monkeypatch.setitem(sys.modules, pipeline_module.__name__, pipeline_module)
    monkeypatch.setattr(gt_model, "_with_mtp_pipeline_runtime", with_runtime)

    result = gt_model.GraphTrainerDeepSeekV3Model.pipeline(
        SimpleNamespace(num_mtp_layers=1),
        parallelism=ParallelismConfig(),
        parallelism_context=SimpleNamespace(),
        model_config=SimpleNamespace(),
    )

    assert result is expected_result
    assert attached["pipeline"] is pipeline
    assert attached["parallelism"] is resolved_parallelism
    assert attached["num_stages"] == 2


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
        parallelism_context=SimpleNamespace(pp=2),
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
        parallelism_context=SimpleNamespace(pp=2),
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
        parallelism_context=SimpleNamespace(pp=2),
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
        (2, 1, False, False, PTRRFlexAttentionCPLoadBalancer.Config(), 64, 64),
        (2, 1, False, False, HeadTailCPLoadBalancer.Config(), 64, 64),
        (2, 2, True, True, PTRRFlexAttentionCPLoadBalancer.Config(), 64, 32),
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
    model_config = build_model_config("debugmodel")
    parallelism_context = SimpleNamespace(cp=cp, tp=tp, tp_enabled=tp_enabled)
    stage_io = _build_decoder_stage_io(
        parallelism_context=parallelism_context,
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
    model_config = build_model_config("debugmodel")
    stage_io = _build_decoder_stage_io(
        parallelism_context=SimpleNamespace(cp=1, tp=1, tp_enabled=False),
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
        parallelism_context=SimpleNamespace(pp=2),
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
    parallelism_context = SimpleNamespace(pp=2)
    model_config = SimpleNamespace(layers=[None] * 10)
    parallelism = ParallelismConfig(
        pipeline_parallel_degree=2,
        pipeline_parallel_schedule="Interleaved1F1B",
        pipeline_parallel_layers_per_stage=2,
    )
    pipeline_parallel.pipeline_with_first_last_stage_modules(
        model,
        first_stage_module_fqns=("vision_encoder",),
        parallelism_context=parallelism_context,
        parallelism=parallelism,
        model_config=model_config,
    )

    handed = captured["parallelism"]
    split = handed.pipeline_parallel_module_fqns_per_model_part
    assert handed.pipeline_parallel_layers_per_stage == 2
    assert parallelism.pipeline_parallel_module_fqns_per_model_part is None
    assert split[0][0] == "vision_encoder"
    num_stages = _get_pipeline_metadata(parallelism_context, handed, model_config)[0]
    unsized = dataclasses.replace(parallelism, pipeline_parallel_layers_per_stage=None)
    assert len(split) == num_stages
    assert (
        num_stages
        != _get_pipeline_metadata(parallelism_context, unsized, model_config)[0]
    )


def test_parallelism_config_refuses_a_split_with_layers_per_stage():
    with pytest.raises(ValueError, match="set only one of them"):
        ParallelismConfig(
            pipeline_parallel_layers_per_stage=2,
            pipeline_parallel_module_fqns_per_model_part=[["tok_embeddings"], ["norm"]],
        )


@pytest.mark.parametrize(
    ("backward_requires_autograd", "expected"),
    [(True, True), (False, False)],
)
def test_build_pipeline_schedule_sets_deferred_reduce_grad_wait(
    monkeypatch, backward_requires_autograd, expected
):
    schedule_kwargs = {}

    class TestSchedule(pipeline_parallel.PipelineScheduleMulti):
        def __init__(self, *args, **kwargs):
            schedule_kwargs.update(kwargs)

    monkeypatch.setattr(
        "torchtitan.distributed.pipeline_parallel.get_schedule_class",
        lambda _: TestSchedule,
    )
    parallelism = ParallelismConfig(
        pipeline_parallel_degree=2,
        pipeline_parallel_schedule="Interleaved1F1B",
    )

    _build_pipeline_schedule(
        parallelism=parallelism,
        num_microbatches=4,
        stages=[object(), object()],
        loss_fn=lambda *args, **kwargs: (object(), object()),
        backward_requires_autograd=backward_requires_autograd,
    )

    assert schedule_kwargs["defer_reduce_grad_wait"] is expected


def test_build_pipeline_schedule_forwards_max_outstanding_sends(monkeypatch):
    schedule_kwargs = {}

    class TestSchedule(pipeline_parallel.PipelineScheduleMulti):
        def __init__(self, *args, **kwargs):
            schedule_kwargs.update(kwargs)

    monkeypatch.setattr(
        "torchtitan.distributed.pipeline_parallel.get_schedule_class",
        lambda _: TestSchedule,
    )
    parallelism = ParallelismConfig(
        pipeline_parallel_degree=2,
        pipeline_parallel_schedule="Interleaved1F1B",
        pipeline_parallel_max_outstanding_sends=2,
    )

    _build_pipeline_schedule(
        parallelism=parallelism,
        num_microbatches=4,
        stages=[object(), object()],
        loss_fn=lambda *args, **kwargs: (object(), object()),
    )

    assert schedule_kwargs["max_outstanding_sends"] == 2
