# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.pipeline_parallel import (
    _generate_llm_fqn_per_model_part,
    _get_pipeline_metadata,
    _get_pp_rank_to_stage_indices_mapping,
)


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
