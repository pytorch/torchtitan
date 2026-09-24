# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn

from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.config_utils import (
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.multimodal import get_packed_vision_grids
from torchtitan.models.flops import (
    active_parameter_flops_per_unit,
    get_parameter_counts,
)


class _EmbeddingModel(nn.Module):
    def __init__(self, *, tie_weights: bool) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(5, 3)
        self.lm_head = nn.Linear(3, 5, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight


def test_active_parameter_flops_per_unit_counts_dense_module_parameters() -> None:
    with torch.device("meta"):
        module = nn.Linear(3, 2)

    assert active_parameter_flops_per_unit(module) == 48


@pytest.mark.parametrize("tie_weights", [False, True])
def test_parameter_flops_exclude_input_embedding(tie_weights: bool) -> None:
    with torch.device("meta"):
        model = _EmbeddingModel(tie_weights=tie_weights)

    assert active_parameter_flops_per_unit(model) == 90


@pytest.mark.parametrize(
    ("tie_weights", "expected_count"),
    [(False, 30), (True, 15)],
)
def test_get_parameter_counts_includes_embeddings_and_deduplicates_tied_weights(
    tie_weights: bool,
    expected_count: int,
) -> None:
    with torch.device("meta"):
        model = _EmbeddingModel(tie_weights=tie_weights)

    assert get_parameter_counts(model) == (expected_count, expected_count)


def test_parameter_flops_weight_routed_experts_by_active_ratio() -> None:
    num_experts = 3
    top_k = 2
    with torch.device("meta"):
        model = make_moe_config(
            num_experts=num_experts,
            router=make_router_config(
                dim=2,
                num_experts=num_experts,
                gate_param_init={"weight": nn.init.zeros_},
                score_func=Sigmoid.Config(),
                top_k=top_k,
            ),
            routed_experts=make_routed_experts_config(
                dim=2,
                hidden_dim=3,
                num_experts=num_experts,
                top_k=top_k,
                param_init={},
                comm_backend="standard",
            ),
            load_balance_coeff=None,
        ).build()

    assert get_parameter_counts(model) == (60, 42)
    assert active_parameter_flops_per_unit(model) == 252


def test_parameter_flops_exclude_module_subtrees() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.included = nn.Linear(3, 2, bias=False)
            self.excluded = nn.Sequential(
                nn.Linear(2, 4),
                nn.LayerNorm(4),
            )

    with torch.device("meta"):
        model = Model()

    assert get_parameter_counts(model) == (26, 26)

    assert (
        active_parameter_flops_per_unit(
            model,
            excluded_modules=(model.excluded,),
        )
        == 36
    )


class _GridRows:
    def __init__(self, rows: list[tuple[int, int, int]]) -> None:
        self.rows = rows
        self.num_tolist_calls = 0

    def tolist(self) -> list[tuple[int, int, int]]:
        self.num_tolist_calls += 1
        return self.rows


def test_packed_vision_grids_reads_each_grid_once() -> None:
    image_grids = _GridRows([(1, 2, 4)])
    video_grids = _GridRows([(2, 2, 4)])
    batch = {
        "pixel_values": object(),
        "grid_thw": image_grids,
        "pixel_values_videos": object(),
        "grid_thw_videos": video_grids,
    }

    vision_grids = get_packed_vision_grids(
        batch,
        modality_fields=(
            ("pixel_values", "grid_thw"),
            ("pixel_values_videos", "grid_thw_videos"),
        ),
    )

    assert vision_grids == ((1, 2, 4), (2, 2, 4))
    assert image_grids.num_tolist_calls == 1
    assert video_grids.num_tolist_calls == 1
