# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.moe import MoE
from torchtitan.models.minimax_m3 import build_model_config, MODEL_FLAVORS
from torchtitan.models.minimax_m3.attention import (
    MiniMaxSparseAttention,
    MSAInnerAttention,
)
from torchtitan.models.minimax_m3.model import MiniMaxM3Model


def _layout(flavor: str) -> list[tuple[bool, bool]]:
    config = build_model_config(flavor, parallelism=ParallelismConfig())
    return [
        (
            isinstance(layer.attention, MiniMaxSparseAttention.Config),
            isinstance(layer.feed_forward, MoE.Config),
        )
        for layer in config.layers
    ]


@pytest.mark.parametrize("flavor", list(MODEL_FLAVORS))
def test_every_flavor_builds(flavor: str) -> None:
    config = build_model_config(flavor, parallelism=ParallelismConfig())
    assert isinstance(config, MiniMaxM3Model.Config)
    assert config.vision_encoder is not None


@pytest.mark.parametrize(
    "flavor, num_dense, num_full_attention",
    [
        ("debugmodel", 1, 1),
        ("debugmodel_flash", 1, 0),
        ("M3", 3, 3),
        ("M3.1-Flash", 3, 0),
    ],
)
def test_layer_layout(flavor: str, num_dense: int, num_full_attention: int) -> None:
    layout = _layout(flavor)
    assert [moe for _, moe in layout] == [i >= num_dense for i in range(len(layout))]
    assert [sparse for sparse, _ in layout] == [
        i >= num_full_attention for i in range(len(layout))
    ]


def test_m3_matches_the_released_checkpoint_size() -> None:
    config = build_model_config("M3", parallelism=ParallelismConfig())
    with torch.device("meta"):
        model = config.build()
    assert sum(t.numel() for t in model.state_dict().values()) == 427_040_140_160


def test_dense_attention_keeps_the_indexer() -> None:
    config = build_model_config(
        "debugmodel", sparse_attention=False, parallelism=ParallelismConfig()
    )
    sparse_layers = [
        layer.attention
        for layer in config.layers
        if isinstance(layer.attention, MiniMaxSparseAttention.Config)
    ]
    assert sparse_layers
    for attention in sparse_layers:
        assert isinstance(attention.inner_attention, MSAInnerAttention.Config)
        assert not attention.inner_attention.sparse
        assert attention.indexer.wq.out_features > 0


def test_seq_len_above_the_flavor_context_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds max context length"):
        build_model_config(
            "debugmodel", seq_len=1 << 20, parallelism=ParallelismConfig()
        )
