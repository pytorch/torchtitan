# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import torch
from torchtitan.config import ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.experiments.transformers_modeling_backend.model import (
    HFTransformerModel,
)


def _run(m: HFTransformerModel):
    B, S = 2, 4
    batch = cast(
        "dict[str, torch.Tensor]",
        {
            "input": torch.zeros(B, S),
            "positions": torch.arange(S).repeat(B, 1),
            "labels": torch.zeros(B, S),
        },
    )
    pd = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1)
    return (
        m.preprocess_inputs(
            batch,
            parallel_dims=pd,
            device=torch.device("cpu"),
            parallelism=ParallelismConfig(),
        ),
        B,
        S,
    )


def test_hf_has_no_build_forward_inputs():
    assert not hasattr(HFTransformerModel, "_build_forward_inputs")


def test_hf_has_no_input_sharding():
    assert not hasattr(HFTransformerModel, "input_sharding")


def test_hf_builds_mask_when_present(monkeypatch):
    monkeypatch.setattr(
        HFTransformerModel, "get_attention_masks", lambda self, positions: "MASK"
    )
    m = cast(HFTransformerModel, object.__new__(HFTransformerModel))
    (inputs, labels, extra, ntok), B, S = _run(m)
    assert extra["attention_masks"] == "MASK"
    assert ntok == B * S
    assert "input" not in extra and "labels" not in extra


def test_hf_no_mask_when_get_attention_masks_returns_none(monkeypatch):
    monkeypatch.setattr(
        HFTransformerModel, "get_attention_masks", lambda self, positions: None
    )
    m = cast(HFTransformerModel, object.__new__(HFTransformerModel))
    (inputs, labels, extra, ntok), B, S = _run(m)
    assert "attention_masks" not in extra
