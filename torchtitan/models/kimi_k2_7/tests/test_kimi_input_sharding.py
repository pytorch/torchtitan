# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import torch
from torchtitan.config import ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.kimi_k2_7.model import KimiK25Model


def test_kimi_has_no_build_forward_inputs():
    assert "_build_forward_inputs" not in vars(KimiK25Model)


def test_kimi_has_no_input_sharding():
    assert "input_sharding" not in vars(KimiK25Model)


def test_kimi_preprocess_builds_mask_and_returns(monkeypatch):
    # Fully self-contained preprocess_inputs: with the default spmd_backend the
    # SPMD branches are skipped, so this exercises the inlined mask gate, merged
    # sharding path, ntokens, and pop without building the heavy Kimi model.
    monkeypatch.setattr(
        KimiK25Model, "get_attention_masks", lambda self, positions: "MASK"
    )
    m = cast(KimiK25Model, object.__new__(KimiK25Model))
    inner = object.__new__(FlexAttention.Config)
    first_attention = type("_FA", (), {"inner_attention": inner})()
    # object.__setattr__ bypasses nn.Module.__setattr__ to inject a plain config.
    object.__setattr__(
        m, "config", type("_Cfg", (), {"first_attention": first_attention})()
    )

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
    inputs, labels, extra, ntok = m.preprocess_inputs(
        batch,
        parallel_dims=pd,
        device=torch.device("cpu"),
        parallelism=ParallelismConfig(),
    )
    assert extra["attention_masks"] == "MASK"
    assert ntok == B * S
    assert "input" not in extra and "labels" not in extra
    assert tuple(inputs.shape) == (B, S)
