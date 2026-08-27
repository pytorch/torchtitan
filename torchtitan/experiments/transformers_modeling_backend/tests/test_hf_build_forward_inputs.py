# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import cast

import spmd_types as spmd
import torch
from torchtitan.config import ParallelismConfig
from torchtitan.distributed.parallel_dims import MeshAxisName, ParallelDims
from torchtitan.experiments.transformers_modeling_backend.model import (
    HFTransformerModel,
)


def _run(
    m: HFTransformerModel,
    *,
    spmd_backend: str = "partial_dtensor",
):
    B, S = 2, 4
    batch = cast(
        "dict[str, torch.Tensor]",
        {
            "input": torch.zeros(B, S),
            "positions": torch.arange(S).repeat(B, 1),
            "labels": torch.zeros(B, S),
        },
    )
    pd = ParallelDims(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=1,
        spmd_backend=spmd_backend,
    )
    return (
        m.preprocess_inputs(
            batch,
            parallel_dims=pd,
            parallelism=ParallelismConfig(spmd_backend=spmd_backend),
        ),
        B,
        S,
    )


def test_hf_builds_mask_when_present(monkeypatch):
    monkeypatch.setattr(
        HFTransformerModel, "get_attention_masks", lambda self, positions: "MASK"
    )
    m = cast(HFTransformerModel, object.__new__(HFTransformerModel))
    (inputs, labels, extra), B, S = _run(m)
    assert extra["attention_masks"] == "MASK"
    assert labels.numel() == B * S
    assert "input" not in extra and "labels" not in extra


def test_hf_no_mask_when_get_attention_masks_returns_none(monkeypatch):
    monkeypatch.setattr(
        HFTransformerModel, "get_attention_masks", lambda self, positions: None
    )
    m = cast(HFTransformerModel, object.__new__(HFTransformerModel))
    (inputs, labels, extra), B, S = _run(m)
    assert "attention_masks" not in extra


def test_hf_cp_shards_before_spmd_annotation(monkeypatch):
    calls = []

    def prepare(batch, input_shardings, cp_mesh, *_args):
        assert input_shardings is None
        assert cp_mesh == "cp_mesh"
        calls.append("cp")
        return batch

    def annotate(_parallel_dims, batch, input_sharding):
        assert calls == ["cp"]
        assert set(batch) == {"input", "labels", "positions"}
        assert input_sharding["input"].local_type[MeshAxisName.TP] is spmd.R
        assert input_sharding["labels"].local_type[MeshAxisName.TP] is spmd.I
        assert input_sharding["positions"].local_type[MeshAxisName.TP] is spmd.R
        calls.append("spmd")
        return batch

    monkeypatch.setattr(
        "torchtitan.distributed.context_parallel.api.prepare_context_parallel_input",
        prepare,
    )
    monkeypatch.setattr(
        "torchtitan.distributed.spmd_types.annotate_input_spmd_types", annotate
    )
    dense_attention_mask = torch.zeros(1, 1, 4, 4)
    monkeypatch.setattr(
        HFTransformerModel,
        "get_attention_masks",
        lambda self, positions: dense_attention_mask,
    )
    model = cast(HFTransformerModel, object.__new__(HFTransformerModel))
    batch = {
        "input": torch.zeros(2, 4),
        "labels": torch.zeros(2, 4),
        "positions": torch.arange(4).repeat(2, 1),
    }
    parallel_dims = cast(
        ParallelDims,
        SimpleNamespace(cp_enabled=True, get_mesh=lambda _name: "cp_mesh"),
    )

    _, _, extra_kwargs = model.preprocess_inputs(
        batch,
        parallel_dims=parallel_dims,
        parallelism=ParallelismConfig(spmd_backend="spmd_types"),
    )

    assert calls == ["cp", "spmd"]
    assert extra_kwargs["attention_masks"] is dense_attention_mask
