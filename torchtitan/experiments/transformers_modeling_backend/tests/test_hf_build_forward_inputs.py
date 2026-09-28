# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import spmd_types as spmd
import torch
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.context_parallel import ContextParallelLoadBalancer
from torchtitan.distributed.parallelism_context import MeshAxisName, ParallelismContext
from torchtitan.experiments.transformers_modeling_backend.model import (
    HFTransformerModel,
)
from torchtitan.models.common.cp_attention import KVAllGatherCPFlexInnerAttention


def _run(
    m: HFTransformerModel,
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
    pd = ParallelismContext(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=1,
        enable_sequence_parallel=False,
    )
    with patch(
        "torchtitan.distributed.spmd_types.annotate_input_spmd_types",
        side_effect=lambda _parallelism_context, batch, _input_sharding: batch,
    ):
        return (
            m.preprocess_inputs(
                batch,
                parallelism_context=pd,
                parallelism=ParallelismConfig(),
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
    expected_permutation = object()

    class TestLoadBalancer(ContextParallelLoadBalancer):
        def generate_permutation(self):
            calls.append("permutation")
            return expected_permutation

    load_balancer = TestLoadBalancer()
    load_balancer_config = Mock()
    load_balancer_config.build.return_value = load_balancer

    def shard_tensors(input_dict, **kwargs):
        assert kwargs["permutation"] is expected_permutation
        calls.append("cp_input")
        return input_dict

    def prepare_cp_metadata(attention_metadata, *, permutation):
        assert permutation is expected_permutation
        calls.append("cp_metadata")
        return attention_metadata

    def annotate(_parallelism_context, batch, input_sharding):
        assert calls == ["permutation", "cp_metadata", "cp_input"]
        assert set(batch) == {"input", "labels", "positions"}
        assert input_sharding["input"].local_type[MeshAxisName.TP] is spmd.R
        assert input_sharding["labels"].local_type[MeshAxisName.TP] is spmd.I
        assert input_sharding["positions"].local_type[MeshAxisName.TP] is spmd.R
        calls.append("spmd")
        return batch

    monkeypatch.setattr(
        "torchtitan.distributed.context_parallel.shard_tensors",
        shard_tensors,
    )
    monkeypatch.setattr(
        KVAllGatherCPFlexInnerAttention,
        "prepare_cp_metadata",
        staticmethod(prepare_cp_metadata),
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
    parallelism_context = cast(
        ParallelismContext,
        SimpleNamespace(cp_enabled=True),
    )

    _, _, extra_kwargs = model.preprocess_inputs(
        batch,
        parallelism_context=parallelism_context,
        parallelism=ParallelismConfig(
            context_parallel_load_balancer=load_balancer_config
        ),
    )

    load_balancer_config.build.assert_called_once()
    build_kwargs = load_balancer_config.build.call_args.kwargs
    assert build_kwargs["seq_len"] == 2
    assert build_kwargs["attention_metadata"] is dense_attention_mask
    assert "cp_mesh" not in build_kwargs
    assert calls == ["permutation", "cp_metadata", "cp_input", "spmd"]
    assert extra_kwargs["attention_masks"] is dense_attention_mask
