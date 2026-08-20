# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import spmd_types as spmd
from torchtitan.distributed.context_parallel.api import cp_shard_dims
from torchtitan.distributed.parallel_dims import MeshAxisName, SpmdLayout


def _layout(dp, cp, tp):
    return SpmdLayout({MeshAxisName.DP: dp, MeshAxisName.CP: cp, MeshAxisName.TP: tp})


def test_shard_dim_reads_cp_axis():
    layout = _layout(spmd.S(0), spmd.S(1), spmd.R)
    assert layout.shard_dim(MeshAxisName.CP) == 1
    assert layout.shard_dim(MeshAxisName.DP) == 0
    assert layout.shard_dim(MeshAxisName.TP) is None  # R is not a Shard


def test_cp_shard_dims_includes_only_cp_sharded():
    # An image stream sharded on its own seq dim (dim 1) is CP-sharded;
    # a globally-replicated conditioning vector is not.
    input_sharding = {
        "input": _layout(spmd.S(0), spmd.S(1), spmd.R),
        "labels": _layout(spmd.S(0), spmd.S(1), spmd.I),
        "positions": _layout(spmd.S(0), spmd.S(1), spmd.R),
        "image_tokens": _layout(spmd.S(0), spmd.S(1), spmd.R),
        "cond": _layout(spmd.S(0), spmd.R, spmd.R),
    }
    dims = cp_shard_dims(input_sharding)
    assert dims == {"input": 1, "labels": 1, "positions": 1, "image_tokens": 1}
    assert "cond" not in dims


def test_prepare_cp_input_empty_input_sharding_is_noop():
    import torch
    from torchtitan.distributed.context_parallel import prepare_context_parallel_input

    inputs = torch.zeros(2, 4)
    labels = torch.ones(2, 4)
    extra = {"positions": torch.arange(4)}
    # An input_sharding with no CP-sharded tensors must be a graceful no-op that
    # returns the inputs untouched without touching cp_mesh (passed None here).
    out_inputs, out_labels, out_extra = prepare_context_parallel_input(
        inputs,
        labels,
        extra,
        cp_mesh=None,
        device=torch.device("cpu"),
        input_sharding={},
    )
    assert out_inputs is inputs
    assert out_labels is labels
    assert out_extra is extra
