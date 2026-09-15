# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re

import pytest
import torch

from torchtitan.distributed.flex_shard import BlockShard, ComputeLayout, Owned
from torchtitan.distributed.flex_shard.dist_muon import (
    _compute_muon_direction,
    _matrix_batch_view_from_compute_layout,
    _zeropower_via_newtonschulz,
)
from torchtitan.models.common.attention import QKVLinear
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.kimi_k2_7.config_registry import kimi_k2_5_debugmodel


_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def _assert_independent_ns(matrix_batch: torch.Tensor) -> None:
    original = matrix_batch.clone()
    expected = (
        torch.stack(
            [
                _zeropower_via_newtonschulz(
                    matrix,
                    ns_coefficients=_NS_COEFFICIENTS,
                    ns_steps=2,
                    eps=1e-7,
                )
                for matrix in original.reshape(-1, *original.shape[-2:])
            ]
        )
        .view_as(original)
        .to(matrix_batch.dtype)
    )

    _compute_muon_direction(
        matrix_batch,
        ns_coefficients=_NS_COEFFICIENTS,
        ns_steps=2,
        eps=1e-7,
        out=matrix_batch,
    )

    torch.testing.assert_close(matrix_batch, expected, rtol=0, atol=2e-2)


def test_stacked_gate_up_applies_ns_independently() -> None:
    input_dim = 2
    hidden_dim = 3
    feed_forward = FeedForward.Config(
        w13=Linear.Config(
            in_features=input_dim,
            out_features=hidden_dim,
            num_linears=2,
        ),
        w2=Linear.Config(in_features=hidden_dim, out_features=input_dim),
    ).build()
    storage = torch.arange(
        feed_forward.w13.weight.numel(), dtype=torch.float32
    ).reshape_as(feed_forward.w13.weight)
    layout = ComputeLayout(shardings_by_mesh_axis={"dp_shard": Owned()})

    assert storage.shape == (2, hidden_dim, input_dim)
    assert (
        _matrix_batch_view_from_compute_layout(
            "layers.0.feed_forward.w13.weight", storage, layout
        )
        is None
    )
    _assert_independent_ns(storage)


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4], ids=["mqa", "gqa", "mha"])
def test_fused_qkv_applies_ns_per_head(num_kv_heads: int) -> None:
    input_dim = 5
    head_dim = 3
    num_heads = 4
    num_fused_heads = num_heads + 2 * num_kv_heads
    fused_qkv = QKVLinear.Config(
        head_dim=head_dim,
        n_heads=num_heads,
        n_kv_heads=num_kv_heads,
        wqkv=Linear.Config(
            in_features=input_dim,
            out_features=num_fused_heads * head_dim,
        ),
    ).build()
    storage = torch.arange(
        fused_qkv.wqkv.weight.numel(), dtype=torch.float32
    ).reshape_as(fused_qkv.wqkv.weight)
    layout = ComputeLayout(
        shardings_by_mesh_axis={
            "dp_shard": BlockShard(dim=0, block_size=head_dim),
        },
    )

    view = _matrix_batch_view_from_compute_layout(
        "layers.0.attention.wqkv.weight", storage, layout
    )

    assert view is not None
    matrix_batch = view.view_as_matrix_batch(storage)
    assert matrix_batch.shape == (num_fused_heads, head_dim, input_dim)
    _assert_independent_ns(matrix_batch)


def test_kimi_dist_muon_configures_fused_weights() -> None:
    config = kimi_k2_5_debugmodel()
    assert config.model_spec is not None
    dense_feed_forward = config.model_spec.model.layers[0].feed_forward
    moe = config.model_spec.model.layers[1].moe
    assert dense_feed_forward is not None
    assert moe is not None and moe.shared_experts is not None
    assert dense_feed_forward.w13.num_linears == 2
    assert moe.shared_experts.w13.num_linears == 2

    factory_kwargs = config.optimizer.optimizer_factory_kwargs_by_name["DistMuon"]
    compute_layouts = factory_kwargs["compute_sharding_by_fqn"]
    fused_fqns = (
        "layers.0.feed_forward.w13.weight",
        "layers.1.moe.shared_experts.w13.weight",
    )

    for fqn in fused_fqns:
        assert type(compute_layouts[fqn].shardings_by_mesh_axis["dp_shard"]) is Owned

    muon_pattern = config.optimizer.param_groups[0].pattern
    assert all(re.search(muon_pattern, fqn) for fqn in fused_fqns)
    assert not re.search(muon_pattern, "layers.0.feed_forward.w1.weight")
    assert not re.search(muon_pattern, "layers.0.feed_forward.w3.weight")
    bucket_configs = factory_kwargs["bucket_configs"]
    for fqn in fused_fqns:
        assert sum(fqn in bucket.patterns for bucket in bucket_configs) == 1

    config.parallelism.expert_parallel_degree = 2
    config.__post_init__()
    config.__post_init__()
    factory_kwargs = config.optimizer.optimizer_factory_kwargs_by_name["DistMuon"]
    compute_layouts = factory_kwargs["compute_sharding_by_fqn"]
    routed_layouts = tuple(
        layout
        for fqn, layout in compute_layouts.items()
        if ".moe.routed_experts.inner_experts." in fqn
    )
    assert routed_layouts
    for layout in routed_layouts:
        assert set(layout.shardings_by_mesh_axis) == {"ep", "efsdp"}
        assert layout.shard_order_by_tensor_dim[0] == ("ep", "efsdp")
