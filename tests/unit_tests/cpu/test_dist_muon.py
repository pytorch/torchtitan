# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.distributed.flex_shard import (
    BlockShard,
    ComputeLayout,
    MatrixBatchLayout,
    Owned,
)
from torchtitan.distributed.flex_shard.dist_muon import (
    _compute_muon_direction,
    _matrix_batch_view_from_compute_layout,
    _MatrixBatchView,
    _zeropower_via_newtonschulz,
)
from torchtitan.models.common.attention import FusedQKVLinear
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.overrides.fused_swiglu import fused_swiglu


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
        .view(original.shape)
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


def test_interleaved_matrix_batch_applies_ns_to_gate_and_up_independently() -> None:
    num_groups = 2
    matrix_rows = 3
    matrix_columns = 2
    storage = torch.arange(
        num_groups * matrix_rows * 2 * matrix_columns,
        dtype=torch.float32,
    ).reshape(num_groups * matrix_rows * 2, matrix_columns)
    original = storage.clone()
    view = _MatrixBatchView.from_storage_shape(
        torch.Size(storage.shape),
        matrix_rows=matrix_rows,
        num_interleaved_matrices=2,
        uses_block_shard=True,
    )

    matrix_batch = view.view_as_matrix_batch(storage)

    assert matrix_batch.shape == (num_groups, 2, matrix_rows, matrix_columns)
    grouped = original.view(num_groups, matrix_rows, 2, matrix_columns)
    torch.testing.assert_close(matrix_batch[:, 0], grouped[:, :, 0])
    torch.testing.assert_close(matrix_batch[:, 1], grouped[:, :, 1])
    _assert_independent_ns(matrix_batch)
    torch.testing.assert_close(
        storage.view(num_groups, matrix_rows, 2, matrix_columns),
        matrix_batch.permute(0, 2, 1, 3),
    )


def test_concatenated_matrix_batch_applies_ns_to_gate_and_up_independently() -> None:
    num_groups = 2
    matrix_rows = 3
    matrix_columns = 2
    storage = torch.arange(
        num_groups * 2 * matrix_rows * matrix_columns,
        dtype=torch.float32,
    ).reshape(num_groups * 2 * matrix_rows, matrix_columns)
    original = storage.clone()
    layout = ComputeLayout(
        shardings_by_mesh_axis={
            "dp_shard": BlockShard(dim=0, block_size=matrix_rows),
        },
    )
    view = _matrix_batch_view_from_compute_layout(
        "layers.0.feed_forward.w13.weight",
        storage,
        layout,
    )

    assert view is not None
    assert view.transport_shape(torch.Size(storage.shape)) == (
        num_groups * 2,
        matrix_rows,
        matrix_columns,
    )
    matrix_batch = view.view_as_matrix_batch(storage)
    assert matrix_batch.shape == (num_groups * 2, matrix_rows, matrix_columns)
    torch.testing.assert_close(matrix_batch, original.view_as(matrix_batch))
    _assert_independent_ns(matrix_batch)
    torch.testing.assert_close(storage, matrix_batch.view_as(storage))


def test_fused_swiglu_applies_ns_to_gate_and_up_independently() -> None:
    input_dim = 2
    hidden_dim = 3
    fused = fused_swiglu(
        FeedForward.Config(
            w1=Linear.Config(in_features=input_dim, out_features=hidden_dim),
            w2=Linear.Config(in_features=hidden_dim, out_features=input_dim),
            w3=Linear.Config(in_features=input_dim, out_features=hidden_dim),
        )
    ).build()
    with torch.no_grad():
        fused.w13.weight.copy_(
            torch.arange(
                fused.w13.weight.numel(),
                dtype=fused.w13.weight.dtype,
            ).reshape_as(fused.w13.weight)
        )
    storage = fused.w13.weight.detach()
    layout = ComputeLayout(
        shardings_by_mesh_axis={
            "dp_shard": BlockShard(dim=0, block_size=2 * hidden_dim),
        },
        matrix_batch=MatrixBatchLayout(
            num_interleaved_matrices=2,
        ),
    )
    view = _matrix_batch_view_from_compute_layout(
        "layers.0.feed_forward.w13.weight",
        storage,
        layout,
    )

    assert view is not None
    matrix_batch = view.view_as_matrix_batch(storage)
    assert matrix_batch.shape == (1, 2, hidden_dim, input_dim)
    torch.testing.assert_close(
        matrix_batch[0],
        storage.unflatten(0, (hidden_dim, 2)).permute(1, 0, 2),
    )
    _assert_independent_ns(matrix_batch)


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4], ids=["mqa", "gqa", "mha"])
def test_fused_qkv_matrix_batch_applies_ns_per_head(num_kv_heads: int) -> None:
    input_dim = 5
    head_dim = 3
    num_heads = 4
    num_fused_heads = num_heads + 2 * num_kv_heads
    fused_qkv = FusedQKVLinear.Config(
        head_dim=head_dim,
        n_heads=num_heads,
        n_kv_heads=num_kv_heads,
        wqkv=Linear.Config(
            in_features=input_dim,
            out_features=num_fused_heads * head_dim,
            bias=False,
        ),
    ).build()
    with torch.no_grad():
        fused_qkv.wqkv.weight.copy_(
            torch.arange(
                fused_qkv.wqkv.weight.numel(),
                dtype=fused_qkv.wqkv.weight.dtype,
            ).reshape_as(fused_qkv.wqkv.weight)
        )
    storage = fused_qkv.wqkv.weight.detach()
    per_head_layout = ComputeLayout(
        shardings_by_mesh_axis={
            "dp_shard": BlockShard(dim=0, block_size=head_dim),
        },
    )
    view = _matrix_batch_view_from_compute_layout(
        "layers.0.attention.wqkv.weight",
        storage,
        per_head_layout,
    )
    assert view is not None
    assert view.transport_shape(torch.Size(storage.shape)) == (
        num_fused_heads,
        head_dim,
        input_dim,
    )

    matrix_batch = view.view_as_matrix_batch(storage)

    assert matrix_batch.shape == (num_fused_heads, head_dim, input_dim)
    torch.testing.assert_close(matrix_batch, storage.view_as(matrix_batch))
    _assert_independent_ns(matrix_batch)


def test_interleaved_matrix_batch_rejects_partial_storage_group() -> None:
    with pytest.raises(ValueError, match="matrix storage groups"):
        _MatrixBatchView.from_storage_shape(
            torch.Size((6, 2)),
            matrix_rows=2,
            num_interleaved_matrices=2,
        )


def test_compute_layout_supports_owned_interleaved_matrix_batch() -> None:
    layout = ComputeLayout(
        shardings_by_mesh_axis={"dp_shard": Owned()},
        matrix_batch=MatrixBatchLayout(
            matrix_rows=4,
            num_interleaved_matrices=2,
        ),
    )

    assert layout.matrix_batch == MatrixBatchLayout(
        matrix_rows=4,
        num_interleaved_matrices=2,
    )
    view = _matrix_batch_view_from_compute_layout(
        "layers.0.feed_forward.w13.weight",
        torch.empty(8, 3),
        layout,
    )
    assert view is not None
    assert view.transport_shape(torch.Size((8, 3))) == (8, 3)
    assert view.matrix_batch_shape(torch.Size((8, 3))) == (1, 2, 4, 3)


def test_interleaved_block_shard_requires_complete_storage_groups() -> None:
    layout = ComputeLayout(
        shardings_by_mesh_axis={
            "dp_shard": BlockShard(dim=0, block_size=4),
        },
        matrix_batch=MatrixBatchLayout(
            matrix_rows=4,
            num_interleaved_matrices=2,
        ),
    )

    with pytest.raises(ValueError, match="storage group size 8"):
        _matrix_batch_view_from_compute_layout(
            "layers.0.feed_forward.w13.weight",
            torch.empty(8, 3),
            layout,
        )


def test_interleaved_block_shard_transports_complete_storage_groups() -> None:
    layout = ComputeLayout(
        shardings_by_mesh_axis={
            "dp_shard": BlockShard(dim=0, block_size=8),
        },
        matrix_batch=MatrixBatchLayout(num_interleaved_matrices=2),
    )

    view = _matrix_batch_view_from_compute_layout(
        "layers.0.feed_forward.w13.weight",
        torch.empty(16, 3),
        layout,
    )

    assert view is not None
    assert view.transport_shape(torch.Size((16, 3))) == (2, 8, 3)
    assert view.matrix_batch_shape(torch.Size((16, 3))) == (2, 2, 4, 3)


def test_interleaved_block_shard_requires_divisible_block_size() -> None:
    layout = ComputeLayout(
        shardings_by_mesh_axis={
            "dp_shard": BlockShard(dim=0, block_size=7),
        },
        matrix_batch=MatrixBatchLayout(num_interleaved_matrices=2),
    )

    with pytest.raises(ValueError, match="block size 7 must be divisible"):
        _matrix_batch_view_from_compute_layout(
            "layers.0.feed_forward.w13.weight",
            torch.empty(14, 3),
            layout,
        )


def test_matrix_batch_layout_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError, match="matrix_rows"):
        MatrixBatchLayout(matrix_rows=0)
    with pytest.raises(ValueError, match="num_interleaved_matrices"):
        MatrixBatchLayout(matrix_rows=4, num_interleaved_matrices=0)


def test_owned_matrix_batch_requires_matrix_rows() -> None:
    layout = ComputeLayout(
        shardings_by_mesh_axis={"dp_shard": Owned()},
        matrix_batch=MatrixBatchLayout(num_interleaved_matrices=2),
    )

    with pytest.raises(
        ValueError,
        match="without BlockShard requires MatrixBatchLayout.matrix_rows",
    ):
        _matrix_batch_view_from_compute_layout(
            "layers.0.feed_forward.w13.weight",
            torch.empty(8, 3),
            layout,
        )


def test_compute_layout_rejects_invalid_matrix_batch() -> None:
    with pytest.raises(ValueError, match="matrix_batch"):
        ComputeLayout(
            shardings_by_mesh_axis={"dp_shard": Owned()},
            matrix_batch="invalid",  # pyrefly: ignore[bad-argument-type]
        )
