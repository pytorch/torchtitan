# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.distributed.flex_shard import BlockShard, ComputeLayout, Owned
from torchtitan.distributed.flex_shard.dist_muon import (
    _compute_muon_direction,
    _matrix_batch_view_from_compute_layout,
    _zeropower_via_newtonschulz,
)


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


def test_segments_apply_ns_per_segment() -> None:
    num_blocks = 2
    num_rows_per_segment = (3, 2)
    block_rows = sum(num_rows_per_segment)
    matrix_columns = 4
    storage = (
        torch.arange(num_blocks * block_rows * matrix_columns)
        .reshape(num_blocks * block_rows, matrix_columns)
        .float()
    )
    layout = ComputeLayout(
        shardings_by_mesh_axis={"dp_shard": BlockShard(dim=0, block_size=block_rows)},
        num_rows_per_segment=num_rows_per_segment,
    )
    view = _matrix_batch_view_from_compute_layout(
        "layers.0.attention.wkv_b.weight", storage, layout
    )

    assert view is not None
    assert view.matrix_rows == block_rows
    segments = view.segment_matrix_batches(storage)
    assert [tuple(segment.shape) for segment in segments] == [
        (num_blocks, 3, matrix_columns),
        (num_blocks, 2, matrix_columns),
    ]
    blocks = storage.view(num_blocks, block_rows, matrix_columns)
    torch.testing.assert_close(segments[0], blocks[:, :3])
    torch.testing.assert_close(segments[1], blocks[:, 3:])
    for segment in segments:
        _assert_independent_ns(segment)
    # NS wrote through the narrowed views into the flat storage.
    torch.testing.assert_close(storage.view_as(blocks)[:, :3], segments[0])
    torch.testing.assert_close(storage.view_as(blocks)[:, 3:], segments[1])


def test_segments_require_matching_block_size() -> None:
    layout = ComputeLayout(
        shardings_by_mesh_axis={"dp_shard": BlockShard(dim=0, block_size=4)},
        num_rows_per_segment=(3, 2),
    )
    with pytest.raises(ValueError, match="sum of num_rows_per_segment"):
        _matrix_batch_view_from_compute_layout("w", torch.zeros(20, 4), layout)


def test_segments_require_block_shard() -> None:
    layout = ComputeLayout(
        shardings_by_mesh_axis={"dp_shard": Owned()},
        num_rows_per_segment=(3, 2),
    )
    with pytest.raises(ValueError, match="requires a BlockShard"):
        _matrix_batch_view_from_compute_layout("w", torch.zeros(10, 4), layout)


def test_segments_reject_invalid_values() -> None:
    for bad in ((3,), (3, 0), [3, 2], (3, True)):
        with pytest.raises(ValueError, match="num_rows_per_segment"):
            ComputeLayout(
                shardings_by_mesh_axis={"dp_shard": BlockShard(dim=0, block_size=5)},
                num_rows_per_segment=bad,  # type: ignore[arg-type]
            )
