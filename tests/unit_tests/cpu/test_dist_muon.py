# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import pytest
import torch

from torchtitan.distributed.flex_shard.dist_muon import _MatrixBatchView
from torchtitan.distributed.flex_shard.optimizer_reshard import ComputeLayout, Owned


def test_interleaved_matrix_batch_view_round_trips_gate_and_up() -> None:
    hidden_dim = 3
    input_dim = 2
    storage = torch.arange(2 * hidden_dim * input_dim, dtype=torch.float32).reshape(
        2 * hidden_dim, input_dim
    )
    view = _MatrixBatchView.from_storage_shape(
        torch.Size(storage.shape),
        matrix_rows=hidden_dim,
        interleaved=True,
    )

    matrix_batch = view.view_as_matrix_batch(storage)

    torch.testing.assert_close(matrix_batch[0], storage[0::2])
    torch.testing.assert_close(matrix_batch[1], storage[1::2])
    torch.testing.assert_close(view.view_as_storage(matrix_batch), storage)


def test_interleaved_matrix_batch_view_rejects_non_pair_shape() -> None:
    with pytest.raises(ValueError, match="cannot be partitioned"):
        _MatrixBatchView.from_storage_shape(
            torch.Size((6, 2)),
            matrix_rows=2,
            interleaved=True,
        )


def test_compute_layout_declares_supported_matrix_layouts() -> None:
    layout = ComputeLayout(
        shardings_by_mesh_axis={"dp_shard": Owned()},
        matrix_layout="interleaved_gated",
    )

    assert layout.matrix_layout == "interleaved_gated"


def test_compute_layout_rejects_unknown_matrix_layout() -> None:
    with pytest.raises(ValueError, match="matrix_layout"):
        ComputeLayout(
            shardings_by_mesh_axis={"dp_shard": Owned()},
            matrix_layout="unknown",
        )
