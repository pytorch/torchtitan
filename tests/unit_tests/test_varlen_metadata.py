# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.models.common.attention import create_varlen_metadata_for_document


def test_varlen_metadata_uses_fixed_sequence_capacity() -> None:
    positions = torch.tensor(
        [
            [0, 1, 0, 1],
            [0, 0, 1, 2],
        ]
    )

    metadata = create_varlen_metadata_for_document(
        positions,
        include_host_offsets=True,
        max_sequences_per_sample=3,
    )

    expected_offsets = torch.tensor([0, 2, 4, 5, 8, 8, 8], dtype=torch.int32)
    torch.testing.assert_close(metadata.cu_seq_q, expected_offsets)
    assert metadata.cu_seq_k is metadata.cu_seq_q
    assert metadata.cu_seq_q_host == (0, 2, 4, 5, 8, 8, 8)
    assert metadata.max_q == positions.shape[1]
    assert metadata.max_k == positions.shape[1]


def test_varlen_metadata_capacity_keeps_shape_across_packing_patterns() -> None:
    positions_a = torch.tensor([[0, 1, 0, 1], [0, 1, 2, 3]])
    positions_b = torch.tensor([[0, 0, 1, 0], [0, 1, 0, 1]])

    metadata_a = create_varlen_metadata_for_document(
        positions_a, max_sequences_per_sample=3
    )
    metadata_b = create_varlen_metadata_for_document(
        positions_b, max_sequences_per_sample=3
    )

    assert metadata_a.cu_seq_q.shape == metadata_b.cu_seq_q.shape == (7,)
    assert metadata_a.max_q == metadata_b.max_q == positions_a.shape[1]


def test_varlen_metadata_rejects_sequence_capacity_overflow() -> None:
    positions = torch.zeros((1, 4), dtype=torch.int64)

    with pytest.raises(
        ValueError,
        match=r"sample 0 contains 4 sequences.*max_packed_sequences_per_sample=3",
    ):
        create_varlen_metadata_for_document(
            positions,
            max_sequences_per_sample=3,
        )
