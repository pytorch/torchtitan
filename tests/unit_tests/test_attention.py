# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.models.common.attention import create_varlen_metadata_for_document


def test_single_document():
    # One document per row: positions never reset after 0, so the only boundary
    # is the sequence end.
    positions = torch.arange(50).unsqueeze(0)
    meta = create_varlen_metadata_for_document(positions)
    assert meta.cu_seq_q.tolist() == [0, 50]
    assert meta.cu_seq_k.tolist() == [0, 50]
    assert meta.max_q == 50


def test_packed_documents():
    # Two documents packed in one row (positions restart at 0 at the boundary).
    positions = torch.cat([torch.arange(30), torch.arange(40)]).unsqueeze(0)
    meta = create_varlen_metadata_for_document(positions)
    assert meta.cu_seq_q.tolist() == [0, 30, 70]
    assert meta.max_q == 40


def test_batched_rows_are_flattened():
    # Boundaries are cumulative over the flattened [batch * seq_len] layout, so
    # each row start is also a boundary.
    positions = torch.arange(20).unsqueeze(0).repeat(2, 1)
    meta = create_varlen_metadata_for_document(positions)
    assert meta.cu_seq_q.tolist() == [0, 20, 40]
