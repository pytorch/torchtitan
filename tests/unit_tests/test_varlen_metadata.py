# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for create_varlen_metadata_for_document function.

Verifies that both position-based and EOS-based boundary detection
produce correct cumulative sequence lengths.
"""

import unittest

import torch

from torchtitan.models.attention import create_varlen_metadata_for_document, VarlenMetadata


class TestCreateVarlenMetadataForDocument(unittest.TestCase):
    """Test create_varlen_metadata_for_document with positions and input_ids+eos_id."""

    def test_single_document_with_positions(self):
        """Single document per batch item - positions are sequential."""
        # positions: [0, 1, 2, 3, 4] - single document
        positions = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64)

        metadata = create_varlen_metadata_for_document(positions=positions)

        # Expected: one document of length 5
        # cu_seqlens: [0, 5]
        self.assertIsInstance(metadata, VarlenMetadata)
        expected_cu_seqlens = torch.tensor([0, 5], dtype=torch.int32)
        self.assertTrue(
            torch.equal(metadata.cu_seq_q, expected_cu_seqlens),
            f"Expected {expected_cu_seqlens}, got {metadata.cu_seq_q}"
        )
        self.assertTrue(torch.equal(metadata.cu_seq_k, expected_cu_seqlens))
        self.assertEqual(metadata.max_q, 5)
        self.assertEqual(metadata.max_k, 5)

    def test_two_documents_with_positions(self):
        """Two documents detected via position reset."""
        # positions: [0, 1, 2, 0, 1] - doc1 has 3 tokens, doc2 has 2 tokens
        positions = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.int64)

        metadata = create_varlen_metadata_for_document(positions=positions)

        # Expected: two documents, lengths 3 and 2
        # cu_seqlens: [0, 3, 5]
        expected_cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
        assert torch.equal(metadata.cu_seq_q, expected_cu_seqlens)
        assert torch.equal(metadata.cu_seq_k, expected_cu_seqlens)
        assert metadata.max_q == 3
        assert metadata.max_k == 3

    def test_single_document_with_eos(self):
        """Single document with EOS at the end."""
        # input_ids: [1, 2, 3, 4, EOS] where EOS=0
        input_ids = torch.tensor([[1, 2, 3, 4, 0]], dtype=torch.int64)
        eos_id = 0

        metadata = create_varlen_metadata_for_document(input_ids=input_ids, eos_id=eos_id)

        # Expected: one document of length 5
        # cu_seqlens: [0, 5]
        expected_cu_seqlens = torch.tensor([0, 5], dtype=torch.int32)
        assert torch.equal(metadata.cu_seq_q, expected_cu_seqlens)
        assert metadata.max_q == 5

    def test_two_documents_with_eos(self):
        """Two documents separated by EOS token."""
        # input_ids: [1, 2, EOS, 3, 4, EOS] where EOS=0
        # doc1: [1, 2, EOS], doc2: [3, 4, EOS]
        input_ids = torch.tensor([[1, 2, 0, 3, 4, 0]], dtype=torch.int64)
        eos_id = 0

        metadata = create_varlen_metadata_for_document(input_ids=input_ids, eos_id=eos_id)

        # Expected: two documents, lengths 3 and 3
        # cu_seqlens: [0, 3, 6]
        expected_cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
        assert torch.equal(metadata.cu_seq_q, expected_cu_seqlens)
        assert metadata.max_q == 3

    def test_batch_size_two_with_positions(self):
        """Batch of 2 samples with different document structures."""
        # Sample 0: [0, 1, 2, 3, 4] - one document of length 5
        # Sample 1: [0, 1, 0, 1, 2] - two documents of lengths 2 and 3
        positions = torch.tensor([
            [0, 1, 2, 3, 4],
            [0, 1, 0, 1, 2],
        ], dtype=torch.int64)

        metadata = create_varlen_metadata_for_document(positions=positions)

        # Expected cu_seqlens for packed format:
        # Sample 0 contributes: starts at 0, length 5
        # Sample 1 contributes: starts at 5 (offset), doc1 starts at 5, doc2 starts at 7
        expected_cu_seqlens = torch.tensor([0, 5, 7, 10], dtype=torch.int32)
        assert torch.equal(metadata.cu_seq_q, expected_cu_seqlens), \
            f"Expected {expected_cu_seqlens}, got {metadata.cu_seq_q}"
        assert metadata.max_q == 5  # max of [5, 2, 3]

    def test_batch_size_two_with_eos(self):
        """Batch of 2 samples with different document structures using EOS."""
        # Sample 0: [1, 2, 3, 4, EOS] - one document
        # Sample 1: [1, EOS, 2, 3, EOS] - two documents
        input_ids = torch.tensor([
            [1, 2, 3, 4, 0],
            [1, 0, 2, 3, 0],
        ], dtype=torch.int64)
        eos_id = 0

        metadata = create_varlen_metadata_for_document(input_ids=input_ids, eos_id=eos_id)

        # Sample 0: one doc of length 5, cu_seqlens contribution: [0]
        # Sample 1: doc1 length 2, doc2 length 3, cu_seqlens contribution: [5, 7]
        expected_cu_seqlens = torch.tensor([0, 5, 7, 10], dtype=torch.int32)
        assert torch.equal(metadata.cu_seq_q, expected_cu_seqlens), \
            f"Expected {expected_cu_seqlens}, got {metadata.cu_seq_q}"

    def test_positions_and_eos_produce_same_result(self):
        """Verify that positions and EOS-based detection produce equivalent results."""
        # Create equivalent inputs
        # Two documents: lengths 3 and 2
        # Positions: [0, 1, 2, 0, 1]
        # EOS-based: [A, B, EOS, C, D] where EOS marks end of doc1

        positions = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.int64)
        input_ids = torch.tensor([[1, 2, 0, 3, 4]], dtype=torch.int64)  # 0 is EOS
        eos_id = 0

        metadata_positions = create_varlen_metadata_for_document(positions=positions)
        metadata_eos = create_varlen_metadata_for_document(input_ids=input_ids, eos_id=eos_id)

        assert torch.equal(metadata_positions.cu_seq_q, metadata_eos.cu_seq_q), \
            f"Position-based: {metadata_positions.cu_seq_q}, EOS-based: {metadata_eos.cu_seq_q}"
        assert metadata_positions.max_q == metadata_eos.max_q


if __name__ == "__main__":
    unittest.main()
