# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F
from torchtitan.models.deepseek_v3.model.dsa_attention import (
    DSASparseAttentionWrapper,
    DSAVarlenMetadata,
    DSAVarlenSparseAttention,
    DSAVarlenSparseAttentionOptimized,
)
from torchtitan.models.deepseek_v3.model.dsa_indexer import (
    _generate_hadamard_matrix,
    DSAConfig,
    DSALightIndexer,
)


class TestDSAConfig(unittest.TestCase):
    """Test DSAConfig dataclass."""

    def test_default_values(self):
        """Test that DSAConfig has correct default values."""
        config = DSAConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.indexer_dim, 128)
        self.assertEqual(config.topk, 2048)
        self.assertTrue(config.use_fp8)
        self.assertTrue(config.hadamard_transform)
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.start_layer, 0)
        self.assertTrue(config.combine_with_indexer_scores)

    def test_custom_values(self):
        """Test DSAConfig with custom values."""
        config = DSAConfig(
            enabled=True,
            indexer_dim=64,
            topk=1024,
            use_fp8=False,
            hadamard_transform=False,
            temperature=0.5,
            start_layer=3,
        )
        self.assertTrue(config.enabled)
        self.assertEqual(config.indexer_dim, 64)
        self.assertEqual(config.topk, 1024)
        self.assertFalse(config.use_fp8)
        self.assertFalse(config.hadamard_transform)
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.start_layer, 3)


class TestHadamardMatrix(unittest.TestCase):
    """Test Hadamard matrix generation."""

    def test_hadamard_1x1(self):
        """Test 1x1 Hadamard matrix."""
        h = _generate_hadamard_matrix(1)
        self.assertEqual(h.shape, (1, 1))
        self.assertEqual(h[0, 0].item(), 1.0)

    def test_hadamard_2x2(self):
        """Test 2x2 Hadamard matrix."""
        h = _generate_hadamard_matrix(2)
        self.assertEqual(h.shape, (2, 2))
        expected = torch.tensor([[1, 1], [1, -1]], dtype=torch.float)
        self.assertTrue(torch.allclose(h, expected))

    def test_hadamard_orthogonality(self):
        """Test that normalized Hadamard matrix is orthogonal."""
        for dim in [4, 8, 16, 32]:
            h = _generate_hadamard_matrix(dim)
            h_normalized = h / (dim**0.5)
            # H * H^T should be identity
            product = torch.matmul(h_normalized, h_normalized.T)
            identity = torch.eye(dim)
            self.assertTrue(
                torch.allclose(product, identity, atol=1e-6),
                f"Hadamard matrix of size {dim} is not orthogonal",
            )


class TestDSALightIndexer(unittest.TestCase):
    """Test DSALightIndexer module."""

    def setUp(self):
        """Set up test fixtures."""
        self.q_dim = 256
        self.kv_lora_rank = 512
        self.indexer_dim = 128
        self.n_heads = 8
        self.topk = 64
        self.bsz = 2
        self.seq_len = 128

    def test_output_shapes(self):
        """Test that indexer outputs have correct shapes."""
        indexer = DSALightIndexer(
            q_dim=self.q_dim,
            kv_lora_rank=self.kv_lora_rank,
            indexer_dim=self.indexer_dim,
            n_heads=self.n_heads,
            topk=self.topk,
            use_fp8=False,  # Disable FP8 for CPU testing
            hadamard_transform=True,
        )

        q_compressed = torch.randn(self.bsz, self.seq_len, self.q_dim)
        kv_compressed = torch.randn(self.bsz, self.seq_len, self.kv_lora_rank)

        topk_indices, topk_scores = indexer(q_compressed, kv_compressed)

        self.assertEqual(topk_indices.shape, (self.bsz, self.seq_len, self.topk))
        self.assertEqual(topk_scores.shape, (self.bsz, self.seq_len, self.topk))

    def test_output_shapes_without_hadamard(self):
        """Test indexer without Hadamard transform."""
        indexer = DSALightIndexer(
            q_dim=self.q_dim,
            kv_lora_rank=self.kv_lora_rank,
            indexer_dim=64,  # Non-power-of-2 is okay without Hadamard
            n_heads=self.n_heads,
            topk=self.topk,
            use_fp8=False,
            hadamard_transform=False,
        )

        q_compressed = torch.randn(self.bsz, self.seq_len, self.q_dim)
        kv_compressed = torch.randn(self.bsz, self.seq_len, self.kv_lora_rank)

        topk_indices, topk_scores = indexer(q_compressed, kv_compressed)

        self.assertEqual(topk_indices.shape, (self.bsz, self.seq_len, self.topk))
        self.assertEqual(topk_scores.shape, (self.bsz, self.seq_len, self.topk))

    def test_causal_masking(self):
        """Test that causal masking is applied correctly."""
        indexer = DSALightIndexer(
            q_dim=self.q_dim,
            kv_lora_rank=self.kv_lora_rank,
            indexer_dim=self.indexer_dim,
            n_heads=self.n_heads,
            topk=16,
            use_fp8=False,
            hadamard_transform=True,
        )

        seq_len = 32
        q_compressed = torch.randn(1, seq_len, self.q_dim)
        kv_compressed = torch.randn(1, seq_len, self.kv_lora_rank)

        topk_indices, _ = indexer(q_compressed, kv_compressed)

        # Check that each position only attends to previous positions
        for q_pos in range(seq_len):
            max_valid_kv = q_pos
            selected_indices = topk_indices[0, q_pos, :]
            # All selected indices should be <= q_pos
            self.assertTrue(
                (selected_indices <= q_pos).all(),
                f"Position {q_pos} attends to future position: {selected_indices.max().item()}",
            )

    def test_effective_topk_small_seq(self):
        """Test that effective topk is clamped for small sequences."""
        indexer = DSALightIndexer(
            q_dim=self.q_dim,
            kv_lora_rank=self.kv_lora_rank,
            indexer_dim=self.indexer_dim,
            n_heads=self.n_heads,
            topk=1000,  # Larger than sequence length
            use_fp8=False,
            hadamard_transform=True,
        )

        seq_len = 32  # Smaller than topk
        q_compressed = torch.randn(1, seq_len, self.q_dim)
        kv_compressed = torch.randn(1, seq_len, self.kv_lora_rank)

        topk_indices, topk_scores = indexer(q_compressed, kv_compressed)

        # effective_topk should be min(topk, kv_len) = 32
        self.assertEqual(topk_indices.shape[-1], seq_len)

    def test_scores_sum_to_one(self):
        """Test that topk scores sum to approximately 1 (softmax property)."""
        indexer = DSALightIndexer(
            q_dim=self.q_dim,
            kv_lora_rank=self.kv_lora_rank,
            indexer_dim=self.indexer_dim,
            n_heads=self.n_heads,
            topk=self.topk,
            use_fp8=False,
            hadamard_transform=True,
        )

        q_compressed = torch.randn(self.bsz, self.seq_len, self.q_dim)
        kv_compressed = torch.randn(self.bsz, self.seq_len, self.kv_lora_rank)

        _, topk_scores = indexer(q_compressed, kv_compressed)

        # Scores should sum to 1 along the topk dimension
        score_sums = topk_scores.sum(dim=-1)
        self.assertTrue(
            torch.allclose(score_sums, torch.ones_like(score_sums), atol=1e-5),
            "Softmax scores should sum to 1",
        )

    def test_hadamard_dimension_error(self):
        """Test that non-power-of-2 dimension raises error with Hadamard."""
        with self.assertRaises(ValueError) as context:
            DSALightIndexer(
                q_dim=self.q_dim,
                kv_lora_rank=self.kv_lora_rank,
                indexer_dim=100,  # Not a power of 2
                n_heads=self.n_heads,
                topk=self.topk,
                use_fp8=False,
                hadamard_transform=True,  # This should fail
            )
        self.assertIn("power-of-2", str(context.exception))


class TestDSASparseAttentionWrapper(unittest.TestCase):
    """Test DSASparseAttentionWrapper module."""

    def setUp(self):
        """Set up test fixtures."""
        self.bsz = 2
        self.n_heads = 4
        self.seq_len = 64
        self.head_dim = 32
        self.v_head_dim = 32
        self.topk = 16

    def test_output_shape(self):
        """Test that sparse attention output has correct shape."""
        attn = DSASparseAttentionWrapper()

        q = torch.randn(self.bsz, self.n_heads, self.seq_len, self.head_dim)
        k = torch.randn(self.bsz, self.n_heads, self.seq_len, self.head_dim)
        v = torch.randn(self.bsz, self.n_heads, self.seq_len, self.v_head_dim)
        topk_indices = torch.randint(
            0, self.seq_len, (self.bsz, self.seq_len, self.topk)
        )

        output = attn(q, k, v, topk_indices)

        self.assertEqual(
            output.shape, (self.bsz, self.n_heads, self.seq_len, self.v_head_dim)
        )

    def test_with_topk_scores(self):
        """Test sparse attention with indexer score weighting."""
        attn = DSASparseAttentionWrapper()

        q = torch.randn(self.bsz, self.n_heads, self.seq_len, self.head_dim)
        k = torch.randn(self.bsz, self.n_heads, self.seq_len, self.head_dim)
        v = torch.randn(self.bsz, self.n_heads, self.seq_len, self.v_head_dim)
        topk_indices = torch.randint(
            0, self.seq_len, (self.bsz, self.seq_len, self.topk)
        )
        topk_scores = F.softmax(
            torch.randn(self.bsz, self.seq_len, self.topk), dim=-1
        )

        output = attn(q, k, v, topk_indices, topk_scores=topk_scores)

        self.assertEqual(
            output.shape, (self.bsz, self.n_heads, self.seq_len, self.v_head_dim)
        )

    def test_equivalence_full_selection(self):
        """Test that selecting all positions equals full attention (for small seq)."""
        attn = DSASparseAttentionWrapper()

        bsz = 1
        n_heads = 2
        seq_len = 8
        head_dim = 16

        q = torch.randn(bsz, n_heads, seq_len, head_dim)
        k = torch.randn(bsz, n_heads, seq_len, head_dim)
        v = torch.randn(bsz, n_heads, seq_len, head_dim)

        # Select all positions for each query
        topk_indices = (
            torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(bsz, seq_len, -1)
        )

        sparse_output = attn(q, k, v, topk_indices, scale=head_dim**-0.5)

        # Full attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim**-0.5)
        attn_weights = F.softmax(scores, dim=-1)
        full_output = torch.matmul(attn_weights, v)

        self.assertTrue(
            torch.allclose(sparse_output, full_output, atol=1e-5),
            "Sparse attention with full selection should equal full attention",
        )


class TestDSAVarlenSparseAttention(unittest.TestCase):
    """Test DSAVarlenSparseAttention module."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_heads = 4
        self.head_dim = 32
        self.v_head_dim = 32
        self.topk = 16

    def test_output_shape(self):
        """Test that varlen sparse attention output has correct shape."""
        attn = DSAVarlenSparseAttention()

        total_tokens = 128
        q = torch.randn(total_tokens, self.n_heads, self.head_dim)
        k = torch.randn(total_tokens, self.n_heads, self.head_dim)
        v = torch.randn(total_tokens, self.n_heads, self.v_head_dim)
        topk_indices = torch.randint(0, total_tokens, (total_tokens, self.topk))
        cu_seqlens = torch.tensor([0, 64, 128])

        metadata = DSAVarlenMetadata(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=64,
            max_seqlen_k=64,
            topk_indices=topk_indices,
            topk_scores=None,
        )

        output = attn(q, k, v, metadata)

        self.assertEqual(output.shape, (total_tokens, self.n_heads, self.v_head_dim))

    def test_with_scores(self):
        """Test varlen sparse attention with scores."""
        attn = DSAVarlenSparseAttention()

        total_tokens = 64
        q = torch.randn(total_tokens, self.n_heads, self.head_dim)
        k = torch.randn(total_tokens, self.n_heads, self.head_dim)
        v = torch.randn(total_tokens, self.n_heads, self.v_head_dim)
        topk_indices = torch.randint(0, total_tokens, (total_tokens, self.topk))
        topk_scores = F.softmax(torch.randn(total_tokens, self.topk), dim=-1)
        cu_seqlens = torch.tensor([0, 32, 64])

        metadata = DSAVarlenMetadata(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=32,
            max_seqlen_k=32,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
        )

        output = attn(q, k, v, metadata)

        self.assertEqual(output.shape, (total_tokens, self.n_heads, self.v_head_dim))


class TestDSAVarlenSparseAttentionOptimized(unittest.TestCase):
    """Test DSAVarlenSparseAttentionOptimized module."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_heads = 4
        self.head_dim = 32
        self.v_head_dim = 32
        self.topk = 16

    def test_output_shape(self):
        """Test that optimized varlen sparse attention output has correct shape."""
        attn = DSAVarlenSparseAttentionOptimized()

        total_tokens = 128
        q = torch.randn(total_tokens, self.n_heads, self.head_dim)
        k = torch.randn(total_tokens, self.n_heads, self.head_dim)
        v = torch.randn(total_tokens, self.n_heads, self.v_head_dim)
        # Indices are absolute positions in packed sequence
        topk_indices = torch.randint(0, total_tokens, (total_tokens, self.topk))
        cu_seqlens = torch.tensor([0, 64, 128])

        metadata = DSAVarlenMetadata(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=64,
            max_seqlen_k=64,
            topk_indices=topk_indices,
            topk_scores=None,
        )

        output = attn(q, k, v, metadata)

        self.assertEqual(output.shape, (total_tokens, self.n_heads, self.v_head_dim))

    def test_multiple_documents(self):
        """Test with multiple documents of varying lengths."""
        attn = DSAVarlenSparseAttentionOptimized()

        # 3 documents: 32, 48, 20 tokens
        cu_seqlens = torch.tensor([0, 32, 80, 100])
        total_tokens = 100

        q = torch.randn(total_tokens, self.n_heads, self.head_dim)
        k = torch.randn(total_tokens, self.n_heads, self.head_dim)
        v = torch.randn(total_tokens, self.n_heads, self.v_head_dim)
        topk_indices = torch.randint(0, total_tokens, (total_tokens, self.topk))

        metadata = DSAVarlenMetadata(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=48,
            max_seqlen_k=48,
            topk_indices=topk_indices,
            topk_scores=None,
        )

        output = attn(q, k, v, metadata)

        self.assertEqual(output.shape, (total_tokens, self.n_heads, self.v_head_dim))
        # Output should not contain NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestDSAIntegration(unittest.TestCase):
    """Integration tests for DSA components working together."""

    def test_indexer_to_attention_pipeline(self):
        """Test full pipeline from indexer to sparse attention."""
        # Setup dimensions
        bsz = 2
        seq_len = 64
        q_dim = 256
        kv_lora_rank = 512
        indexer_dim = 128
        n_heads = 8
        head_dim = 32
        v_head_dim = 32
        topk = 16

        # Create indexer
        indexer = DSALightIndexer(
            q_dim=q_dim,
            kv_lora_rank=kv_lora_rank,
            indexer_dim=indexer_dim,
            n_heads=n_heads,
            topk=topk,
            use_fp8=False,
            hadamard_transform=True,
        )

        # Create sparse attention
        sparse_attn = DSASparseAttentionWrapper()

        # Create inputs
        q_compressed = torch.randn(bsz, seq_len, q_dim)
        kv_compressed = torch.randn(bsz, seq_len, kv_lora_rank)

        # Get indices from indexer
        topk_indices, topk_scores = indexer(q_compressed, kv_compressed)

        # Create full Q, K, V for attention
        q = torch.randn(bsz, n_heads, seq_len, head_dim)
        k = torch.randn(bsz, n_heads, seq_len, head_dim)
        v = torch.randn(bsz, n_heads, seq_len, v_head_dim)

        # Run sparse attention with indices from indexer
        output = sparse_attn(q, k, v, topk_indices, topk_scores)

        # Verify output
        self.assertEqual(output.shape, (bsz, n_heads, seq_len, v_head_dim))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestDSACPModules(unittest.TestCase):
    """Test DSA CP modules (basic unit tests without distributed environment)."""

    def test_dsa_context_parallel_wrapper_init(self):
        """Test that DSAContextParallelWrapper initializes correctly."""
        from torchtitan.models.deepseek_v3.model.dsa_cp_attention import (
            DSAContextParallelWrapper,
        )

        # Create indexer
        indexer = DSALightIndexer(
            q_dim=256,
            kv_lora_rank=512,
            indexer_dim=128,
            n_heads=8,
            topk=64,
            use_fp8=False,
            hadamard_transform=True,
        )

        # Create CP wrapper
        cp_wrapper = DSAContextParallelWrapper(
            indexer=indexer,
            combine_with_indexer_scores=True,
        )

        # Verify wrapper has correct attributes
        self.assertIsNotNone(cp_wrapper.ring_indexer)
        self.assertIsNotNone(cp_wrapper.sparse_attention)
        self.assertTrue(cp_wrapper.combine_scores)

    def test_dsa_ring_indexer_init(self):
        """Test that DSARingIndexer wraps indexer correctly."""
        from torchtitan.models.deepseek_v3.model.dsa_cp_attention import DSARingIndexer

        # Create indexer
        indexer = DSALightIndexer(
            q_dim=256,
            kv_lora_rank=512,
            indexer_dim=128,
            n_heads=8,
            topk=64,
            use_fp8=False,
            hadamard_transform=True,
        )

        # Create ring indexer
        ring_indexer = DSARingIndexer(indexer=indexer)

        # Verify it wraps the indexer
        self.assertIs(ring_indexer.indexer, indexer)

    def test_dsa_sparse_attention_cp_init(self):
        """Test that DSASparseAttentionCP initializes correctly."""
        from torchtitan.models.deepseek_v3.model.dsa_cp_attention import (
            DSASparseAttentionCP,
        )

        # Should be able to create without any arguments
        sparse_attn = DSASparseAttentionCP()
        self.assertIsNotNone(sparse_attn)

    def test_dsa_cp_metadata_namedtuple(self):
        """Test DSACPMetadata namedtuple structure."""
        from torchtitan.models.deepseek_v3.model.dsa_cp_attention import DSACPMetadata

        # Create a mock metadata (without actual DeviceMesh)
        metadata = DSACPMetadata(
            cp_mesh=None,  # Would be DeviceMesh in real usage
            local_seq_len=128,
            global_seq_len=512,
            cp_rank=0,
            cp_world_size=4,
        )

        self.assertEqual(metadata.local_seq_len, 128)
        self.assertEqual(metadata.global_seq_len, 512)
        self.assertEqual(metadata.cp_rank, 0)
        self.assertEqual(metadata.cp_world_size, 4)


if __name__ == "__main__":
    unittest.main()
