# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for attention sink LSE renormalization.

The LSE renormalization approach should be mathematically equivalent to the
HuggingFace implementation which:
1. Concatenates sink logits to attention logits
2. Applies softmax over K+1 positions (including sink)
3. Drops the sink position after softmax (absorbing probability mass)
"""

import pytest
import torch


def lse_renormalization(output: torch.Tensor, lse: torch.Tensor, sinks: torch.Tensor) -> torch.Tensor:
    """
    Apply attention sinks using LSE renormalization.

    This is the implementation from model.py - extracted for testing.

    Args:
        output: Attention output [B, H, Q, D]
        lse: Log-sum-exp from attention [B, H, Q]
        sinks: Per-head sink weights [H]

    Returns:
        Rescaled output [B, H, Q, D]
    """
    batch_size, num_heads, seq_len_q, head_dim = output.shape

    # Expand dimensions for broadcasting
    lse_expanded = lse.unsqueeze(-1)  # [B, H, Q, 1]
    sinks_expanded = sinks.view(1, -1, 1, 1).expand(
        batch_size, num_heads, seq_len_q, 1
    )

    # Compute combined LSE that includes the sink
    combined_lse = torch.logsumexp(
        torch.cat([lse_expanded, sinks_expanded], dim=-1), dim=-1, keepdim=True
    )

    # Renormalization factor: exp(old_lse - new_lse)
    renorm_factor = torch.exp(
        torch.clamp(lse_expanded - combined_lse, min=-20.0, max=0.0)
    )
    return output * renorm_factor


def hf_concat_softmax_reference(
    scores: torch.Tensor, values: torch.Tensor, sinks: torch.Tensor
) -> torch.Tensor:
    """
    HuggingFace-style reference implementation using concat+softmax.

    This explicitly concatenates sink to attention scores, applies softmax
    over K+1 positions, and drops the sink column after weighting.

    Args:
        scores: Raw attention scores [B, H, Q, K]
        values: Value tensor [B, H, K, D]
        sinks: Per-head sink weights [H]

    Returns:
        Attention output [B, H, Q, D]
    """
    batch_size, num_heads, seq_len_q, _ = scores.shape

    # Expand sinks to [B, H, Q, 1] for concatenation
    sinks_expanded = sinks.view(1, -1, 1, 1).expand(batch_size, num_heads, seq_len_q, 1)

    # Concatenate sink logits to attention scores: [B, H, Q, K+1]
    scores_with_sink = torch.cat([scores, sinks_expanded], dim=-1)

    # Softmax over K+1 positions
    probs_with_sink = torch.softmax(scores_with_sink, dim=-1)

    # Extract probabilities for real keys (drop sink column)
    probs = probs_with_sink[..., :-1]  # [B, H, Q, K]

    # Weighted sum of values
    output = torch.matmul(probs, values)  # [B, H, Q, D]
    return output


class TestAttentionSinkLSE:
    """Test suite for attention sink LSE renormalization."""

    @pytest.fixture
    def setup_tensors(self):
        """Create test tensors with reasonable attention values."""
        torch.manual_seed(42)
        batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 16

        # Random attention scores (pre-softmax)
        scores = torch.randn(batch_size, num_heads, seq_len, seq_len)

        # Random values
        values = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Per-head sink weights (typically small negative to positive)
        sinks = torch.randn(num_heads) * 2  # Range roughly [-6, 6]

        return scores, values, sinks

    def test_equivalence_with_hf_reference(self, setup_tensors):
        """Verify LSE renormalization matches HuggingFace concat+softmax."""
        scores, values, sinks = setup_tensors

        # HuggingFace reference implementation
        hf_output = hf_concat_softmax_reference(scores, values, sinks)

        # Our LSE renormalization implementation
        # First compute standard attention
        probs = torch.softmax(scores, dim=-1)
        lse = torch.logsumexp(scores, dim=-1)  # [B, H, Q]
        standard_output = torch.matmul(probs, values)  # [B, H, Q, D]

        # Apply LSE renormalization
        lse_output = lse_renormalization(standard_output, lse, sinks)

        # Should match within numerical tolerance
        torch.testing.assert_close(lse_output, hf_output, rtol=1e-5, atol=1e-5)

    def test_probability_mass_preserved(self, setup_tensors):
        """Verify that renormalization doesn't increase total probability mass."""
        scores, values, sinks = setup_tensors

        # Compute LSE for renormalization
        lse = torch.logsumexp(scores, dim=-1)

        # Compute renorm factor
        batch_size, num_heads, seq_len_q = lse.shape
        lse_expanded = lse.unsqueeze(-1)
        sinks_expanded = sinks.view(1, -1, 1, 1).expand(batch_size, num_heads, seq_len_q, 1)
        combined_lse = torch.logsumexp(
            torch.cat([lse_expanded, sinks_expanded], dim=-1), dim=-1, keepdim=True
        )
        renorm_factor = torch.exp(lse_expanded - combined_lse)

        # Renorm factor should always be <= 1 (probability can only decrease)
        assert (renorm_factor <= 1.0 + 1e-6).all(), "Renorm factor should not increase probability"

    def test_edge_case_very_negative_sinks(self, setup_tensors):
        """Test with very negative sinks (should absorb almost no probability)."""
        scores, values, sinks = setup_tensors
        sinks = torch.full_like(sinks, -100.0)  # Very negative

        hf_output = hf_concat_softmax_reference(scores, values, sinks)
        probs = torch.softmax(scores, dim=-1)
        lse = torch.logsumexp(scores, dim=-1)
        standard_output = torch.matmul(probs, values)
        lse_output = lse_renormalization(standard_output, lse, sinks)

        # With very negative sinks, output should be nearly unchanged
        torch.testing.assert_close(lse_output, standard_output, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(lse_output, hf_output, rtol=1e-4, atol=1e-4)

    def test_edge_case_very_positive_sinks(self, setup_tensors):
        """Test with very positive sinks (should absorb most probability)."""
        scores, values, sinks = setup_tensors
        sinks = torch.full_like(sinks, 100.0)  # Very positive

        hf_output = hf_concat_softmax_reference(scores, values, sinks)
        probs = torch.softmax(scores, dim=-1)
        lse = torch.logsumexp(scores, dim=-1)
        standard_output = torch.matmul(probs, values)
        lse_output = lse_renormalization(standard_output, lse, sinks)

        # With very positive sinks, output should be near zero (due to clamping)
        # Note: Our implementation clamps at exp(-20) ≈ 2e-9
        assert (lse_output.abs() < 1e-8).all(), "Very positive sinks should absorb all attention"
        torch.testing.assert_close(lse_output, hf_output, rtol=1e-4, atol=1e-6)

    def test_edge_case_zero_sinks(self, setup_tensors):
        """Test with zero sinks."""
        scores, values, sinks = setup_tensors
        sinks = torch.zeros_like(sinks)

        hf_output = hf_concat_softmax_reference(scores, values, sinks)
        probs = torch.softmax(scores, dim=-1)
        lse = torch.logsumexp(scores, dim=-1)
        standard_output = torch.matmul(probs, values)
        lse_output = lse_renormalization(standard_output, lse, sinks)

        torch.testing.assert_close(lse_output, hf_output, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
