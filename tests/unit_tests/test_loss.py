# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
from torchtitan.components.loss import (
    ChunkedCELoss,
    cross_entropy_loss,
    GradAccumulator,
    IGNORE_INDEX,
)


class TestLoss(unittest.TestCase):
    def test_ignore_index_equal_per_token_contribution(self):
        """Test that each valid token contributes equally to the loss.

        This test verifies that:
        1. Tokens marked with IGNORE_INDEX don't contribute to the loss
        2. Each valid token contributes the same amount to the total loss
        3. The sum-based loss calculation is correct for token normalization
        """
        torch.manual_seed(42)
        batch_size = 4
        seq_len = 8
        vocab_size = 100

        # Create predictions (logits) - same for all test cases
        predictions = torch.randn(batch_size, seq_len, vocab_size)

        # Create base labels with some tokens as IGNORE_INDEX, others valid
        # This ensures we test on the same subset of valid tokens
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Mark specific positions as IGNORE_INDEX
        labels[0, 1] = IGNORE_INDEX
        labels[1, 3] = IGNORE_INDEX
        labels[2, 5] = IGNORE_INDEX
        labels[3, 7] = IGNORE_INDEX

        # Test case 1: Compute loss on this label set
        loss1 = cross_entropy_loss(predictions, labels)
        num_valid_tokens1 = (labels != IGNORE_INDEX).sum().item()

        # Test case 2: Use the exact same predictions and labels in multiple microbatches
        # Simulating gradient accumulation with identical data
        loss2 = cross_entropy_loss(predictions, labels) + cross_entropy_loss(
            predictions, labels
        )
        num_valid_tokens2 = num_valid_tokens1 * 2

        # Per-token loss should be identical
        per_token_loss1 = loss1 / num_valid_tokens1
        per_token_loss2 = loss2 / num_valid_tokens2

        self.assertAlmostEqual(
            per_token_loss1.item(),
            per_token_loss2.item(),
            places=6,
            msg="Per-token loss should be the same across gradient accumulation steps",
        )

        # Test case 3: Verify loss scaling with batch replication
        # Stack the batch to create 2x the data
        predictions_doubled = torch.cat([predictions, predictions], dim=0)
        labels_doubled = torch.cat([labels, labels], dim=0)

        loss_doubled = cross_entropy_loss(predictions_doubled, labels_doubled)
        num_valid_tokens_doubled = (labels_doubled != IGNORE_INDEX).sum().item()

        per_token_loss_doubled = loss_doubled / num_valid_tokens_doubled

        self.assertAlmostEqual(
            per_token_loss1.item(),
            per_token_loss_doubled.item(),
            places=6,
            msg="Per-token loss should remain constant when scaling batch size",
        )

        # Verify that total loss scales linearly with number of valid tokens
        expected_ratio = num_valid_tokens_doubled / num_valid_tokens1
        actual_ratio = loss_doubled / loss1

        self.assertAlmostEqual(
            expected_ratio,
            actual_ratio.item(),
            places=6,
            msg=f"Loss should scale linearly with valid token count. "
            f"Expected ratio: {expected_ratio}, got: {actual_ratio.item()}",
        )

    def test_ignore_index_gradient_accumulation_consistency(self):
        """Test that loss is consistent across gradient accumulation steps.

        This simulates the scenario where we have:
        - Multiple microbatches with different numbers of valid tokens
        - Total loss should equal sum of individual losses
        - Per-token loss should be consistent
        """
        torch.manual_seed(123)
        vocab_size = 100

        # Microbatch 1: 2x4 with all valid tokens
        pred1 = torch.randn(2, 4, vocab_size)
        labels1 = torch.randint(0, vocab_size, (2, 4))
        loss1 = cross_entropy_loss(pred1, labels1)
        tokens1 = (labels1 != IGNORE_INDEX).sum()

        # Microbatch 2: 2x4 with half valid tokens
        pred2 = torch.randn(2, 4, vocab_size)
        labels2 = torch.randint(0, vocab_size, (2, 4))
        labels2[:, ::2] = IGNORE_INDEX  # Mask every other token
        loss2 = cross_entropy_loss(pred2, labels2)
        tokens2 = (labels2 != IGNORE_INDEX).sum()

        # Microbatch 3: 2x4 with 1/4 valid tokens
        pred3 = torch.randn(2, 4, vocab_size)
        labels3 = torch.randint(0, vocab_size, (2, 4))
        labels3[:, 1:] = IGNORE_INDEX  # Only first token valid
        loss3 = cross_entropy_loss(pred3, labels3)
        tokens3 = (labels3 != IGNORE_INDEX).sum()

        # Simulate gradient accumulation: sum losses, sum tokens, then normalize
        total_loss = loss1 + loss2 + loss3
        total_tokens = tokens1 + tokens2 + tokens3
        global_avg_loss = total_loss / total_tokens

        # Verify this equals the average of individual per-token losses weighted by token count
        weighted_avg = (
            (loss1 / tokens1) * tokens1
            + (loss2 / tokens2) * tokens2
            + (loss3 / tokens3) * tokens3
        ) / total_tokens

        self.assertAlmostEqual(
            global_avg_loss.item(),
            weighted_avg.item(),
            places=5,
            msg="Global averaged loss should equal weighted average of per-token losses",
        )


class TestGradAccumulator(unittest.TestCase):
    def test_accumulate_matches_cat(self):
        """Verify GradAccumulator produces the same result as torch.cat."""
        torch.manual_seed(42)
        B, L, D = 2, 8, 16
        num_chunks = 4
        reference = torch.randn(B, L, D)
        chunks = torch.chunk(reference, num_chunks, dim=1)

        acc = GradAccumulator(reference, num_chunks=num_chunks, seq_dim=1)
        for chunk in chunks:
            acc.add(chunk)

        result = acc.result()
        torch.testing.assert_close(result, reference)

    def test_accumulate_with_dtype_conversion(self):
        """Verify fp32 accumulation from bf16 chunks."""
        torch.manual_seed(42)
        B, L, D = 2, 8, 16
        num_chunks = 4
        reference = torch.randn(B, L, D)
        bf16_chunks = [c.bfloat16() for c in torch.chunk(reference, num_chunks, dim=1)]

        acc = GradAccumulator(reference, num_chunks=num_chunks, dtype=torch.float32)
        for chunk in bf16_chunks:
            acc.add(chunk)

        result = acc.result()
        self.assertEqual(result.dtype, torch.float32)
        # Verify values match (allowing for bf16 precision loss)
        expected = torch.cat([c.float() for c in bf16_chunks], dim=1)
        torch.testing.assert_close(result, expected)

    def test_too_many_adds_raises(self):
        """Verify error when adding more chunks than expected."""
        reference = torch.randn(2, 8, 16)
        acc = GradAccumulator(reference, num_chunks=2)
        acc.add(torch.randn(2, 4, 16))
        acc.add(torch.randn(2, 4, 16))
        with self.assertRaises(ValueError):
            acc.add(torch.randn(2, 4, 16))


class _FakeDecoder(nn.Module):
    """Minimal Decoder-like model for testing ChunkedCELoss."""

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.output = nn.Linear(dim, vocab_size, bias=False)
        # Make it look like a Decoder to ChunkedCELoss
        self.layers = nn.ModuleDict()
        self.tok_embeddings = None
        self.norm = None

    def forward(self, tokens, skip_lm_head=False):
        if skip_lm_head:
            return tokens  # return hidden states directly
        return self.output(tokens)


class TestChunkedCELoss(unittest.TestCase):
    def _make_model_and_loss(self, dim=32, vocab_size=64, num_chunks=4):
        """Create a fake Decoder and ChunkedCELoss for testing."""
        from torchtitan.models.common.decoder import Decoder

        model = _FakeDecoder(dim, vocab_size)
        # Patch class check: ChunkedCELoss asserts isinstance(model, Decoder)
        # For unit testing without full model infra, we monkey-patch.
        original_init = ChunkedCELoss.__init__

        def patched_init(self_loss, m, num_chunks, loss_fn):
            assert hasattr(m, "output") and m.output is not None
            self_loss.model = m
            self_loss.lm_head = m.output
            self_loss.num_chunks = num_chunks
            self_loss.loss_fn = loss_fn

        ChunkedCELoss.__init__ = patched_init
        chunked_loss = ChunkedCELoss(model, num_chunks=num_chunks, loss_fn=cross_entropy_loss)
        ChunkedCELoss.__init__ = original_init
        return model, chunked_loss

    def test_numerical_equivalence(self):
        """ChunkedCELoss must produce the same loss and gradients as the standard path."""
        torch.manual_seed(42)
        B, L, D, V = 2, 8, 32, 64
        num_chunks = 4

        model_std, _ = self._make_model_and_loss(D, V, num_chunks)
        model_chunked, chunked_loss = self._make_model_and_loss(D, V, num_chunks)

        # Share the same lm_head weights
        model_chunked.output.load_state_dict(model_std.output.state_dict())

        hidden_states = torch.randn(B, L, D, requires_grad=True)
        labels = torch.randint(0, V, (B, L))
        labels[0, 1] = IGNORE_INDEX
        labels[1, 3] = IGNORE_INDEX
        global_valid_tokens = (labels != IGNORE_INDEX).sum().float()

        # Standard path: lm_head + ce_loss + backward
        hidden_std = hidden_states.detach().requires_grad_(True)
        logits_std = model_std.output(hidden_std)
        loss_std = cross_entropy_loss(logits_std, labels)
        scaled_loss_std = loss_std / global_valid_tokens
        scaled_loss_std.backward()
        grad_std = hidden_std.grad.clone()
        lm_head_grad_std = model_std.output.weight.grad.clone()

        # Chunked path
        hidden_chunked = hidden_states.detach().requires_grad_(True)
        loss_chunked = chunked_loss(hidden_chunked, labels, global_valid_tokens)
        grad_chunked = hidden_chunked.grad.clone()
        lm_head_grad_chunked = model_chunked.output.weight.grad.clone()

        # Verify loss values match
        torch.testing.assert_close(
            loss_chunked, scaled_loss_std, atol=1e-5, rtol=1e-5,
            msg="Chunked and standard loss values should match",
        )

        # Verify hidden state gradients match
        torch.testing.assert_close(
            grad_chunked.float(), grad_std.float(), atol=1e-5, rtol=1e-5,
            msg="Chunked and standard hidden state gradients should match",
        )

        # Verify lm_head weight gradients match
        torch.testing.assert_close(
            lm_head_grad_chunked.float(), lm_head_grad_std.float(),
            atol=1e-5, rtol=1e-5,
            msg="Chunked and standard lm_head gradients should match",
        )

    def test_different_chunk_counts(self):
        """Loss should be the same regardless of num_chunks."""
        torch.manual_seed(42)
        B, L, D, V = 2, 8, 32, 64
        labels = torch.randint(0, V, (B, L))
        global_valid_tokens = (labels != IGNORE_INDEX).sum().float()
        hidden_states = torch.randn(B, L, D)

        losses = []
        for num_chunks in [1, 2, 4, 8]:
            model, chunked_loss = self._make_model_and_loss(D, V, num_chunks)
            # Use same lm_head weights
            if losses:
                model.output.load_state_dict(ref_state_dict)
            else:
                ref_state_dict = model.output.state_dict()

            h = hidden_states.detach().requires_grad_(True)
            loss = chunked_loss(h, labels, global_valid_tokens)
            losses.append(loss.item())

        for i in range(1, len(losses)):
            self.assertAlmostEqual(
                losses[0], losses[i], places=5,
                msg=f"Loss with {2**i} chunks should match loss with 1 chunk",
            )


if __name__ == "__main__":
    unittest.main()
