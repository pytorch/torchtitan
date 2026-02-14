# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchtitan.components.loss import cross_entropy_loss, IGNORE_INDEX


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


if __name__ == "__main__":
    unittest.main()
