# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock, patch

import torch

from torchtitan.trainer import Trainer


class TestTokensSeen(unittest.TestCase):
    @patch("torchtitan.trainer.prepare_context_parallel_input")
    def test_cp_token_normalization(self, mock_prepare_cp: MagicMock) -> None:
        """
        Verify that ntokens_seen is accumulated *after* context parallel sharding
        inside post_dataloading_process to prevent metric inflation across the loss mesh.
        """
        seq_len = 100
        global_batch_size = 8
        original_labels = torch.zeros(global_batch_size, seq_len)
        actual_tokens_in_batch = original_labels.numel()  # 800

        for cp_degree in [1, 2, 4]:
            # 1. Mock the trainer and its dependencies
            mock_trainer = MagicMock(spec=Trainer)
            mock_trainer.ntokens_seen = 0

            # 2. Mock ParallelDims with our cp_degree
            mock_trainer.parallel_dims = MagicMock()
            mock_trainer.parallel_dims.cp_enabled = cp_degree > 1
            mock_trainer.parallel_dims.cp = cp_degree
            mock_trainer.model_config = MagicMock()
            mock_trainer.model_parts = []
            mock_trainer.config = MagicMock()
            # 3. Setup mock output for context parallel slicing simulation
            expected_normalized_tokens = actual_tokens_in_batch // cp_degree
            # We simulate prepare_context_parallel_input physically slicing the tensor
            sliced_labels = torch.zeros(expected_normalized_tokens)
            mock_prepare_cp.return_value = (torch.zeros(1), sliced_labels, {})

            # Dummy input with correct shape (batch_size, seq_len)
            mock_trainer.device = "cpu"
            input_dict = {"input": torch.zeros(global_batch_size, seq_len)}

            # 4. Call the method where token tracking now natively lives
            Trainer.post_dataloading_process(mock_trainer, input_dict, original_labels)

            # 5. Verify the token counts incremented correctly using the sliced labels
            self.assertEqual(mock_trainer.ntokens_seen, expected_normalized_tokens)

            # 6. Verify mathematically that dist_sum over CP ranks reconstructs actual
            simulated_dist_sum = mock_trainer.ntokens_seen * cp_degree
            self.assertEqual(simulated_dist_sum, actual_tokens_in_batch)


if __name__ == "__main__":
    unittest.main()
