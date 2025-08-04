# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
from torchtitan.components.loss import (
    is_liger_kernel_enabled,
    liger_fused_linear_cross_entropy_loss,
    cross_entropy_loss,
    LIGER_KERNEL_AVAILABLE,
)
from torchtitan.config import JobConfig


class TestLigerKernel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 4
        self.hidden_dim = 8
        self.vocab_size = 16
        
        # Check if CUDA is available for GPU tests
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create test tensors (on GPU if available, CPU otherwise)
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device)
        self.weight = torch.randn(self.vocab_size, self.hidden_dim, device=self.device)
        self.target = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        
        # Create job config
        self.job_config = JobConfig()

    def test_is_liger_kernel_enabled_default(self):
        """Test that liger kernel is disabled by default."""
        self.assertFalse(is_liger_kernel_enabled(self.job_config))
    
    def test_is_liger_kernel_enabled_when_enabled(self):
        """Test that liger kernel detection works when enabled."""
        self.job_config.liger_kernel.enable_fused_linear_cross_entropy = True
        self.assertTrue(is_liger_kernel_enabled(self.job_config))

    @unittest.skipIf(not LIGER_KERNEL_AVAILABLE, "Liger-Kernel not available")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_liger_fused_loss_shapes(self):
        """Test that liger fused loss handles tensor shapes correctly."""
        loss = liger_fused_linear_cross_entropy_loss(
            self.weight, self.hidden_states, self.target
        )
        
        # Loss should be a scalar tensor
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss))

    @unittest.skipIf(not LIGER_KERNEL_AVAILABLE, "Liger-Kernel not available")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_liger_vs_standard_loss_equivalence(self):
        """Test that liger fused loss produces equivalent results to standard approach."""
        # Standard approach: linear + cross entropy
        logits = torch.nn.functional.linear(self.hidden_states, self.weight)
        standard_loss = cross_entropy_loss(logits, self.target)
        
        # Liger fused approach
        liger_loss = liger_fused_linear_cross_entropy_loss(
            self.weight, self.hidden_states, self.target
        )
        
        # Should be very close (allowing for small numerical differences)
        self.assertTrue(torch.allclose(standard_loss, liger_loss, rtol=1e-5, atol=1e-6))

    def test_liger_loss_import_error(self):
        """Test that proper error is raised when liger-kernel is not available."""
        with patch('torchtitan.components.loss.LIGER_KERNEL_AVAILABLE', False):
            with self.assertRaises(ImportError) as context:
                liger_fused_linear_cross_entropy_loss(
                    self.weight, self.hidden_states, self.target
                )
            
            self.assertIn("Liger-Kernel is not installed", str(context.exception))
            self.assertIn("pip install liger-kernel", str(context.exception))

    @unittest.skipIf(not LIGER_KERNEL_AVAILABLE, "Liger-Kernel not available")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")  
    def test_tensor_reshaping(self):
        """Test that tensor reshaping works correctly."""
        # Test different batch sizes and sequence lengths
        test_cases = [
            (1, 2, 4, 8),   # small
            (4, 8, 16, 32), # medium
            (2, 1, 8, 16),  # seq_len = 1
        ]
        
        for batch_size, seq_len, hidden_dim, vocab_size in test_cases:
            with self.subTest(batch_size=batch_size, seq_len=seq_len):
                hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
                weight = torch.randn(vocab_size, hidden_dim, device=self.device)
                target = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
                
                loss = liger_fused_linear_cross_entropy_loss(weight, hidden_states, target)
                self.assertEqual(loss.shape, torch.Size([]))
                self.assertFalse(torch.isnan(loss))

    @unittest.skipIf(not LIGER_KERNEL_AVAILABLE, "Liger-Kernel not available")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gradient_flow(self):
        """Test that gradients flow correctly through liger fused loss."""
        # Enable gradients
        weight = self.weight.clone().requires_grad_(True)
        hidden_states = self.hidden_states.clone().requires_grad_(True)
        
        loss = liger_fused_linear_cross_entropy_loss(weight, hidden_states, self.target)
        loss.backward()
        
        # Check that gradients exist and are not zero
        self.assertIsNotNone(weight.grad)
        self.assertIsNotNone(hidden_states.grad)
        self.assertFalse(torch.allclose(weight.grad, torch.zeros_like(weight.grad)))
        self.assertFalse(torch.allclose(hidden_states.grad, torch.zeros_like(hidden_states.grad)))


if __name__ == "__main__":
    unittest.main()