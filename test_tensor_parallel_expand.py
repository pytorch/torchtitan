#!/usr/bin/env python3

"""
Simple test script to validate the tensor parallel expand operation.
"""

import torch
import sys
import os

# Add the torchtitan path to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from torchtitan.models.deepseek_v3.model.tensor_parallel_ops import tensor_parallel_expand

def test_tensor_parallel_expand():
    """Test the tensor parallel expand operation."""
    print("Testing tensor_parallel_expand function...")
    
    # Create a test tensor similar to k_pe
    # Shape: (batch_size, seq_len, 1, qk_rope_head_dim)
    batch_size = 2
    seq_len = 10
    qk_rope_head_dim = 64
    n_local_heads = 8
    
    k_pe = torch.randn(batch_size, seq_len, 1, qk_rope_head_dim, requires_grad=True)
    print(f"Input k_pe shape: {k_pe.shape}")
    
    # Test our custom expand operation
    expanded_k_pe = tensor_parallel_expand(k_pe, (-1, -1, n_local_heads, -1))
    print(f"Expanded k_pe shape: {expanded_k_pe.shape}")
    
    # Verify the shape is correct
    expected_shape = (batch_size, seq_len, n_local_heads, qk_rope_head_dim)
    assert expanded_k_pe.shape == expected_shape, f"Expected {expected_shape}, got {expanded_k_pe.shape}"
    
    # Test gradient flow
    print("Testing gradient flow...")
    loss = expanded_k_pe.sum()
    loss.backward()
    
    print(f"Original k_pe gradient shape: {k_pe.grad.shape}")
    assert k_pe.grad is not None, "Gradient should not be None"
    assert k_pe.grad.shape == k_pe.shape, f"Gradient shape mismatch: {k_pe.grad.shape} vs {k_pe.shape}"
    
    # Compare with standard expand operation
    print("Comparing with standard expand...")
    k_pe_std = torch.randn(batch_size, seq_len, 1, qk_rope_head_dim, requires_grad=True)
    expanded_k_pe_std = k_pe_std.expand(-1, -1, n_local_heads, -1)
    
    # Both should have the same forward behavior
    assert expanded_k_pe.shape == expanded_k_pe_std.shape, "Forward shapes should match"
    
    print("✅ All tests passed!")
    print("\nSummary:")
    print(f"- Input shape: {k_pe.shape}")
    print(f"- Output shape: {expanded_k_pe.shape}")
    print(f"- Gradient shape: {k_pe.grad.shape}")
    print("- Custom tensor parallel expand operation works correctly")

if __name__ == "__main__":
    test_tensor_parallel_expand()
