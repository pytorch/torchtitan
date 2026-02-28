#!/usr/bin/env python3
"""
Test script to validate the attention sink LSE renormalization implementation.

This test verifies that the LSE renormalization approach is mathematically
equivalent to the HuggingFace concatenation approach.
"""

import torch
import torch.nn.functional as F


def test_lse_renormalization_equivalence():
    """
    Verify that LSE renormalization equals concat + softmax + drop.

    The two approaches should produce identical results:
    1. Concatenate sink to logits, softmax, drop sink position
    2. Compute LSE of original, compute combined LSE with sink, renormalize
    """
    print("=" * 60)
    print("Testing LSE Renormalization Equivalence")
    print("=" * 60)

    torch.manual_seed(42)

    B, H, Q, K = 2, 64, 128, 512

    # Random attention logits and sink values
    logits = torch.randn(B, H, Q, K, dtype=torch.float32)
    sinks = torch.randn(H, dtype=torch.float32)

    # Method 1: Explicit concatenation (HuggingFace eager attention approach)
    sinks_for_concat = sinks.view(1, -1, 1, 1).expand(B, H, Q, 1)
    combined = torch.cat([logits, sinks_for_concat], dim=-1)
    probs_concat = F.softmax(combined, dim=-1)
    probs_without_sink = probs_concat[..., :-1]  # Drop sink position

    # Method 2: LSE renormalization (TorchTitan/HuggingFace flex_attention approach)
    # This is what the fixed code does
    lse = torch.logsumexp(logits, dim=-1)  # [B, H, Q]
    lse_expanded = lse.unsqueeze(-1)  # [B, H, Q, 1]
    sinks_expanded = sinks.view(1, -1, 1, 1).expand(B, H, Q, 1)
    combined_lse = torch.logsumexp(
        torch.cat([lse_expanded, sinks_expanded], dim=-1), dim=-1, keepdim=True
    )
    renorm_factor = torch.exp(lse_expanded - combined_lse)
    probs_lse = F.softmax(logits, dim=-1) * renorm_factor

    # Compare
    max_diff = (probs_lse - probs_without_sink).abs().max().item()
    mean_diff = (probs_lse - probs_without_sink).abs().mean().item()

    print(f"\nResults:")
    print(f"  Max absolute difference:  {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")

    # Both should sum to less than 1 (since sink absorbs some probability)
    probs_sum_concat = probs_without_sink.sum(dim=-1)
    probs_sum_lse = probs_lse.sum(dim=-1)

    print(f"\n  Probability mass (should be < 1.0 due to sink absorption):")
    print(f"    Concat method mean: {probs_sum_concat.mean().item():.6f}")
    print(f"    LSE method mean:    {probs_sum_lse.mean().item():.6f}")

    # Assert equivalence
    try:
        torch.testing.assert_close(probs_lse, probs_without_sink, rtol=1e-4, atol=1e-5)
        print(f"\n✓ PASSED: LSE renormalization is mathematically equivalent to concatenation")
        return True
    except AssertionError as e:
        print(f"\n✗ FAILED: {e}")
        return False


def test_sink_edge_cases():
    """Test numerical stability with extreme sink values."""
    print("\n" + "=" * 60)
    print("Testing Edge Cases")
    print("=" * 60)

    torch.manual_seed(42)
    B, H, Q, D = 2, 4, 16, 64

    all_passed = True

    # Simulate attention output and LSE
    output = torch.randn(B, H, Q, D, dtype=torch.float32)
    lse = torch.randn(B, H, Q, dtype=torch.float32)

    def apply_sink_renorm(output, lse, sinks):
        """Apply the sink renormalization."""
        batch_size, num_heads, seq_len_q, head_dim = output.shape
        lse_expanded = lse.unsqueeze(-1)
        sinks_expanded = sinks.view(1, -1, 1, 1).expand(batch_size, num_heads, seq_len_q, 1)
        combined_lse = torch.logsumexp(
            torch.cat([lse_expanded, sinks_expanded], dim=-1), dim=-1, keepdim=True
        )
        renorm_factor = torch.exp(lse_expanded - combined_lse)
        return output * renorm_factor, renorm_factor

    # Test 1: Very negative sinks (should have minimal effect)
    print("\n1. Very negative sinks (sink = -100):")
    sinks_negative = torch.full((H,), -100.0)
    result_neg, factor_neg = apply_sink_renorm(output, lse, sinks_negative)
    print(f"   Renorm factor mean: {factor_neg.mean().item():.6f} (should be ~1.0)")
    print(f"   Output change: {(result_neg - output).abs().mean().item():.6f} (should be ~0)")
    if factor_neg.mean().item() > 0.99:
        print("   ✓ PASSED")
    else:
        print("   ✗ FAILED")
        all_passed = False

    # Test 2: Very positive sinks (should heavily attenuate)
    print("\n2. Very positive sinks (sink = +100):")
    sinks_positive = torch.full((H,), 100.0)
    result_pos, factor_pos = apply_sink_renorm(output, lse, sinks_positive)
    print(f"   Renorm factor mean: {factor_pos.mean().item():.2e} (should be ~0)")
    print(f"   Output norm: {result_pos.norm().item():.2e} (should be ~0)")
    if factor_pos.mean().item() < 1e-10:
        print("   ✓ PASSED")
    else:
        print("   ✗ FAILED")
        all_passed = False

    # Test 3: Zero sinks
    print("\n3. Zero sinks (neutral case):")
    sinks_zero = torch.zeros(H)
    result_zero, factor_zero = apply_sink_renorm(output, lse, sinks_zero)
    print(f"   Renorm factor mean: {factor_zero.mean().item():.6f}")
    print(f"   No NaN/Inf: {not (torch.isnan(result_zero).any() or torch.isinf(result_zero).any())}")
    if not (torch.isnan(result_zero).any() or torch.isinf(result_zero).any()):
        print("   ✓ PASSED")
    else:
        print("   ✗ FAILED")
        all_passed = False

    # Test 4: Mixed positive/negative sinks
    print("\n4. Mixed sinks (random * 10):")
    sinks_mixed = torch.randn(H) * 10
    result_mixed, factor_mixed = apply_sink_renorm(output, lse, sinks_mixed)
    has_nan_inf = torch.isnan(result_mixed).any() or torch.isinf(result_mixed).any()
    print(f"   Renorm factor range: [{factor_mixed.min().item():.4f}, {factor_mixed.max().item():.4f}]")
    print(f"   No NaN/Inf: {not has_nan_inf}")
    if not has_nan_inf:
        print("   ✓ PASSED")
    else:
        print("   ✗ FAILED")
        all_passed = False

    return all_passed


def test_gradient_flow():
    """Verify gradients flow correctly through sink parameters."""
    print("\n" + "=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)

    torch.manual_seed(42)
    B, H, Q, D = 2, 4, 16, 64

    output = torch.randn(B, H, Q, D, requires_grad=True)
    lse = torch.randn(B, H, Q, requires_grad=True)
    sinks = torch.randn(H, requires_grad=True)

    # Apply renormalization
    lse_expanded = lse.unsqueeze(-1)
    sinks_expanded = sinks.view(1, -1, 1, 1).expand(B, H, Q, 1)
    combined_lse = torch.logsumexp(
        torch.cat([lse_expanded, sinks_expanded], dim=-1), dim=-1, keepdim=True
    )
    renorm_factor = torch.exp(lse_expanded - combined_lse)
    result = output * renorm_factor

    # Backprop
    loss = result.sum()
    loss.backward()

    all_passed = True

    # Verify gradients exist and are not NaN
    print("\nGradient checks:")

    print(f"  sinks.grad exists: {sinks.grad is not None}")
    if sinks.grad is not None:
        print(f"  sinks.grad has NaN: {torch.isnan(sinks.grad).any().item()}")
        print(f"  sinks.grad norm: {sinks.grad.norm().item():.4f}")
        if torch.isnan(sinks.grad).any():
            all_passed = False
    else:
        all_passed = False

    print(f"  output.grad exists: {output.grad is not None}")
    print(f"  lse.grad exists: {lse.grad is not None}")

    if all_passed and sinks.grad is not None and output.grad is not None and lse.grad is not None:
        print("\n✓ PASSED: Gradients flow correctly through sink parameters")
    else:
        print("\n✗ FAILED: Gradient flow issues detected")
        all_passed = False

    return all_passed


def main():
    print("\n" + "=" * 60)
    print("GPT-OSS Attention Sink Validation Tests")
    print("=" * 60 + "\n")

    results = []

    results.append(("LSE Renormalization Equivalence", test_lse_renormalization_equivalence()))
    results.append(("Edge Cases", test_sink_edge_cases()))
    results.append(("Gradient Flow", test_gradient_flow()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
