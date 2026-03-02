import pytest
import torch
from torchtitan.models.common.moe.moe import ExpertCombineFunction

NUM_TOKENS = 42
# Use some Qwen3-30B-A3B dimensions for testing.
DIM = 2048
NUM_EXPERTS = 128
TOP_K = 8

def original_token_combine(s: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Original token combine implementation using bmm, for testing against the custom autograd function."""
    return torch.bmm(s.unsqueeze(1), R.float()).squeeze(1).to(torch.bfloat16)

@pytest.mark.gpu
def test_token_combine_fwd():
    routed_expert_activations = torch.randn(NUM_TOKENS, TOP_K, DIM, device="cuda", dtype=torch.bfloat16)
    router_scores = torch.randn(NUM_TOKENS, TOP_K, device="cuda", dtype=torch.float32)

    combined_output_ground_truth = original_token_combine(router_scores, routed_expert_activations)
    combined_output = ExpertCombineFunction.apply(router_scores, routed_expert_activations, torch.bfloat16)

    assert combined_output.dtype == torch.bfloat16, "Expected combined output to be bfloat16"
    assert combined_output.shape == (NUM_TOKENS, DIM), f"Expected combined output shape to be {(NUM_TOKENS, DIM)}"
    assert torch.allclose(combined_output, combined_output_ground_truth, rtol=1e-4, atol=1e-4), "Combined output does not match ground truth"


@pytest.mark.gpu
def test_token_combine_bwd():
    routed_expert_activations = torch.randn(NUM_TOKENS, TOP_K, DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    router_scores = torch.randn(NUM_TOKENS, TOP_K, device="cuda", dtype=torch.float32, requires_grad=True)

    routed_expert_activations2 = routed_expert_activations.clone().detach().requires_grad_(True)
    router_scores2 = router_scores.clone().detach().requires_grad_(True)

    combined_output = ExpertCombineFunction.apply(router_scores, routed_expert_activations, torch.bfloat16)
    loss = combined_output.sum()
    loss.backward()

    combined_output2 = original_token_combine(router_scores2, routed_expert_activations2)
    loss2 = combined_output2.sum()
    loss2.backward()

    assert router_scores.grad is not None, "Expected gradient w.r.t. router scores"
    assert routed_expert_activations.grad is not None, "Expected gradient w.r.t. routed expert activations"

    assert torch.allclose(router_scores.grad, router_scores2.grad, rtol=1e-4, atol=1e-4), "Gradient w.r.t. router scores does not match ground truth"
    assert torch.allclose(routed_expert_activations.grad, routed_expert_activations2.grad, rtol=1e-4, atol=1e-4), "Gradient w.r.t. routed expert activations does not match ground truth"
