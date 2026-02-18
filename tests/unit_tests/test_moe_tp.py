import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import DTensor, Replicate
from torchtitan.models.moe.moe import GroupedExperts

def test_moe_experts_grad_reduction():
    # 1. Setup a fake distributed world (Simulated on CPU)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    
    # 2. Define a dummy Mesh (TP=1 simulation)
    mesh = init_device_mesh("cpu", (1,))
    
    # 3. Initialize GroupedExperts (This is where your fix lives!)
    # We use use_grouped_mm=False to force the for-loop path or standard MM
    # but the key is that we want to trigger the .to_local() call.
    experts = GroupedExperts(
        dim=16,
        hidden_dim=32,
        num_experts=4,
        use_grouped_mm=False 
    )
    
    # 4. Mock the weights as DTensors
    # This forces the code to hit the "if isinstance(self.w1, DTensor)" block in your fix
    w1_local = experts.w1.detach()
    w2_local = experts.w2.detach()
    w3_local = experts.w3.detach()
    
    experts.w1 = nn.Parameter(DTensor.from_local(w1_local, mesh, [Replicate()]))
    experts.w2 = nn.Parameter(DTensor.from_local(w2_local, mesh, [Replicate()]))
    experts.w3 = nn.Parameter(DTensor.from_local(w3_local, mesh, [Replicate()]))

    # 5. Run Forward Pass
    # We need dummy inputs: x and num_tokens_per_expert
    x = torch.randn(10, 16) # 10 tokens, dim 16
    # Create a dummy distribution of tokens (e.g., 2 tokens for exp0, 3 for exp1, etc.)
    num_tokens_per_expert = torch.tensor([2, 3, 2, 3], dtype=torch.long)
    
    # Run forward
    output = experts(x, num_tokens_per_expert)
    
    # 6. Run Backward Pass
    loss = output.sum()
    loss.backward()
    
    # 7. Verification
    # If your fix works, gradients should propagate back to the DTensor parameters.
    assert experts.w1.grad is not None, "Error: Gradients failed to flow to w1"
    assert experts.w2.grad is not None, "Error: Gradients failed to flow to w2"
    
    print("\nSUCCESS: Test Passed! Gradients flowed correctly through the GroupedExperts DTensors.\n")

if __name__ == "__main__":
    test_moe_experts_grad_reduction()