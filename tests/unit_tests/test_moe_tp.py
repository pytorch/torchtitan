import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import DTensor, Replicate, Partial

from torchtitan.distributed.__init__ import NoParallel

def test_noparallel_router_grad_reduction():
    # 1. Setup Fake Distributed World
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    
    mesh = init_device_mesh("cpu", (1,))
    
    # 2. Initialize a dummy module (simulating the Router Gate)
    class DummyRouterGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(16, 4, bias=False)
            
        def forward(self, x):
            return self.gate(x)
            
    module = DummyRouterGate()
    
    # 3. Apply NoParallel wrapper (This triggers our new fix!)
    parallel_style = NoParallel()
    distributed_module = parallel_style._apply(module, mesh)
    
    # 4. Run Forward Pass
    x_local = torch.randn(10, 16)
    x_dtensor = DTensor.from_local(x_local, mesh, [Replicate()])
    
    # The output will be a local tensor because use_local_output=True by default
    output_local = distributed_module(x_dtensor)
    
    # 5. Run Backward Pass
    loss = output_local.sum()
    loss.backward()
    
    # 6. Verification
    # If our fix works, gradients should successfully propagate through the erasure 
    # back to the underlying DTensor parameters.
    assert distributed_module.gate.weight.grad is not None, "Error: Gradients failed to flow through NoParallel"
    
    print("\nSUCCESS: Test Passed! Gradients flowed correctly through NoParallel DTensor erasure.\n")

if __name__ == "__main__":
    test_noparallel_router_grad_reduction()
