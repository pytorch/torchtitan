## Triton Group GEMM for MOE training and inference.
## M*G Version
This is a v1 of a Triton Group GEMM for use with Moe models in both training and inference.
This is an M*G group gemm, and returns an M*G,K output.  This may be more suitable for all2all EP MoE *with Token Choice routing*.

Currently passes basic verification vs PyTorch reference on *simple* sizes and shapes:
Forward: 
~~~
Max absolute difference: 0.000000e+00
Max relative difference: 0.000000e+00
✓ Outputs match within tolerance

Backwards: 
===== Running basic backward pass test =====
2025-03-15 11:50:43,513 - INFO - Test setup - G: 4, M_total: 6144, N: 512, K: 256
2025-03-15 11:50:43,526 - INFO - Group sizes: tensor([1024, 1024, 2048, 2048], device='cuda:0', dtype=torch.int32)
2025-03-15 11:50:43,526 - INFO - Input x shape: torch.Size([6144, 256])
2025-03-15 11:50:43,526 - INFO - Weight w shape: torch.Size([512, 256])
2025-03-15 11:50:43,526 - INFO - Running forward pass
2025-03-15 11:50:44,026 - INFO - Forward result shape: torch.Size([6144, 512])
2025-03-15 11:50:44,026 - INFO - Created gradient with shape: torch.Size([6144, 512])
2025-03-15 11:50:44,026 - INFO - Running backward pass directly
2025-03-15 11:50:44,026 - INFO - Starting grouped_gemm_backward with optimized scheduling
2025-03-15 11:50:44,026 - INFO - K dimension: 256
2025-03-15 11:50:44,108 - INFO - EVEN_K optimization enabled: False (K=256)
2025-03-15 11:50:44,108 - INFO - Computing grad_x with optimized kernel (TMA=False)
2025-03-15 11:50:44,284 - INFO - Kernel run success: grad_x computation successful
2025-03-15 11:50:44,284 - INFO - Computing grad_w with optimized kernel
2025-03-15 11:50:44,473 - INFO - Kernel run success: grad_w computation successful
2025-03-15 11:50:44,473 - INFO - Gradient shapes - grad_x: torch.Size([6144, 256]), grad_w: torch.Size([512, 256])
2025-03-15 11:50:44,474 - INFO - Running PyTorch reference implementation
2025-03-15 11:50:44,935 - INFO - Comparing gradients with PyTorch reference
2025-03-15 11:50:44,989 - INFO - Maximum gradient error - grad_x: 0.0625, grad_w: 0.25
2025-03-15 11:50:45,072 - INFO - ✓ SUCCESS! grad_X matches the PyTorch reference (allclose check passed)
2025-03-15 11:50:45,085 - INFO - ✓ SUCCESS! grad_W matches the PyTorch reference (allclose check passed)
2025-03-15 11:50:45,085 - INFO - Gradients allclose check - grad_x: True, grad_w: True
2025-03-15 11:50:45,085 - INFO - ✓ SUCCESS: Gradients match the PyTorch reference (allclose check passed)
2025-03-15 11:50:45,113 - INFO - Largest grad_x difference at (np.int64(1288), np.int64(167)): 72.3125 vs 72.25
2025-03-15 11:50:45,140 - INFO - Zeros in grad_x: 0/1572864 (0.00%)
2025-03-15 11:50:45,140 - INFO - Zeros in x_autograd.grad: 0/1572864 (0.00%)
2025-03-15 11:50:45,154 - INFO - Largest grad_w difference at (np.int64(8), np.int64(237)): -174.5 vs -174.25
2025-03-15 11:50:45,181 - INFO - Zeros in grad_w: 0/131072 (0.00%)
2025-03-15 11:50:45,181 - INFO - Zeros in w_autograd.grad: 8/131072 (0.01%)
2025-03-15 11:50:45,208 - INFO - Basic test succeeded
~~~
Todo:
1 - Add BF16 backward pass support  
2 - Add autotuning  
3 - FP8 support backward.  
4 - Benchmarking.  
