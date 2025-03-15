## Triton Group GEMM for MOE training and inference.
## M*G Version
This is a v1 of a Triton Group GEMM for use with Moe models in both training and inference.
This is an M*G group gemm, and returns an M*G,K output.  This should be more suitable for all2all EP MoE *with Token Choice routing*.

Currently passes basic verification vs PyTorch reference on simple and DeepSeek v3 shapes:

~~~
===== Testing DeepSeek Config 4 =====
2025-03-15 16:19:10,494 - INFO - G=8, M=4096, K=2048, N=7168
2025-03-15 16:19:10,508 - INFO - Input x shape: torch.Size([4096, 2048]), Weight w shape: torch.Size([7168, 2048])
2025-03-15 16:19:10,657 - INFO - Forward result shape: torch.Size([4096, 7168])
2025-03-15 16:19:11,013 - INFO - Largest Forward output difference: 0.0 at (np.int64(0), np.int64(0))
2025-03-15 16:19:11,013 - INFO - Values: 73.75 vs 73.75
2025-03-15 16:19:11,028 - INFO - ✓ SUCCESS: Forward output matches PyTorch reference
2025-03-15 16:19:11,028 - INFO - Starting grouped_gemm_backward with optimized scheduling
2025-03-15 16:19:11,028 - INFO - K dimension: 2048
2025-03-15 16:19:11,150 - INFO - EVEN_K optimization enabled: True (K=2048)
2025-03-15 16:19:11,151 - INFO - Computing grad_x with optimized kernel (TMA=False)
2025-03-15 16:19:11,151 - INFO - Kernel run success: grad_x computation successful
2025-03-15 16:19:11,151 - INFO - Computing grad_w with optimized kernel
2025-03-15 16:19:11,151 - INFO - Kernel run success: grad_w computation successful
2025-03-15 16:19:11,506 - INFO - Largest grad_x difference: 0.25 at (np.int64(2), np.int64(843))
2025-03-15 16:19:11,506 - INFO - Values: 282.25 vs 282.5
2025-03-15 16:19:11,520 - INFO - ✓ SUCCESS: grad_x matches PyTorch reference
2025-03-15 16:19:11,547 - INFO - Largest grad_w difference: 0.375 at (np.int64(707), np.int64(1871))
2025-03-15 16:19:11,547 - INFO - Values: 186.5 vs 186.125
2025-03-15 16:19:11,561 - INFO - ✓ SUCCESS: grad_w matches PyTorch reference
2025-03-15 16:19:11,561 - INFO - ✓ SUCCESS: Config 4 passed all tests!
~~~
Todo:
1 - DONE: Add BF16 backward pass support
2 - Add autotuning
3 - FP8 support backward.
4 - Benchmarking.
