## Triton Group GEMM for MOE training and inference.
## M*G Version
This is a v1 of a Triton Group GEMM for use with Moe models in both training and inference.
This is an M*G group gemm, and returns an M*G,K output.  This may be more suitable for all2all EP MoE.

Currently passes basic verification vs PyTorch reference on *simple* sizes and shapes:
Max absolute difference: 0.000000e+00
Max relative difference: 0.000000e+00
✓ Outputs match within tolerance

Todo:
1 - Add BF16 backward pass support
2 - Add autotuning
3 - FP8 support backward.
4 - Benchmarking.
