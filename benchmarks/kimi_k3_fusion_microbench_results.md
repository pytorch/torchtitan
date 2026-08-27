# Kimi K3 Fusion Microbenchmark Results

## Scope

This benchmark compares forward and backward performance for three Kimi K3
patterns:

1. SiTU-GLU activation
2. Attention residual aggregation
3. Sigmoid-gated RMSNorm

Implementations include eager PyTorch, the current `torch.compile` helpers,
FLA's handwritten Triton kernels where available, and a benchmark-local
handwritten Triton SiTU kernel.

Benchmark source: `benchmarks/kimi_k3_fusion_microbench.py`

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GB300 |
| PyTorch | `2.15.0a0+gitd62ee83` |
| Triton | `3.8.0` |
| flash-linear-attention | `0.5.2` |
| Tensor dtype | BF16 |
| Warmup budget | 500 ms per measurement |
| Measurement budget | 1000 ms per measurement |

Forward latency is measured directly. Backward latency is derived as
`median(forward + backward) - median(forward)` because AOTAutograd's donated
buffers prevent repeatedly invoking backward on one retained compiled graph.

Peak delta is peak allocated GPU memory above the persistent benchmark inputs.

## Command

```bash
CUDA_VISIBLE_DEVICES=2 \
PATH=/home/bahuang/.conda/envs/pytorch-3.12/bin:$PATH \
python benchmarks/kimi_k3_fusion_microbench.py \
    --preset k3 \
    --warmup-ms 500 \
    --rep-ms 1000
```

## Results

### SiTU-GLU

| Shape | Implementation | Forward | Backward | F+B | Speedup | Peak delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Routed expert `[8192, 3072]` | eager | 0.480 ms | 0.616 ms | 1.096 ms | 1.00x | 960 MiB |
| Routed expert `[8192, 3072]` | `torch.compile` | 0.112 ms | 0.209 ms | 0.321 ms | 3.41x | 192 MiB |
| Routed expert `[8192, 3072]` | handwritten Triton | **0.071 ms** | **0.197 ms** | **0.268 ms** | **4.08x** | 192 MiB |
| Shared expert `[32768, 6144]` | eager | 3.432 ms | 4.447 ms | 7.879 ms | 1.00x | 7680 MiB |
| Shared expert `[32768, 6144]` | `torch.compile` | **0.360 ms** | 0.572 ms | **0.933 ms** | **8.45x** | 1536 MiB |
| Shared expert `[32768, 6144]` | handwritten Triton | 0.443 ms | **0.525 ms** | 0.967 ms | 8.14x | 1536 MiB |

Conclusion: the handwritten kernel is about 16% faster than Inductor for the
routed-expert shape, but Inductor is about 4% faster overall for the larger
shared-expert shape. A shape-based dispatch is possible, but the full-model
benefit must justify maintaining a custom backward kernel.

External SiTU kernels found during research are mostly fused-MoE serving or
quantized kernels rather than portable BF16 training replacements:

- [FlashInfer CuTe-DSL SiTU](https://github.com/flashinfer-ai/flashinfer/commit/e44dae5b36ab5a8fc0c8834d4614906879c80659)
- [ROCm AITER SiTUv2](https://github.com/ROCm/aiter/pull/4397)
- [DeepGEMM SiTU support](https://github.com/deepseek-ai/DeepGEMM/pull/396)

### Attention residual

`saved` is the number of tensors already stored in `block_residual_TND`. The
operation has one additional current `prefix_sum`, so `saved=3` means four
softmax sources.

| Sources | Implementation | Forward | Backward | F+B | Speedup | Peak delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | eager | 6.852 ms | 12.603 ms | 19.455 ms | 1.00x | 11648 MiB |
| 2 | `torch.compile` | 0.597 ms | **0.778 ms** | **1.375 ms** | **14.15x** | 2857 MiB |
| 2 | FLA Triton | **0.406 ms** | 1.021 ms | 1.426 ms | 13.64x | **2689 MiB** |
| 4 | eager | 12.859 ms | 24.437 ms | 37.296 ms | 1.00x | 22400 MiB |
| 4 | `torch.compile` | 2.690 ms | 3.326 ms | 6.016 ms | 6.20x | 8150 MiB |
| 4 | FLA Triton | **2.098 ms** | **1.757 ms** | **3.855 ms** | **9.67x** | **5376 MiB** |
| 6 | eager | 19.258 ms | 36.476 ms | 55.734 ms | 1.00x | 33152 MiB |
| 6 | `torch.compile` | **3.105 ms** | 5.072 ms | 8.178 ms | 6.82x | 11776 MiB |
| 6 | FLA Triton | 3.336 ms | **2.489 ms** | **5.825 ms** | **9.57x** | **8064 MiB** |
| 8 | eager | 25.148 ms | 48.079 ms | 73.227 ms | 1.00x | 43904 MiB |
| 8 | `torch.compile` | **3.518 ms** | 6.506 ms | 10.025 ms | 7.30x | 15403 MiB |
| 8 | FLA Triton | 4.561 ms | **3.321 ms** | **7.882 ms** | **9.29x** | **10752 MiB** |

The sweep reuses the same compiled callable across changing source counts, as
the model does. Inductor recompiles and eventually produces a dynamic-source
kernel. An isolated static four-source invocation measured about 2.42 ms F+B,
but the dynamic sweep measured 6.02 ms. FLA remains stable as the number of
sources changes and wins clearly from four sources onward, mostly through a
faster backward and lower activation memory.

FLA's `fused_attnres` is an exact algorithmic match and includes handwritten
Triton forward/backward kernels. It also supports folding the immediately
following output RMSNorm through `output_rms_weight`, which this benchmark does
not yet include and could provide another gain.

Sources:

- [FLA AttnRes operator](https://github.com/fla-org/flash-linear-attention/pull/878)
- [Liger Kimi AttnRes kernel](https://github.com/linkedin/Liger-Kernel/pull/1161)
- [vLLM AMD AttnRes fusion](https://github.com/vllm-project/vllm/pull/50593)

### Gated RMSNorm

| Shape | Implementation | Forward | Backward | F+B | Speedup | Peak delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `[8, 4096, 96, 128]` | eager | 6.435 ms | 11.188 ms | 17.624 ms | 1.00x | 12300 MiB |
| `[8, 4096, 96, 128]` | `torch.compile` | **0.348 ms** | **0.633 ms** | **0.980 ms** | **17.98x** | 3120 MiB |
| `[8, 4096, 96, 128]` | FLA Triton | 0.389 ms | 0.768 ms | 1.157 ms | 15.24x | **3084 MiB** |

Conclusion: the current Inductor kernel is approximately 15% faster F+B than
FLA's handwritten Triton kernel at the exact K3 batch-8 shape. FLA saves only
about 36 MiB of peak temporary memory, so replacing the current implementation
is not justified by this result.

Source:

- [FLA gated-normalization benchmarks](https://github.com/fla-org/flash-linear-attention/pull/1044)
- [FLA gated-normalization backward fix](https://github.com/fla-org/flash-linear-attention/pull/1071)

## Numerical comparison

All implementations passed the benchmark's BF16 checks. The maximum relative
L2 gradient error across the full-shape tests was `3.42e-4`; output maximum
absolute error was at most `3.125e-2`.

The larger absolute gradient differences for AttnRes, up to 16, occur in
BF16-reduced projection/norm weight gradients whose reference magnitudes reach
several thousand. Their relative L2 errors remain below `3.5e-4`. Full-model
loss/gradient validation is still required before replacing the production
implementation.

## Recommendation

1. Keep the current compiled gated RMSNorm.
2. Keep compiled SiTU as the general implementation. Consider the handwritten
   Triton path only for the routed-expert shape after measuring end-to-end gain.
3. Prototype FLA `fused_attnres` for source counts of four or more. Preserve the
   current compiled path for two-source aggregation.
4. In that prototype, benchmark FLA's `output_rms_weight` option so AttnRes and
   the following attention/FFN RMSNorm become one kernel.
5. Require deterministic full-model loss and gradient comparison before
   retaining any production change.
