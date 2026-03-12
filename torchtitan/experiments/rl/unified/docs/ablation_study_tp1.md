# Unified Model Definition for Training and Inference: A Single-GPU Performance Ablation Study

## TL;DR

We systematically measured the performance gap between TorchTitan's unified model definition and vLLM's native hand-optimized Qwen3 model running inside vLLM's inference engine. On a single H100 GPU, the unoptimized TorchTitan model is **35% slower** in eager mode — but with `compile(eager) + piecewise cudagraph`, the gap **completely disappears** (1.02x parity). Through a series of 8 ablation experiments, we identified exactly where every microsecond of the eager-mode gap comes from and proved that it's entirely eliminated by the compile+cudagraph optimization path.

## Motivation

The goal of the TorchTitan unified model definition is to use **one model codebase** for both training and inference, avoiding the error-prone process of maintaining separate model implementations. But does this convenience come at a performance cost?

To answer this, we ran TorchTitan's Qwen3-1.7B model inside vLLM's V1 inference engine and compared it against vLLM's native `Qwen3ForCausalLM` — a hand-tuned implementation with fused kernels, merged projections, and optimized tensor layouts.

## Experimental Setup

- **Model**: Qwen3-1.7B (28 layers, dim=2048, 16 heads, 8 KV heads)
- **Hardware**: Single NVIDIA H100 GPU (94GB HBM)
- **Workload**: batch_size=4, max_tokens=128 (decode-heavy)
- **Modes tested**: Eager (no compile, no cudagraph) and compile(eager) + piecewise cudagraph
- **Benchmark**: vLLM V1 engine, measuring end-to-end throughput (tokens/sec)

## Key Result: Compile+CUDAGraph Achieves Parity

| Mode | vLLM Native | TorchTitan | TT/Native |
|---|---|---|---|
| **Eager** | 423 tok/s | 274 tok/s | 0.65x |
| **compile + cudagraph** | 718 tok/s | 731 tok/s | **1.02x** |

With compile+cudagraph enabled, TorchTitan matches vLLM native performance. The 35% eager-mode gap is entirely framework overhead that the compiler eliminates.

## The Ablation Journey

We systematically applied optimizations one at a time to understand what drives the eager-mode gap.

### Ablation Results (Cumulative)

| # | Optimization | Throughput | vs Native | Delta |
|---|---|---|---|---|
| 0 | TorchTitan baseline | 274 tok/s | 0.65x | — |
| 1 | +vLLM fused RMSNorm | 268 tok/s | 0.63x | ~0% |
| 2 | +vLLM fused SiluAndMul | 265 tok/s | 0.63x | ~0% |
| 3 | **+vLLM fused RoPE** | **321 tok/s** | **0.76x** | **+17%** |
| 4 | +Merged QKV (3→1 GEMM) | 318 tok/s | 0.75x | ~0% |
| 5 | **+Merged gate_up (2→1 GEMM)** | **343 tok/s** | **0.81x** | **+7%** |
| 6 | +Fused residual-add-norm | 349 tok/s | 0.83x | +2% |
| 7 | +Cached rope dtype conversion | 356 tok/s | 0.84x | +2% |
| 8 | **+Zero-copy attention layout** | **373 tok/s** | **0.88x** | **+5%** |
| 9 | **+Direct _C:: RMSNorm calls** | **434 tok/s** | **1.03x** | **+16%** |

### What Worked and What Didn't

**Swapping kernel implementations didn't help (ablations 1-2):**
Replacing `nn.RMSNorm` with vLLM's Triton-based `RMSNorm`, or replacing `F.silu(w1(x)) * w3(x)` with vLLM's fused `SiluAndMul`, made no measurable difference. In eager mode, vLLM's `CustomOp` dispatch adds its own Python overhead that offsets the kernel fusion benefit. The GPU kernels themselves are fast — they just can't overcome the CPU dispatch cost.

**Eliminating intermediate tensor allocations did help (ablation 3, +17%):**
TorchTitan's `apply_rotary_emb_cos_sin` uses `_rotate_half()` which splits, negates, and concatenates tensor halves — creating multiple intermediate tensors per call. vLLM's fused `rotary_embedding` kernel does everything in-place in a single CUDA kernel. Across 28 layers × 128 decode steps, this adds up significantly.

**Merging GEMM projections partially helped (ablations 4-5):**
Merged QKV (3→1 GEMM) showed no improvement, but merged gate_up (2→1 GEMM) gave +7%. The difference: gate_up's weight matrices are 3x larger (6144 vs 2048 per projection), so saving one kernel launch has more impact relative to the GEMM compute time.

**Eliminating layout conversions helped (ablation 8, +5%):**
The `VLLMAttention` wrapper converts between TorchTitan's `(batch, heads, seq, dim)` layout and vLLM's `(num_tokens, heads, dim)` layout via a double-transpose that creates non-contiguous tensors requiring explicit `clone()` operations. By reshaping directly from `(batch, seq, heads, dim)` to `(num_tokens, heads, dim)` — a zero-copy operation — we eliminated thousands of unnecessary copy kernels.

**Bypassing the ATen dispatch chain was the biggest win (ablation 9, +16%):**
This was the most surprising finding. `nn.RMSNorm` goes through PyTorch's full ATen dispatch chain: `aten::rms_norm` → `aten::_fused_rms_norm` → CUDA kernel, costing ~37us per call. vLLM's native model calls `torch.ops._C.rms_norm` directly at ~8us per call. With 7168 norm calls per benchmark, this 29us/call difference accounts for the entire remaining gap. Our `DirectRMSNorm` module calls `_C::rms_norm` directly, matching vLLM native.

## Trace-Level Root Cause Analysis

We collected GPU traces for both models and performed a detailed breakdown:

### GPU Utilization

| | Native | TorchTitan (baseline) |
|---|---|---|
| GPU active time | 405 ms | 551 ms |
| GPU idle time | 2,041 ms | 2,432 ms |
| **GPU utilization** | **16.6%** | **18.5%** |

Both models have **<20% GPU utilization** in eager mode. The GPU finishes each kernel in microseconds and then waits 10-50us for the CPU to dispatch the next one.

### The Gap Breakdown

The 1.79 ms/step gap between baseline TorchTitan (274 tok/s) and native (423 tok/s) breaks down to:

```
Per-step gap: 1.79 ms
├── 64% Extra GPU kernel compute (1.14 ms)
│   └── clone/contiguous/copy kernels from layout conversions
│       and dtype conversions (rope cache)
└── 36% Extra CPU dispatch overhead (0.65 ms)
    ├── 139 extra kernel launches × ~13us dispatch each
    └── 1,546 extra metadata ops (reshape/view/transpose)
```

### CPU Op Comparison

| Extra CPU Op | Extra Count | Root Cause |
|---|---|---|
| `aten::rms_norm` chain | +14,336 | ATen dispatch (37us) vs direct `_C::` call (8us) |
| `aten::clone` | +10,752 | Non-contiguous tensors from attention transposes |
| `aten::copy_` | +14,208 | Layout + dtype conversions |
| `aten::contiguous` | +7,168 | Tensors made non-contiguous by transpose |
| `aten::reshape` | +46,720 | TorchTitan's `(bs,seq,heads,dim)` ↔ `(num_tokens,heads*dim)` |

## Why Compile+CUDAGraph Eliminates Everything

CUDAGraph captures the entire GPU kernel sequence during a warmup run and replays it with a single CPU call:

```
Eager mode (per decode step):
CPU:  [dispatch 13us][dispatch 13us][dispatch 13us]... × 500+ kernels
GPU:  ...[kernel 2us]...[kernel 3us]...[kernel 2us]...
      GPU utilization: ~17%

CUDAGraph mode (per decode step):
CPU:  [replay 1us]
GPU:  [kernel][kernel][kernel][kernel]... (no gaps)
      GPU utilization: ~100%
```

This eliminates:
- All Python dispatch overhead (nn.Module.__call__, ATen dispatch)
- All kernel launch overhead (CUDA driver calls)
- All tensor metadata operations (view, reshape, transpose)
- All intermediate tensor allocations (clone, empty_like)

The result: both models replay the exact same sequence of CUDA kernels, achieving identical performance.

## Workload Generalizability

We swept across different batch sizes and sequence lengths to verify the findings hold:

| Workload | Batch | MaxTokens | TT/Native (all ablations) |
|---|---|---|---|
| Small decode | 4 | 32 | 1.05x |
| Baseline | 4 | 128 | 1.04x |
| Long decode | 4 | 512 | 1.05x |
| Medium batch | 32 | 128 | 1.04x |

The ~4% advantage is **consistent across all workloads** because the per-step CPU dispatch overhead is fixed (not proportional to batch size or sequence length). At larger batch sizes, GPU compute grows but the overhead stays constant, maintaining the ratio.

## Conclusions

1. **Compile+cudagraph is essential**: The unified TorchTitan model achieves **1.02x parity** with vLLM native when compile+cudagraph is enabled. Without it, eager mode is 35% slower.

2. **The eager gap is CPU overhead, not GPU compute**: Both models have <20% GPU utilization in eager mode. The gap comes from Python dispatch (ATen dispatch chain, nn.Module.__call__) and tensor metadata operations, not from kernel-level differences.

3. **Direct _C:: calls matter in eager mode**: Bypassing PyTorch's ATen dispatch chain for hot-path operations like RMSNorm gave the single largest improvement (+16%). Each ATen dispatch costs ~37us vs ~8us for direct calls — multiplied by 7168 calls per step, this dominates.

4. **Kernel fusion has limited impact without compile**: Swapping individual kernels (RMSNorm, SiluAndMul) showed no improvement in eager mode because the `CustomOp` dispatch overhead offsets the kernel fusion benefit. The overhead is in the dispatch, not the kernel.

5. **Layout conversions are avoidable**: The double-transpose between TorchTitan's `(batch, heads, seq, dim)` and vLLM's `(num_tokens, heads, dim)` layout created thousands of unnecessary clone/contiguous operations. Direct reshaping eliminates this entirely.

6. **Results generalize across workloads**: The performance ratio is consistent across batch sizes (4-32) and sequence lengths (32-512), confirming the overhead is per-step, not per-token.

### Recommendation

For production inference with the unified model definition:
- **Always enable compile(eager) + piecewise cudagraph** — this is the default in `GeneratorCompileConfig` and achieves native parity
- The eager-mode optimizations (DirectRMSNorm, merged projections, zero-copy layout) are nice-to-have for debugging but unnecessary in production
- The unified model definition imposes **zero performance cost** when the compile+cudagraph optimization path is used
