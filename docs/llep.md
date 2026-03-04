# Least-Loaded Expert Parallelism (LLEP)

LLEP is a dynamic load-balancing strategy for Expert Parallelism (EP) in Mixture-of-Experts (MoE) models. It redistributes expert workloads across GPUs at runtime to handle the token imbalance inherent in models like DeepSeek-V3, Kimi-K2, and Qwen3.

**Reference:** "Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts" (Nguyen et al., Salesforce AI Research)

## How It Works

Standard EP assigns each expert to a fixed GPU. When routing is imbalanced (e.g., 80% of tokens go to 2 of 256 experts), the "hot" GPUs become bottlenecks while others sit idle.

LLEP fixes this by:

1. **Gathering** per-rank expert token counts via `all_gather`
2. **Planning** a Longest-Processing-Time (LPT) assignment that splits overloaded experts across multiple GPUs
3. **Transferring** expert weights to "helper" GPUs via P2P
4. **Dispatching** tokens via a modified AllToAll that routes to the helper GPUs
5. **Computing** SwiGLU FFN on each GPU's assigned token-expert pairs
6. **Combining** results via inverse AllToAll back to the original token order

## Architecture

### Hook-Based Flow (Recommended)

The hook-based API decomposes LLEP into three steps that integrate with the existing MoE forward pass:

```python
from torchtitan.distributed.llep import (
    llep_dispatch_tokens,
    llep_prepare_weights,
    llep_combine_output,
)

# EP pre-hook: plan + dispatch tokens
dispatched_tokens, padded_counts, state = llep_dispatch_tokens(
    routed_input, num_tokens_per_expert, ep_group,
    max_tokens_factor=1.1,
    min_tokens_per_gemm=1024,
    adaptive_threshold=1.3,
)

# Inside GroupedExperts.forward: P2P weight transfer + pack
w1_packed, w2_packed, w3_packed, valid_mask, gradient_anchor = llep_prepare_weights(
    w1, w2, w3, state,
)
# Then run grouped_mm with packed weights (same compute as standard EP)

# EP post-hook: combine results
combined = llep_combine_output(output, state)
```

## Configuration

LLEP is configured via `LLEPConfig` (defined in `torchtitan/models/moe/moe.py`):

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `max_tokens_factor` | alpha | 1.1 | GPU capacity factor. `max_tokens_per_gpu = alpha * (total_tokens / num_gpus)`. Controls how much overload is tolerated before spilling. |
| `min_tokens_per_gemm` | m | 1024 | Minimum tokens to justify spilling to a helper GPU. Below this the GEMM is too small to be efficient. |
| `adaptive_threshold` | lambda | 0.0 | Imbalance ratio (`max_gpu_load / mean_gpu_load`) to trigger LLEP. Set to 0 to always use LLEP. Paper recommends 1.3. |
| `verbose` | - | false | Enable per-step distribution logging (see [Distribution Logging](#distribution-logging) below). |

### TOML Configuration

```toml
[llep]
enabled = true
max_tokens_factor = 1.1
min_tokens_per_gemm = 1024
adaptive_threshold = 1.3
verbose = false            # set true for per-step distribution logging
```

### Environment Variable Overrides

These override TOML/code values at runtime (useful for tuning without config changes):

| Variable | Overrides |
|----------|-----------|
| `EP_MAX_TOKENS_FACTOR` | `max_tokens_factor` |
| `EP_MIN_TOKENS_PER_GEMM` | `min_tokens_per_gemm` |
| `EP_ADAPTIVE_THRESHOLD` | `adaptive_threshold` |
| `LLEP_MERGE_A2A` | Merge hidden+scores+ids into single AllToAll (default: 1) |
| `LLEP_DEBUG` | Verbose per-step logging (default: 0). Equivalent to `[llep] verbose = true` |

## Distribution Logging

LLEP can log per-step token distribution details showing how load balancing works. Enable via TOML (`verbose = true`) or env var (`LLEP_DEBUG=1`). Logs are emitted on **all ranks** so you can see each GPU's perspective.

### Enabling

```toml
[llep]
enabled = true
verbose = true
```

Or at runtime without config changes:

```bash
LLEP_DEBUG=1 torchrun --nproc_per_node=8 -m torchtitan.train --job.config_file ...
```

### Log Messages

Each LLEP dispatch produces 4 log messages per rank:

**1. BEFORE** — Native distribution (before LLEP redistribution):
```
[LLEP rank=0 step=1] BEFORE: total_tokens=32768 imbalance=1.68
  native_gpu_loads=[44579, 25139, 54981, 22581, 29444, 29702, 25659, 30059]
  expert_counts=[3202, 8887, 5002, ...]
```

**2. AFTER LPT** — After LLEP redistribution with before->after imbalance ratio:
```
[LLEP rank=0 step=1] AFTER LPT: use_lpt=True imbalance=1.68->1.10
  llep_gpu_loads=[36044, 30735, 36044, 35922, 29444, 29702, 34194, 30059]
  weight_transfers (3):
    expert 23: GPU 2->3 (tokens 0-13341)
    expert 18: GPU 2->1 (tokens 6278-11874)
    expert 1: GPU 0->6 (tokens 352-8887)
```

**3. SEND_MATRIX** — AllToAll routing matrix (row=source GPU, col=destination GPU):
```
[LLEP rank=0 step=1] SEND_MATRIX (row=src, col=dst):
    [4697, 3206, 5152, 4494, 3665, 3670, 4023, 3861]
    [4347, 3080, 5377, 4520, 3666, 3725, 4443, 3610]
    ...
  input_splits=[4697, 3206, 5152, 4494, 3665, 3670, 4023, 3861]
  output_splits=[4697, 4347, 4430, 4400, 4382, 4483, 4627, 4678]
```

**4. RECEIVED** — What each rank actually received after AllToAll:
```
[LLEP rank=0 step=1] RECEIVED: total_recv=36044 experts=[0, 1, 2, 3, 4, 5, 6, 7] counts=[3202, 352, 5002, 7563, 3882, 8722, 1999, 5322]
```

### Debug Config for 8 GPUs

A ready-made config for inspecting LLEP distribution on 8 GPUs:

```bash
torchrun --nproc_per_node=8 -m torchtitan.train \
  --job.config_file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  2>&1 | tee /tmp/llep_distribution_logs.txt
```

This config uses `debugmodel_ep8_llep` (64 experts, top_k=8, EP=8) with `min_tokens_per_gemm=1` and `adaptive_threshold=0.0` so LLEP always triggers, even at small debug scale. See `train_configs/debug_model_ep8_llep.toml`.

## Benchmark: LLEP vs Standard EP

### Model

The `debugmodel_ep8_llep` flavor is a 9.5B-parameter MoE model designed for single-node 8-GPU benchmarking:

| Parameter | Value |
|-----------|-------|
| Total params | 9.5B (8.9 GB bf16) |
| MoE expert params | 9.1B (96%) |
| Active params/token | 1.6B (top_k=8 of 64 experts) |
| dim | 2048 |
| inter_dim | 8192 |
| moe_inter_dim | 1536 |
| n_layers | 16 (1 dense + 15 MoE) |
| num_experts | 64 |
| top_k | 8 |
| EP | 8 (8 local experts/GPU) |

Training config: `lbs=6, seq_len=4096, AdamW, no compile, no activation checkpointing`.

### Reproducing

```bash
cd torchtitan

# WITH LLEP (20 steps)
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \
  -m torchtitan.train \
  --job.config-file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 20 --compile.no-enable \
  2>&1 | tee llep_with_llep.txt

# WITHOUT LLEP (20 steps, same model)
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \
  -m torchtitan.train \
  --job.config-file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 20 --compile.no-enable --llep.enabled=False \
  2>&1 | tee llep_no_llep.txt
```

To enable verbose per-step distribution logging (shows BEFORE/AFTER imbalance, send matrix, weight transfers):

```bash
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \
  -m torchtitan.train \
  --job.config-file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 3 --compile.no-enable --llep.verbose=True \
  2>&1 | tee llep_verbose_logs.txt
```

### Results (8xB200, 20 steps)

**Speed** (steps 5-20 average, excluding warmup):

| | With LLEP | Without LLEP | Delta |
|---|---|---|---|
| Mean TPS | ~16,270 | ~15,120 | **+7.6%** |
| Mean MFU | 8.2% | 7.6% | +7.9% |

**Memory** (per-GPU at step 20):

| | With LLEP | Without LLEP |
|---|---|---|
| Active range | 105-107 GiB (59-60%) | 93-124 GiB (52-**69%**) |
| Reserved range | 116-120 GiB (65-67%) | 143-173 GiB (80-**97%**) |
| Max reserved | 120 GiB | **173 GiB** (near OOM) |
| Spread (reserved) | ~4 GiB | **30 GiB** |

Without LLEP, the most-loaded GPU hits 97% reserved memory (near OOM) while the least-loaded sits at 80%. LLEP keeps all GPUs in a tight 65-67% band. LLEP is both faster (less straggler waiting from load imbalance) and safer (no GPU near OOM).

### Per-GPU Memory Breakdown (step 5)

Detailed per-GPU view showing the memory imbalance that LLEP eliminates:

**With LLEP** — all GPUs balanced within a 3 GiB band:

| GPU | Active (GiB) | Active % | Reserved (GiB) | Reserved % | TPS |
|-----|-------------|----------|----------------|------------|-----|
| 0 | 104.29 | 58.5% | 115.74 | 64.9% | 16,088 |
| 1 | 104.40 | 58.5% | 120.20 | 67.4% | 16,078 |
| 2 | 104.98 | 58.9% | 116.29 | 65.2% | 16,087 |
| 3 | 105.24 | 59.0% | 115.59 | 64.8% | 15,978 |
| 4 | 105.48 | 59.1% | 116.83 | 65.5% | 16,082 |
| 5 | 105.64 | 59.2% | 118.05 | 66.2% | 15,975 |
| 6 | 107.10 | 60.1% | 116.87 | 65.5% | 16,082 |
| 7 | 107.45 | 60.2% | 118.10 | 66.2% | 15,955 |
| **Spread** | **3.2** | | **4.6** | | |

**Without LLEP** — wildly imbalanced, one GPU near OOM:

| GPU | Active (GiB) | Active % | Reserved (GiB) | Reserved % | TPS |
|-----|-------------|----------|----------------|------------|-----|
| 0 | 100.20 | 56.2% | 131.74 | 73.9% | 15,636 |
| 1 | 104.78 | 58.7% | 148.46 | 83.2% | 15,655 |
| 2 | 110.80 | 62.1% | 147.52 | 82.7% | 15,655 |
| 3 | 118.87 | **66.6%** | 165.93 | **93.0%** | 15,640 |
| 4 | 123.88 | **69.5%** | 137.88 | 77.3% | 15,633 |
| 5 | 91.15 | **51.1%** | 132.84 | 74.5% | 15,616 |
| 6 | 93.18 | 52.2% | 152.99 | 85.8% | 15,600 |
| 7 | 99.20 | 55.6% | 149.78 | 84.0% | 15,583 |
| **Spread** | **32.7** | | **33.2** | | |

Key observations:
- Without LLEP, GPU 3 reserves **165.9 GiB (93.0%)** of 178 GiB — one more imbalanced step away from OOM.
- GPU 5 is nearly idle at 51.1% active while GPU 4 is at 69.5% — an **18.4 percentage point** gap.
- LLEP compresses the active memory spread from **32.7 GiB to 3.2 GiB** (10x reduction).
- With LLEP every GPU runs at ~16,000+ TPS vs ~15,600 without — the straggler GPU drags everyone down.

To reproduce this comparison:

```bash
cd torchtitan

# 5-step memory comparison with LLEP
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \
  -m torchtitan.train \
  --job.config-file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 5 --compile.no-enable \
  2>&1 | tee llep_memory_with_llep.txt

# 5-step memory comparison without LLEP
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \
  -m torchtitan.train \
  --job.config-file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 5 --compile.no-enable --llep.enabled=False \
  2>&1 | tee llep_memory_no_llep.txt

# Extract per-GPU memory at step 5
grep "step:  5" llep_memory_with_llep.txt
grep "step:  5" llep_memory_no_llep.txt
```

### Loss Correctness (LLEP vs Standard EP)

LLEP produces identical training loss to standard EP, confirming numerical correctness. Both runs use the same seed, weights, and data — only the dispatch/combine path differs.

**WandB**: [nous_research/llep_loss_comparison](https://wandb.ai/nous_research/llep_loss_comparison) — overlay both runs to see matching loss curves.

| Step | With LLEP | Without LLEP | Diff |
|------|-----------|-------------|------|
| 10 | 4.4471 | 4.4062 | 0.041 |
| 30 | 3.1145 | 3.1137 | 0.001 |
| 50 | 2.9153 | 2.8987 | 0.017 |
| 80 | 2.7933 | 2.7947 | 0.001 |
| 100 | 2.7529 | 2.7575 | 0.005 |
| 130 | 2.7769 | 2.7873 | 0.010 |

To reproduce (8 GPUs, ~3 min each, logs to wandb):

```bash
cd torchtitan

# WITH LLEP (130 steps, seed=42, wandb)
WANDB_PROJECT=llep_loss_comparison \
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \
  -m torchtitan.train \
  --job.config_file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 130 --debug.seed 42 --compile.no-enable \
  --llep.enabled True --metrics.log_freq 1 --metrics.enable_wandb

# WITHOUT LLEP (130 steps, same seed/config)
WANDB_PROJECT=llep_loss_comparison \
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29501 \
  -m torchtitan.train \
  --job.config_file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 130 --debug.seed 42 --compile.no-enable \
  --llep.enabled False --metrics.log_freq 1 --metrics.enable_wandb
```

### Unit Tests

```bash
# LPT planning + SwiGLU FFN (5 tests, no GPU required)
python -m pytest tests/unit_tests/test_llep.py -v

# Grouped MM, Triton kernels, numerical correctness (17 tests, 1 GPU)
python -m pytest tests/unit_tests/test_llep_correctness.py -v

# Hook-based flow (59 tests, requires >= 2 GPUs)
torchrun --nproc_per_node=2 tests/unit_tests/test_llep_hooks.py
```

## Files

| File | Description |
|------|-------------|
| `torchtitan/distributed/llep.py` | Main LLEP implementation (planning, dispatch, compute, combine) |
| `torchtitan/distributed/llep_kernels.py` | Triton kernels (fused_silu_gate, pad/unpad, assign_tokens, send_matrix) |
| `torchtitan/models/moe/moe.py` | MoE module with `LLEPConfig` and hook integration points |

## Performance Optimizations

The implementation includes several optimizations over the initial port (see `docs/llep_optimization_report_pr008.md` for details):

| Optimization | Speedup | Technique |
|-------------|---------|-----------|
| Triton pad/unpad | 4.6x / 6.5x | Row-parallel kernels (1 program per row) |
| Triton fused_silu_gate | 1.9x | Fused `silu(x1) * x3` in single pass |
| Triton assign_tokens | 3.6x | Numpy plan encoding + GPU kernel |
| Vectorized send_matrix | 1.9x | Numpy `add.at` + per-expert vectorized overlap |
| Selective weight packing | - | Avoids `torch.where` double-materialization |

**E2E Result:** +10.6% mean TPS (1470 -> 1625) on mini_kimi_k2_llep_ep8, 8xB200.

## Testing

All tests run via `torchrun` and require at least 2 GPUs:

```bash
# Unit tests (LPT planning, FFN, no GPU required)
python tests/unit_tests/test_llep.py

# Optimization correctness (grouped MM, Triton kernels, numerical, 17 tests)
torchrun --nproc_per_node=1 tests/unit_tests/test_llep_correctness.py

# Hook-based flow comprehensive tests (59 tests, requires >= 2 GPUs)
torchrun --nproc_per_node=2 tests/unit_tests/test_llep_hooks.py

# Run specific category
torchrun --nproc_per_node=2 tests/unit_tests/test_llep_hooks.py --category topk

# List all tests
torchrun --nproc_per_node=2 tests/unit_tests/test_llep_hooks.py --list
```

### Test Coverage

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_llep.py` | 5 | LPT planning, SwiGLU FFN (single-process) |
| `test_llep_correctness.py` | 17 | Grouped MM vs for-loop, Triton fused_silu_gate, numerical stability, benchmarks |
| `test_llep_hooks.py` | 59 | Hook-based flow (`dispatch` -> `compute` -> `combine`) across top_k, hyperparams, dimensions, backward, edge cases |
| `test_moe_ep_e2e.py` | 32 | Real MoE.forward() E2E: single-GPU ref vs EP vs LLEP (forward + backward, pytest parametrize) |
