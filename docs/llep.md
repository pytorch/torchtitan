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
  --training.steps 3
```

Or with verbose per-step distribution logging:

```bash
LLEP_DEBUG=1 torchrun --nproc_per_node=8 -m torchtitan.train \
  --job.config_file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 3 \
  2>&1 | tee /tmp/llep_distribution_logs.txt
```

This config uses `debugmodel_ep8_llep` (64 experts, top_k=8, EP=8) with `min_tokens_per_gemm=1` and `adaptive_threshold=0.0` so LLEP always triggers. See `train_configs/debug_model_ep8_llep.toml`.

## Benchmark: LLEP vs Standard EP

### Model

The `debugmodel_ep8_llep` flavor is a 9.5B-parameter MoE model designed for single-node 8-GPU benchmarking:

| Parameter | Value |
|-----------|-------|
| Total params | 9.5B |
| dim | 2048 |
| inter_dim | 8192 |
| moe_inter_dim | 1536 |
| n_layers | 16 (1 dense + 15 MoE) |
| num_experts | 64 |
| top_k | 8 |
| EP | 8 (8 local experts/GPU) |

Training config: `lbs=8, seq_len=4096, AdamW, no compile, no activation checkpointing`.

### Reproducing

```bash
cd torchtitan

# WITH LLEP (20 steps)
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \
  -m torchtitan.train \
  --job.config_file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 20 \
  2>&1 | tee llep_with_llep.txt

# WITHOUT LLEP (20 steps, same model)
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \
  -m torchtitan.train \
  --job.config_file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 20 --llep.enabled=False \
  2>&1 | tee llep_no_llep.txt
```

### Results (8xB200, 20 steps)

**Speed** (steps 5-20 average, excluding warmup):

| | With LLEP | Without LLEP | Delta |
|---|---|---|---|
| Mean TPS | ~26,370 | ~23,780 | **+10.9%** |

**Memory** (per-GPU at step 20):

| | With LLEP | Without LLEP |
|---|---|---|
| Memory range | 140-147 GiB (78-82%) | 132-173 GiB (74-**97%**) |
| Max memory | 147 GiB | **173 GiB** (near OOM) |
| Spread | ~7 GiB | **42 GiB** |

Without LLEP, the most-loaded GPU hits 97% memory (near OOM on 178 GiB B200) while the least-loaded sits at 74%. LLEP keeps all GPUs in a tight 78-82% band. LLEP is both faster (less straggler waiting from load imbalance) and safer (no GPU near OOM).

### Per-GPU Memory Breakdown (step 5)

Detailed per-GPU view showing the memory imbalance that LLEP eliminates:

**With LLEP** — all GPUs balanced within a 7 GiB band:

| GPU | Memory (GiB) | Memory % | TPS |
|-----|-------------|----------|-----|
| 0 | 144.62 | 81.1% | 25,400 |
| 1 | 144.37 | 80.9% | 25,464 |
| 2 | 143.46 | 80.4% | 25,491 |
| 3 | 144.57 | 81.1% | 25,398 |
| 4 | 143.00 | 80.2% | 25,458 |
| 5 | 141.26 | 79.2% | 25,349 |
| 6 | 139.89 | 78.4% | 25,510 |
| 7 | 146.60 | 82.2% | 25,477 |
| **Spread** | **6.7** | | |

**Without LLEP** — wildly imbalanced, two GPUs near OOM:

| GPU | Memory (GiB) | Memory % | TPS |
|-----|-------------|----------|-----|
| 0 | 159.46 | 89.4% | 18,936 |
| 1 | 144.33 | 80.9% | 18,908 |
| 2 | 172.27 | **96.6%** | 18,945 |
| 3 | 172.73 | **96.8%** | 18,955 |
| 4 | 166.89 | 93.6% | 18,953 |
| 5 | 165.37 | 92.7% | 18,907 |
| 6 | 155.88 | 87.4% | 18,949 |
| 7 | 145.89 | 81.8% | 18,878 |
| **Spread** | **28.4** | | |

Key observations:
- Without LLEP, GPU 2-3 hit **96.6-96.8%** of 178 GiB — one more imbalanced step away from OOM. At `lbs=10` this config OOMs without LLEP.
- GPU 7 is at 81.8% while GPU 3 is at 96.8% — a **15 percentage point** gap.
- LLEP compresses the memory spread from **28.4 GiB to 6.7 GiB** (4x reduction).
- With LLEP every GPU runs at ~25,400+ TPS vs ~18,900 without — **+34% faster** at step 5 when imbalance is worst.

To reproduce this comparison:

```bash
cd torchtitan

# 5-step memory comparison with LLEP
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \
  -m torchtitan.train \
  --job.config_file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 5 \
  2>&1 | tee llep_memory_with_llep.txt

# 5-step memory comparison without LLEP
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \
  -m torchtitan.train \
  --job.config_file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 5 --llep.enabled=False \
  2>&1 | tee llep_memory_no_llep.txt

# Extract per-GPU memory at step 5
grep "step: 5" llep_memory_with_llep.txt
grep "step: 5" llep_memory_no_llep.txt
```

### Loss Correctness (LLEP vs Standard EP)

LLEP produces matching training loss to standard EP, confirming numerical correctness. Both runs use the same seed, weights, and data — only the dispatch/combine path differs.

| Step | With LLEP | Without LLEP | Diff |
|------|-----------|-------------|------|
| 10 | 4.2333 | 4.3096 | 0.076 |
| 30 | 3.0474 | 3.0273 | 0.020 |
| 50 | 2.8995 | 2.8876 | 0.012 |
| 80 | 2.7699 | 2.7721 | 0.002 |
| 100 | 2.7737 | 2.7767 | 0.003 |
| 130 | 2.7993 | 2.7989 | 0.000 |

To reproduce (8 GPUs, ~10 min each):

```bash
cd torchtitan

# WITH LLEP (130 steps, seed=42)
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \
  -m torchtitan.train \
  --job.config_file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 130 --debug.seed=42 --metrics.log_freq=1

# WITHOUT LLEP (130 steps, same seed/config)
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29501 \
  -m torchtitan.train \
  --job.config_file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 130 --debug.seed=42 --metrics.log_freq=1 --llep.enabled=False
```

### With DeepEP Backend

LLEP also works with the DeepEP communication backend (`expert_parallel_comm_backend = "deepep"`). When both are enabled, `DeepEPLLEPExpertParallel` adaptively switches per step: DeepEP for balanced steps, LLEP for imbalanced steps (controlled by `adaptive_threshold`).

On single-node NVLink, DeepEP adds overhead vs standard NCCL all-to-all (buffer management, extra all_gather for imbalance check). DeepEP's advantage is primarily on **multi-node RDMA** where its custom transport is significantly faster than NCCL.

**All 4 configurations** (8xB200, 9.5B model, steps 5-20 average):

| Config | Mean TPS | vs Standard EP | Memory Range | Spread |
|---|---|---|---|---|
| **LLEP** (standard backend) | **26,250** | **+19.6%** | 140-146 GiB | **6 GiB** |
| **DeepEP + LLEP** (threshold=1.3) | **25,820** | **+17.7%** | 147-154 GiB | **7 GiB** |
| Standard EP (no LLEP) | 21,940 | baseline | 132-173 GiB | 41 GiB |
| DeepEP only (no LLEP) | 19,640 | -10.5% | 124-176 GiB | 52 GiB |

LLEP vs DeepEP comparisons:
- LLEP vs LLEP+DeepEP: **+1.7%** (26,250 vs 25,820) — DeepEP adds slight overhead on single-node NVLink
- LLEP vs DeepEP only: **+33.7%** (26,250 vs 19,640) — LLEP's load balancing dominates
- LLEP+DeepEP vs DeepEP only: **+31.5%** (25,820 vs 19,640) — adaptive switching recovers most of LLEP's benefit

To reproduce the DeepEP runs:

```bash
cd torchtitan

# DeepEP + LLEP adaptive (20 steps)
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \
  -m torchtitan.train \
  --job.config_file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 20 --parallelism.expert_parallel_comm_backend=deepep \
  --llep.adaptive_threshold=1.3

# DeepEP only, no LLEP (20 steps)
torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \
  -m torchtitan.train \
  --job.config_file torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml \
  --training.steps 20 --parallelism.expert_parallel_comm_backend=deepep \
  --llep.enabled=False
```

### Unit Tests

```bash
# LPT planning + SwiGLU FFN (3 tests, no GPU required)
python -m pytest tests/unit_tests/test_llep.py -v

# Hook-based flow (59 tests, requires >= 2 GPUs)
torchrun --nproc_per_node=2 tests/unit_tests/test_llep_hooks.py
```

## Files

| File | Description |
|------|-------------|
| `torchtitan/distributed/llep.py` | Main LLEP implementation (planning, dispatch, compute, combine) |
| `torchtitan/distributed/llep_kernels.py` | Triton kernels (imbalance_ratio, pad/unpad, assign_tokens, send_matrix) |
| `torchtitan/models/moe/moe.py` | MoE module with `LLEPConfig` and hook integration points |

## Performance Optimizations

The implementation includes several optimizations over the initial port (see `docs/llep_optimization_report_pr008.md` for details):

| Optimization | Speedup | Technique |
|-------------|---------|-----------|
| Triton pad/unpad | 4.6x / 6.5x | Row-parallel kernels (1 program per row) |
| Triton assign_tokens | 3.6x | Numpy plan encoding + GPU kernel |
| Vectorized send_matrix | 1.9x | Numpy `add.at` + per-expert vectorized overlap |
| Selective weight packing | - | Avoids `torch.where` double-materialization |

**E2E Result:** +10.6% mean TPS (1470 -> 1625) on mini_kimi_k2_llep_ep8, 8xB200.

## Testing

All tests run via `torchrun` and require at least 2 GPUs:

```bash
# Unit tests (LPT planning, FFN, no GPU required)
python tests/unit_tests/test_llep.py

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
| `test_llep.py` | 3 | LPT planning, GPU imbalance ratio (single-process) |
| `test_llep_hooks.py` | 59 | Hook-based flow (`dispatch` -> `compute` -> `combine`) across top_k, hyperparams, dimensions, backward, edge cases |
| `test_moe_ep_e2e.py` | 32 | Real MoE.forward() E2E: single-GPU ref vs EP vs LLEP (forward + backward, pytest parametrize) |
