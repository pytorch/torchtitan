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
    llep_compute_with_weights,
    llep_combine_output,
)

# EP pre-hook: plan + dispatch tokens
dispatched_tokens, padded_counts, state = llep_dispatch_tokens(
    routed_input, num_tokens_per_expert, ep_group,
    max_tokens_factor=1.1,
    min_tokens_per_gemm=1024,
    adaptive_threshold=1.3,
)

# Inside GroupedExperts.forward: P2P weight transfer + compute
output = llep_compute_with_weights(
    dispatched_tokens, padded_counts,
    w1, w2, w3, state,
    use_grouped_mm=True,
)

# EP post-hook: combine results
combined = llep_combine_output(output, state)
```

### Legacy Monolithic API

For standalone use outside the MoE module:

```python
from torchtitan.distributed.llep import llep_moe_forward

output = llep_moe_forward(
    hidden_states, top_scores, selected_experts,
    w1, w2, w3, ep_group, num_experts,
    score_before_experts=True,
    max_tokens_factor=1.1,
    min_tokens_per_gemm=1024,
    adaptive_threshold=1.3,
)
```

## Configuration

LLEP is configured via `LLEPConfig` (defined in `torchtitan/models/moe/moe.py`):

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `max_tokens_factor` | alpha | 1.1 | GPU capacity factor. `max_tokens_per_gpu = alpha * (total_tokens / num_gpus)`. Controls how much overload is tolerated before spilling. |
| `min_tokens_per_gemm` | m | 1024 | Minimum tokens to justify spilling to a helper GPU. Below this the GEMM is too small to be efficient. |
| `adaptive_threshold` | lambda | 0.0 | Imbalance ratio (`max_gpu_load / mean_gpu_load`) to trigger LLEP. Set to 0 to always use LLEP. Paper recommends 1.3. |

### TOML Configuration

```toml
[training.model.moe]
use_llep = true

[training.model.moe.llep]
max_tokens_factor = 1.1
min_tokens_per_gemm = 1024
adaptive_threshold = 1.3
```

### Environment Variable Overrides

These override TOML/code values at runtime (useful for tuning without config changes):

| Variable | Overrides |
|----------|-----------|
| `EP_MAX_TOKENS_FACTOR` | `max_tokens_factor` |
| `EP_MIN_TOKENS_PER_GEMM` | `min_tokens_per_gemm` |
| `EP_ADAPTIVE_THRESHOLD` | `adaptive_threshold` |
| `LLEP_USE_GROUPED_MM` | Use grouped GEMM vs for-loop (default: 1) |
| `LLEP_W_TRANSFER_AUTOGRAD` | Enable autograd for weight transfer (default: 1) |
| `LLEP_MERGE_A2A` | Merge hidden+scores+ids into single AllToAll (default: 1) |
| `LLEP_DEBUG` | Verbose per-step logging (default: 0) |

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
# Unit tests (LPT planning, FFN, basic distributed)
python tests/unit_tests/test_llep.py
torchrun --nproc_per_node=2 tests/unit_tests/test_llep.py --distributed

# Forward/backward correctness vs standard EP (55 tests)
torchrun --nproc_per_node=2 tests/unit_tests/test_llep_correctness.py

# Hook-based flow comprehensive tests (63 tests)
torchrun --nproc_per_node=2 tests/unit_tests/test_llep_hooks.py

# Run specific category
torchrun --nproc_per_node=2 tests/unit_tests/test_llep_hooks.py --category topk

# List all tests
torchrun --nproc_per_node=2 tests/unit_tests/test_llep_hooks.py --list

# Triton kernel correctness + benchmarks
torchrun --nproc_per_node=2 tests/unit_tests/test_new_kernels.py
torchrun --nproc_per_node=2 tests/unit_tests/test_llep_bench.py
```

### Test Coverage

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_llep.py` | 6 | LPT planning, SwiGLU FFN, basic distributed forward |
| `test_llep_correctness.py` | 55 | Legacy `llep_moe_forward` vs standard EP and single-GPU reference (forward, backward, kernels, numerical) |
| `test_llep_hooks.py` | 63 | Hook-based flow (`dispatch` -> `compute` -> `combine`) across top_k, hyperparams, dimensions, backward, edge cases, and legacy parity |
| `test_llep_bench.py` | - | Benchmark: old vs optimized implementations |
| `test_new_kernels.py` | - | Triton kernel correctness (pad, unpad, send_matrix) |
