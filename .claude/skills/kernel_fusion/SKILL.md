---
name: kernel_fusion
description: >
  Fused kernel workflow: extract fusible regions from FX graph, generate
  optimized Triton kernels, and auto-select the best backend per op.
  Use when the user asks to find kernel fusion opportunities, generate
  custom kernels, or benchmark fused kernels. Invokes /kernel_fusion.
disable-model-invocation: true
---

# Fused Kernel Workflow

Single-pass design: the fused kernel pass discovers fusible regions,
writes problem.py for offline kernel generation, and only replaces
regions where a proven kernel beats eager. Replacement uses direct
`call_function(kernel_fn)` in the FX graph — zero dispatch overhead.

## Workflow

```bash
# Step 1: Train — extracts problems, zero graph modification
FUSED_KERNEL_DIR=/tmp/kernels ./run_graph_trainer_llama3_8b.sh

# Step 2: Generate kernels + benchmark (single command, offline)
python -m torchtitan.experiments.graph_trainer.kernel_gen.generate --dir /tmp/kernels

# Step 3: Train — only replaces regions with proven kernels
FUSED_KERNEL_DIR=/tmp/kernels ./run_graph_trainer_llama3_8b.sh
```

Step 1 and 3 are the same command. Step 2 runs offline.

## How It Works

### Conditional extract + replace (`fused_kernel_registry.py`)

The `fused_kernel_pass` runs after all graph restructuring passes,
before inductor compilation:

1. **Discovers** fusible regions using a pluggable extractor:
   - `fqn` (default): segments at module_fqn boundaries + union-find
   - `inductor`: uses inductor's `is_fusible_node` + `CapabilityBasedPartitioner`
   Config: `--compile.fused_kernel_extractor fqn|inductor`
2. **Splits** disconnected components via union-find
3. **Filters** unfusable ops: all `_c10d_functional.*`, `ao.*`,
   `bucketing.*` (collectives/offload), flash attention, embedding.
   Keeps matmul only with epilog compute (mm→cast, mm→add)
4. **Computes** a stable hash per region (ops + shapes)
5. **Writes** `problem.py` for all regions (step 1 — no graph modification)
6. **Only replaces** regions where `benchmark.json` proves a non-eager
   backend wins (step 3). Inserts `call_function(kernel_fn)` directly
   into the parent FX graph — no `call_module` wrapper, zero overhead.
   Regions without kernels or where eager wins stay untouched.

### Backend auto-selection

For each hash, reads `benchmark.json` and picks:
- **triton** if `triton_ms < eager_ms`
- **torch.compile** if `compile_ms < eager_ms` and `compile_ms < triton_ms`
- **eager** (no replacement) otherwise

No benchmark.json = no replacement. Unbenchmarked kernels are never used.

### Correctness validation

`benchmark.py` validates with:
- **Bitwise parity** (atol=0, rtol=0) for elementwise/view ops
- **Relaxed tolerance** for reductions (rtol=1e-3) and complex ops (rtol=1e-3)
- **5 random input trials** per kernel
- Auto-detects tolerance tier from ops in problem.py

### Hash-based directory layout

```
/tmp/kernels/
  a3f7b2c1/         # hash of (ops + shapes)
    problem.py       # written by step 1
    kernel.py        # written by step 2 (KernelAgent)
    benchmark.json   # written by step 2 (eager/compile/triton times)
  e9d1f4a0/
    problem.py
    ...
```

## Step 2: Generate Kernels

```bash
# Generate + benchmark (single command)
python -m torchtitan.experiments.graph_trainer.kernel_gen.generate \
    --dir /tmp/kernels --skip-optimize

# Specific problems only
python -m torchtitan.experiments.graph_trainer.kernel_gen.generate \
    --dir /tmp/kernels --problems a3f7b2c1 e9d1f4a0
```

### Handle Failures

If generation fails:
1. Check `{hash}/logs/session_*/test_0.py` for what was tested
2. Simplify the problem: single output, expand backward math, reduce sizes
3. Retry with `--problems <hash>`

## Benchmarking

### Microbenchmark (per-kernel, offline)

Run as part of step 2 (automatic) or standalone:

```bash
python -m torchtitan.experiments.graph_trainer.kernel_gen.benchmark --dir /tmp/kernels
```

Compares eager vs torch.compile vs triton per kernel with bitwise
validation. Writes `benchmark.json`.

### E2E compute benchmark (FAKE_BACKEND)

Use `FAKE_BACKEND=1` to run on a single GPU with collectives as no-ops.
This isolates pure compute time without needing 8 GPUs.

```bash
# Baseline:
FAKE_BACKEND=1 ./run_graph_trainer_llama3_8b.sh

# With fused kernels:
FAKE_BACKEND=1 FUSED_KERNEL_DIR=/tmp/kernels ./run_graph_trainer_llama3_8b.sh
```

Compare the last step's tps/tflops/mfu between the two runs.

A/B comparison with profiling + Perfetto upload:
```bash
FUSED_KERNEL_DIR=/tmp/kernels ./run_benchmark_fused_kernels.sh
FUSED_KERNEL_DIR=/tmp/kernels FAKE_BACKEND=1 ./run_benchmark_fused_kernels.sh
```

### E2E distributed benchmark (real NCCL)

```bash
# Baseline:
./run_graph_trainer_llama3_8b.sh

# With fused kernels:
FUSED_KERNEL_DIR=/tmp/kernels ./run_graph_trainer_llama3_8b.sh
```

## Files

```
torchtitan/experiments/graph_trainer/
  fused_kernel_registry.py    # single-pass: discover + replace + accelerate
  configs.py                  # fused_kernel_dir, fused_kernel_extractor
  passes.py                   # wiring into compile_time_passes
  kernel_gen/
    generate.py               # step 2: generate kernels + benchmark
    benchmark.py              # standalone microbenchmark + validation
    kernelagent_bridge.py     # KernelAgent API wrapper
    generated/.gitignore

run_graph_trainer_llama3_8b.sh     # FUSED_KERNEL_DIR + FAKE_BACKEND
run_benchmark_fused_kernels.sh     # A/B benchmark with profiling
```

## Don'ts

- Don't replace regions without benchmark.json — unbenchmarked = eager
- Don't benchmark at training time — use offline step 2
- Don't hardcode device indices in problems — use `torch.device('cuda')`
- Don't include collective ops in fused regions — blocked by namespace
- Don't use `call_module` for kernel dispatch — use `call_function` directly
