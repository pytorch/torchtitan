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
replaces each with a `call_module` wrapping the original subgraph
(zero-overhead eager fallback), and writes problem.py for offline
kernel generation. If kernel.py + benchmark.json exist, auto-selects
the fastest backend (eager, torch.compile, or triton) per op.

## Workflow

```bash
# Step 1: Train — extracts problems, runs with eager fallback
FUSED_KERNEL_DIR=/tmp/kernels ./run_graph_trainer_llama3_8b.sh

# Step 2: Generate kernels + benchmark (single command, offline)
python -m torchtitan.experiments.graph_trainer.kernel_gen.generate --dir /tmp/kernels

# Step 3: Train — auto-picks best backend per op from benchmark.json
FUSED_KERNEL_DIR=/tmp/kernels ./run_graph_trainer_llama3_8b.sh
```

Step 1 and 3 are the same command. Step 2 runs offline.

## How It Works

### Single-pass extract + replace (`fused_kernel_registry.py`)

The `fused_kernel_pass` runs after all graph restructuring passes,
before inductor compilation:

1. **Discovers** fusible regions using `module_fqn` boundaries
   (`layers.*.attention`, `layers.*.feed_forward`, etc.)
2. **Splits** disconnected components via union-find
3. **Filters** unfusable ops: all `_c10d_functional.*`, `ao.*`,
   `bucketing.*` (collectives/offload), flash attention, embedding.
   Keeps matmul only with epilog compute (mm→cast, mm→add)
4. **Computes** a stable hash per region (ops + shapes) — same region
   always maps to the same directory regardless of extraction order
5. **Replaces** each region with `call_module` wrapping a subgraph
   `GraphModule` (zero-overhead eager fallback)
6. **Checks** `{fused_kernel_dir}/{hash}/kernel.py` — if exists and
   `benchmark.json` shows it's fastest, swaps the module's forward
7. **Writes** `problem.py` for regions without kernels yet

### Backend auto-selection

At step 3, for each hash the pass reads `benchmark.json` and picks:
- **triton** if `kernel.py` exists and `triton_ms` is lowest
- **torch.compile** if `compile_ms` is lowest (compiled at graph time)
- **eager** if nothing beats the original subgraph

If no `benchmark.json` exists, uses triton if `kernel.py` exists,
otherwise eager.

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

# With NCU optimization (slower, better kernels)
python -m torchtitan.experiments.graph_trainer.kernel_gen.generate \
    --dir /tmp/kernels

# Specific problems only
python -m torchtitan.experiments.graph_trainer.kernel_gen.generate \
    --dir /tmp/kernels --problems a3f7b2c1 e9d1f4a0
```

`generate.py` does:
1. Generates Triton kernels via KernelAgent (parallel, 4 workers × 10 rounds)
2. Benchmarks each kernel (eager vs torch.compile vs triton)
3. Writes `benchmark.json` for backend selection

### Prerequisites

- KernelAgent at `~/local/KernelAgent` (or `$KERNEL_AGENT_ROOT`)
- `pip install anthropic omegaconf python-dotenv`
- API key: `claude-meta inference get-secret OPUS_FAST_API_KEY`

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

Compares eager vs torch.compile vs triton per kernel. Writes `benchmark.json`.

### E2E compute benchmark (fake_backend, no collectives)

```bash
FUSED_KERNEL_DIR=/tmp/kernels ./run_benchmark_fused_kernels.sh
```

Uses `--comm.mode=fake_backend` to replace all collectives with no-ops,
isolating pure compute time on a single GPU. Runs both baseline and
fused kernel training, compares tps/tflops/mfu.

With profiling enabled, uploads Perfetto traces for both runs. Open in
Perfetto to compare kernel-by-kernel.

Example results (Llama3 8B, compute-only):
```
BASELINE:     tps: 6,141  tflops: 355.68  mfu: 35.96%
WITH FUSED:   tps: 6,613  tflops: 382.96  mfu: 38.72%  (+7.7%)
```

### E2E distributed benchmark (real NCCL)

```bash
FUSED_KERNEL_DIR=/tmp/kernels ./run_graph_trainer_llama3_8b.sh
```

Full 8-GPU training with real collectives. Compare step 10 tps between
runs with and without `FUSED_KERNEL_DIR`.

## Files

```
torchtitan/experiments/graph_trainer/
  fused_kernel_registry.py    # single-pass: discover + replace + accelerate
  configs.py                  # fused_kernel_dir config
  passes.py                   # wiring into compile_time_passes
  kernel_gen/
    generate.py               # step 2: generate kernels + benchmark
    benchmark.py              # standalone microbenchmark
    kernelagent_bridge.py     # KernelAgent API wrapper
    generated/.gitignore

run_graph_trainer_llama3_8b.sh     # FUSED_KERNEL_DIR=<path> to enable
run_benchmark_fused_kernels.sh     # A/B compute-only benchmark
```

## Don'ts

- Don't include collective ops in fused regions — blocked by namespace
- Don't benchmark at training time — use offline step 2
- Don't hardcode device indices in problems — use `torch.device('cuda')`
- Don't assume kernel.py is always faster — benchmark.json decides
