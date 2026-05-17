---
name: kernel_fusion
description: >
  Analyze a dumped FX graph and profiling trace to identify fusible kernel patterns,
  formulate them as KernelAgent problems, and generate Triton kernels.
  Use when the user asks to find kernel fusion opportunities, generate custom kernels
  from an FX graph dump, or invokes /kernel_fusion.
disable-model-invocation: true
argument-hint: <graph_dump_path> [profile_trace_path]
---

# Kernel Fusion: Graph Analysis → Problem Extraction → Kernel Generation

End-to-end workflow that takes a dumped FX graph (and optional profiling trace),
identifies the most impactful fusion opportunities, writes KernelAgent-compatible
problem files, and generates Triton kernels.

## Prerequisites

- The graph dump file must exist (produced by `--compile.debug_graph_passes` in
  graph_trainer, written to `/tmp/final_graph_after_all_passes.txt`).
- Optional: a profiling trace JSON from `--profiler.enable_profiling` for
  kernel-level timing data.
- KernelAgent must be installed at `~/local/KernelAgent` (or `$KERNEL_AGENT_ROOT`).
- The `anthropic` pip package must be installed.
- Output goes to `autoresearch/kernel_agent/generated/<pattern_name>/`.

## Phase 1: Collect Inputs

Ask the user for:

1. **Graph dump path** — e.g., `/tmp/final_graph_after_all_passes.txt`
2. **Profile trace path** (optional) — e.g., `~/tmp/profile_traces/profiling/traces/iteration_10/rank0_trace.json`
3. **Model config** — which model was traced (e.g., Llama3 8B, DeepSeek-v3 16B) for context in descriptions

If the user provided these as arguments, use them directly.

Validate:
- Graph dump file exists and is non-empty
- Profile trace file exists if provided
- `~/local/KernelAgent` directory exists
- `python3 -c "import anthropic"` succeeds

## Phase 2: Analyze the Graph

### 2a. Op Frequency Analysis

Extract all `torch.ops.*` calls from the graph dump and count them:

```python
import re
from collections import Counter

with open(graph_dump_path) as f:
    content = f.read()

ops = re.findall(r'torch\.ops\.([\w.]+)\(', content)
op_counts = Counter(ops)
```

Show the top 40 ops by frequency. The user needs this to understand the graph composition.

### 2b. Profile Trace Analysis (if provided)

Parse the Chrome trace JSON and categorize GPU kernel time:

```python
import json
from collections import defaultdict

with open(trace_path) as f:
    trace = json.load(f)

events = trace.get('traceEvents', [])
gpu_kernels = [e for e in events if e.get('cat') == 'kernel' and 'dur' in e]
```

Categorize kernels into groups:
- **NCCL** (AllGather, ReduceScatter, AllReduce) — communication, not fusible
- **GEMM** (nvjet, cublas, cutlass) — already optimized
- **Flash Attention** (flash_fwd, flash_bwd) — already optimized
- **Fused Optimizer** (FusedOptimizerTensor) — already optimized
- **Copy/Cast** (direct_copy, bfloat16_copy, elementwise copy) — fusible
- **Elementwise** (add, mul, silu, silu_backward, exp, etc.) — fusible
- **RMSNorm** (vectorized_layer_norm, layer_norm_grad, GammaBeta) — fusible
- **Cat** (CatArrayBatchedCopy) — fusible
- **Reduce** (reduce_kernel) — fusible
- **Other**

Report: total GPU time, time per category, percentage breakdown. The **fusible**
categories are the optimization targets.

### 2c. Extract Fusible Regions

Use the extraction script to automatically discover fusible patterns from
the graph dump. It requires **no hardcoded pattern definitions** — it
discovers everything from graph structure:

```bash
# Dry run — show discovered regions without writing files
python3 -m autoresearch.kernel_agent.extract_from_dump /tmp/final_graph_after_all_passes.txt --dry-run

# Extract top 15 regions to generated/<name>/problem.py
python3 -m autoresearch.kernel_agent.extract_from_dump /tmp/final_graph_after_all_passes.txt --top-n 15

# Stricter filtering: only regions with ≥3 compute ops appearing ≥4 times
python3 -m autoresearch.kernel_agent.extract_from_dump graph.txt --min-compute-ops 3 --min-count 4
```

The script (`autoresearch/kernel_agent/extract_from_dump.py`) works as follows:

1. **Parse** the `print_readable` dump into structured nodes with shape/dtype/target.
2. **Segment** the graph at heavy-op boundaries (matmuls, collectives, flash
   attention, offload/reload). Everything between two heavy ops is one region.
3. **Deduplicate** by op-target signature — regions with the same sequence of
   op targets (e.g., `(_to_copy, reshape, view_as_complex, mul, view_as_real,
   reshape, _to_copy)`) are grouped together. Repeated across 32 layers = 1 unique
   signature with count 128.
4. **Filter** by `--min-ops`, `--min-compute-ops` (excluding metadata ops like
   reshape/view/transpose), and `--min-count`.
5. **Rank** by `count × compute_ops × tensor_bytes` (frequency × complexity × size).
6. **Generate** problem.py for the top-N regions, using the first instance as the
   representative (with concrete shapes from the graph dump).

Extracted problems use **raw aten IR** (`torch.ops.aten.silu.default(x)`) —
no translation to high-level PyTorch. Aten ops are valid callable Python and
preserve exact semantics. Type annotations from the dump are preserved as comments.

Flags:
- `--min-ops N`: minimum total ops per region (default: 2)
- `--min-compute-ops N`: minimum non-metadata ops (default: 1)
- `--min-count N`: minimum repetitions in the graph (default: 2)
- `--top-n N`: how many regions to extract (default: 20)

### 2d. Review Extracted Regions

The script prints a ranked table:

```
  #    Count  Ops Compute        Bytes Slug
  0       62   30      15   67108864  _to_copy_view_as_complex__to_copy_view_as_complex
  1      128   15       7   67108864  _to_copy_view_as_complex_mul_view_as_real
  5       64   14       6  117440512  mul_mul_silu_backward_view.dtype
  ...
```

Present this to the user. They can adjust filters and re-run, or proceed to
Phase 4 with the generated problem files.

### 2e. Limitations

- **Cross-collective patterns** (e.g., cross-entropy with allreduce between
  amax/sum) get split into separate regions at each allreduce boundary. The
  extracted subgraphs have external inputs for allreduce results. These need
  manual reformulation as self-contained algorithms (e.g., online softmax).
- **Metadata-only regions** (pure reshape chains) are filtered out by default
  via `--min-compute-ops 1`. They have zero compute but may have memory
  traffic (clone/contiguous). Set `--min-compute-ops 0` to include them.
- **Duplicate node names** across forward/recompute passes: the graph dump
  may reuse names. The extractor picks the first occurrence.

## Phase 3: Manual Problem Adjustments (if needed)

Most regions extract cleanly. For regions that need manual editing:

1. **Cross-collective patterns**: reformulate as self-contained online
   algorithms (the extraction gives you the op sequence to replicate).
2. **Backward ops** like `silu_backward`, `_fused_rms_norm_backward`:
   the kernel agent may struggle with these since they're compound ops.
   Expand into explicit arithmetic so the agent can write Triton.
   Keep the auto-extracted version as reference, write the expanded
   version alongside it.
3. **Include dtype casts explicitly** — the kernel must handle mixed bf16/f32.
4. **Single output preferred** — if a pattern has two outputs, consider splitting
   into separate problems or returning a tuple. Single output is easier for the agent.
5. **Description must include**: what the fusion does, how many times per step,
   the tensor shapes, and why it's worth fusing (memory traffic, kernel launch count).
6. **For complex backward math** (silu_backward, rmsnorm_backward): expand into
   explicit arithmetic rather than using PyTorch's backward ops. Show the math
   with intermediate variables. The agent generates Triton, not PyTorch.
7. **Validate syntax**: `ast.parse()` the code portion (after `import torch`) to
   catch errors before invoking the agent.

### Pattern-Specific Guidance

**RoPE**: The frequency tensor broadcasts over the head dimension. Use shape
`[1, seq_len, 1, head_dim//2]` for `complex64` freqs. Include both bf16→f32
cast and f32→bf16 cast in the problem.

**Cross-entropy loss**: The graph has `allreduce` between amax and sum for TP.
Exclude the allreduce from the problem — focus on the local compute only
(cast + amax + sub + exp + sum + log + gather + masking + mean).

**SwiGLU backward**: Expand `silu_backward(g, x)` as:
`sig = sigmoid(x); g * sig * (1 + x * (1 - sig))`. The agent needs explicit
arithmetic to write a Triton kernel.

**RMSNorm + residual backward**: Only include the input gradient, not the weight
gradient (which is a reduction). The input gradient is:
`d_input = (grad_normed - normed_input * mean(grad_normed * normed_input)) * rstd`.

**FSDP split→cat**: This is a layout transform — `split(x, chunk, dim=1)` then
`cat(chunks, dim=0)`. Simple but high-frequency.

## Phase 4: Generate Kernels

### 4a. Setup

Ensure the environment is ready:

```python
python3 -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/local/KernelAgent'))
from autoresearch.kernel_agent.bridge import _ensure_api_key, _ensure_proxy
_ensure_api_key()
_ensure_proxy()
from utils.providers import get_model_provider
p = get_model_provider('claude-sonnet-4-20250514', None)
print(f'Provider: {p.name}, available: {p.is_available()}')
"
```

If this fails:
- Missing `anthropic` → `pip install anthropic`
- Missing API key → check `claude-meta inference get-secret OPUS_FAST_API_KEY`
- Missing proxy → check `with-proxy env`

### 4b. Run the Agent

Use the runner script at `autoresearch/kernel_agent/run_all.py`. By default it
runs **both** generation and NCU-guided optimization in sequence:

```bash
python3 -m autoresearch.kernel_agent.run_all --max-parallel 5
```

This will:
1. **Generate** kernels for problems missing `kernel.py` (parallel, 4 workers × 10 rounds)
2. **Optimize** each kernel with NCU profiling (sequential, since NCU needs exclusive GPU)
3. Save results to `generated/<name>/kernel.py` (initial) and `optimized_kernel.py` (best)

The optimization step uses `bridge.optimize_kernel()` which:
- Auto-detects the GPU name (maps `torch.cuda.get_device_name()` to KernelAgent's spec DB)
- Strips the description preamble from problem.py (writes `problem_clean.py`)
- Auto-finds the test code from the generation session logs
- Uses `claude-sonnet` via `AnthropicProvider` (not the Meta AI gateway)
- Benchmarks against PyTorch eager, torch.compile, and the initial Triton kernel
- Runs NCU to profile bottlenecks and iteratively improve the kernel

For specific problems only:
```bash
python3 -m autoresearch.kernel_agent.run_all --problems rope_fwd swiglu_bwd
```

For generation only (skip NCU optimization):
```bash
python3 -m autoresearch.kernel_agent.run_all --skip-optimize
```

For NCU optimization only (kernels already generated):
```bash
python3 -m autoresearch.kernel_agent.run_all --optimize-only --opt-rounds 10
```

For debugging a single problem:
```bash
python3 -m autoresearch.kernel_agent.run_all --problems rope_fwd --sequential
```

### 4c. Handle Failures

If generation fails after 10 rounds:

1. **Check the logs** in `generated/<name>/logs/` — look at the latest
   `session_*/test_0.py` and `session_*/seed_*.py` to see what the agent tried.
2. **Simplify the problem** — common fixes:
   - Reduce from multi-output (`tuple[Tensor, Tensor]`) to single output
   - Expand complex backward math into explicit arithmetic
   - Use `.float()` explicitly instead of `dtype=torch.float32` kwarg
   - Reduce tensor size if the test times out (e.g., seq_len 8192 → 2048)
3. **Retry** with the simplified problem.

If NCU optimization fails:
- Check `generated/<name>/opt_logs/` for worker logs
- Common issues: GPU name not recognized (check `_GPU_NAME_MAP` in `bridge.py`),
  NCU binary not found, or kernel too large for NCU timeout (300s default)

## Phase 5: Standalone Benchmark (optional)

The benchmark script is useful for quick validation without running the full
optimizer, or to compare initial vs optimized kernels:

```bash
python3 -m autoresearch.kernel_agent.benchmark_all
python3 -m autoresearch.kernel_agent.benchmark_all --problems rope_fwd swiglu_bwd
```

This measures correctness (max error) and wall-clock timing (CUDA events, 100 iters)
for each kernel vs its PyTorch reference.

### Interpreting Results

- **Speedup > 1.0x** means the Triton kernel is faster than the PyTorch reference.
  Good kernels typically show 2-15x speedup on elementwise fusions.
- **Max error** for bf16 kernels: values up to 0.1-0.25 are normal when the tensor
  range is large (e.g., [-40, 40]). Check that mean error is small (~1e-3).
  The key metric is that relative error stays within bf16 precision (~0.8%).
- **Failures**: if a kernel fails correctness, check if it's a Triton compilation
  error (e.g., `multi_tensor_norm` with pointer-to-pointer patterns) vs. a
  numerical error. Compilation errors need problem reformulation; numerical errors
  may just need looser tolerances.

## Phase 6: Report

The runner prints a summary table with NCU optimization results:

```
Kernel                       Status  Eager     Compile   Initial   Best      vs Eager
-------------------------------------------------------------------------------------------------
dtype_cast_add               OK      0.5121    0.1744    0.0971    0.0970    5.28x
cross_entropy_loss           OK      9.7080    1.2340    1.1660    0.9800    9.91x
```

Present the final summary to the user:

1. How many patterns were identified and their estimated GPU time
2. How many kernels were successfully generated
3. **NCU benchmark table**: PyTorch eager, torch.compile, initial kernel,
   optimized kernel, and speedup vs eager for each
4. Total estimated GPU time savings (sum of `(eager_ms - best_ms) * instances_per_step`)
5. Next steps: integrate kernels as graph passes using `autoresearch.kernel_agent.integrate`
   (see `example_swiglu.py` for the integration pattern)

## Don'ts

- Don't modify any core torchtitan code — all output goes to `autoresearch/`.
- Don't run the kernel agent without first validating the problem.py syntax.
- Don't include collective ops (allreduce, allgather) in problem formulations — only local compute.
- Don't use ProcessPoolExecutor without setting up API key/proxy in the main process first — child processes inherit env vars from the parent.
- Don't include more than ~15 ops in a single problem — split complex patterns into smaller pieces.
- Don't use PyTorch backward ops (like `silu_backward`) in problems — expand to explicit forward math so the agent can write Triton.
