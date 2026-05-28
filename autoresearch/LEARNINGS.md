# Autoresearch Learnings

Living document of what works, what doesn't, and how to approach kernel
fusion optimization effectively. The agent owns this file.

Re-read at the start of each loop iteration alongside `IDEAS.md` and
`EXPERIMENT_LOG.md`. Update after meaningful experiments (especially
surprising results, both positive and negative). Keep concise and
actionable — per-experiment details belong in `EXPERIMENT_LOG.md`.

## Methodology

- Production starting graph baseline = 7,162 tps. Treat ±1% (≈±70 tps) as
  noise. A single benchmark within ±1% should be re-run before keeping or
  discarding if the change is otherwise promising.
- Each experiment edits only `passes.py`. Revert with
  `git checkout -- torchtitan/experiments/graph_trainer/passes.py` before
  starting a new experiment so the baseline graph is consistent.
- Numerics gate FIRST, perf SECOND: a perf win that fails
  `aot_fx_trace_vs_eager` is `crash`, not `keep`.

## Key structural findings (graph after production passes)

- **Regional Inductor compiles nothing in this config.** Llama3 uses
  `_scaled_dot_product_cudnn_attention` (cuDNN SDPA), not FlexAttention
  HOPs, so `annotate_flex_attention_for_regional_inductor_pass` finds no
  HOPs to tag and `regional_inductor_pass` is a no-op. The whole
  regional-Inductor chain is dormant — `compile_with_inductor`
  annotations are absent in the dumped graph.
- **TP collectives are unbucketed**: 130 `all_gather_into_tensor` + 130
  `reduce_scatter_tensor` on group `'22'` (TP-2). Each is a separate
  small NCCL launch.
- **TP weight-grad all_reduces are unbucketed**: 67 small `all_reduce('sum')`
  on group `'22'` for RMSNorm-weight gradients (8 KB bf16 each), launched
  serially in backward.
- **FSDP collectives ARE bucketed**: 34 bucketed AGs (forward) + 34
  bucketed RSs (backward) via `bucketing._pre_bucket_*` wrappers. Healthy.
- **Hot unfused pointwise patterns** outside any Inductor region: 32×
  RoPE blocks (mul/view_as_complex/view_as_real), 32× SwiGLU
  (`silu→mul`), 32× residual `add` immediately preceding RMSNorm.
- **290 FSDP weight `clone(contiguous_format)` calls** — one per param,
  caused by bf16 bucket-buffer slice being non-contiguous. Large memcpy.

## Patterns that worked

(none yet)

## Patterns that didn't work

- Adding upstream Inductor `joint_graph_passes` at the joint-graph stage
  (after `normalize_view_ops_as_reshape`, before bucketing): bitwise-safe
  but +0.11% (noise). Llama3 8B has little algebraic redundancy at this
  stage; existing no-op cleanup already covers what it would remove.
- Hard-enabling `async_tensor_parallel_pass` **after** bucketing/view
  normalization: 0 fusions engaged because reshape/view ops between
  `mm` and the adjacent collective break upstream `micro_pipeline_tp_pass`'s
  adjacency matcher. Net -1.9% from pass overhead alone. The pass MUST
  run before view normalization and bucketing for the matcher to find
  `mm→RS` / `AG→mm` adjacency.

## Tooling tips

- `bash autoresearch/scripts/run_benchmark.sh` produces `run.log`.
- `grep "step:" run.log | tail -1` → steady-state tps/mfu/memory line.
- `grep "benchmark_wall_time_s" run.log` → total wall time.
- Numerics check (must pass): `pytest torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py -k "TestLlama3BitwiseDeterministic and aot_fx_trace_vs_eager" -x > numerics.log 2>&1`.
  Failure → `crash` regardless of perf.
- Inspect graph: add a one-line `gm.print_readable(...)` inside a temp pass,
  run once, then remove before benchmarking.
