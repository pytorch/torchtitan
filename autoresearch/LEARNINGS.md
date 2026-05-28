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
  discarding if the change is otherwise promising. For wins above the
  threshold, run 3× to confirm consistency.
- Each experiment edits only `passes.py`. Revert with
  `git checkout -- torchtitan/experiments/graph_trainer/passes.py` before
  starting a new experiment so the baseline graph is consistent.
- Numerics gate FIRST, perf SECOND: a perf win that fails
  `aot_fx_trace_vs_eager` is `crash`, not `keep`.

## Key structural findings (graph after production passes)

- **Regional Inductor compiles nothing in the default config.** Llama3 uses
  `_scaled_dot_product_cudnn_attention` (cuDNN SDPA), not FlexAttention
  HOPs, so `annotate_flex_attention_for_regional_inductor_pass` finds no
  HOPs to tag and `regional_inductor_pass` is a no-op until WE tag patterns.
- **TP collectives are unbucketed**: 130 `all_gather_into_tensor` + 130
  `reduce_scatter_tensor` on group `'22'` (TP-2). Each is a separate small
  NCCL launch.
- **TP weight-grad all_reduces are unbucketed**: 67 small `all_reduce('sum')`
  on group `'22'` for RMSNorm-weight gradients (8 KB bf16 each), launched
  serially in backward.
- **FSDP collectives ARE bucketed by default**: 34 bucketed AGs (fwd) + 34
  bucketed RSs (bwd) via `bucketing._pre_bucket_*` wrappers.
- **Hot unfused pointwise patterns**: 32× RoPE blocks (complex mul, ~9 ms),
  32× SwiGLU (silu→mul), 32× residual `add` before RMSNorm.

## Patterns that worked (cumulative +5.07% over Run 3 production)

1. **Auto-bucketing replaces manual N-layer FSDP plans (+0.90%)**. Upstream
   `schedule_overlap_bucketing(collective_bucketing=True,
   max_memory_increase_gb=2.0, max_memory_increase_ratio=0.05)` produces
   a tighter, bandwidth-saturating schedule (131 AG / 163 RS buckets vs
   the 18 from a 2-layer manual plan). Compile time +49 s.

2. **Remove redundant `clone(contiguous_format)` (+0.34%)**. Walk the FX
   graph for `aten.clone.default` whose input is already contiguous per
   FakeTensor stride; replace uses with input. 68 redundant clones per step.
   `gm.named_children()` descent and post-bucketing rerun add 0 more.

3. **Custom `mm → reshape → reduce_scatter` fusion (+1.17%)**. Upstream
   `micro_pipeline_tp_pass`'s matcher is too strict (`reshape→mm→reshape`
   only); write a looser custom matcher that walks back through
   view/reshape/_to_copy/permute/transpose/t/_unsafe_view + cat(split())
   to find producer mm, then replace with `symm_mem.fused_matmul_reduce_scatter`.
   65 forward TP fusions; dtype-filter skips the 225 backward weight-grad RS
   that upcast to fp32 (`fused_matmul_reduce_scatter` reduces in mm.A dtype).
   **Bitwise-safe** because chunked matmul + chunked RS doesn't reorder per-position
   rank-axis reduction.

4. **Shape-fix in `fuse_mm_reduce_scatter_pass` (perf-neutral; enables future work)**.
   The fused op output is 2D `[M/G, N]` at runtime but original `wait_tensor`
   meta was 3D `[1, S, D]`. Stamp the correct 2D fake on the fused-op node,
   then insert an `aten.reshape.default` back to 3D so downstream consumers
   see consistent shape. Required for any downstream Inductor scoop touching
   fused-mm-RS outputs.

5. **Residual-add Inductor compilation (+0.26%)**. Tag every bf16
   `aten.add.Tensor` with `compile_with_inductor={}`. Inductor produces a
   `triton_poi_fused_add_0` kernel per region. Bitwise-safe (pure pointwise).
   224 nodes tagged.

6. **Extended SwiGLU chain Inductor compilation (+1.36%)**. Seed with
   bf16 silu/silu_backward, then iteratively expand to adjacent bf16
   `mul.Tensor` whose all-Tensor inputs are bf16. Captures 5-op
   forward+backward SwiGLU chain (vs original 2-op-only). 160 nodes tagged
   yielding 64 fused regions (32 fwd + 32 bwd). Bitwise-safe because pure
   bf16 pointwise.

## Patterns that didn't work

- **Upstream `joint_graph_passes` / `post_grad_passes`** at any position
  (+0.11% / +0.23% — noise). Graph is already too clean for upstream
  pattern matchers; main wins come from MORE Inductor fusion of UNHANDLED
  patterns (silu, add), not from re-CSE/folding.
- **Async TP (`async_tensor_parallel_pass`)** at 3 different pass
  positions — 0 fusions every time. Upstream `_find_producer_matmul`
  whitelist (`reshape`, `_to_copy`, outer `view` only) doesn't match the
  DTensor-lowered `mm → {view|permute|transpose|...}* → RS` chains in our
  graph. Position doesn't matter; the matcher itself is the limit. Our
  custom `fuse_mm_reduce_scatter_pass` is the fix.
- **Upstream `bucket_all_reduce`** for the 67 TP weight-grad ARs:
  topology bug. The merged collective is inserted at `bucket_nodes[-1].next`,
  but consumers of earlier-bucketed ARs were *before* that position →
  graph.lint() fails with "used before defined". Fixing requires custom
  AR coalescer that respects topology (out of scope).
- **Larger / smaller FSDP bucket cap** (8, 64, 128 MB): default 40.2 MB is
  near-optimal at this scale. Smaller doesn't help (singletons), larger
  doesn't help either (bandwidth already saturated).
- **Bucketer param tuning** (`compute_overlap_multipler`,
  `max_memory_increase_gb=8/16`, `max_in_flight_gb=32`, `bucket_mode="custom_ops"`,
  `pre_bucketing_fsdp_collectives=False`): all in noise. The bucketer is
  at a local optimum for this configuration; the binding constraint is the
  roofline cost model (which doesn't see Triton/Inductor compute times)
  not the memory or mode parameters.
- **Re-running `auto_overlap_bucketing_pass` after `regional_inductor_pass`**:
  -2.5%. The bucketer treats `invoke_subgraph` as opaque, so post-Inductor
  rescheduling doesn't help — it just re-shuffles in ways that disrupt
  the original schedule.
- **`overlap_fsdp_ag_rs_pass` (separate stream for AG vs RS)**: 0% perf
  change. Auto-bucketing already overlaps enough.
- **Custom `AG → mm` fusion (mirror of mm-RS)**: bitwise-safe but -0.13%
  with +3.87 GiB memory cost. Forward AG's gathered tensor is reused by
  backward weight-grad mm via `t(gathered)`, forcing `return_A=True`
  which holds the activation live for backward. Auto-bucketing already
  overlapped most forward AG.
- **Extending `fuse_mm_reduce_scatter_pass` to backward dx-RS add-tree
  chains**: -0.74%. The backward `add_tree(RS([...]))` pattern with our
  push-RS-through-leaves rewrite produces 2-3x smaller RS ops, exceeding
  the auto-bucketer's overhead. The original add-tree pattern is more
  efficient as a single large RS.
- **RoPE complex → real-valued rewrite**: numerics CRASH. Aten complex mul
  uses single-FMA (`fma(a,c,-b*d)`) per output component; our 4-mul-2-add
  real decomposition does two roundings → different last bit in bf16.
  Hard floor (~9 ms) under strict bitwise.
- **Broad pointwise Inductor tagging including `view_as_complex/real`**:
  Inductor can't codegen complex ops, fragments graph into many tiny
  regions, -0.51% net.
- **Disabling `custom_codegen_pass`**: 0% — cudagraph replay bypasses
  FX forward at steady state.
- **`remove_noop_to_copy_pass`**: 0 matches — no `_to_copy.default` in the
  graph is a kwargs-only no-op (all are genuine dtype/layout casts).

## Key insights

- **Cumulative wins build from many small + 2-3 large fusion wins.**
  The biggest single experiment was custom mm-RS fusion (+1.17%);
  combined with auto-bucketing (+0.90%) and SwiGLU chains (+1.36%),
  these three account for >70% of the total +5.07%.
- **Inductor regional compilation IS bitwise-safe for pure bf16
  pointwise ops** (proven by silu+mul, then add.Tensor, then
  silu_backward chains). Bitwise breaks when ops involve reductions
  (RMSNorm), complex math (RoPE), or cross-dtype casts where reduction
  order matters (fp32 grad upcasts).
- **Connected-component (seed+expand) tagging > op-by-op tagging**:
  bigger Inductor regions amortize per-region overhead better. SwiGLU
  with iterative expand (160 nodes, +1.36%) >> original SwiGLU pair
  tagging (128 nodes, +0.14%).
- **Bucketer roofline cost model has limits**: post-Inductor compute
  is shorter than roofline thinks, so pre-Inductor bucketing
  over-prefetches AG. Re-bucketing after Inductor doesn't fix this
  because `invoke_subgraph` is treated as opaque.

## Tooling tips

- `bash autoresearch/scripts/run_benchmark.sh` → `run.log`.
- `grep "step:" run.log | tail -1` → steady-state tps/mfu/memory line.
- `grep "benchmark_wall_time_s" run.log` → total wall time.
- Numerics: `pytest torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py -k "TestLlama3BitwiseDeterministic and aot_fx_trace_vs_eager" -x > numerics.log 2>&1`. Failure → `crash` regardless of perf.
- Inspect graph: add a one-line `gm.print_readable(...)` in a temp pass, run 1-2 steps, remove before benchmarking.
- Profile: add `--profiler.enable_profiling --profiler.profile_freq 15` and `--training.steps 25` to the benchmark cmd; trace lands in `outputs/profile_traces/iteration_15/rank0_trace.json`. Parse with Python (top kernels, exposed comm per collective).
