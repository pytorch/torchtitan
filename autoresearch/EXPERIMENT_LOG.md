# Autoresearch Experiment Log

Cumulative log of every experiment in this run. Append-only — never
overwrite previous entries. Re-read at the start of each loop iteration to
learn from past experiments and avoid repeating failed approaches.

## Format

```markdown
## <short title> — <status> (<commit hash>)

- **Idea**: What optimization was attempted and why it was expected to help.
- **Changes**: What was actually modified (brief summary, not a full diff).
- **Result**: Perf numbers (tps, MFU, memory_gib, wall_time_s) or crash/error description.
- **Analysis**: Why it worked, why it didn't help, or why it broke.
- **Lessons**: Key takeaways — what to build on, what to avoid, what to try.
```

---

## Baseline — Run 3 production starting graph (2ffbabc5)

- **Idea**: Establish the Run-3 baseline. `passes.py` runs the production
  pass set: `remove_detach_pass`, `remove_identity_view/slice_pass`,
  `normalize_view_ops_as_reshape`, `joint_transformer_block_bucketing_reordering_pass`
  (with FSDP AG/RS overlap), `annotate_flex_attention_for_regional_inductor_pass`,
  `regional_inductor_pass`, `insert_kernel_annotations_pass`,
  `custom_codegen_pass`, then `cudagraph_pass`.
- **Changes**: None (measure on `2ffbabc5` HEAD).
- **Result**: tps = 7,162, mfu = 41.94%, memory = 47.52 GiB, wall_time = 84 s.
  Bitwise numerics test (`aot_fx_trace_vs_eager`) passes.
- **Analysis**: Within 6 tps of the documented production baseline (7,156).
  Run 2 best on the unoptimized starting graph reached 6,048 tps after 23 iters;
  this Run-3 starting point is already +18.4% above that. Run 2 saturated
  at ~6,040 tps because it couldn't access reference implementations of
  FSDP bucketing, AG/RS overlap, regional Inductor, kernel annotation, or
  custom codegen — all of which are now baked in.
- **Lessons**: Production graph is the new floor. Search must target wins
  on top of regional Inductor + FSDP overlap + CUDA graphs. The Run 2
  "easy wins" (detach removal, joint_graph_passes, schedule_overlap_bucketing,
  cudagraph wrapping) are all already present. Look for: (1) optimizer/loss
  folding into the captured graph (Run 2 noted 2.4% / 40 ms left outside);
  (2) bitwise-safe Inductor flag flips that the production stack hasn't
  set; (3) hot kernels under the regional-Inductor regions still missed;
  (4) idle GPU gaps post-AG/RS-overlap.

---

## joint_graph_passes after no-op cleanup — discard (xxxxxxx)

- **Idea**: Run upstream Inductor `joint_graph_passes` (CSE, noop removal,
  fold_concat_then_split, constant folding of uniform values, etc.) on the
  joint forward+backward graph right after `normalize_view_ops_as_reshape`
  and before `joint_transformer_block_bucketing_reordering_pass`. Hope was
  that a leaner graph would help bucketing/Inductor downstream.
- **Changes**: Added `apply_joint_graph_passes` wrapping
  `torch._inductor.fx_passes.joint_graph.joint_graph_passes`, registered
  in `compile_time_passes`.
- **Result**: tps = 7,170, mfu = 41.98%, memory = 47.52 GiB, wall_time = 81 s.
  Bitwise numerics PASS. Delta +0.11% — well within ±1% noise.
- **Analysis**: After `make_fx` tracing + existing no-op cleanup the joint
  graph already has little algebraic redundancy. Llama3 8B is dominated
  by large matmul/RMSNorm/SDPA/collective kernels; small op-count
  reductions don't move steady-state. The wall_time drop is plausibly
  startup variation.
- **Lessons**: Upstream Inductor's joint_graph_passes is bitwise-safe on
  this graph but a no-op for steady-state perf. Future cleanup wins
  probably need to target the *post-bucketing* / *post-regional-Inductor*
  graph (where new patterns appear) or fuse kernels that Inductor never
  touches (RMSNorm, residual add, embedding+norm). Don't retry generic
  pattern cleanup at the joint-graph stage.

---

## async_tensor_parallel_pass after bucketing — discard (xxxxxxx)

- **Idea**: 130 unbucketed TP all_gathers + 130 reduce_scatters on group `'22'`
  are launched separately every step. `async_tensor_parallel_pass` (already
  in `passes.py`, gated by config) calls upstream
  `micro_pipeline_tp_pass` to fuse `all_gather+mm → fused_all_gather_matmul`
  and `mm+reduce_scatter → fused_matmul_reduce_scatter` via NVLink symm_mem.
  Hard-enable it.
- **Changes**: Removed the `if config.parallelism.enable_async_tensor_parallel:`
  guard so `async_tensor_parallel_pass` always appends.
- **Result**: tps = 7,023 (-1.9%), mfu = 41.13%, memory = 47.56 GiB, wall_time = 80 s.
  Bitwise numerics PASS. Net regression.
- **Analysis**: The pass emitted **99 "no producer matmul found for reduce
  scatter, skipping fuse_matmul_reduce_scatter fusion"** warnings and **zero
  fusions** of either `fused_all_gather_matmul` or
  `fused_matmul_reduce_scatter`. By position #6 (after
  `normalize_view_ops_as_reshape` and
  `joint_transformer_block_bucketing_reordering_pass`) there are
  reshape/view ops between `mm` and the adjacent collective that break the
  upstream matcher's expected adjacency. The pass still walked the graph
  and registered symm_mem groups, adding overhead with no benefit.
- **Lessons**: Pass *position* is critical — `async_tensor_parallel_pass`
  must run **BEFORE** view normalization and bucketing reorder for the
  upstream matcher to find `mm→RS` / `AG→mm` adjacency. Next: move it to
  the very start of `compile_time_passes` (or just after `remove_detach_pass`).

---
