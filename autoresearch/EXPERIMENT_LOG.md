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

## async_TP first in compile_time_passes — discard (xxxxxxx)

- **Idea**: Move `async_tensor_parallel_pass` to position #1 so upstream
  `micro_pipeline_tp_pass` matcher sees `mm→RS` / `AG→mm` adjacency in
  the joint graph before view normalization / FSDP bucketing perturb it.
- **Changes**: Made `async_tensor_parallel_pass` the first appended pass,
  unconditionally (dropped the config gate).
- **Result**: tps = 7,164, mfu = 41.95%, memory = 47.52 GiB, wall_time = 79 s.
  Bitwise numerics PASS. 0 fusions: 421 "no producer matmul found for
  reduce scatter" skips. tps in ±1% of baseline.
- **Analysis**: Position didn't help. `DTensor.redistribute` (which
  produces the TP collectives during `make_fx`) decomposes via aten
  `view`/`_unsafe_view`/`reshape`/`_to_copy`/`permute` between `mm` and
  the collective. Upstream `find_producer_matmul` only traverses through
  a narrow whitelist (`aten.reshape.default`, `aten._to_copy`,
  `aten.view.default`); `_unsafe_view` and `permute` (which sit in the
  Q/K/V reshape path and row-parallel output path) break the chain.
- **Lessons**: `async_tensor_parallel_pass` at the joint-graph stage is
  not engageable from `passes.py` alone — the breakers are intrinsic to
  DTensor lowering, not artifacts of subsequent passes. Either upstream
  needs a looser matcher, or we'd need to rewrite `mm→{view|reshape|
  permute|_to_copy}*→RS` chains into direct adjacency ourselves
  (reimplementing a chunk of `micro_pipeline_tp_pass`). Out of scope.

---

## bucket_all_reduce after FSDP bucketing — crash (xxxxxxx)

- **Idea**: Coalesce 67 small TP `all_reduce` calls (RMSNorm weight grads,
  8 KB bf16 each on TP-2 group) into one merged bucket via upstream
  `torch._inductor.fx_passes.bucketing.bucket_all_reduce`. Bitwise-safe
  in bf16 because per-position rank-axis reduction is unchanged.
- **Changes**: Added `bucket_tp_all_reduce_pass` wrapping
  `bucket_all_reduce(gm)`, registered after
  `joint_transformer_block_bucketing_reordering_pass`.
- **Result**: Bitwise numerics test PASSED. Benchmark CRASHED with
  `RuntimeError: Argument 'view_4062' of Node '_to_copy_432' was used
  before it has been defined!` from `gm.graph.lint()`.
- **Analysis**: Bucketing logically engaged — the post-merge graph has
  one big `all_reduce(cat([67 grads]))` instead of 67 separate AR calls,
  bitwise correct. But the upstream FSDP `enable_fsdp_ag_rs_overlap=True`
  reorder had already moved consumers of those AR-wait outputs EARLIER
  in the graph (so the FSDP RS overlap can start sooner). After AR
  bucketing inserts the merged collective at `bucket_nodes[-1].next`,
  those moved-earlier consumers now reference an output defined LATER.
  Topologically invalid. Numerics test happens to pass because that test
  uses `enable_fsdp_ag_rs_overlap=False`, where the consumer reorder
  doesn't happen.
- **Lessons**: AR bucketing IS bitwise-safe (test passes). The crash is
  pure graph-topology, fixable by either (a) running AR bucketing BEFORE
  FSDP bucketing/overlap so FSDP sees 1 merged AR not 67, or
  (b) inserting the merged AR at an `insertion_point` that precedes the
  earliest moved consumer. Try (a) next — it's a one-line reorder.

---

## bucket_all_reduce before FSDP bucketing — crash (xxxxxxx)

- **Idea**: Move `bucket_tp_all_reduce_pass` to run BEFORE
  `joint_transformer_block_bucketing_reordering_pass` so the FSDP pass
  sees one merged AR instead of 67 (avoiding the consumer-reorder
  conflict that crashed the previous attempt).
- **Changes**: Added `bucket_tp_all_reduce_pass` between
  `normalize_view_ops_as_reshape` and the FSDP bucketing partial.
- **Result**: Numerics PASS. Benchmark CRASHED with the same
  `RuntimeError: Argument 'view_X' of Node '_to_copy_Y' was used before
  it has been defined!` from `gm.graph.lint()` inside our pass.
- **Analysis**: This rules out FSDP-overlap as the cause. The upstream
  `merge_all_reduce_bucket` always inserts the merged collective at
  `bucket_nodes[-1].next` (after the last bucketed AR). The split
  outputs are created at that position. But many consumers of the
  ORIGINAL waits live BETWEEN bucket_nodes[0] and bucket_nodes[-1].next,
  so after redirection they reference split outputs that are defined
  later → topologically invalid. This is a fundamental issue with
  `merge_all_reduce_bucket` when bucketed ARs are spread out across the
  graph: there's no single insertion point where (a) all inputs are
  defined and (b) the position precedes every consumer.
- **Lessons**: Upstream `bucket_all_reduce` is unusable as-is for
  scattered ARs. Fixing it requires either: (i) hoisting all bucket
  inputs to a single earliest position (expensive — disturbs FSDP grad
  accumulation chain), or (ii) moving every consumer of every bucket
  wait to a single latest position (also expensive). Both are
  reimplementations of the upstream pass. Park this direction. The 67×
  TP-AR launch overhead remains an open opportunity but needs a custom
  bucketer that respects topology — or a completely different approach
  (e.g. fold the AR into the downstream `reduce_scatter`'s
  collective-chunk math).

---

## 2-layer FSDP bucket plan — keep (PENDING)

- **Idea**: Profiling showed FSDP AllGather only 28% overlapped, 61.6 ms
  (5.8% of step) exposed. Default `get_default_transformer_block_buckets`
  emits 32 single-layer buckets → only 1 layer of compute behind each AG.
  Override `module_bucket_plans` with **2-layer buckets** (16 layer
  buckets + tok_embeddings + [norm, lm_head] = 18 total) so each AG
  covers 2 layers' params and can prefetch deeper.
- **Changes**: Construct `two_layer_bucket_plan` locally in
  `compile_time_passes` and pass it as `module_bucket_plans` to
  `joint_transformer_block_bucketing_reordering_pass`. Drop unused
  `get_default_transformer_block_buckets` import.
- **Result**: 3 runs at tps 7,239 / 7,214 / 7,236 (avg **7,229,
  +0.94%**), mfu 42.34% avg, memory 47.71 GiB (+0.19 GiB), wall_time
  79–82 s. Bitwise numerics PASS.
- **Analysis**: Small but consistent win — halves AG launch count
  (34→18) and gives 2 layers of compute behind each prefetched AG.
  Memory cost is tiny (2 layers of params live, not 1). The 0.94%
  recovers ~1/6 of the exposed AG; deeper buckets (4-layer, 8-layer)
  may keep paying with diminishing returns, traded against the
  monotonic memory cost.
- **Lessons**: (a) Bucket plan IS a tunable lever for FSDP comm overlap,
  not fixed at default. (b) Profile-driven targeting (find the biggest
  exposed comm bucket, scope an attack at it) works. (c) Worth sweeping
  bucket size — try 4-layer next.

---

## 4-layer FSDP bucket plan — discard (xxxxxxx)

- **Idea**: Continue the bucket-size sweep. Profiling at 2-layer left
  ~50 ms of AG still exposed; 4-layer should hide more.
- **Changes**: Replace 2-layer bucket plan with 4-layer (8 layer
  buckets + tok_embeddings + [norm, lm_head] = 10 total buckets).
- **Result**: 3-run avg tps **7,253** (+0.33% vs 2-layer), mfu 42.47%,
  memory 47.62 GiB, wall ~80 s. Numerics PASS.
- **Analysis**: Diminishing returns: 1→2 layer = +0.94%, 2→4 layer
  = +0.33% (within noise). Likely hit a saturation: either the
  remaining exposed AG is already mostly hidden by 2-layer prefetch,
  or the bottleneck has shifted (TP RS bf16 dominates exposed comm
  next). Memory cost was actually negligible (47.62 vs 47.71 GiB).
- **Lessons**: Don't waste more experiments on bigger bucket sizes —
  the curve is flat past 2-layer. Pivot to TP RS or compute-side.

---

## SwiGLU regional Inductor — discard (xxxxxxx)

- **Idea**: 32× forward `silu→mul` and 32× backward `silu_backward→mul`
  are unfused pointwise pairs (~40 ms / 14.8% of kernel time goes to
  small elementwise kernels). Tag both with `compile_with_inductor`
  so `regional_inductor_pass` produces a single fused Triton kernel
  per site. Pure pointwise math with no add-after-mul → no FMA
  promotion → should be bitwise-safe.
- **Changes**: Added `annotate_swiglu_for_regional_inductor_pass`
  in `passes.py` (mirrors `annotate_rmsnorm_for_regional_inductor_pass`
  template). Tags `aten.silu.default` / `aten.silu_backward.default`
  and their immediate `aten.mul.Tensor` consumers. Registered after
  `annotate_flex_attention_for_regional_inductor_pass` and before
  `regional_inductor_pass`.
- **Result**: 128 SwiGLU nodes tagged across 32 layers; Inductor
  compiled them. tps **7,239** (+0.14% vs 2-layer keep, noise).
  Memory 47.71 GiB, wall 112 s (compile overhead). Bitwise numerics
  PASS — confirms pure pointwise Inductor fusion preserves bitwise
  identity.
- **Analysis**: Fusion engaged but didn't move the needle. The
  silu+mul pair output is consumed by the `w2` matmul whose memory
  traffic dominates the savings from collapsing two pointwise
  kernels into one. Alternatively the bf16 silu+mul eager kernels
  are already well-coalesced, so kernel-launch savings are small.
- **Lessons**: **Important meta-finding — pointwise Inductor fusion
  IS bitwise-safe in this Run-3 setup.** That removes a barrier to
  more aggressive pointwise-tagging experiments (RoPE complex mul,
  residual chain, _to_copy chains). Where SwiGLU didn't pay, broader
  tagging of the 818-count `direct_copy` and 423-count residual `add`
  may still help. Compile time grew 79→112 s, so be mindful.

---
