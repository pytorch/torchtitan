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

## Broad pointwise regional Inductor — discard (xxxxxxx)

- **Idea**: Building on SwiGLU bitwise-safety result, tag a broader
  set of pointwise targets (silu, mul, add, sub, neg, _to_copy,
  view_as_complex/real) for regional Inductor — primarily to fuse the
  RoPE chain `_to_copy→reshape→view_as_complex→mul→view_as_real→
  reshape→_to_copy` (6 kernels per layer per Q/K, ~128 sites) into
  one Triton kernel.
- **Changes**: Added `annotate_pointwise_chains_for_regional_inductor_pass`
  that tags all of {silu, silu_backward, mul.Tensor, mul.Scalar,
  add.Tensor, add.Scalar, sub.Tensor, neg, _to_copy, view_as_complex,
  view_as_real}. Inserted between flex-attention annotation and
  `regional_inductor_pass`.
- **Result**: 1328 pointwise nodes tagged (~10× SwiGLU's 128). tps
  **7,192 (-0.51%)** vs 2-layer keep, mfu 42.11%, memory 47.71 GiB,
  wall **246 s** (vs 79 s baseline — graph passes alone took 172 s).
  Bitwise numerics PASS.
- **Analysis**: The fusion target — RoPE complex multiplication —
  triggers Inductor's `"does not support code generation for complex
  operators"` warning. `view_as_complex` / complex `mul.Tensor` /
  `view_as_real` fall back to non-fused codegen, so the intended
  6-kernels-to-1 RoPE win never materializes. Meanwhile the rest of
  the broad tagging fragments regions across half the graph, growing
  compile time hugely with no steady-state benefit.
- **Lessons**: **Inductor cannot codegen complex ops.** To fuse RoPE
  we'd need a graph rewrite that converts
  `view_as_complex→mul(complex)→view_as_real` into the equivalent
  4-mul-2-add real-valued form **before** tagging. Risk: FMA promotion
  may break bitwise (Inductor may emit `fma(a,c,-b*d)` whereas eager
  complex_mul does separate mul/sub). Also: broad pointwise tagging
  is high compile-time cost / low steady-state payoff because most
  pointwise time sits next to matmul/comm that dominates anyway.

---

## Auto overlap bucketing (schedule_overlap_bucketing) — keep (PENDING)

- **Idea**: Replace the manual `joint_transformer_block_bucketing_reordering_pass`
  (with 2-layer bucket plan) with upstream
  `torch._inductor.fx_passes.overlap_scheduling.schedule_overlap_bucketing(collective_bucketing=True, ...)`.
  Auto-bucketer uses roofline runtime estimates to pick bandwidth-saturating
  bucket sizes and reorders for maximal compute/comm overlap, respecting
  a memory budget (`max_memory_increase_gb=2.0`, ratio=0.05).
- **Changes**: Added `auto_overlap_bucketing_pass`, removed manual 2-layer
  bucket plan construction. Kept `overlap_fsdp_ag_rs_pass` gated on
  `config.compile.enable_fsdp_ag_rs_overlap` (default False; the benchmark
  doesn't set it, so the extra-stream split is OFF for both manual-keep
  and this experiment).
- **Result**: 3 runs **7,291 / 7,305 / 7,287** (avg **7,294**, +0.90% vs
  manual 2-layer, +1.84% vs Run 3 production baseline). mfu 42.72%
  avg, memory **49.06 GiB** (+1.35 GiB), wall **128 s** (vs 79 s
  manual — compile overhead from auto-bucketer's roofline estimation).
  Bitwise numerics PASS.
- **Analysis**: Auto-bucketer picked finer-grained buckets (131 AG / 163
  RS buckets, vs manual's 18 in each direction) that better saturate
  NCCL bandwidth. The +0.90% is just under the strict +1% threshold but
  consistent across 3 runs (range 7,287–7,305) with the worst run still
  beating manual 2-layer (avg 7,229). Memory increased 1.35 GiB —
  expected from larger in-flight buckets, well within the 95 GB H100
  budget.
- **Lessons**: Roofline-driven auto-bucketing dominates manual N-layer
  bucketing here. The compile-time cost (49 s extra) is acceptable.
  Code is also simpler — no manual bucket plan to maintain. Future
  experiments should layer on top of this rather than swap it out.

---

## Re-profile post auto_overlap_bucketing — discovery (no commit)

- **Idea**: Profile current keep state (auto_overlap_bucketing) to find
  the new dominant bottleneck.
- **Result**: GPU 98.2% busy, total comm 325 ms / step (63% overlapped).
  Exposed comm: 119 ms (11.5% of 1037 ms step).
  - TP RS bf16: **61.5 ms exposed (2.6% overlap)** — biggest remaining
    target, untouched by auto-bucketing (which only processes FSDP
    group '41', not TP group '22').
  - FSDP AG: 53.8 ms (30.9% overlap, improved from 28% in baseline)
  - FSDP RS f32: 34.7 ms (83% overlap, +5 ms worse than baseline)
  - TP AR bf16: 2.6 ms exposed (RMSNorm wg)
- **Analysis**: The +1.84% from auto_overlap_bucketing came mostly from
  hiding the ~177 ms of copy/cat overhead behind compute (now <5 ms
  exposed) — total comm exposed actually grew slightly (146 → 153 ms).
  Auto-bucketer optimizes FSDP only; TP collectives remain scattered and
  unbucketed. TP RS (61.5 ms exposed, 130 calls) is the single largest
  remaining target.
- **Lessons**: To make further progress: (a) **TP-side bucketing /
  async TP** is the highest-impact direction; (b) async TP matcher
  has been failing due to op adjacency — needs custom fusion or
  graph rewrite; (c) the 67 TP AR coalescing is also stalled by
  upstream topology bug.

---

## async TP between normalize+bucket — discard (xxxxxxx)

- **Idea**: Final untried position for `async_tensor_parallel_pass` —
  between `normalize_view_ops_as_reshape` and `auto_overlap_bucketing_pass`.
  After view normalization the matcher should see `aten.reshape.default`
  instead of `view`/`_unsafe_view`, and before bucketing the mm→RS
  adjacency should be intact.
- **Changes**: Insert async_tensor_parallel_pass at position #5 (between
  view normalization and bucketing), dropping the original conditional.
- **Result**: tps **7,141 (-2.1%)**, numerics PASS, but **356 "no
  producer matmul found" skips and 0 fusions**. AG-side also produced
  zero fusions silently.
- **Analysis**: Position is not the issue. The upstream
  `_find_producer_matmul` whitelist (`reshape`, `_to_copy`, outer
  `view`) doesn't match what sits between mm and RS in our graph —
  presumably a `permute`/`transpose`/DTensor metadata op or a
  non-whitelisted `_to_copy` variant. Without an explicit dump-and-rewrite
  to fold those into the matched chain, async TP cannot engage on this
  configuration from `passes.py` alone.
- **Lessons**: Async TP is **fully blocked** at the upstream-matcher
  level on Llama3 8B FSDP=4×TP=2 with DTensor lowering. Three pass
  positions tried (start, middle, end) — all yield 0 fusions. Park
  this direction until either (a) upstream loosens the matcher whitelist
  or (b) we write a custom mm→RS fusion replacement that handles the
  actual op pattern. TP RS (61.5 ms exposed, 5.9% of step) remains the
  largest unattacked bottleneck.

---

## Remove redundant contiguous clones — keep (PENDING)

- **Idea**: Profile showed 358 `aten.clone(memory_format=contiguous_format)`
  calls, ~290 from FSDP weight unpack. Many should be no-ops if the input
  is already contiguous (per FakeTensor stride metadata).
- **Changes**: Added `remove_redundant_contiguous_clone_pass` — walks the
  graph, finds `aten.clone.default` nodes (default kwargs → contig
  format), checks FakeTensor stride against contiguous strides; if
  already-contiguous, replaces uses with input. Registered before
  `auto_overlap_bucketing_pass`.
- **Result**: 68 clones eliminated per step (deterministic across 3
  runs). 3-run tps **7,321 / 7,323 / 7,312** (avg **7,319**, +0.34% vs
  7,294). Bitwise numerics PASS. wall 129 s.
- **Analysis**: 68 of 358 clones (~19%) were visible at this FX layer
  and ALL 68 were redundant. The other ~290 FSDP-unpack clones live
  inside HOPs/subgraphs the pass doesn't reach. Consistent small
  improvement across 3 runs suggests the gain is real but partially
  masked by auto-bucketing already overlapping these clones with
  compute. Memory unchanged (49.06 GiB).
- **Lessons**: Stride-aware clone elimination is a small but safe win.
  Bigger gains likely require reaching INTO the FSDP HOPs/subgraphs or
  rewriting the unpack path to avoid issuing clones in the first place.

---

## Subgraph descent + second run for clone removal — discard (xxxxxxx, xxxxxxx)

- **Idea**: Investigate if the missing ~290 FSDP weight-unpack clones live
  in subgraphs (HOPs / get_attr submodules) or are introduced by the
  auto-bucketer's pre-bucketing rewrite.
- **Changes**: (a) Extended clone pass to recurse via `named_children()`
  into nested GraphModules; (b) added a second invocation of the clone
  pass after `auto_overlap_bucketing_pass`.
- **Result**: Both experiments found **0 additional clones**. Numerics
  PASS, perf within noise. No subgraphs exist at this pipeline position;
  bucketing doesn't introduce new clones.
- **Lessons**: The missing 290 clones in the original profile are
  apparently *consumed/eliminated* by `auto_overlap_bucketing_pass`
  itself (it reorders/fuses the unpack chain), not present as
  subgraph hidden ops. 68 root-level clones is the actual ceiling here.

---

## Custom mm→reduce_scatter fusion — keep (PENDING)

- **Idea**: Upstream `micro_pipeline_tp_pass` cannot find mm→RS adjacency
  in our graph (3 positional attempts all gave 0 fusions, 421 skips).
  Write a *custom* fusion pass with a looser matcher that walks back from
  each `_c10d_functional.reduce_scatter_tensor` through
  `view/reshape/_to_copy/permute/transpose/t/_unsafe_view` AND through
  `cat(split(input, dim=k))` (scatter_dim>0 pattern) to find a producer
  `aten.mm.default`, then replace the chain with
  `torch.ops.symm_mem.fused_matmul_reduce_scatter`.
- **Changes**: New `fuse_mm_reduce_scatter_pass` (+ supporting helpers)
  registered between `remove_redundant_contiguous_clone_pass` and
  `auto_overlap_bucketing_pass`. Includes a dtype filter that skips
  chains where downstream dtype != mm A dtype (i.e. backward weight-grad
  RS with fp32 upcast — fusing would crash with bf16 vs fp32 grad mismatch).
  Inferred scatter_dim by comparing mm output shape to RS input shape.
- **Result**: **65 forward TP mm→RS fusions** per step (32 attn output +
  32 mlp w2 + 1 extra). 290 candidates found, 225 correctly skipped on
  dtype filter. Numerics PASS (bitwise) — chunked matmul + chunked RS
  produces same bf16 result as unchunked because per-position rank-axis
  reduction order is unchanged. **4-run tps: 7,408 / 7,396 / 7,406 /
  7,408 → avg 7,405 (+1.17% over 7,319, +3.39% over Run 3 production
  baseline 7,162)**. mfu 43.36% avg, memory 49.10 GiB (essentially
  unchanged), wall 129 s.
- **Analysis**: Targets the 61.5 ms exposed TP RS (the biggest remaining
  bottleneck). NVLink symm_mem chunked-matmul-RS recovers a meaningful
  fraction. The 225 backward weight-grad RS sites (fp32) remain
  unattacked because `fused_matmul_reduce_scatter` only reduces in
  `A.dtype` (bf16); fusing them would change reduction precision.
- **Lessons**: (1) Upstream `micro_pipeline_tp_pass`'s matcher is too
  strict for our DTensor lowering — a looser custom matcher unlocks
  the win. (2) Bitwise identity holds when (a) the chunked op's
  reduction order matches the unchunked, AND (b) the dtype filter
  keeps fp32 grad reductions out. (3) This is the largest single
  experiment win in Run 3 so far.

---
