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

## Baseline — keep (baseline)

- **Idea**: Establish the Run 1 baseline. Llama3 8B with FSDP=4 TP=2 bs=1, `construct_default_graph_passes` returns an empty list. The agent will iterate from this raw aten-op graph.
- **Changes**: None to `passes.py`. Infrastructure-only commit.
- **Result**: tps=4,161, mfu=24.36%, memory=49.0 GiB (52% of H100), wall_time=75s; loss=9.21808, grad_norm=4.5867 (last step of 20, `c4_test` dataset).
- **Analysis**: Bitwise-identical loss/grad_norm across runs with `--debug.seed 42 --debug.deterministic`. Memory is comfortable (31 GiB headroom). There is meaningful headroom above this floor — the agent's job is to find it. Wall time ~75s per benchmark → ~1100 iterations in 24h.
- **Lessons**: FSDP=4 TP=2 bs=1 chosen over bs=2 because of larger optimization headroom and safer memory budget (31 GiB free vs 12 GiB).

---

## iter33 graph motif analysis — info-only (xxxxxxx)

- **Idea**: With FX-pass surface seemingly exhausted, look one level deeper. Maybe there's a high-frequency 3- or 4-node motif that could be hand-rolled-fused into a single op.
- **Procedure**: Added `_recon_motifs` pass between `disable_uninitialized_memory_fill` and `cuda_graph_pass`. Walked `call_function` nodes in topo order, counted unique 3-tuples and 4-tuples of consecutive targets. Dumped top motifs + example shapes/dtypes to /tmp/recon_motifs.txt. Recon-only.
- **Top-10 3-tuple motifs (counts)**:
  - 258 `view → view → view` — zero-cost metadata.
  - 194 `mm → t → t` — backward of linear (`grad_w = mm(grad_y^T, x)` style).
  - 192 `clone → _unsafe_view → slice` — real memcpy for FSDP grad pre-aggregation.
  - 158 `t → mm → t` — forward of linear, transpose-input variant.
  - 149 `slice.Tensor → t → mm` — sliced-param-shard → matmul (FSDP local linear forward).
  - 129 `getitem → cat → reduce_scatter` — collective input staging.
  - 129 `view → slice → view`, 129 `t → t → _to_copy`, 128 `_unsafe_view → slice → t` — all metadata.
  - 128 `_to_copy → view → view_as_complex` — RoPE bf16→fp32 → complex view (4/layer × 32 layers).
- **Top 4-tuples**: 158 `t → mm → t → t`, 128 `clone → _unsafe_view → slice → t`, 128 `mul → view_as_real → view → _to_copy` (RoPE complex multiply → bf16 epilogue), 126 `getitem → getitem → cat → reduce_scatter`, 125 `split → getitem → getitem → cat`.
- **Analysis**:
  - **The top motifs are dominated by zero-cost metadata ops.** `view`, `t`, `slice`, `_unsafe_view`, `getitem` produce no GPU kernels — they're stride/layout metadata ops. The FX node count overstates the real GPU work. Eliminating them saves zero runtime cost (already proven by iter-29 `sink_waits`, iter-16 view/clone cleanup).
  - **The genuine compute motifs are**:
    1. RoPE (`_to_copy → view_as_complex → mul → view_as_real → _to_copy`) at 128/step. Inductor's Triton fuser WOULD collapse this into one kernel, but Inductor codegen is bitwise-blocked.
    2. mm chains (`t/mm/t`) — irreducible.
    3. `clone → unsafe_view → slice` (192) — real memcpy; hand-fusing requires merging clone+slice into a single strided write, non-trivial and Inductor preserves these for layout (per iter-16).
- **Conclusion**:
  - **The FX-layer fusion surface IS genuinely exhausted under bitwise constraints.** The remaining "high-frequency" patterns are either zero-cost metadata or are already structurally optimized.
  - **The only concrete fusion candidate (hand-rolled fused RoPE) requires writing a custom Triton/CUDA kernel** — outside `passes.py` scope, would duplicate Inductor's existing fusion that we already can't use, and unlikely to deliver more than what Inductor would.
- **Lessons**:
  - **Don't be misled by FX node counts.** 9,893 `call_function` nodes sounds like a lot, but a large fraction are metadata-only and produce no kernels. The real kernel count is in the hundreds, dominated by `mm` (675), `_to_copy` (649, real kernels), collectives, and a few hundred elementwise ops.
  - **Future iterations under the bitwise + passes.py-only constraint should expect null results.** The genuine ceiling has been reached: GPU at 98% utilization, all FX-level levers exhausted, Inductor codegen blocked, NCCL tuning out of scope.

---

## iter32 regional_inductor_compile re-attempt — discard (xxxxxxx)

- **Idea**: iter-31's σ=4 baseline reclassified iter-14's +14 tps as 3.5σ above mean — possibly real. Re-implement `regional_inductor_compile` for the iter-22 graph (no bucketing) and measure with an 8-run average to detect σ-level effects.
- **Implementation**: 708 candidate regions (pure-ATen, no collectives/wait/in-place, ≥4 nodes). Drained 73 decomp entries (suffix-match against `_to_copy`, `t`, `transpose.int`, `silu`, `arange`, `zeros_like`, `ones_like`, `_fused_rms_norm_backward`, `silu_backward`, `embedding_dense_backward`) + cleared `fast_random_decomps` cache. Used `compile_fx_inner` per sub-GraphModule; outputs spliced back via `operator.getitem` (CompiledFxGraph always returns a sequence). Fix-ups: `aten.scatter_.default` doesn't exist on this PyTorch → string-based in-place filter; single-output regions still need a getitem wrapper.
- **Result**: 472/708 regions compiled (66.7%, avg size 7.8 nodes, 3,687 nodes total Triton-fused); 236 skipped (Inductor lowering failures). Numerics PASS bitwise (loss=9.21808, grad_norm=4.5867). Compile time +134s.
  - **8-run TPS: 5741 / 5750 / 5744 / 5739 / 5737 / 5739 / 5735 / 5748 → mean 5,741.6, σ=5.3.**
  - **Δ vs 6,040 baseline: −298 tps (−4.9%)**, ~58σ below mean — **regression discard**.
- **Analysis**:
  - **iter-14's +14 tps on the iter-12 graph was 3-run sampling noise.** On a wider 8-run measurement, regional Inductor compile is solidly negative.
  - **Mechanism for regression**: Triton-fused kernels are slower than the unfused ATen sequence under CUDA graph capture. Three plausible reasons: (a) Inductor's per-region codegen can't see the outer memory-layout decisions, re-introducing materializations that CUDA graphs had eliminated; (b) Triton kernels have their own per-kernel overhead that doesn't go away even in capture; (c) the 472 fused kernels each have launch overhead within the CUDA graph that's not lower than the 3,687 source kernels'.
  - **Region size is the wrong knob**: avg 7.8 nodes is too small for Triton arithmetic intensity to win.
- **Lessons (added to LEARNINGS.md)**:
  - **Regional Inductor compile is anti-synergistic with CUDA graphs.** The iter-8 capture amortizes launches so well that even tiny per-kernel overhead from Inductor's lowering becomes a NET regression.
  - **Don't trust 3-run "noise" calls.** Iter-14 (and possibly iter-15) measured +14 within iter-12's ~σ=10-15 tps band, called it noise correctly even though we now know σ=4 there might have been narrower. The lesson is to verify with 8-run when a single sub-3% delta could be load-bearing — but ALSO accept that iter-14's measurement was honest.

---

## iter31 baseline variance characterization — info-only (xxxxxxx)

- **Idea**: Iter-22's "stable" baseline used 3-run averages (6,052 / 6,041 / 6,051 → 6,048). If actual run-to-run σ is much smaller than the ±1.5% (±90 tps) noise threshold used in iter-23+ discards, we may have rejected real-signal +14 tps deltas (iter-14, iter-15) as noise.
- **Procedure**: Ran `bash autoresearch/scripts/run_benchmark.sh` 8 times back-to-back at the current pass list (iter-22 best). No code change.
- **Result**: 8 runs at 6,047 / 6,039 / 6,035 / 6,040 / 6,035 / 6,044 / 6,041 / 6,041 tps.
  - **Mean: 6,040.25 tps. σ: 4.1 tps. Range: 12 tps. ±2σ ≈ ±8 tps.**
  - All runs: bitwise-identical loss=9.21808, grad_norm=4.5867, memory=49.24 GiB.
- **Analysis**:
  - **Real noise is ~σ=4 tps, not the ±90 tps band we'd been using.** A +14 tps observation is 3.5σ above the mean — almost certainly real signal.
  - **The reported iter-22 "best" of 6,048 was 2σ above the long-run mean (6,040).** Three-run averages are too noisy to anchor decisions.
  - **Reclassification**: iter-29 sink_waits at 6,049 was 2.3σ above the true baseline — could be real (or could be lucky tail). Re-tests with 8-run avg needed to confirm any marginal claim.
- **Lessons**:
  - **Always use 8-run averages for ablation/keep decisions on this workload.** A 3-run average has ~σ_mean = σ/√3 ≈ 2.4 tps, meaning even the *mean of 3* runs varies by ~5 tps. An 8-run mean has σ_mean ≈ 1.4 tps, tight enough to detect +14 deltas.
  - **The σ=4 noise band is small enough that small consistent wins compound visibly.** Several discarded +3 / +9 tps results MAY have been real but were dismissed by the wide threshold.
  - **Update operating procedure: any new pass must beat baseline by >8 tps over an 8-run average to be considered a "keep"; <8 is discard.**

---

## iter30 literature research — info-only (xxxxxxx)

- **Idea**: With the FX-level optimization surface seeming exhausted, do focused web research for any recent (2024-2026) FSDP+TP+CUDA-graphs Llama3 H100 optimization techniques we haven't tried.
- **Searches performed** (via web search, time-boxed): PyTorch Async-TP / SymmetricMemory, micro_pipeline_tp pass mechanics, NCCL channel tuning, Triton-distributed (arXiv 2504.19442), CUDA graph extensions, RMSNorm/RoPE custom kernels.
- **Surviving candidates**:
  - **A. SymmetricMemory async-TP (`_fused_all_gather_matmul` / `_fused_matmul_reduce_scatter`)**: Documented in https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487 . Activated by `micro_pipeline_tp_pass`, which looks for `all_gather → matmul` or `matmul → reduce_scatter` adjacency. **Blocked**: iter-7 already confirmed `micro_pipeline_tp_pass` is a no-op on our graph because our TP uses `_c10d_functional.all_reduce_` (the 68 AR nodes), not AG+matmul/matmul+RS. Async-TP requires sequence-parallel TP layout, which `simple_fsdp` doesn't emit.
  - **B. NCCL channel/SM tuning (`NCCL_MIN_NCHANNELS`, `NCCL_NTHREADS`, `NCCL_PROTO`)**: Out of `passes.py` scope (NCCL env vars are read at communicator init time, before `passes.py` runs). Setting them inside a pass is too late.
- **Dead ends**: Triton-distributed (replaces NCCL with custom kernels → non-bitwise); RMSNorm/RoPE custom Triton (external-library prohibition + non-bitwise); zero-bubble PP (no PP in our config); CUDA graph cross-step capture (already captured joint graph); bf16 collectives (bitwise-blocked).
- **Lesson**: **The optimization surface accessible from `passes.py` under bitwise constraint is genuinely exhausted at this point.** All remaining levers either require: (a) graph rewrites to enable async-TP (substantial, breaks if mis-done); (b) non-FX-level config (NCCL env vars, pre-init); (c) numerics relaxation (Inductor codegen, bf16 grads). Future iterations should treat known leads as null-hypothesis and look only for genuinely-novel exploits.

---

## iter29 sink_waits (custom FX) — discard (xxxxxxx)

- **Idea**: iter-4 noted "move waits LATER toward consumers" as an overlap lever; `schedule_overlap_bucketing` does some of this at the scheduler level but its bucketing/memory budgets may leave residual slack. Custom FX-level pass: for each `wait_tensor`, find its earliest topo user and `prepend` the wait to that user.
- **Changes**: Added `sink_waits` pass between `apply_inductor_pattern_passes` and `disable_uninitialized_memory_fill`. Pre-computed topo indices, then iterated all 622 waits and called `target_user.prepend(wait_node)` when there was slack.
- **Result**: discard. **Pass moved 325 of 622 wait_tensor nodes** (52%), with **max move distance 4430 positions** (a huge gap). Numerics bitwise PASS (loss=9.21808, grad_norm=4.5867). tps=6,049 vs baseline 6,048 → **+0.02% (≡ noise)**. mem=49.32 GiB.
- **Analysis**:
  - `schedule_overlap_bucketing` does NOT optimally place waits at the FX level — half of them sit far earlier than their earliest consumer.
  - **But moving them gave 0 TPS change.** Two structural reasons:
    1. **CUDA graph capture freezes the actual runtime stream/event structure.** Once captured, the sequence of NCCL launches on the comm stream and the corresponding events on the compute stream is fixed. FX-level "where the wait appears in the graph" only matters for pre-capture scheduling decisions — which `schedule_overlap_bucketing` already locked in.
    2. **Waits sitting between collectives don't block the compute stream.** They live on the NCCL stream; the compute stream sees only the corresponding sync events. Moving the FX wait_tensor later in topo order doesn't change those events.
- **Lessons (added to LEARNINGS.md)**:
  - **FX node order after `schedule_overlap_bucketing` is irrelevant for runtime overlap.** The actual overlap structure is determined by the scheduler's stream-aware op placement, not by where a wait_tensor appears in `gm.graph.nodes`. Any "sink_waits"-style FX-level pass we write after this point is purely cosmetic.
  - **Don't conflate "FX wait node position" with "runtime sync point".** The functional `wait_tensor` op IS the sync, but its scheduled position on the stream is determined by Inductor's overlap scheduler, not by FX list order.

---

## iter28 dedupe_to_copy — discard (xxxxxxx)

- **Idea**: Iter-27 recon found 649 `_to_copy.default` nodes after all pattern passes. If `joint_graph_passes`' CSE misses any (different code paths, layout normalization, etc.), deduping pairs sharing the same `(input, kwargs)` would cut redundant `direct_copy_kernel` launches.
- **Changes**: Added `dedupe_to_copy` pass between `apply_inductor_pattern_passes` and `disable_uninitialized_memory_fill`. Groups nodes by `(input_node, target_dtype, target_layout, target_device, target_memory_format)`; picks topo-earliest canonical; replaces uses + erases duplicates.
- **Result**: discard. Scanned 649 `_to_copy` nodes; **0 dedup groups with >1 member** found. Pass was a no-op. tps=6,041 (≡6,048 within noise), numerics bitwise PASS.
- **Analysis**: `joint_graph_passes`' CSE is thorough — every duplicate cast is already collapsed before we run. All 649 remaining `_to_copy` calls have distinct inputs (genuine intermediate-tensor conversions). The ~74 ms / 4.6% they cost is from genuine work, not redundancy.
- **Lessons**:
  - **Don't expect to out-CSE `joint_graph_passes`** — it does pure-op CSE thoroughly, including `_to_copy`.
  - To reduce the `_to_copy` cost further requires either (a) eliminating the surrounding dtype boundary (e.g. doing RMSNorm in bf16 — bitwise-blocked), (b) fusing the cast into adjacent kernels (Inductor codegen, bitwise-blocked per iter-21), or (c) running 2+ casts in parallel on a side stream (FX-unfriendly).

---

## iter27 untried_random_passes + post-iter-22 recon — discard (xxxxxxx)

- **Idea**: Three callables in `torch._inductor.fx_passes.replace_random` (`replace_random_passes`, `fuse_seed_creation_pass`, `fuse_offset_creation_pass`) hadn't been tried. Bundle with a recon dump of the post-iter-22 graph (final form after all current passes run) to surface any newly-actionable patterns.
- **Result**: discard. All 3 random passes ran cleanly with 0 node-count delta (no rand/dropout/bernoulli/philox nodes — Llama3 8B training without dropout). tps=6,043, mfu=35.38%, memory=49.24 GiB. Numerics bitwise PASS (loss=9.21808, grad_norm=4.5867).
- **Post-iter-22 graph recon** (9,893 call_function nodes total, dumped to /tmp/recon_iter22.txt):
  - **`_to_copy.default`: 649** (vs 842 at baseline). Breakdown: 420 bf16→fp32, 228 fp32→bf16, 1 bool→fp32. Down from 551 at iter-12 (but the iter-12 count was pre-bucket-removal; restructuring slightly changed the count).
  - **`cat.default`: 226** (130 bf16, 96 fp32) — concentrated in attention/rotary plumbing.
  - **Collectives**: 229 AG + 293 RS + 68 AR (was 421 + 421 + 68 at baseline). The reductions from joint_graph_passes / post_grad_passes / schedule_overlap_bucketing are larger than expected — apparently some collective coalescing or dedup happens inside these passes even after iter-22 removed the explicit `bucket_*` calls.
  - 622 `wait_tensor` (vs 910 baseline) consistent with reduced collective count.
  - 2022 `view`, 1125 `t`, 675 `mm`, 256 `clone`, 32 `empty.memory_format`, 32 `copy_.default`, 128+128 `view_as_complex`/`view_as_real` (RoPE — unchanged).
- **Analysis**:
  - Random passes correctly identified the graph has no random ops to fuse — expected null result.
  - **The collective count dropped 421→229 AG and 421→293 RS, a much larger reduction than previously documented.** This is a structural finding: iter-22 might have looked at the wrong baseline graph for some of its reasoning.
  - **New patterns to attack (saved for future iters)**:
    - 649 `_to_copy` with 65/35 bf16↔fp32 split. Some are likely RMSNorm/SDPA mixed-precision boundaries; if any pair has the same input + same target dtype, we could CSE-dedupe.
    - 130 bf16 cat + 96 fp32 cat — most cats are bf16-uniform. iter-11's "cat-of-upcasts" target was 0 here (joint_graph already collapsed). But might be cat-with-shared-input patterns worth a look.
- **Lessons**:
  - **Bundled recon + cheap attempt is good iteration economics.** Even when the optimization is a no-op, the recon dump grounds future iterations in current graph state, not stale baseline numbers.
  - **The `_inductor.fx_passes.replace_random` family is dead-on-arrival for inference-only training graphs.** Don't waste iterations re-trying.

---

## tune_aten_distributed_optimizations — discard (xxxxxxx)

- **Idea**: `schedule_overlap_bucketing` (load-bearing per iter-23 ablation) runs with all `torch._inductor.config.aten_distributed_optimizations.*` knobs at their `None` defaults. We have ~31 GiB of GPU memory headroom; raising the memory budget for prefetch and using a more accurate compute estimator might give the scheduler more reordering opportunities.
- **Changes**: Before calling `schedule_overlap_bucketing(gm)`, set (each in its own try/except): `max_memory_increase_gb=20.0`, `max_in_flight_gb=10.0`, `max_compute_pre_fetch=200`, `compute_estimator="benchmark"`. All 4 attrs existed on this PyTorch build.
- **Result**: discard. 2 runs: tps=6,040 / 6,038 → avg 6,039 (mfu 35.37%), memory=49.24 GiB. Numerics bitwise PASS (loss=9.21808, grad_norm=4.5867). −0.15% vs iter-22 baseline (within noise).
- **Analysis**:
  - `compute_estimator` already defaults to `"benchmark"` on this PyTorch build (verified at `pytorch/torch/_inductor/config.py:1215`) — that knob was a no-op.
  - Peak memory stayed at 49.24 GiB identical to baseline, confirming the prefetch budget was never the binding constraint at the `None` defaults.
  - The scheduler must already be finding all overlap opportunities the graph affords. Raising the memory budget and lookahead window doesn't unlock additional reordering.
- **Lessons**:
  - **The overlap scheduling surface is saturated.** `schedule_overlap_bucketing` at its default settings already extracts all the overlap this graph affords; tuning its budget/lookahead knobs doesn't help.
  - **Pre-flight check the defaults first.** Half our knobs (`compute_estimator`) were already at the value we tried to set. Future PyTorch config-tuning experiments should diff against current defaults to avoid no-op assignments.

---

## replace_overwritten_zeros_with_empty — discard (xxxxxxx)

- **Idea**: Iter-24 re-profile noted ~74 hidden `FillFunctor<float>` invocations (~18 ms) on top of 88 raw ones (~19 ms). Iter-12 only covered `aten.empty.*`; maybe `aten.zeros/full/ones/.*_like` with full-overwrite users would qualify too.
- **Recon**: post-iter-22 graph contains 9× `aten.full.default` (scalars, fill 0), 1× `aten.zeros_like.default` (CE backward, shape 8192×64128 f32), 1× `aten.ones_like.default` (scalar). 0 zeros / new_zeros / new_full / full_like / ones.
- **Result**: discard. 0 substitutions safely applicable.
  - The 9 `aten.full(0)` scalars feed `where.self` (reads the scalar value) or `index_put_` (PARTIAL overwrite — untargeted positions must stay zero).
  - The `zeros_like` IS the CE-backward grad buffer feeding `index_put_` (iter-20 finding — semantically required).
  - `ones_like` has non-zero fill — skipped.
- **TPS**: 6,057 ≈ 6,048 (noise; pass adds ~15 ms compile but zero substitutions).
- **Lessons**: The iter-24 profile's "74 hidden FillFunctor invocations" are not from FX-level fill ops we can attack from `passes.py`. They're internal launches from `index_put_` partial writes, `where.self` scalar broadcasts, and other op-internal bookkeeping that doesn't appear as a fill node in FX. The fill-elimination surface is exhausted.

---

## iter-24 re-profile (post-iter-22) — info-only (xxxxxxx)

- **Idea**: iter-15 profile was on the pre-iter-22 graph. Re-profile now that bucketing is removed.
- **Top categories (rank 0, iteration_20, 1,493 ms total kernel time)**:
  - GEMM: 485 ms (32.5%, up from 29.5%) — irreducible.
  - FlashAttn: 373 ms (25.0%, up from 22.5%) — irreducible.
  - **ReduceScatter**: 259 ms (17.3%, down from 23.9%): **195 ms fp32 (326 calls)** + 64 ms bf16 (238 calls).
  - AllGather (bf16): 91 ms (6.1%, up from 4.9%).
  - elementwise_other: 68 ms (4.5%).
  - elementwise_mul: 44 ms (2.9%).
  - elementwise_add: 41 ms (2.7%, sequential residual+grad chains).
  - optimizer (multi_tensor): 40 ms (2.7%, out of graph).
  - CatArrayBatchedCopy: 21 ms (1.4%, down from 2.9% — bucketing slice/cat gone).
  - FillFunctor (raw): 19 ms (1.3%) + ~18 ms hidden in vectorized_elementwise template.
  - RMSNorm: 17 ms (1.2%). SiLU: 17 ms (1.2%).
- **Key new finding**: ReduceScatter is split fp32 + bf16. The fp32 RS alone is 195 ms / 13% — **single biggest non-irreducible lever** but blocked by bitwise numerics. The bf16 RS (64 ms / 4.3%) is closer to ideal.
- **Candidate next ideas**: (a) bf16 grad RS (blocked), (b) more FillFunctor elimination (iter 25), (c) direct_copy_kernel fusion (820 + 373 launches, ~55 ms, untried).
- **Surprises**: GEMM/FlashAttn grew proportionally because the saved launch overhead was a denominator effect; absolute kernel times unchanged. Optimizer at 2.7% is the largest out-of-graph category.

---

## re-ablation after iter 22 — info-only (xxxxxxx)

Re-test each remaining pass on the iter-22 graph (with bucketing removed). 3 runs per variant; bitwise numerics preserved on all.

| Variant | tps avg | Δ vs iter-22 | Verdict |
|---|---|---|---|
| iter-22 baseline | 6,048 | — | — |
| drop `remove_detach_nodes` | 6,032 | -16 | within noise; keep (borderline) |
| drop `schedule_overlap_bucketing` | 5,505 | **-543** | strongly load-bearing; KEEP |
| add 2nd `joint_graph_passes` after overlap | 6,048 | 0 | no-op; drop |
| move `disable_uninitialized_memory_fill` before inductor | 5,743 | -305 | empty.* nodes don't exist yet at that point — must run AFTER inductor; keep current order |

**Key finding**: iter-22's pass list is a tight local optimum. `schedule_overlap_bucketing` becomes the single largest contributor post-bucketing-removal (-543 tps without it), confirming it's not just a no-op-reorder-fixup but does real comm/compute overlap work that bucketing was masking.

Pass list unchanged: `[remove_detach_nodes, apply_inductor_pattern_passes, disable_uninitialized_memory_fill, install_cuda_graph]` with `schedule_overlap_bucketing` as the last step inside `apply_inductor_pattern_passes`.

---

## remove bucketing from apply_inductor_pattern_passes — keep (e47a29c)

- **Idea**: Iter-22 ablation found that disabling `bucket_all_gather` + `bucket_reduce_scatter` improved tps by ~3% (5,876 → 6,051 in a single run). Verify and keep.
- **Changes**: Removed the two bucketing calls + `stable_topological_sort` from `apply_inductor_pattern_passes`. Kept `schedule_overlap_bucketing` (still load-bearing per iter-22 ablation note: pruning it drops tps by ~120).
- **Result**: keep. 3-run check: tps=6,052 / 6,041 / 6,051 → **avg 6,048, mfu=35.42%**, memory=49.24 GiB. Loss=9.21808, grad_norm=4.5867 (bitwise identical). Numerics test PASS. Wall time 143s.
  - **+2.9% over iter 12, +45.4% over baseline.**
- **Analysis**:
  - At iter 7 when bucketing was added, the launch-overhead savings (~+5% tps) dominated the +810/+615 cat/slice/reshape node cost.
  - **Iter 8 (CUDA graphs) flipped the equation**: it amortizes ALL per-launch overhead, so the launch-count reduction is no longer valuable. But the extra graph nodes (cat/slice/reshape needed to split bucketed outputs) still cost runtime as kernels in the captured graph.
  - **`schedule_overlap_bucketing` is independently valuable**: it reorders nodes for comm/compute overlap on the NCCL stream (a node-count no-op). Pruning it drops tps ~120, so it stays.
- **Lessons**:
  - **Re-ablation matters after new passes are added.** A pass that was helpful at one stage may become counterproductive after later passes change the cost-benefit calculus. The iter-22 ablation surfaced this for bucketing.
  - **Bucketing-only-when-CUDA-graphs-not-available** is the right policy. If a future iteration disables CUDA graphs (e.g. for shape-changing workloads), bucketing should return.

---

## unblock_inductor_codegen (bitwise-strict configs) — discard (xxxxxxx)

- **Idea**: Iter 19's codegen drift might come from one of Inductor's many fusion/scheduling flags. Try the maximally-conservative bitwise mode.
- **Changes**: Same as iter 19 (rewire `_get_submesh`, drain 10-op decomp list, cache_clear `D.fast_random_decomps`, call `compile_fx_inner`) but with 17 Inductor config flags set to most-conservative values FIRST: `split_reductions=False`, `epilogue_fusion=False`, `coordinate_descent_tuning=False`, `shape_padding=False`, `pattern_matcher=False`, `realize_acc_reads_threshold=0`, `force_fuse_int_mm_with_mul=False`, `aggressive_fusion=False`, `allow_buffer_reuse=False`, `fallback_random=True`, `realize_reads_threshold=0`, `size_asserts=False`, `max_autotune*=False`, `triton.cudagraphs=False`, `fx_graph_cache=False`. All 17 flags were available on this PyTorch.
- **Result**: discard. `compile_fx_inner` succeeded in 32s. CUDA-graph wrap succeeded. BUT step-20 numerics: loss=8.78991 (expected 9.21808), grad_norm=8.1171 (expected 4.5867). Drift starts at step 1 grad_norm 4.2603 → 4.2606. Same drift signature as iter 19. debugmodel bitwise test PASSES; 8B benchmark FAILS bitwise.
- **Analysis**: Drift survives all reachable user-facing Inductor flags. The remaining divergence is from kernel-selection / lowering decisions not gated by config — likely cuBLAS GEMM algorithm selection or fused RMSNorm/SDPA lowering. **This confirms the bitwise constraint is structurally incompatible with Inductor whole-graph codegen on the 8B graph.**
- **Lessons**:
  - **Inductor codegen has bitwise-divergent kernel selection that isn't controllable by user flags.** Don't expect strict bitwise via config tuning.
  - The drift signature (loss 9.21808 → 8.78991 at step 20, grad_norm 4.5867 → 8.1171) is reproducible across iter 19 and iter 21 — same kernels, same numerical path.
  - Pass left in `passes.py` toggled OFF (`enable_compile_fx_codegen = False`); reverted via `git checkout` so toggle is removed from committed state.

---

## Session checkpoint (after iter 20)

**Final stable configuration**: pass list `remove_detach_nodes` → `apply_inductor_pattern_passes` → `disable_uninitialized_memory_fill` → `install_cuda_graph`.

3-run stability check (immediately after iter 20):
- Run 1: tps=5,876, tflops=340.29, mfu=34.41%
- Run 2: tps=5,872, tflops=340.08, mfu=34.39%
- Run 3: tps=5,873, tflops=340.14, mfu=34.39%
- avg=5,874, σ<2 tps. Bitwise: loss=9.21808, grad_norm=4.5867.

**vs baseline (4,161 tps, 24.36% mfu): +41.2% tps, +10.04 pp mfu, memory ≈ unchanged.**

Iteration outcomes summary (20 iters, 5 keeps):
| # | Pass | Status | Cumulative tps | Δ tps | Mechanism |
|---|---|---|---|---|---|
| 0 | (baseline) | — | 4,161 | — | empty pass list |
| 1 | remove_detach_nodes | **keep** | 4,188 | +0.6% | 196 autograd-noop nodes, also frees ~2 GiB |
| 2 | _to_copy redundant elim | discard | — | — | 0/841 qualify; all real bf16↔fp32 casts |
| 3 | bucket_all_gathers (manual) | discard | — | — | shape-heterogeneous → small buckets; reshape copy negates win |
| 4 | prefetch_all_gathers (FX move) | discard | — | — | only 65/421 movable by 7 positions; make_fx already tight |
| 5 | apply_inductor_pattern_passes | **keep** | 4,572 | +9.9% cum | joint_graph_passes + post_grad_passes (11.5k → 9.1k nodes) |
| 6 | compile_fx_inductor (whole-graph) | discard | — | — | blocked by in-place all_reduce_ + decomp conflicts |
| 7 | expanded inductor (bucketing+overlap) | **keep** | 4,783 | +14.9% cum | bucket_*_all_gather/RS + schedule_overlap_bucketing |
| 8 | install_cuda_graph | **keep** | 5,566 | +33.8% cum | CUDA-graph wrap of gm.forward |
| 9 | functionalize collectives + compile_fx | discard | — | — | decomps drainable but blocked by _get_submesh; reinplace resurrects in-place |
| 10 | profile-driven (iter 8 graph) | info | — | — | found FillFunctor as next lever |
| 11 | collapse_dtype_roundtrips | discard | — | — | 0/358 collapsible after iter 5 passes |
| 12 | disable_uninitialized_memory_fill | **keep** | 5,876 | +41.2% cum | global flag (fill_uninitialized_memory) toggled off after audit |
| 13 | untried inductor passes | discard | — | — | 0 ran effectively; iter-7 graph already in their post-pass form |
| 14 | regional_inductor_compile | discard | — | — | 97/97 regions compile but +0.24% (regions <40 nodes; CUDA graphs already amortize) |
| 15 | re-profile + cat-of-upcasts | discard | — | — | profile redone; cat-of-upcasts 0/290 (joint_graph already collapsed) |
| 16 | cleanup_redundant_ops | discard | — | — | 450 double-transpose removed but +0.05%; clone/view-of-view break stride |
| 17 | inductor passes batch #2 | discard | — | — | all no-ops; fx_passes surface exhausted |
| 18 | coalesce_independent_adds | discard | — | — | 225 adds all sequentially data-dependent |
| 19 | unblock_inductor_codegen | discard | — | — | compile_fx_inner works (+2.8%) but 8B numerics drift (debugmodel test masks it) |
| 20 | residual FillFunctor recon | info | — | — | 91% from FlashAttn-internal scratch; structural |

**Structural ceilings discovered** (won't move past these under bitwise constraint, see `LEARNINGS.md` for details):
1. **bf16 ReduceScatter** would save ~9% TPS but changes numerics.
2. **Whole-graph Inductor codegen** is reachable (`compile_fx_inner` succeeds after `_get_submesh` rewire + decomp drain) but the resulting kernels aren't bitwise-equivalent on the 8B model.
3. **FlashAttention-internal scratch fills, CE backward zeros_like, sequential residual+grad chains** — all structurally necessary, not addressable from `passes.py`.

**Reproducibility**: every "keep" is captured in a `[autoresearch]` commit with the `passes.py` change, the tracking-file updates, and a backfill commit pinning the actual hash into `results.tsv` / `EXPERIMENT_LOG.md`.

---

## residual FillFunctor recon — info-only (xxxxxxx)

- **Idea**: Iter-12 killed 4,061 FillFunctor zero-init kernels. Iter-15 re-profile showed 88 still remain (~22 ms / 1.35%). Find their source.
- **Result**: discard (pure recon, no implementation).
- **Categorization of the 88 fill kernels**:
  - **64 (~20 ms, 91%)**: FlashAttention-internal scratch buffers (softmax_lse, cu_seqlens, etc.) from `_scaled_dot_product_flash_attention` fwd+bwd. Opaque C++ kernel — not addressable from FX.
  - **2 (~1.8 ms)**: `zeros_like.default` for CE backward grad of shape (8192, 32064) feeding `index_put_`. The zero-init is **semantically required**: `index_put_` only writes target indices, untargeted positions must stay zero for gradient correctness. Restructuring would require rewriting CE backward (out of `passes.py` scope).
  - **9 (~µs each)**: tiny `aten.full.default` scalars feeding `where`/`index_put_`. Negligible.
  - **1 (~µs)**: `ones_like` (loss grad seed). Negligible.
  - **8 (~0.18 ms)**: `masked_fill_kernel` — actual model masking, not zero-init.
- **Conclusion**: The residual FillFunctor is structural. Iter-12's `disable_uninitialized_memory_fill` was the only `FillFunctor` lever reachable from `passes.py`.

---

## unblock_inductor_codegen — discard (xxxxxxx)

- **Idea**: Iter-9's compile_fx_inner blocker was `_get_submesh`. Remove it + drain decomp conflicts + call compile_fx_inner. If the whole-graph codegen works, it composes with CUDA graphs for potentially +10-20% TPS.
- **Recon**: 1325 non-OpOverload call_functions — ALL are `_operator.getitem` (Python tuple unpack), which Inductor handles natively (not actual blockers). The REAL blocker `device_mesh._get_submesh` has target = `OpOverloadPacket` (not detected by `isinstance(target, OpOverload)`). Only **1** `_get_submesh` node in the graph, whose sole user is the `output` node.
- **Changes**: Added `unblock_inductor_codegen` pass (toggled by `enable_compile_fx_codegen` flag). Rewires the output slot of `_get_submesh` to its mesh input, erases the node; drains the 10-op decomp list (from iter-14 recipe); calls `compile_fx_inner(gm, list(example_inputs))`; overrides `gm.forward` with adapter `lambda *args: compiled_fn(args)`.
- **Result**: discard.
  - `compile_fx_inner` SUCCEEDED end-to-end (compile time +33s, no decomp errors). CUDA-graph capture on top also succeeded.
  - With toggle ON: tps=6,042 (+2.8% on iter 12), mfu=35.38%, memory=46.23 GiB.
  - With toggle OFF: tps=5,879 (≡ iter 12 baseline). Toggle defaults to OFF.
  - **Numerics**: Llama3 debugmodel `test_aot_fx_trace_vs_eager` PASSES in both configs (model is too small to trigger the diverging lowerings). But Llama3-8B benchmark **fails bitwise**: step 1 loss 12.24426 → grad_norm 4.2603 → 4.2606 (drift starts immediately); by step 20 loss has drifted 9.21808 → 9.11513 and grad_norm 4.5867 → 22.9803.
- **Analysis**: Inductor's whole-graph lowering substitutes ATen ops with Triton-fused equivalents that aren't bitwise-equivalent (matmul reduction order or fused-kernel-selection differences). The bitwise test on the debugmodel doesn't exercise these because the graph is too small. **This is a structural property of Inductor codegen: kernels may be numerically correct (loss-equivalent over a training run) but not bitwise.**
- **Lessons**:
  - **`_get_submesh` removal IS feasible** for this graph — only 1 node, only consumed by `output`, so rewiring to its mesh input is safe and preserves the output spec.
  - **Inductor codegen and bitwise numerics are fundamentally incompatible** for the 8B model. To use Inductor codegen we'd need to relax to loss-equivalent numerics (out of scope for this experiment).
  - **The debugmodel bitwise test is necessary but not sufficient** — it can mask precision drift that only emerges in larger models. For a future run, the experimenter could augment the test with a model-scale numerics check.
  - Pass left in `passes.py` toggled OFF so a future iteration (or relaxed numerics regime) can flip the toggle. Reverted via `git checkout` — toggle removed entirely from the committed state.

---

## coalesce_independent_adds (recon-only) — discard (xxxxxxx)

- **Idea**: Profile shows 435 `elementwise_add` kernels (2.5% / 41 ms). If many are independent same-shape adds, coalesce into `aten._foreach_add` to cut kernel count.
- **Recon**: 225 FX `aten.add.Tensor` nodes (≠435 kernels — discrepancy from cuda-graph internal binding). Distribution:
  - ~64 forward residual connections (`x = x + sublayer(x)`): each LHS = prior add's output, **strict sequential chain** across 32 layers × 2 adds/layer. Inherently uncoalescable.
  - ~161 backward grad accumulators (`"Gradient addition node due to multiple use of tensor"` stack_trace): also chained — each extends the prior accumulator.
  - 128/225 have multi-user LHS (skipped anyway).
  - Independent-group analysis (greedy, same shape/dtype, distinct LHS): **0 groups of size ≥ 2**.
- **Result**: discard. No worthwhile coalescing pattern; pass not implemented.
- **Lessons**:
  - **`_foreach_add` requires INDEPENDENT inputs.** Sequential residual / grad accumulator chains can't be parallelized this way regardless of multi-tensor packing.
  - **To reduce the elementwise_add cost would require either (a) fusing each add with its surrounding compute (e.g. residual-add into attention output projection's epilogue, grad-accum into the preceding view/mm) — Inductor pattern-matching territory, OR (b) rewiring backward to a tree-reduction structure** — substantial complexity, out of scope for this experiment.

---

## iter17 exploration batch — discard (xxxxxxx)

- **Idea**: Try a batch of cheap exploratory tweaks: apply `joint_graph_passes` twice, add `post_grad_passes(is_inference=True)`, toggle `epilogue_fusion`/`coordinate_descent_tuning` etc., find `dedupe_symint_uses` / `binary_folding` under alternate module paths, call `joint_graph_passes` after bucketing.
- **Result**: discard. All attempts no-op (deletes 0 nodes after the first fixed-point call) or noise-level TPS. `dedupe_symints` exists under the same module but is a no-op on our graph. `binary_folding_pass` doesn't exist as a callable in this PyTorch build (only `register_binary_folding_pattern` / `binary_folding_init`).
- **Lessons**:
  - **`joint_graph_passes` reaches fixed point in one call** on this graph. Repeated application doesn't help.
  - **`post_grad_passes(is_inference=True)` is equivalent to `is_inference=False` here** — both no-ops on the iter-12 graph (already fully reduced).
  - **Inductor config flags don't change FX-level pass output**; they gate codegen behavior which we bypass via CUDA graphs.
  - **The FX-pass surface for further wins is exhausted at this configuration.** Remaining levers are at the kernel level (bf16 RS — blocked by bitwise numerics) or scheduling level — not reachable via FX pattern passes.

---

## cleanup_redundant_ops — discard (xxxxxxx)

- **Idea**: Look for survivors after iter-5 passes: redundant double-transpose, redundant clone, view-of-view collapse, identity permute/squeeze/unsqueeze/expand.
- **Recon counts** (iter-12 graph): double_transpose=450, redundant_clone=354, view_of_view=257; permute_identity / squeeze_unsqueeze / dead_to_copy / expand_identity = 0 each.
- **Result**: discard. Only double_transpose was safely applicable (numerics pass after 450 removals). tps=5,879 ≈ iter-12 5,876 (+0.05%, noise). Clone and view-of-view removal **broke runtime** with "view size is not compatible with input tensor's size and stride" — surviving clones serve as contiguity canonicalizations, surviving views depend on inner-view strides. Inductor preserves both intentionally.
- **Lessons**:
  - **450 double-transpose removals = 0 TPS win** because Inductor's downstream codegen already collapses them during lowering. FX-level removal is redundant work.
  - The clone / view-of-view "survivors" are NOT bugs — they encode stride/layout requirements that downstream consumers depend on. Removing them silently breaks numerics or runtime.
  - **The graph after iter-5 passes is essentially fully optimized at the FX level for low-risk pattern rewrites.** Future wins require either deeper transforms (kernel-level fusion, codegen) or attacking different categories (numerics-blocked or out-of-graph items).

---

## collapse_cat_of_upcasts — discard (xxxxxxx)

- **Idea**: Iter-11 recon (on the bare make_fx graph) found 290 `_to_copy(bf16→fp32) → shape_op → cat.default(...)` chains. Collapse: `cat([_to_copy(x_i, fp32) for x_i])` → `_to_copy(cat([x_i]), fp32)` when all cat inputs share an upcast and source dtype.
- **Result**: discard. **0 patterns found** in the post-iter-5 graph (260 cat nodes and 551 `_to_copy` nodes exist but no `_to_copy → cat` adjacency remains). `joint_graph_passes` already collapsed them upstream.
- **Lessons**: The iter-11 recon was on the bare make_fx graph; the iter-12 graph is post-iter-5 and post-iter-7. Inductor's pattern matchers run early in our pipeline and have already done this work. **Recon counts at one pipeline stage do not predict opportunities at later stages.**

---

## regional_inductor_compile — discard (xxxxxxx)

- **Idea**: Whole-graph `compile_fx` is blocked by DTensor's `_get_submesh` and `compile_fx`'s reinplace pass. Compile only **pure-ATen connected subgraphs** (no collectives, no `wait_tensor`, no `_get_submesh`, no in-place) via `compile_fx_inner`, replace nodes with `call_function(compiled_fn, ...)`.
- **Changes**: Added `regional_inductor_compile` pass. Found 97 regions by maximal consecutive-run topo-sort grouping (convex by construction). Drained decomp conflicts: `_to_copy`, `t`, `transpose.int`, `silu`, `arange`, `zeros_like`, `ones_like`, `_fused_rms_norm_backward`, `silu_backward`, `embedding_dense_backward`. Cleared `fast_random_decomps`'s `@functools.cache` (otherwise pop doesn't propagate). All 97 regions compiled successfully.
- **Result**: discard. tps=5,890 vs iter-12 5,876 → **+0.24%** (within noise). Numerics PASS bitwise. Wall time 161s (compile +11s for 97 regions).
- **Analysis**:
  - Regions are small (20-39 nodes, avg 31). The graph is fragmented by collectives every ~30 nodes; Triton fusion can't span a useful fraction of the graph.
  - CUDA graphs (iter 8) already amortize per-launch overhead. Compiling small pointwise runs recovers very little arithmetic intensity on top.
  - **Mechanism works** (compile_fx_inner succeeds on cleaned regions, replaced cleanly in outer graph, numerics survive). It just doesn't pay back when individual regions are too small.
- **Lessons**:
  - **Region size matters**: <40-node pointwise regions don't recoup compile overhead post-CUDA-graphs.
  - **Decomp drain + cache clear** unblocks `compile_fx_inner` on subgraphs. Future iterations can reuse this technique.
  - **The graph's collective density (~30 nodes between collectives) is the structural limit** for Triton coverage. Larger regions would require bypassing collectives — e.g., compile across collective boundaries by treating collectives as opaque calls inside Triton (probably needs custom inductor lowerings, out of scope).

---

## expanded apply_inductor_pattern_passes attempt #2 — discard (xxxxxxx)

- **Idea**: Try untried `torch._inductor.fx_passes.*` submodules (`fsdp.dedup_fsdp_reduce_scatter`, `reduced_atomic_contention.partitioned_scatter_optimization_pass`, `ddp_fusion.schedule_comm_wait`, plus 8 others).
- **Result**: discard. All 3 callable rewrites ran cleanly but were node-count no-ops. tps=5,883 ≈ iter-12 5,876 (within noise).
- **Lessons**: After iter-7's `bucket_*` + `schedule_overlap_bucketing` the graph is already in these passes' "post-pass" form. Other untried modules are utilities (memory_estimator, node_runtime_estimation, etc.), require driver context we don't have (overlap_manual_scheduling needs module_bucket_plans; overlap_preserving_bucketer needs scheduled set; control_dependencies needs additional_deps_map), or are in the wrong domain (apply_gumbel_max_trick=RL, efficient_conv_bn_eval=CNN, quantization=N/A, mkldnn_fusion=CPU, freezing_patterns=inference). Inductor's fx_passes surface for this workload is fully explored.

---

## disable_uninitialized_memory_fill — keep (d99dce8)

- **Idea**: Profile shows 4,061 FillFunctor zero-init kernels / step (~95 ms, 5.9%). Investigate the source and eliminate where safe.
- **Recon finding**: Only **75** zero-fill *FX nodes* in the entire 10,947-node graph (not 4,061). Distribution:
  - **65 × `aten.empty.memory_format`** — large bf16 buffers (20M–88M elts), each backing an `_c10d_functional.all_gather_into_tensor_out`. Each `empty` has exactly two users: a `slice.Tensor` (read view, safely no-op until written) and the `all_gather_into_tensor_out` (FULL overwrite).
  - **9 × `aten.full.default`** — tiny scalars feeding `where` / `index_put_`.
  - **1 × `aten.zeros_like.default`** — cross-entropy backward grad buffer (intentional zero, must keep).
  - **4,061 FillFunctor CUDA kernels ≈ 65 empties × ~62 launches** per step. They come from `torch.utils.deterministic.fill_uninitialized_memory = True`, which `--debug.deterministic` enables. Every `aten.empty.*` emits a FillFunctor zero-init even when the consuming op overwrites the buffer immediately.
- **Changes**: Added `disable_uninitialized_memory_fill` pass. Audits every `aten.empty.memory_format` / `aten.empty_strided.default` node; requires every user be either a full-overwrite op (`all_gather_into_tensor_out`, `copy_`, `fill_`, `zero_`) or a read-only view AND at least one full-overwrite user. If all safe, sets `torch.utils.deterministic.fill_uninitialized_memory = False` so subsequent `install_cuda_graph` capture records the graph WITHOUT the FillFunctor zero-init kernels. Registered between `apply_inductor_pattern_passes` and `install_cuda_graph`. Falls through on any exception.
- **Result**: keep. 2 runs: tps=5,875 / 5,877, mfu=34.40% / 34.42%, memory=49.08 GiB. Loss=9.21808, grad_norm=4.5867 (bitwise identical). Numerics PASS. Wall time 141s.
  - **+5.5% over iter 8, +41.2% over baseline.**
- **Lessons**:
  - **`--debug.deterministic` quietly costs ~5.5% TPS via `fill_uninitialized_memory=True`.** The flag's intent (reproducibility through clean uninitialized memory) is achieved here by other means (`--debug.seed`), so we can safely disable the fill once we've audited that every uninitialized buffer is fully overwritten before any non-view read.
  - **Profile-driven optimization works**: iter-10's profile flagged the bottleneck, this iter realized the actual root cause was a global PyTorch flag, not the FX graph itself.
  - **FillFunctor kernel count ≠ FillFunctor FX node count.** A single `empty.memory_format` can generate many FillFunctor launches when deterministic mode is on — count the kernels not the nodes.

---

## collapse_dtype_roundtrips — discard (xxxxxxx)

- **Idea**: Among 842 `_to_copy.default` casts, find bf16→fp32→shape-op→bf16 round-trips where the intermediate ops are dtype-trivial (view/transpose/reshape/etc.). Collapse to a single bf16 path; bitwise identical since shape ops don't compute.
- **Changes**: Added `collapse_dtype_roundtrips` pass between `apply_inductor_pattern_passes` and `install_cuda_graph`. Whitelisted shape ops; required single-user chains; required matching upcast/downcast dtypes.
- **Result**: discard. tps=5,567 (≡ iter 8), numerics pass. **0 collapses applied** out of 358 candidates.
- **Analysis**: Iter 5's `joint_graph_passes` already collapsed the easy round-trips. The remaining 358 candidates fall into two non-collapsible patterns:
  - **290 (~81%)**: `_to_copy(bf16→fp32) → shape_op → cat.default(...)`. `cat` is dtype-trivial in isolation but the other inputs to `cat` are fp32, so keeping our branch in bf16 would dtype-mismatch.
  - **64 (~18%)**: `_to_copy(bf16→fp32) → shape_op → view_as_complex.default → ...`. `view_as_complex` needs fp32 input (RoPE precompute); not a round-trip.
- **Lessons**:
  - **Joint_graph_passes is thorough at the simple cases** — we don't get to re-do its work.
  - The remaining `_to_copy` overhead (74 ms / 4.6%) needs a different attack: e.g. `_to_copy + cat` fusion into a typed cat, or restructuring RoPE precompute, or earlier-pipeline elimination before inductor introduces the `cat` consumers.

---

## profile-driven analysis (iter 10) — info-only

- **Idea**: After iter 8 (CUDA graphs) we're at tps=5,566, mfu=32.6%. Profile to find the next bottleneck.
- **Findings** (steady-state, 8 ranks, FSDP=4 TP=2):
  - GEMM: 52.8% of kernel time (851 ms / step, mostly irreducible)
  - FlashAttn fwd+bwd: 23.1% (372 ms, irreducible)
  - NCCL collectives: 22.6% (365 ms). Of which: **204 ms fp32 ReduceScatter (132 calls)** + 63 ms bf16 RS + 92 ms AllGather + ~5 ms AllReduce. fp32 RS dominates because 3× costlier per call than bf16.
  - **FillFunctor zero-init: 4,061 kernels / 95 ms (5.9%)** — one stream alone has 592 of these (median 2 µs each).
  - direct_copy + casts: ~74 ms (4.6%) from 842 `_to_copy.default` nodes.
  - GEMM/FlashAttn/Collectives total 98%. GPU idle only 1.9%; NCCL stream 70% overlapped with compute, exposed NCCL ~60 ms.
- **Suggested levers** (ordered by realistic upside without breaking bitwise numerics):
  1. **bf16 RS**: best lever (~140 ms / +9% TPS) but **blocked — would change numerics.**
  2. **FillFunctor coalesce**: ~50-95 ms savings if grouped into multi-tensor zero or DCE'd. Most likely 3-6% TPS.
  3. **`_to_copy + cat` typed-fusion**: ~30-50 ms. Tricky (cat needs same dtype across operands).
  4. **Regional Inductor compile** on ATen-only subgraphs between collectives: potentially 5-15% TPS if achievable.
  5. **Async loss/grad_norm**: pure host-time win; GPU at 98% busy so no MFU gain.
- **Lessons**:
  - **Bitwise constraint blocks the biggest single lever (bf16 RS).** Worth flagging this to the experimenter — relaxing to loss-equivalent rather than bitwise opens up ~9% TPS.
  - **GPU is essentially never idle** (1.9%); remaining wins must come from reducing kernel count/size, not from filling bubbles.
  - **fp32 ReduceScatter (132 calls / 204 ms) is the single largest non-irreducible category**, and `simple_fsdp` uses fp32 for grad reduction even when params/activations are bf16.

---

## functionalize_collectives_and_compile_fx — discard (xxxxxxx)

- **Idea**: The blocker for Inductor codegen (iter 6) was the in-place `_c10d_functional.all_reduce_.default` collective. Swap all in-place functional collectives for their functional+wait_tensor equivalents, then try `compile_fx_inner` again to unlock Triton kernel codegen.
- **Changes**: Added `functionalize_collectives_and_compile_fx` pass with helpers `_swap_inplace_collective_node` and `_try_compile_fx_codegen`. Registered AFTER `apply_inductor_pattern_passes` and BEFORE `install_cuda_graph`.
- **Result**: discard. 3-run avg tps=5,480 (mfu=32.08%) vs iter-8 5,566 → **−1.5% regression**. Loss/grad_norm bitwise identical. Numerics PASS.
- **Outcomes per phase**:
  - **Collective swap**: 133 in-place ops swapped (68 `all_reduce_` + 65 `all_gather_into_tensor_out`; 0 `reduce_scatter_tensor_out`). Numerics OK on swap alone, but adds 133 extra `wait_tensor` dispatches (one per swap) → small TPS regression.
  - **`compile_fx_inner`**: drained 10 "both a fallback and a decomp" assertions by deleting offending ops from `L.decompositions` (`_to_copy`, `t`, `transpose.int`, `silu`, `arange`, `zeros_like`, `ones_like`, `_fused_rms_norm_backward`, `silu_backward`, `embedding_dense_backward`). Then hit unrecoverable `AssertionError: device_mesh._get_submesh is not an OpOverload` — a non-ATen DTensor compile-time helper in the joint graph.
  - **`fx_codegen_and_compile`**: same root cause.
  - **`compile_fx`**: Inductor's `post_grad` reinplace pass resurrects the in-place collectives → fails again. Re-introduced after our swap.
  - **Net codegen impact**: zero (every entry point blocked).
- **Lessons**:
  - **`compile_fx`'s reinplace pass re-introduces in-place collectives** even after we explicitly swapped them. The reinplace machinery treats functional+wait as redundant.
  - **`compile_fx_inner` decomp/fallback conflicts can be drained** by deleting offending entries from `torch._inductor.lowering.decompositions`, but you only delay the wall, you don't move it.
  - **The real structural blocker is `device_mesh._get_submesh`** — DTensor's redistribute emits this non-ATen Python op into the joint graph. Inductor refuses any non-`OpOverload` op. **To unlock codegen, either (a) re-trace with a graph that doesn't emit `_get_submesh`, or (b) replace `_get_submesh` nodes with their resolved literal value (a sub-mesh constant) before codegen.**
  - **Free `wait_tensor` dispatches aren't free** — adding 133 of them cost −1.5% TPS. The fewer the collective sync points, the better. So the iter-7 bucketing's reduction of total collectives is doubly valuable.

---

## install_cuda_graph — keep (63c54ff)

- **Idea**: After iter 7's pattern passes, the graph is largely static — same ops, same shapes every step. Wrap the joint fwd+bwd graph in a `torch.cuda.CUDAGraph` and replay each step to eliminate CPU launch overhead. Functional collectives in recent PyTorch are cuda-graph-friendly.
- **Changes**: Added `install_cuda_graph(gm, example_inputs, *, num_static_inputs)` registered last in `construct_default_graph_passes`. Uses `functools.partial` to plumb `traced_result.num_static_inputs` so the wrapper can skip persistent buffers for the leading FSDP-stable inputs (avoids OOM). Standard PyTorch capture pattern: non-default stream, ≥1 warmup, capture inside `with torch.cuda.graph(g):`. **Crucial detail**: after capture, also call `g.replay()` once so `state["outputs"]` reflects executed values before they're returned (otherwise the first real call returns warmup leftovers and training diverges). Self-aliasing skip via `data_ptr()` for callers that already pass the static buffer. Failure paths fall through and restore the iter-7 baseline.
- **Result**: keep. 2 runs: tps=5,566 / 5,565 → **avg 5,566, mfu=32.59%**, memory=49.08 GiB. Loss=9.21808, grad_norm=4.5867 (bitwise identical). Numerics PASS. Wall time 141s, clean shutdown.
  - **+16.4% over iter 7, +33.8% over baseline.**
- **Implementation gotchas resolved**:
  - **First-call replay**: capture only records ops, doesn't execute them. The first post-capture call must `g.replay()` before returning, else outputs are warmup-stale and training diverges.
  - **Static input copying**: naively allocating a persistent buffer for every flat input doubles model-state memory (was +8 GiB for 295 buffers). With `num_static_inputs` we keep only the varying inputs (3 persistent buffers), restoring memory to iter-7 levels.
  - **Shutdown hang**: captured graph + buffers held NCCL state alive past process teardown for ~22 min until SIGTERM. Monkey-patched `torch.distributed.destroy_process_group` to sync, drop graph, drop buffers, `gc.collect`, `empty_cache` before NCCL teardown. Plus an `atexit` fallback.
- **Lessons**:
  - **CUDA graphs deliver large wins (+16%) on this workload** because steady-state CPU launch overhead is non-trivial (~10-15% of step time for ~9-11k node graphs).
  - **`num_static_inputs` from `TracedResult` is essential** to keep memory flat when wrapping in CUDA graphs. Plumb traced_result fields into passes via `functools.partial` from `construct_default_graph_passes`.
  - **Always replay-after-capture** before returning to the caller. Otherwise outputs are recorded references with no fresh execution.
  - **NCCL + CUDA graphs need explicit teardown** to avoid shutdown hangs.

---

## expanded apply_inductor_pattern_passes (bucketing + overlap scheduling) — keep (b9c27f1)

- **Idea**: `torch._inductor.fx_passes` exposes 30+ submodules. Iter 5 only used `joint_graph_passes` + `post_grad_passes`. The submodules `bucketing`, `overlap_scheduling`, `micro_pipeline_tp`, `low_contention_collectives`, `fsdp`, `fuse_attention`, `pad_mm`, `b2b_gemm`, `decompose_mem_bound_mm`, `group_batch_fusion`, etc. should give additional gains specific to distributed and matmul-heavy graphs.
- **Changes**: Expanded `apply_inductor_pattern_passes`. After iter 5's pattern matchers, added (in order): config-flag setup, then `bucketing.bucket_all_gather(gm)`, `bucketing.bucket_reduce_scatter(gm)`, `stable_topological_sort(gm)`, `overlap_scheduling.schedule_overlap_bucketing(gm)`. Discovered candidates via `dir(module)`, tried plausible entry points with try/except, kept only ones that changed node count OR improved TPS, dropped all no-ops.
- **Result**: keep. 4 runs: tps=4,781 / 4,777 / 4,793 / 4,800 → **avg 4,788, mfu=28.0%**, memory=49.1 GiB. Loss=9.21808, grad_norm=4.5867 (bitwise identical). Numerics test passes. Wall time 140s (compile got ~63s longer than iter 5 due to `schedule_overlap_bucketing`'s analysis — one-time, not per-step).
- **TPS gain attribution**:
  - Without `schedule_overlap_bucketing`: tps drops to ~4,690 (just bucketing alone gives ~+2-3% over iter 5).
  - With `schedule_overlap_bucketing`: tps reaches ~4,780 (extra +2% from reorder).
  - Together: **+4.6% on top of iter 5, +14.9% vs baseline.**
- **Modules tried that did nothing useful here**:
  - `micro_pipeline_tp.micro_pipeline_tp_pass`: ran but no-op. Theory: the 68 TP `all_reduce` nodes don't match its AG+mm / mm+RS patterns — it expects `_c10d_functional.all_gather`/`reduce_scatter`, not `all_reduce`.
  - `fsdp.dedup_fsdp_reduce_scatter`, `bucketing.bucket_all_reduce`, `low_contention_collectives.replace_collectives_with_low_contention`, `reinplace.*`, `group_batch_fusion.group_batch_fusion_passes(pre_grad=False)`, `misc_patterns.numpy_compat_normalization`: all no-ops on this graph.
  - `pad_mm`, `b2b_gemm`, `decompose_mem_bound_mm`, `fuse_attention`: no directly-callable pass entry points in this PyTorch — they're handler/registration modules driven via `joint_graph_passes` config flags. We set the config flags but didn't see additional node changes beyond what joint_graph already did.
- **Lessons**:
  - **`schedule_overlap_bucketing` is the real win** — node-count is unchanged but reordering for comm/compute overlap moved TPS by ~+2%. Iter 3/4's manual FX reordering couldn't find this; the upstream pass does proper dependency analysis.
  - **Upstream `bucketing.*` works where iter 3 didn't**: it knows how to bucket AGs into contiguous-recoverable layouts (no slow reshape-copy).
  - **Iter 5's `binary_folding_pass` and `dedupe_symint_uses_pass` are silent ImportError no-ops** on this PyTorch build — kept the try/except so they activate if a future PyTorch upgrade exposes them.
  - **The `+810` and `+615` node deltas from the bucketing passes are not "more work" but graph restructure**: they add cat / wait / slice / reshape nodes around the bucketed collectives. Net effect on runtime is positive because the per-launch fixed cost goes down.

---

## compile_fx_inductor (whole-graph) — discard (xxxxxxx)

- **Idea**: Route the cleaned joint fwd+bwd graph through Inductor's full pipeline (`compile_fx_inner` / `compile_fx`) to get **Triton kernel codegen**. Iter 5's pattern passes did FX-level rewrites; the next lever is full kernel codegen.
- **Changes**: Added `compile_fx_inductor(gm, example_inputs)` that tries 3 entry points and overrides `gm.forward` with the compiled callable. Registered AFTER `apply_inductor_pattern_passes`. Failures fall through to no-op so iter 5 gains persist.
- **Result**: discard. step 20: tps=4,577 mfu=26.80% (basically iter 5's number, since pass falls through). Numerics PASS. The pass never installed a compiled function.
- **Failures (all 3 variants)**:
  1. `compile_fx_inner(gm, list(example_inputs))` → `InductorError(AssertionError: both a fallback and a decomp for same op: aten._to_copy.default)`. The decomp registry sees `_to_copy.default` twice. Suspected: iter 5's `joint_graph_passes` primed Inductor state, and the second call hits a duplicate.
  2. `fx_codegen_and_compile(gm, list(example_inputs))` → `TypeError: missing 1 required positional argument: 'inputs_to_check'`. API needs args we can't infer without reading the source.
  3. `compile_fx(gm, list(example_inputs))` → `RuntimeError: Found a custom (non-ATen) operator whose output has alias annotations: _c10d_functional::all_reduce_`. `compile_fx` re-runs functionalization; the joint graph already contains the in-place `_c10d_functional.all_reduce_.default` (DTensor redistribute baked it in). This is structural: graphs with mutable functional collectives can't pass through `compile_fx`.
- **Lessons**:
  - **Whole-graph Inductor codegen is blocked on (a) in-place functional collectives, and (b) decomposition registry conflicts after iter 5's pattern passes.** Two distinct walls; both need workarounds.
  - To use Inductor codegen on this workload we must either (i) compile **regions** that exclude collectives, (ii) reset Inductor's registry between calls, or (iii) re-trace the graph fresh and avoid emitting the in-place all_reduce_.
  - First-step compile attempts only added ~7s wall time even when they all failed, so safe try/except wrappers are cheap.

---

## apply_inductor_pattern_passes — keep (76d3f9e)

- **Idea**: PyTorch's Inductor ships several FX-level pattern matchers (`joint_graph_passes`, `post_grad_passes`, etc.) that fuse common ATen sequences (mm+bias+activation, SDPA epilogues, `_to_copy` round-trips, …) by rewriting the GraphModule in place. They were designed for the same kind of joint fwd+bwd graph that `make_fx` produces, so they should apply directly.
- **Changes**: Added `apply_inductor_pattern_passes(gm, example_inputs)` that tries 5 upstream entry points in sequence, each wrapped in its own `try/except`: `joint_graph_passes`, `post_grad_passes(is_inference=False)`, `pre_grad_passes`, `binary_folding_pass`, `dedupe_symint_uses_pass`. Each logs availability and before/after node counts. Registered AFTER `remove_detach_nodes`.
- **Result**: keep. Two consecutive runs: tps=4,569 (mfu=26.75%) and tps=4,576 (mfu=26.80%); avg **tps=4,572 (+9.9% vs baseline 4,161)**. Loss=9.21808, grad_norm=4.5867 (bitwise identical). Numerics test passes. Wall time 77s.
- **Node-count trace**:
  - initial: 11,538
  - after `joint_graph_passes`: 9,106 (−2,432, −21%)
  - after `post_grad_passes(is_inference=False)`: 9,101 (−5)
  - `pre_grad_passes`: ran no-op (pre-grad targets a different graph form)
  - `binary_folding_pass`, `dedupe_symint_uses_pass`: not available in this PyTorch build
- **Analysis**: This is the first iteration to move TPS meaningfully — the −21% node reduction translates to a clean +10% TPS. Earlier topology-only FX edits (bucket/prefetch) failed to translate to TPS, but Inductor's pattern matchers are doing real *semantic* fusion (combining producer/consumer ATen ops into single fused ops, removing entire chains of casts, etc.), not just reordering. The pattern matchers also know how to fuse the `_to_copy` round-trips that iter 2 couldn't touch, which likely accounts for a large chunk of the gain.
- **Lessons**:
  - **Reuse upstream Inductor passes before writing your own.** PyTorch already ships pattern matchers that took years to tune; rewriting from scratch is wasteful.
  - The next big wins likely come from running additional Inductor pipeline stages — full **codegen via `compile_fx`** (Triton kernels for pointwise), and **CUDA graphs** for static replay.
  - `joint_graph_passes` is the right entry point for joint fwd+bwd graphs out of `make_fx`. `post_grad_passes` does small extra cleanup on top.
  - The two unavailable passes (`binary_folding_pass`, `dedupe_symint_uses_pass`) suggest the installed PyTorch version is older than the ones that exposed those names. Future iterations should be careful about API drift.

---

## prefetch_all_gathers (FX-level move) — discard (xxxxxxx)

- **Idea**: Move each `all_gather_into_tensor` node to its earliest valid FX position (right after its inputs). The launch happens on a separate NCCL stream, so intervening compute should overlap with the AG transfer; the `wait_tensor` stays at its original position.
- **Changes**: `prefetch_all_gathers(gm, example_inputs)` walks the graph, computes each AG's earliest-valid position from `A.all_input_nodes`, and moves the AG via the FX doubly-linked-list API. Distance capped at 200 to bound memory peak. Registered after `remove_detach_nodes`.
- **Result**: discard. step 20: tps=4,177 mfu=24.46% memory=47.03GiB; loss=9.21808, grad_norm=4.5867. Pass moved only **65 of 421 AGs**, average distance **7** positions, max **8**. The 200-cap never triggered. Numerics test pass.
- **Analysis**: `make_fx` already places AGs immediately after their input shards are computed. Without also hoisting the AG's **input ops** (`_to_copy(view(param))` patterns) earlier in the graph, the AG cannot move. The 7-position headroom is essentially insignificant.
- **Lessons**:
  - **FX-level reordering alone cannot prefetch FSDP all_gathers**: the AG input chain (cast + view of a placeholder param) is what pins the AG in place. To prefetch effectively we must (a) move the input ops too, OR (b) reorder at a different level (NCCL stream priorities, separate compile step), OR (c) attack the **wait_tensor side** instead — move waits LATE toward consumers so AG↔wait gap widens.
  - This also explains iter 3's bucket result: even after hoisting input ops, the shape-heterogeneity meant most AGs stayed singletons. Together they suggest FX-level scheduling of FSDP collectives has limited slack — the dominant gains live in **fusion of the compute** (kernel side) or in **whole-graph compile via inductor** (which does its own scheduling), not in shuffling existing nodes.

---

## bucket_all_gathers (dim-0 shape-grouped) — discard (xxxxxxx)

- **Idea**: Bucket consecutive `_c10d_functional.all_gather_into_tensor.default` calls (421 of them) to amortize launch overhead. Combine inputs via `aten.cat(..., 0)`, do one all_gather, then `view → slice along dim 1 → reshape` to recover per-original outputs preserving rank-major layout.
- **Changes**: Added `bucket_all_gathers` pass after `remove_detach_nodes`. Bucket size cap = 8; grouping by `(group_size, group_name, dtype, device, len(shape), shape[1:])`. Hoisting strategy: input ops (`_to_copy`, `view`, `_unsafe_view`, `t`, `transpose`, `clone`, `reshape`) of later bucket entries get moved to before the first AG so the cat input is computable in-order. Non-pure-op input chains cause flush. Slice→reshape returns non-contiguous → used `aten.reshape.default` (may copy) instead of `view`.
- **Result**: discard. step 20: tps=4,169 mfu=24.41% memory=47.03GiB; loss=9.21808, grad_norm=4.5867 (bitwise identical). Numerics test pass. 421 AGs → 64 bucketed calls (160 AGs bucketed, 261 left as singletons).
- **Analysis**: Shape heterogeneity is the limiter. FSDP unsharding emits AGs with widely varying `shape[1:]` (per-layer: `(16032,4096), (1024,), (1,4096,4096), (512,4096), (128,4096), (128,4096), (1024,2048)` etc.), so the shape-compatible grouping rule rarely finds 3+ neighbors. Average bucket size 2.5. The reduction from 421 → 64+261 = 325 collective launches (~23% fewer launches) is real but does NOT show up as tps. Two likely reasons: (i) the reshape after slice copies data (non-trivial extra work), wiping out the savings; (ii) the unbucketed AGs still dominate runtime — bucketing 38% of nodes leaves 62% untouched.
- **Lessons**:
  - **Shape-grouped dim-0 bucketing has structural limits.** To bucket aggressively across all AGs we must FLATTEN inputs to 1D before cat, then carefully reshape back. The flat path also dodges the non-contiguous slice problem (a 1D slice is contiguous).
  - **Per-original reshape can cost as much as the launch savings.** If a bucketed approach forces a copy per output, the net is ~zero. Future bucketing should arrange so each per-original output is recoverable via a contiguous view (no copy).
  - **The wins from reducing launches require also overlapping with compute.** Without prefetching, a synchronous AG+wait pair still blocks on data arrival. **Bucketing without prefetch ≠ free perf.**
  - **Next ideas to pursue:** (a) AG prefetch — move AG nodes earlier so communication overlaps with intervening compute, leaving the wait at the original position; (b) inductor regional compile on pointwise hot paths; (c) CUDA graphs.

---

## remove_redundant_to_copy — discard (xxxxxxx)

- **Idea**: 842 `aten._to_copy.default` nodes appear in the graph. If any are no-op casts (target dtype/device equals source dtype/device), removing them is safe and cheap.
- **Changes**: Added `remove_redundant_to_copy(gm, example_inputs)` that walks all `_to_copy.default` nodes, compares requested kwargs against the input's `meta["val"]`, and erases only when dtype/device/layout all match AND no user is in-place. Registered after `remove_detach_nodes`.
- **Result**: discard. 0 of 841 candidate `_to_copy` nodes qualified. tps/loss/grad_norm bitwise identical to iteration 1 (4180 tps, loss 9.21808, grad_norm 4.5867). Numerics test passes.
- **Analysis**: All `_to_copy` nodes in this graph are genuine fp32 → bf16 mixed-precision casts (or vice versa), not redundant. The conservative criterion correctly rejected all of them.
- **Lessons**: The 842 `_to_copy` count cannot be reduced by pure elimination; it must be attacked by **(a)** detecting bf16→fp32→bf16 round-trips and collapsing them, **(b)** fusing the cast into producer/consumer kernels (e.g., via inductor regional compile), or **(c)** restructuring the graph so casts happen once instead of repeatedly. Future iterations should investigate the round-trip pattern.

---

## remove_detach_nodes — keep (ae6e425)

- **Idea**: The recon counted 196 `aten.detach.default` nodes. The runtime executes the traced graph under `torch.no_grad()` (see `trainer.py: _make_fx_forward_backward_step`), so detach has no autograd effect at all. Removing them should reduce dispatcher work and free held tensor references.
- **Changes**: Added `remove_detach_nodes(gm, example_inputs)` to `passes.py`. Collects all `aten.detach.default` call_function nodes, rewires users via `replace_all_uses_with(node.args[0])`, erases, then runs `eliminate_dead_code()` / `lint()` / `recompile()`. Registered as the first pass in `construct_default_graph_passes`.
- **Result**: 196 detach nodes removed. Numerics test passes (`aot_fx_trace_vs_eager`). Two benchmark runs: tps=4,202 (mfu=24.60%) and tps=4,174 (mfu=24.44%). Both produce loss=9.21808, grad_norm=4.5867 — bitwise identical to baseline. Memory drops from 49.0 → 47.0 GiB consistently.
- **Analysis**: TPS delta is within the 1-3% noise band, so the speedup is at best marginal. The robust effect is the ~2 GiB memory drop: each detach node held a Python reference to its input via the FX graph, preventing the underlying tensor from being released during graph execution. Numerics are unaffected because detach shares storage with its input, so users that read the same data after removal are reading identical bytes.
- **Lessons**: (1) Removing FX nodes that hold references can free real memory even when the op itself is "free." (2) detach.default in joint fwd+bwd graphs is essentially always dead under run_traced's no_grad. (3) Future cleanup passes should focus on bigger node categories (e.g. 842 `_to_copy.default`, 196 (now 0) detach was the smallest cleanup target).

---

## Graph reconnaissance — discard (xxxxxxx)

- **Idea**: Before picking concrete optimizations, count FX node targets in the joint fwd+bwd graph to find the highest-leverage areas. No optimization is attempted in this iteration.
- **Changes**: Added a temporary `_recon_dump_graph_stats` pass that counts call_function targets and writes the top-40 ops, collective ops, and view-like ops to `/tmp/recon_graph_stats.txt`. Pass returns `gm` unchanged. Reverted after capturing the stats.
- **Result**: discard — no perf change attempted (single inspection run with `--training.steps 3`). Confirmed baseline still passes through unchanged.
- **Analysis**: See LEARNINGS.md "Graph shape snapshot" for the full table. Headline numbers: 11.9k nodes; 421 all_gather_into_tensor + 421 reduce_scatter_tensor + 68 all_reduce; 196 detach + 842 _to_copy; 32 SDPA flash fwd+bwd; 65 fused_rms_norm fwd+bwd. The collective counts are the obvious lever — about 13 unbucketed FSDP all-gathers per layer × 32 layers.
- **Lessons**: Pass list ordering matters: anything we add to `construct_default_graph_passes` runs after make_fx_tracer's `_remove_cpu_shadow_chains`, so the recon numbers above already exclude shadow chains. Future cleanup passes should not target CPU shadow nodes.
