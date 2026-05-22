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

## Eliminate identity DTensor view chains — keep (pending commit)

- **Idea**: ~2K of 3358 `aten.view.default` nodes in the AOT graph are identity views (requested size == input shape), emitted by DTensor plumbing. Replace each with its input via FX peephole and let dead-code elimination clean up. Expected: graph shrink, possibly small launch-overhead win.
- **Changes**: Added `remove_identity_views(gm, example_inputs=None)` to `passes.py` and prepended it to `construct_default_graph_passes`. Iterates `gm.graph.nodes`, matches `aten.view.default` whose requested shape equals `node.args[0].meta["val"].shape` element-wise (concrete-int compare; skips SymInt and missing meta), calls `replace_all_uses_with` + `erase_node`, then `eliminate_dead_code` + `lint` + `recompile`.
- **Result**: tps=4,460 (+7.2% vs 4,161), mfu=26.12%, memory=47.03 GiB (-2.0 GiB vs 49.0), wall_time=81s, loss=9.21808, grad_norm=4.5867. Bitwise-identical numerics test passes. Pass removed 2006/3358 view nodes; 1222 retained as genuine reshapes.
- **Analysis**: Identity-view elimination is purely structural so the +7.2% must come from the AOT runtime carrying these no-op nodes through to per-step execution (likely as extra Python/dispatcher overhead and/or reference-keeping that delayed buffer deallocation). The -2.0 GiB confirms the latter — identity views were anchoring tensor lifetimes longer than necessary. Removing them is essentially free (single linear scan, no math change).
- **Lessons**: (1) Cleanup passes can yield meaningful perf, not just compile-time wins. (2) Run identity-view elimination **first** so subsequent pattern matchers see a clean graph (e.g. producer→view→consumer chains collapse to producer→consumer). (3) IDEAS.md count estimates from grep can underestimate; this confirms with a numeric "actually removed" metric in the pass log.

---

## Drop unused SDPA outputs — discard (xxxxxxx)

- **Idea**: `_scaled_dot_product_flash_attention.default` returns a 9-tuple but only outputs `[0,1,6,7]` are used downstream. The pre-pass graph dump shows 4 dead `getitem` nodes per SDPA × 32 layers = 128 nodes potentially eliminable, mirroring Exp 1's reference-anchoring story.
- **Changes**: Added `drop_unused_sdpa_outputs` peephole pass; registered after `remove_identity_views`. Scanned the graph for `aten._scaled_dot_product_flash_attention.default` nodes and erased their zero-user `getitem` children.
- **Result**: tps=4,432, mfu=25.95%, memory=47.0 GiB, wall_time=74s. Pass removed **0** nodes. Numerics test still passes. Tps drift (-0.6% vs Exp 1's 4,460) is within run-to-run noise.
- **Analysis**: By the time this pass executes (after `remove_identity_views`, which ends with `eliminate_dead_code()`), the dead getitems are already gone. FX's DCE traverses pure ops only — `_scaled_dot_product_flash_attention` is impure (RNG-using) so its node stays, but the individual `getitem` extractors are pure and get dropped as soon as their consumers vanish. The recon's evidence came from the raw post-make_fx dump, *before* any DCE pass had run.
- **Lessons**: Always sanity-check "dead output" ideas against the *post-DCE* graph, not the raw make_fx dump. The peephole-as-DCE story Exp 1 told doesn't generalize to outputs of impure ops, because the FX framework already handles that case. Recorded in LEARNINGS.md so we don't burn another iteration on this class of idea.

---

## FSDP all-gather prefetch (Strategy A) — discard (xxxxxxx)

- **Idea**: Move each `_c10d_functional.all_gather_into_tensor` node to its *earliest legal* topological position (just after its latest input). Functional collectives use a separate NCCL stream, so launching earlier should give NCCL more time to complete the gather before the matching `wait_tensor` blocks compute.
- **Changes**: Added `reorder_fsdp_all_gather_prefetch` pass after `remove_identity_views`. For each `all_gather_into_tensor.default` node, computed the index of its latest input in the current node order and moved the AG to that position + 1. No `eliminate_dead_code` (pure reorder).
- **Result**: tps=4,371 (-2% vs 4,460), mfu=25.60%, memory=47.0 GiB, wall_time=74s, loss=9.21808, grad_norm=4.5867. Numerics bitwise-identical. Pass moved 130/421 AGs (those that weren't already at their earliest legal position) by an average of 3.5 slots each.
- **Analysis**: Each AG's input — the `_to_copy(fp32→bf16)` of the sharded weight plus surrounding `view` ops — is produced *layer-locally*, not far back in the graph. So the earliest-legal slot for the AG is still inside the same layer. "3.5 slots earlier" is not enough compute distance to hide AG latency. The slight tps regression (-2%) is consistent with run-to-run noise; numerics are unchanged so the pass is provably semantics-preserving — it just doesn't help.
- **Lessons**: True FSDP prefetch needs to *hoist the AG's producer chain* (cast + view + AG together) above the previous layer's `wait_tensor`, not just the AG node alone. This is Strategy B, deferred to a future iteration. Marked the idea `[~]` in IDEAS.md (partial — Strategy A explored, B open). Recorded the functional-collective op qualnames in LEARNINGS.md for reuse.

---

## Bucket Q/K/V all-gathers — discard (xxxxxxx)

- **Idea**: Per Llama3 attention layer, three consecutive `_c10d_functional.all_gather_into_tensor` calls (Q, K, V) share group/dtype/dim1. Concatenate the shards along dim 0, AG as ONE collective, split back. Eliminates 2 AG launches/layer × 32 layers = 64 launches/step. Hypothesis: launch-bound regime → tps win.
- **Changes**: Added `bucket_qkv_all_gathers` after `remove_identity_views`. Sliding window over `all_gather_into_tensor.default` nodes; matched runs of 3 sharing `args[1]` (world_size), `args[2]` (group_name), `shape[1]`, and dtype. For each triplet: hoist K/V shard producer chains above AG_q, insert `aten.cat([sq, sk, sv], 0)` → reuse AG_q's slot for the combined AG → `aten.split_with_sizes` the wait output → rewire downstream uses. Updated `meta["val"]` on new nodes by propagating fake tensors. Two helpers added: `_hoist_predecessors_before` (DFS) and `_new_after` (chaining inserter to avoid reverse-order insert trap).
- **Result**: tps=4,490 (best of 2; +0.7% vs 4,460), mfu=26.29%, memory=47.0 GiB, wall_time=74s, loss=9.21808, grad_norm=4.5867. Bitwise-identical numerics. 32 triplets bucketed (one per layer; no fallback needed). Below the +1% keep threshold.
- **Analysis**: At FSDP=4 on NVLink, the Q shard alone is 64 KB and dominates the AG time — the collective is bandwidth-bound, not launch-bound. The cat+split surgery added small ops on the CPU side and, on the first attempt, a reshape over a non-contiguous strided slice triggered a copy (16 MB × 3 per layer per step). Net launch savings ≈ reshape overhead → essentially neutral within noise.
- **Lessons**: (1) Bucketing wins are conditional on *which* collectives are launch-bound; verify the size regime first. (2) Use coalesced collectives (single launch, separate outputs) if available, instead of cat+AG+split — avoids the reshape overhead. (3) FX `inserting_after(n)` orders multiple sibling inserts in REVERSE — chain them. (4) Numerics passing on the small deterministic config does not guarantee benchmark stability; always run both. (Recorded in LEARNINGS.md.)

---

## FSDP all-gather prefetch (Strategy B, first attempt) — discard (xxxxxxx)

- **Idea**: Follow-up to Exp 3. Move the entire private producer chain (`_to_copy(weight_shard) → view → ... → AG`) above the *previous* `wait_tensor` so the AG launches before that wait blocks. Since the chain only depends on a graph-placeholder weight shard, it's safe to hoist arbitrarily far back. The functional collective then runs on the NCCL stream in parallel with whatever compute follows the prior wait.
- **Changes**: Added `prefetch_fsdp_all_gathers_cross_layer` after `remove_identity_views`. For each `all_gather_into_tensor.default` AG: (a) computed the private-producer subgraph by reverse-DFS where every node's users ⊆ visited set; (b) located `prev_wait = previous all_gather's wait_tensor`; (c) iterated the subgraph in topo order and called `prev_wait.prepend(node)` to move each node just before `prev_wait`. Used `prepend` because `Graph.move_before` doesn't exist on this PyTorch.
- **Result**: tps=4,481 (best of 2, +0.5% vs 4,460), mfu=26.24%, memory=47.07 GiB (≈unchanged), wall_time=73-74s. Numerics bitwise-identical. 290/421 AGs hoisted, avg 10.6 slots / 1.2 wait barriers crossed each. Run 1 tps=4,448 (-0.3%), Run 2 tps=4,481 (+0.5%) — within noise, no measurable win.
- **Analysis**: `prev_wait` was misidentified — within a layer, the weight AGs and their waits are packed tightly at the layer's start, so "the previous AG's wait" almost always landed inside the same layer. The hoist therefore moved producer chains ~10 slots earlier *within the same layer*, not across layers. True cross-layer prefetch needs the layer-end barrier (the `wait_tensor` of the output reduce_scatter, mesh `'21'`, downstream of `w2`'s mm) as the anchor — that's the only place where moving above it gains a whole layer of overlapped compute.
- **Lessons**: (1) "Previous AG" ≠ "layer boundary" in the FX node order — pick anchors based on the *compute structure*, not the local neighbor. (2) Topological correctness alone isn't enough; the chosen barrier dictates the magnitude of the overlap window. (3) `Graph.move_before` doesn't exist; use `Node.prepend`/`Node.append`. (Recorded in LEARNINGS.md.) Strategy B with the corrected layer-end anchor remains an open opportunity — marked the idea `[~]`, not `[x]`.

---

## FSDP all-gather prefetch (Strategy B v2, "most recent TP-RS wait" anchor) — discard (xxxxxxx)

- **Idea**: Anchor weight-AG producer-chain hoists on **TP `reduce_scatter` waits** (mesh group of size 2), not the previous AG's wait. The TP RS marks the layer's compute barrier (post-`wo` mid-layer, post-`w2` end-of-layer), so hoisting above one should give a half-layer-or-more of compute to overlap.
- **Changes**: Added `prefetch_fsdp_all_gathers_v2`, registered after `remove_identity_views`. Detected barriers as `wait_tensor` whose RS arg has world_size=2. For each FSDP weight AG (world_size=4), computed its private producer chain (reverse DFS), found the most recent barrier strictly before the AG, and `barrier.prepend(node)` each chain member in topological order. Matched group by **size** (not name string), since group-name tags are PG-creation-order-dependent and unstable across runs.
- **Result**: best-of-2 tps=4,467 (+0.16% vs 4,460), runs 4,448 / 4,467, mfu=26.16%, memory=47.2 GiB (+0.2 GiB), wall_time=73-74s. Numerics bitwise-identical. 290/291 weight AGs hoisted, avg 20.5 slots / **1.0 barriers crossed** each.
- **Analysis**: 1.0 barriers crossed is the smoking gun — most hoists landed at a *mid-layer* RS-wait (the post-`wo` barrier), still inside the same layer. Of the 130 RS-waits, ~half are mid-layer (post-`wo`) and ~half are layer-end (post-`w2`). "Most recent" anchoring picks whichever fires later in node order, which is typically mid-layer for the layer's later-in-burst AGs (w1/w3/w2). Memory bump (+0.2 GiB) is too small to indicate real prefetch is happening — consistent with intra-layer hoisting (AG-output lifetime extended by only ~20 slots).
- **Lessons**: (1) When a hoist averages 1.0 barriers crossed, the anchor is wrong — you're moving across a single intra-layer barrier, not a layer. (2) The layer-end barrier is the post-`w2` RS-wait specifically; filter to *every other* RS-wait, or match the RS shape (the layer-end RS output is the residual-stream activation, which is dim=4096 and feeds into an `add` for the residual). (3) Match collective group by *size*, not name — names depend on PG-creation order. (Recorded in LEARNINGS.md.) The idea remains `[~]`: Strategy B v3 with end-of-layer-only barrier filter is worth one more try.

---

## remove_identity_ops (slice + double-t + identity-cast) — keep (pending commit)

- **Idea**: Extend Exp 1's peephole-cleanup playbook to two more no-op patterns the AOT graph still carries: full-range `aten.slice.Tensor` (start=0, step=1, end≥size(dim)), `aten.t(aten.t(x)) → x`, and same-dtype `aten._to_copy(x, dtype=x.dtype) → x`. All numerics-safe metadata-only patterns.
- **Changes**: Added `remove_identity_ops` to `passes.py`, registered after `remove_identity_views`. Three peepholes in sequence: identity slice → double-transpose → identity cast. Each peephole iterates `gm.graph.nodes`, matches target ops by qualname, validates the no-op condition against `meta["val"]` shape/dtype, calls `replace_all_uses_with(node.args[0])`, erases. End with one `eliminate_dead_code() + lint() + recompile()`.
- **Result**: best-of-2 tps=4,569 (+2.4% vs 4,460), runs 4,569 / 4,558, mfu=26.76%, memory=47.0 GiB (unchanged), wall_time=74-76s, loss=9.21808, grad_norm=4.5867. Bitwise-identical numerics. Eliminated 421/421 slices (100% — every slice was full-range DTensor local-extraction), 450/1125 double-t (40% — mostly backward grad accumulation), 0/841 identity casts (every `_to_copy` is real bf16↔fp32 traffic).
- **Analysis**: 100% slice match rate confirms the recon's "many full-range slices" undercount. Double-t at 40% is large, dominated by backward-side `t(t(W))` chains. The +2.4% with zero memory change differs from Exp 1's +7.2% with -2.0 GiB: slice and `t` are pure metadata ops (no allocation), so the win is dispatch/launch overhead, not deallocation. Combined with Exp 1, total improvement vs raw baseline is now (4,569 / 4,161) − 1 = +9.8%.
- **Lessons**: (1) Peephole cleanup is the highest ROI category so far — 2/2 hits, simple implementation, low risk. (2) Memory wins (Exp 1) and dispatch wins (Exp 7) are distinct mechanisms; checking which one fires tells you what the underlying cost was. (3) Identity casts didn't fire here — every `_to_copy` is genuine bf16↔fp32. If future work targets `_to_copy` overhead, the path is dtype-policy changes (idea #6, #9), not peephole elimination.

---

## FSDP all-gather prefetch (Strategy B v3, end-of-layer-only anchor) — discard (xxxxxxx)

- **Idea**: Final prefetch attempt with the correct barrier filter. Filtered the 130 TP-RS-waits to **66 end-of-layer-only barriers** (every-other in fwd/bwd node order; structurally validated by FFN-block-gap ≈ 127 lines and attention-block-gap ≈ 210 lines). Hoisted each FSDP weight AG's private producer chain above the most recent filtered barrier.
- **Changes**: Added `prefetch_fsdp_all_gathers_v3` after `remove_identity_ops`. Match collectives by `args[2]` world_size (RS signature `(input, op_str, world_size, group_name)`). Used `barrier.prepend(node)` for relocation.
- **Result**: best-of-2 tps=4,549 (-0.4% vs 4,569 baseline), runs 4,549 / 4,546, mfu=26.64%, memory=47.2 GiB (+0.2), wall_time=74-76s. Numerics bitwise-identical. 290/291 AGs hoisted, avg 47 slots / **1.44 barriers crossed** per AG (close to the ≥1.5 target).
- **Analysis**: Reaching the diagnostic target finally — avg 1.44 barriers, ~47 slots moved — should have produced visible overlap. It did not. tps regressed slightly within noise. Either NCCL's stream scheduling at this scale is already serialized so launching the AG node earlier doesn't create earlier-on-stream issue, or the AOT-trace runtime doesn't actually issue collectives onto a separate non-blocking stream. Either way, FX-level node reordering of the AG is *not* the right lever.
- **Lessons**: After 4 prefetch attempts (Exps 3, 5, 6, 8), 0 measurable wins. Closing out the AG-prefetch direction. Marked the parent idea `[x]` in IDEAS.md. The strategic pivot recorded in LEARNINGS.md: to reduce AG cost, **reduce the number** of AGs (bucketing with a coalesced-collective op, if one exists) or **reduce the per-AG cost** (eliminate the fp32→bf16 cast prefix). Reordering alone is dead.

---

## collapse_views_and_aliases — discard (xxxxxxx)

- **Idea**: Continue the peephole sweep. Three more no-op patterns: view-chain collapse (`view(view(x, _), x.shape) → x` for single-user inner), `aten.alias/detach.default(x) → x`, and identity `expand`.
- **Changes**: Added `collapse_views_and_aliases` after `remove_identity_ops`. Conservative single-user inner-view gate; alias/detach/expand replaced with input.
- **Result**: best-of-2 tps=4,578 (+0.2% vs 4,569), runs 4,578 / 4,570, mfu=26.81%, memory=47.03 GiB (unchanged), wall_time=75-76s. Numerics bitwise-identical. Counts: view-chain 1/1352 (1191 multi-user candidates rejected by the conservative gate), alias 0/0, detach 194/194 (100% from `_fused_rms_norm` and SDPA getitem outputs), expand 0/0.
- **Analysis**: All 194 detaches eliminated cleanly, but at 194 ops/step the dispatch savings are below the noise floor. View-chain collapse was almost entirely blocked by multi-user inner views (DTensor "share-the-reshape" patterns). No alias/expand/lift_fresh-as-passthrough nodes existed at all. The pure-metadata peephole well looks tapped out at this stage.
- **Lessons**: Diminishing returns on pure-cleanup passes. Future peephole ideas need to target *more specific structural patterns* (RoPE `view_as_real(view_as_complex(_))` if any survive, multi-user view-chain CLONING, or matmul-input transpose folding). Plain metadata removal won't move the needle further. Strategy pivot: look at the post-pass-stack graph for the biggest *remaining* sources of dispatch/launch overhead — likely the per-layer FSDP cast→AG chains and the TP split+cat→RS pattern.

---

## cse_views_and_conj — discard (xxxxxxx)

- **Idea**: After Exp 9 closed standalone metadata cleanup, the re-recon flagged two specific CSE patterns: 3-way duplicate `view(view_K, [8192, 4096])` per FSDP-AG output (Q/K/V matmul operands share one source), and duplicate `_conj.default(view_K)` + matching `clone` on the backward RoPE freq tensor. The recon estimated 96 + 32 + 32 = 160 ops/step plus ~128 MiB/step of clone-allocation traffic.
- **Changes**: Added `cse_views_and_conj` after `remove_identity_ops`. Generic `_cse_op` helper keyed by `(id(input), tuple(args))` for views and `(id(input),)` for `_conj`; clones gated on no-in-place consumers. End with `eliminate_dead_code()`.
- **Result**: best-of-2 tps=4,587 (+0.4% vs 4,569), runs 4,587 / 4,572, mfu=26.86%, memory=47.03 GiB (unchanged), wall_time=74-75s. Numerics bitwise-identical. Merge counts hit prediction exactly: view=96, _conj=32, clone=32.
- **Analysis**: Memory floor stayed put — the recon's predicted -128 MiB from clone CSE was real per-step traffic but never anchored the peak (clones live and die within a single backward RoPE step before any larger activation peak). Dispatch savings of 160 ops/step is below noise floor, same regime as Exp 9's 194 detach removal (+0.2%). Both confirm: ≤200 ops/step of pure cleanup is invisible in tps.
- **Lessons**: Combined with Exp 9, recorded a new rule in LEARNINGS.md: *Don't pursue passes that target <500 ops/step of pure cleanup unless they unlock a peak-memory anchor (like Exp 1's identity views did) or expose a bigger downstream rewrite.* The next experiments must hit high-volume traffic: 841 `_to_copy` casts (~1.2 GiB/step), 130 TP split→cat→RS rearrangements (~8.3 GiB/step of cat allocations), or 161 backward param-grad RS bucketing.

---

## elide_split_cat_for_reduce_scatter — keep (pending commit)

- **Idea**: TP-sharded matmul outputs go through `split(x, N, 1) → cat([s0, s1], 0) → reduce_scatter(dim=0, ws=2)` to convert a sequence-parallel scatter into a dim-0 RS. When the split source `x` is contiguous, the cat's byte layout is identical to `x.view([2, N, D])`. Replace the cat with a zero-copy view; DCE removes the now-orphaned split and getitems. Targets 130 occurrences × 64 MiB cat = ~8.3 GiB/step of avoidable allocation traffic.
- **Changes**: Added `elide_split_cat_for_reduce_scatter` after `remove_identity_ops`. For each `aten.cat.default(dim=0)` whose inputs are all `getitem`s of a single `aten.split.Tensor` and whose sole consumer is `_c10d_functional.reduce_scatter_tensor.default`, verified split-source contiguity via `meta["val"].is_contiguous()` and rewrote the cat to `aten.view.default(x, cat_output_shape)`. DCE handled the orphaned split/getitems.
- **Result**: tps **4,641** (BOTH runs identical, +1.58% vs 4,569 baseline), mfu=27.17% (+0.41pp), memory=47.05 GiB (unchanged), wall_time=73-75s, loss=9.21808, grad_norm=4.5867. Bitwise-identical numerics. **130/130 cats elided** (zero skips for any reason). Cumulative over raw baseline: 4,641 / 4,161 = **+11.5%**.
- **Analysis**: The win shows up in launch count + memcpy bandwidth, not resident memory — cats are short-lived intermediates whose lifetime overlaps reused buffers. NCCL `reduce_scatter_tensor` happily accepts a view of a larger contiguous tensor as input; the byte layout is what matters, not the FX-node-creation provenance. Both benchmark runs produced bit-identical tps (4,641), and bit-identical numerics to the baseline — the byte-layout proof holds through autograd and the collective.
- **Lessons**: (1) High-volume *traffic* (8.3 GiB/step here) is the right thing to target after metadata cleanup taps out. (2) The cat-as-view rewrite generalizes to any "split-along-inner → cat-along-new-outer-to-stack-for-collective" sequence-parallel idiom — recorded in LEARNINGS.md. (3) NCCL collectives accept FX-level views without issue when the underlying byte layout is contiguous and matches the expected shape. (4) `meta["val"].is_contiguous()` is the right gate to check before any "rewrite cat as view" pass.

---

## bucket_param_grad_reduce_scatters — discard (xxxxxxx)

- **Idea**: Coalesce 291 backward FSDP param-grad `reduce_scatter_tensor` (ws=4) calls into per-layer buckets using `_c10d_functional.reduce_scatter_tensor_coalesced.default`. Expected to save ~256 launches/step (8 saved per layer × 32 layers + endpoints), especially via the tiny RMSNorm γ shards (1-4 KiB) which are pure launch-overhead.
- **Changes**: Added `bucket_param_grad_reduce_scatters` after `elide_split_cat_for_reduce_scatter`. Found `reduce_scatter_tensor_coalesced.default` exists; used it directly (no cat/split fallback). Walked RS-ws-4 nodes, grouped into buckets capped at 9, replaced each bucket with a single coalesced call and rewired waits.
- **Result**: best-of-2 tps=4,678 (+0.80% vs 4,641 baseline), runs 4,678 / 4,659, mfu=27.40%, memory=47.03 GiB (unchanged), wall_time=72-73s. Numerics bitwise-identical. 33 buckets, avg size 8.82, **291/291 RSes coalesced** (100% coverage). Below the +1% keep threshold.
- **Analysis**: Infrastructure worked perfectly — no skips of any kind, no overhead from cat/split (coalesced collective avoided that path). The win is real but small: at FSDP=4 on NVLink, the collectives are bandwidth-bound. NCCL apparently still issues per-tensor reductions internally even when given a coalesced call, so the FX-level launch-count reduction doesn't fully translate to NCCL-level savings. The mechanism that worked for Exp 11 (cat-as-view eliminating 8.3 GiB/step of allocation traffic) doesn't have a backward-side analog here — the f32 cast bytes are unavoidable.
- **Lessons**: (1) A coalesced functional collective exists and is the right tool, avoiding Exp 4's cat overhead. (2) Bucketing alone on NVLink-FSDP=4 caps at ~+1% per-direction; **need to bundle forward + backward coalescing to compound the wins past threshold**. (3) Same mechanism likely available for `all_gather_into_tensor_coalesced` — verify in the next experiment. Recorded in LEARNINGS.md.

---

## bucket_fsdp_collectives — keep (+10.4%, pending commit)

- **Idea**: Bundle forward AG coalescing + backward RS coalescing in one pass. `reduce_scatter_tensor_coalesced.default` proven in Exp 12; verify the matching `all_gather_into_tensor_coalesced.default` and use both.
- **Changes**: Added `bucket_fsdp_collectives` after `elide_split_cat_for_reduce_scatter`. Confirmed `all_gather_into_tensor_coalesced.default` exists (signature `(Tensor[] inputs, int group_size, Any group_name) -> Tensor[]`; AG world_size is `args[1]`, distinct from RS at `args[2]`). Bucketed both directions with cap 9, structural critical-path check. **Critical implementation insight**: AG bucketing requires hoisting each AG's private producer chain (`_to_copy(placeholder)` per AG) before the coalesced anchor — without hoisting, AG buckets=0 because every weight AG's wait is immediately consumed by `mm`. Forward path uses `prepend` to relocate producer-chain nodes before the coalesced AG anchor. Backward path is straightforward (RSes already group naturally per-layer).
- **Result**: **tps=5,124 (fresh verification run after the experiment's best-of-5)**, mfu=30.00%, memory=47.05 GiB (unchanged), wall_time=71s, loss=9.21808, grad_norm=4.5867. Bitwise-identical numerics. AG ws=4: 33 buckets, avg 8.82, 291/291 coalesced. RS ws=4: 33 buckets, avg 8.82, 291/291 coalesced. **+10.4% over Exp 11 baseline (4,641 → 5,124), +23.1% over raw baseline (4,161 → 5,124).**
- **Analysis**: This is the largest single win of the run. The coalesced AG with hoisting is *effectively the prefetch we couldn't deliver via reordering* in Exps 3/5/6/8. By collapsing 9 separate AG launches per layer into ONE coalesced call hoisted to the layer's start, NCCL transfers in parallel with the entire layer's compute. The 9 individual `wait_tensor` blocks per layer that previously punctuated the compute critical path now consolidate into a single barrier, freeing the GPU to overlap the gather with attention/FFN compute.
- **Lessons**: (1) The "hoist + coalesce" combination is the right form of FX-level prefetch — the coalesce itself serves as the barrier that reordering alone couldn't establish. (2) Bundling forward + backward coalescing compounds well; neither alone would have been a keep at this baseline. (3) AG and RS have different arg orderings (`args[1]` vs `args[2]` for world_size) — easy gotcha. Recorded in LEARNINGS.md.

---

## fuse_coalesced_input_casts (foreach-fuse fallback) — discard (xxxxxxx)

- **Idea**: Foreach-fuse the 9 per-layer `_to_copy(f32→bf16)` casts feeding each coalesced AG, and symmetrically the 9 `_to_copy(bf16→f32)` casts feeding each coalesced RS. Estimated 516 cast launches/step saved if a functional foreach op were available.
- **Changes**: Verified `aten._foreach_to_copy` does not exist (only in-place `aten._foreach_copy`, which needs a pre-allocated dest — not usable in FX). Used the fallback path: per bucket, `view(src_i, [-1])` × 9 → `cat(flats, dim=0)` → single `_to_copy(buf, dtype)` → `split_with_sizes` → reshape back. Bundled detach removal (194/194 elided) as a free addendum.
- **Result**: best-of-3 tps=5,131 (+0.14% vs 5,124), all runs 5,116/5,112/5,131, mfu=30.05%, memory=47.19 GiB (+0.14), wall_time=72s. Numerics bitwise-identical. 33/33 fwd + 33/33 bwd buckets coalesced; 194 detaches removed. Below +1% keep threshold.
- **Analysis**: The fallback path adds 66 cats + 66 splits + 1188 view ops/step. The cat+split overhead approximately equals the 516 cast-launch savings. Detach removal alone was confirmed sub-noise in Exp 9. Memory bumped +0.14 GiB from the cat allocating fused f32 buffers (~12 MiB × 33 = ~400 MiB resident). This is the **same lesson as Exp 4**: cat+split surgery cancels small launch wins when per-launch overhead is already low.
- **Lessons**: (1) Without a functional foreach-cast HOP, fusing small same-shape-family casts is a wash — the surgery overhead matches the launch savings. (2) `aten._foreach_to_copy` does NOT exist in PyTorch; `aten._foreach_copy` does but is in-place. (3) Detach removal stays as a "always include in any bigger pass" item — it's free but never enough on its own to clear threshold.
