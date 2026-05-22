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
