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
