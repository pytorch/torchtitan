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
