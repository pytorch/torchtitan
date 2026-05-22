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
