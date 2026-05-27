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
