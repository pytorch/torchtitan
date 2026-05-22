# Autoresearch Learnings

Living document of what works, what doesn't, and how to approach kernel
fusion optimization effectively. The agent owns this file.

Re-read at the start of each loop iteration alongside `IDEAS.md` and
`EXPERIMENT_LOG.md`. Update after meaningful experiments (especially
surprising results, both positive and negative). Keep concise and
actionable — per-experiment details belong in `EXPERIMENT_LOG.md`.

## Methodology

- A reconnaissance pass that counts FX nodes by target (no graph edits)
  is a near-zero-cost way to learn the graph shape. Run it once at the
  start of a session and consult its output before picking ideas.
- Keep `passes.py` minimal between iterations: revert any prior failed
  experiment cleanly before starting the next one.
- Subagent prompts must include reading-scope restrictions verbatim and
  the benchmark/numerics commands, or they will drift.

## Patterns that worked

(empty — agent populates)

## Patterns that didn't work

(empty — agent populates)

## Tooling tips

- A debug pass can call `with open("/tmp/recon_graph_stats.txt", "w") as f`
  to persist data across iterations (logger output is interleaved with
  4 ranks' messages and harder to grep).
- For a quick recon run, pass `--training.steps 3` to `run_benchmark.sh`
  to skip steady-state measurement — but remember tps then is meaningless.

## Graph shape snapshot (baseline, 2026-05-22)

Llama3 8B, FSDP=4 TP=2 bs=1, joint fwd+bwd graph from make_fx:

- Total nodes: ~11,900
- Collectives: 1,820 nodes
  - all_gather_into_tensor: **421** (FSDP unsharding)
  - reduce_scatter_tensor: **421** (FSDP grad reduction)
  - all_reduce: **68** (likely TP)
  - wait_tensor: 910 (one per collective)
- Compute hotpaths: 675 mm, 65 _fused_rms_norm (+65 bwd), 32 SDPA flash
  (+32 bwd), 32 silu (+32 bwd)
- Layout ops: 3358 view, 1125 t, 256 transpose.int, 225 _unsafe_view,
  128 view_as_complex+128 view_as_real (RoPE)
- Other: 842 _to_copy (dtype casts), 196 detach.default, 68 clone

Implications for ideas:
- 421 FSDP all-gathers ≈ 13 per layer × 32 layers — strong **bucketing**
  and **prefetch/overlap** target.
- 196 detach.default likely dead at runtime (no autograd during
  `run_traced` no_grad block) — candidate for cheap **graph cleanup**.
- 842 _to_copy may have round-trips worth eliminating.
- 32 SDPA + RMSNorm + SiLU are well-fused already at the kernel level;
  pointwise around them (RMSNorm-pre, RoPE epilogue) might still benefit
  from **inductor regional fusion**.
