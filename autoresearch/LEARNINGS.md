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

- **Removing `aten.detach.default` nodes from the joint graph.** The
  graph is executed under `torch.no_grad()`, so detach is autograd-only
  and pure overhead. Removing 196 such nodes did not change numerics,
  freed ~2 GiB of memory, and shaved noise-level tps. The mechanic:
  collect nodes whose target is `torch.ops.aten.detach.default`, call
  `node.replace_all_uses_with(node.args[0])`, then `erase_node`,
  finally `eliminate_dead_code() + lint() + recompile()`.

## Patterns that didn't work

- **Same-dtype `_to_copy.default` elimination.** All 842 `_to_copy.default`
  nodes are genuine fp32↔bf16 mixed-precision casts; none satisfy the
  same-dtype/device/layout criterion. Plain elimination is dead-on-arrival
  here. To make progress on these casts, try (a) bf16→fp32→bf16
  round-trip collapse, (b) fuse the cast into the producer/consumer
  kernel via inductor regional compile, or (c) avoid emitting the cast
  altogether by restructuring the graph (e.g. keep a master copy and
  read it as bf16 once).
- **Shape-grouped dim-0 AG bucketing (with reshape-back).** Cut launches
  ~23% but TPS unchanged. Two compounding reasons: (1) FSDP all_gather
  inputs are highly shape-heterogeneous (per-layer shapes differ by both
  parameter and FSDP shard layout), so bucket sizes average ~2.5 even
  with aggressive hoisting; (2) recovering per-original outputs requires
  a slice along the within-rank dim, which is non-contiguous and forces
  a reshape that copies. The copy can erase the launch savings. Bucketing
  is **necessary but not sufficient**: it must be paired with
  contiguous-view recovery (e.g. flatten-then-cat 1D inputs) AND with
  comm/compute overlap (prefetch) to actually move TPS.
- **FX-level AG node prefetching.** Moving AGs to their earliest valid
  FX position only relocates 65/421 nodes by 7 positions max. `make_fx`
  already emits AGs right after their input casts/views. To do
  meaningful prefetch we'd need to hoist the input ops too, OR move
  `wait_tensor` nodes LATER, OR get scheduling from a downstream
  compiler (Inductor) rather than from FX reordering. **Takeaway: pure
  FX-graph topology shuffling has very limited TPS impact on this
  workload.** The win has to come from compute-side optimizations
  (fusion, CUDA graphs, whole-graph compile).

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
