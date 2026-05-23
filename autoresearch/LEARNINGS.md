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
- **Try upstream pattern matchers before writing your own.** Inductor's
  `joint_graph_passes` / `post_grad_passes` etc. encode many fusions
  already; we got our first +10% TPS by simply calling them. Hand-rolled
  passes should target gaps the upstream passes leave behind.
- **TPS is the only metric that matters.** Node-count reduction, fewer
  collective launches, etc. are *necessary but not sufficient*. We've
  seen 23% fewer launches and 7-position prefetches that didn't move
  TPS. Always benchmark, don't trust topology metrics.

## Patterns that worked

- **Removing `aten.detach.default` nodes from the joint graph.** The
  graph is executed under `torch.no_grad()`, so detach is autograd-only
  and pure overhead. Removing 196 such nodes did not change numerics,
  freed ~2 GiB of memory, and shaved noise-level tps. The mechanic:
  collect nodes whose target is `torch.ops.aten.detach.default`, call
  `node.replace_all_uses_with(node.args[0])`, then `erase_node`,
  finally `eliminate_dead_code() + lint() + recompile()`.
- **Reusing upstream Inductor FX pattern matchers
  (`joint_graph_passes` + `post_grad_passes(is_inference=False)`).**
  Single biggest TPS win observed: **+9.9% tps, +2.4 pp MFU, bitwise
  identical numerics, node count 11,538 → 9,101 (−21%)**. These passes
  do real *semantic* fusion (mm+bias+activation, SDPA epilogues,
  collapsing `_to_copy` round-trips that iter 2 couldn't touch), not
  just reordering. **Always try the upstream pattern matchers before
  hand-rolling cleanup passes — they have years of tuning behind them.**

- **Upstream collective bucketing + overlap scheduling
  (`bucketing.bucket_all_gather` + `bucketing.bucket_reduce_scatter` +
  `stable_topological_sort` + `overlap_scheduling.schedule_overlap_bucketing`).**
  **Additional +4.6% TPS (+14.9% cumulative vs baseline), MFU 28.0%**,
  bitwise identical numerics. Bucketing alone adds ~+2% (smarter layout
  than iter 3's manual attempt: no extra memcpy on per-original recovery);
  `schedule_overlap_bucketing` adds another ~+2% by reordering for
  proper comm/compute overlap (a node-count no-op that hand-rolled FX
  prefetch in iter 4 couldn't find). After `bucket_*` you MUST call
  `stable_topological_sort` before `schedule_overlap_bucketing` or you
  hit "Argument was used before defined". Cost: +63s one-time compile
  time for the analysis; no per-step cost.

- **Removing `bucket_all_gather` + `bucket_reduce_scatter` once
  CUDA graphs are in place.**
  **Additional +2.9% TPS (+45.4% cumulative vs baseline), MFU 35.4%**,
  bitwise identical numerics. Bucketing was helpful before CUDA graphs
  (iter 7: +5% from reduced launch count), but iter-8's CUDA graph
  capture amortizes ALL per-launch overhead, so the +810/+615
  cat/slice/reshape nodes that bucketing adds become pure runtime cost
  in the captured graph. `schedule_overlap_bucketing` (which only
  reorders nodes, no node-count change) is independently valuable and
  must stay. **Generalizable rule**: re-ablate kept passes after major
  new passes (especially capture/codegen passes) — the cost-benefit can
  flip.

- **Disabling `torch.utils.deterministic.fill_uninitialized_memory`
  for the captured graph (`disable_uninitialized_memory_fill`).**
  **Additional +5.5% TPS (+41.2% cumulative vs baseline), MFU 34.4%**,
  bitwise identical numerics. `--debug.deterministic` enables this
  global flag, which silently makes every `aten.empty.*` emit a
  FillFunctor zero-init kernel before its consumer overwrites the
  buffer. On this graph that's 65 large empties × ~62 launches/step =
  4,061 FillFunctor kernels / step / 95 ms. By auditing that every
  empty's users are either read-only views or full-overwriters, we can
  safely disable the flag before CUDA-graph capture — the graph then
  records the buffers without the redundant zero. **`--debug.seed=42`
  alone is sufficient for run-to-run reproducibility on this workload;
  the fill is pure overhead.**

- **CUDA-graph wrapping of `gm.forward` (`install_cuda_graph`).**
  **Additional +16.4% TPS (+33.8% cumulative vs baseline), MFU 32.6%**,
  bitwise identical numerics. Steady-state CPU launch overhead on a
  ~10k-node graph turns out to be a much bigger lever than expected
  on this H100 setup. Three implementation gotchas:
  (1) Replay once after the `with torch.cuda.graph(g):` capture block
  before returning outputs — capture only records ops, doesn't execute
  them, and warmup leftovers in the output buffers will corrupt training.
  (2) Use `TracedResult.num_static_inputs` (plumbed via
  `functools.partial` from `construct_default_graph_passes`) to skip
  persistent buffers for FSDP-stable inputs; allocating a buffer per
  flat input doubles model-state memory.
  (3) Monkey-patch `torch.distributed.destroy_process_group` to release
  the graph + buffers + caches before NCCL teardown, else the elastic
  launcher hangs for ~22 min until SIGTERM.

## Patterns that didn't work

- **Whole-graph `compile_fx_inner` IS reachable but breaks bitwise on 8B.**
  Iter-19 successfully unlocked `compile_fx_inner` by (a) rewiring the
  single `_get_submesh` node to its mesh input + erasing (its sole user
  was `output`), (b) draining the 10-op decomp list, and (c) clearing
  `D.fast_random_decomps`'s `@functools.cache`. The compiled callable
  composed with CUDA graphs and delivered **+2.8% TPS (to 6,042)**.
  However the resulting kernels are not bitwise-equivalent on the 8B
  model: loss drifts 9.21808 → 9.11513 and grad_norm 4.5867 → 22.9803
  by step 20 (drift starts at step 1). The debug-model bitwise test
  PASSES (the model is too small to exercise the diverging lowerings)
  but the benchmark loss check fails. **Inductor codegen produces
  numerically-correct-but-not-bitwise-equivalent kernels.** Under the
  bitwise constraint this is a hard ceiling.

- **Whole-graph `compile_fx` / `compile_fx_inner` on the make_fx joint graph.**
  - `compile_fx` re-functionalizes and rejects `_c10d_functional.all_reduce_.default`
    (in-place collective baked in by DTensor redistribute) with "Found a
    custom (non-ATen) operator whose output has alias annotations".
  - `compile_fx_inner` (after iter 5's pattern passes) fails with
    "_to_copy.default registered twice" in Inductor's decomp/fallback
    registry. **Drainable** by deleting offending entries from
    `torch._inductor.lowering.decompositions` for these ops:
    `_to_copy, t, transpose.int, silu, arange, zeros_like, ones_like,
    _fused_rms_norm_backward, silu_backward, embedding_dense_backward`.
  - After draining decomp conflicts, the **real wall** is `device_mesh._get_submesh`
    in the joint graph — a non-ATen Python op emitted by DTensor's
    redistribute. Inductor refuses any non-`OpOverload` op with
    `AssertionError: ... is not an OpOverload`.
  - `compile_fx`'s `post_grad` reinplace pass **resurrects in-place
    collectives** even when we explicitly swapped them out beforehand.
  - **Takeaway**: to unlock Inductor codegen on this graph, we must
    either (a) replace `_get_submesh` nodes with their resolved sub-mesh
    constant before invoking codegen, OR (b) compile only **regions**
    that exclude DTensor helpers and in-place collectives. Drop-in
    whole-graph codegen on the make_fx joint graph won't work.

- **Functionalizing in-place collectives by adding `wait_tensor`.**
  Swapping 133 in-place ops (`all_reduce_`, `all_gather_into_tensor_out`)
  to functional + wait cost **−1.5% TPS** despite preserving numerics:
  each extra `wait_tensor` dispatch has measurable per-step cost. The
  fewer the collective sync points, the better (this also explains why
  iter-7 bucketing's reduction of total collectives is doubly valuable).

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

## Graph shape snapshot (post-iter-22, 2026-05-23 — current best config)

After `remove_detach_nodes` + `apply_inductor_pattern_passes`
(joint_graph_passes + post_grad_passes + schedule_overlap_bucketing) +
`disable_uninitialized_memory_fill`, before `install_cuda_graph`:

- Total `call_function` nodes: **9,893**
- Collectives (down 35% from baseline; joint_graph / overlap consolidated
  more than expected):
  - all_gather_into_tensor: **229** (was 421)
  - reduce_scatter_tensor: **293** (was 421)
  - all_reduce: **68** (unchanged)
  - wait_tensor: 622 (was 910)
- Compute hotpaths: 675 mm, 65 _fused_rms_norm, 65 _fused_rms_norm_backward
- Layout ops: 2022 view, 1125 t, 256 transpose, 417 _unsafe_view, 453 slice,
  256 clone, 128 view_as_complex + 128 view_as_real, 64 _conj
- **`_to_copy.default`: 649** (was 842): **420 bf16→fp32**, **228 fp32→bf16**,
  1 bool→fp32. Mostly mixed-precision boundaries (RMSNorm in fp32, mm in bf16).
- **`cat.default`: 226** (130 bf16, 96 fp32) — concentrated in attention/rotary.
- 225 add.Tensor, 225 mul.Tensor (sequential residual+grad chains).
- 32 empty.memory_format + 32 copy_.default (down from 65 empty — also
  consolidated by passes).

Implications for ideas:
- 421 FSDP all-gathers ≈ 13 per layer × 32 layers — strong **bucketing**
  and **prefetch/overlap** target.
- 196 detach.default likely dead at runtime (no autograd during
  `run_traced` no_grad block) — candidate for cheap **graph cleanup**.
- 842 _to_copy may have round-trips worth eliminating.
- 32 SDPA + RMSNorm + SiLU are well-fused already at the kernel level;
  pointwise around them (RMSNorm-pre, RoPE epilogue) might still benefit
  from **inductor regional fusion**.
