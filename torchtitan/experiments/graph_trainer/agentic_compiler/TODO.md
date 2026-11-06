# Agentic compiler: next steps

Status as of 2026-11-06. There are three loops so far, each with a report directory here and a pack in
github.com/SherlockNoMad/graph-trainer-kernel-packs:

- DSv3 671B DistMoE MXFP8;
- GPT-OSS 120B MXFP8;
- Qwen3 30B-A3B DistMoE MXFP8.

The target workflow design is in `WORKFLOW.md` (draft).

## 1. Verify that the matchers are shape- and rank-agnostic

**Claim.** The lift passes match by topology, not by values:
- slots `S` (sizes) and `V` (scalar operands) match any value;
- slot `D` (devices) matches by type.

So a region should lift at the same sites when shapes, sharding or rank change. Two kinds of evidence exist:
- **Device index 0 vs 1:** shown under the fake PG for DSv3 and GPT-OSS.
- **Real multi-rank:** DSv3 16B on 4 real GPUs.

**Gaps:**
- Every fake-PG run is SPMD coordinate 0, so no other rank's graph has been checked on the large models.
- No size, sharding, sequence-length or batch change has been tried on a published plan.

**Keep two questions apart:**
- **(a) lifting:** do the matchers find the same sites? This must be robust.
- **(b) kernel hits:** does a lifted call hit a cached kernel? l1 contains size literals, so a shape change is
  expected to miss.

This item is about (a); (b) belongs to item 2.

- [ ] **Offline replay tool.** Dump the pre-lift graph for a config. Replay a plan's lift passes on it
  (`curate_regions.py` replay or `lift_regions.find_matches`) and report, per region:
  - sites found vs expected (layers x per-layer instances);
  - "no instance to lift" failures and overlap conflicts;
  - the number of distinct l1 patterns and whether the lifted submodule code is unchanged modulo literals.

  A graph dump is far cheaper than a training run.
- [ ] **Experiment matrix**, one axis at a time from each published recipe. For each cell, run (a), then a short
  training run with the pack to confirm loss and grad_norm are bitwise unchanged (misses fall back to aten).
  - **Sharding:**
    - DSv3: FSDP 256 x EP 64 -> FSDP 128 x EP 32 and FSDP 64 x EP 8;
    - GPT-OSS: EP 16 -> EP 8;
    - Qwen3: FSDP 8 x EP 8 -> FSDP 4 x EP 4;
    - plus TP/CP/PP where the model supports them.
  - **Smaller variants:** DSv3 16B, GPT-OSS 20B, a smaller Qwen3 MoE.
  - **Sequence length:** 4096 -> 2048 and 8192, and Qwen3's 32K -> 8K.
  - **Batch size:** MBS 4 -> 2 / 8, and the number of microbatches (1 vs gradient accumulation).
  - **Ranks:**
    - real multi-GPU runs (4-8 GB300) of the small variants, checking every rank's graph;
    - `FAKE_PP_RANK` / real-PP-fake-SPMD for other pipeline coordinates.
  - **Pass pipeline:** the finder pipeline (1 microbatch) vs the benchmark / GA path vs the default pipeline with
    CUDA graphs.
  - **Modes:** `debug.deterministic` on/off. Lifting should not care; the keys currently do.
- [ ] **Fix what breaks.** Known so far:
  - **Bucketing nodes.** G-031 and G-046 (GPT-OSS) absorbed FSDP bucketing nodes, so they did not match in the
    GA graph, where bucketing runs after the extra passes. G-031 was hand-edited and G-046 skipped. Fix in the
    finder (keep bucketing and collective nodes out of regions), or capture on the pipeline training uses.
  - **Literal comparisons.** Any matcher that still compares a literal which can change with shape or rank: audit
    the generated `is_op` argument patterns for non-slot literals (e.g. split sizes, reshape dims derived from
    the batch).
- [ ] **Regression guard**, if it earns its cost: replay the published plans on small saved graphs (CPU).
- [ ] **Kernel-side portability** (found in the pattern audit; not matcher issues, but they break "run anywhere"):
  - **Qwen3 G-01 (`qnfe`)** hardcodes `NSM = 152`. Its numerics then match quack only on 152-SM parts, and the
    pack target checks compute capability, not SM count. Read `multi_processor_count`, or add the SM count to
    the target.
  - **Five GPT-OSS kernels** (G-011, G-014, G-023, G-025, G-036) use whole-GPU cooperative launches with a grid
    barrier: they can serialize or deadlock against overlapping NCCL kernels. Test them on a real multi-GPU job,
    or replace them with non-cooperative versions.
  - **GPT-OSS G-028 and DSv3 G-001 / G-020** hardcode aten's column-reduction split (64 / 128 CTAs). It is valid
    on any GPU with enough SMs, but unchecked.

## 2. Make kernel generation cheaper by starting from verified kernels

Kernels are specialized to one shape: l1 includes size literals and l2 the input shapes. Every model or shape
change therefore regenerates and re-verifies everything.

**Evidence so far:**
- **Reuse helped where it happened.** Writers that started from an earlier kernel produced closely related
  code:
  - Qwen3 weight prep vs DSv3: 0.81 source similarity;
  - Qwen3 G-19 vs DSv3 G-049 (norm backward + quant): 0.77; its docstring says "structure from the verified
    DSv3 kernel pat_a6rv".
- **Elsewhere, near-identical patterns were rewritten from scratch:**
  - router backward: 0.97 op similarity, 0.39 source similarity;
  - loss backward: the same `reference.py` in GPT-OSS and Qwen3, 0.30 source similarity.

- [ ] **Experiment: does seeding help, and how much?** Take one item-1 shape change, e.g. DSv3 671B -> 16B, or
  GPT-OSS at seq 8192.
  - Run the swarm twice on the same problems: from scratch, and with the nearest verified kernel per problem
    (plus its verifier report) in the writer's prompt.
  - Measure per problem:
    - wall time and tokens to an accepted kernel;
    - writer iterations, and verifier FAILs;
    - final speed vs the from-scratch kernel and vs roofline.
- [ ] **Retrieval.** Index all verified kernels by normalized reference op sequence (literals stripped, as in the
  pattern audit, `/tmp/pack_audit/similarity.py`). For a new problem, return the nearest kernels across packs,
  with family, similarity and verdict evidence.
- [ ] **Re-verify before rewriting.** When a new problem's reference matches an existing pattern modulo sizes,
  first run the existing kernel on the new specialization through the harness. Kernels that take their sizes
  at runtime may already be exact, and then only need re-timing and a fresh verifier.
- [ ] **Shape-generic kernels.**
  - Ask writers for kernels parameterized by shape (compile-time specialization cached per shape, runtime
    checks, exact fallback), so one kernel file can serve many specializations.
  - Needs a cache contract for "this kernel covers these l2 keys", or a size-free key; see item 3 on keys.
- [ ] **Find the cost.** From the three boards' timestamps, measure where swarm time goes: writing, self-checks,
  verification, waiting for GPUs, coordinator promotion. Attack the largest share first. The verifier suites
  (exhaustive 2^32 sweeps, sanitizers) are reusable per family.
- [ ] **Size-free l1.** Consider l1 keys with sizes symbolic, so that one pattern is shared across models (loss
  backward already is). Then per-model work would be new specializations of a known pattern.

## 3. Polish the end-to-end workflow into a production skill

**Today the loop is three parts glued together by per-run code:**
- the fusion_region_finder skill;
- `substitute_kernels` (#5838);
- the autokernel skill (#5839);
- a hand-written `loop_recipe.py` per model;
- coordinator conventions;
- pack tooling in a separate repo.

**Slop to remove** (each seen in at least one of the three loops):

- [ ] **One recipe module, not one per model.** Substitution registration, real-input capture
  (`capture_region_inputs`), `LIFT_OPTIONAL`, `unsubstituted_to_call_function` and the det-key timing override
  are copied or patched in each run. Make them first-class.
- [ ] **GA builder.** Fix the gap in GraphTrainer itself: `split_fsdp_collectives.extract_fsdp_reduce_grad_graph`
  deep-copies and runs DCE, which fails on `call_module`. Then delete `unsubstituted_to_call_function`.
- [ ] **Capture on the training pipeline.** Capture on the same pass pipeline and microbatch setup that training
  uses, or make regions independent of it (see item 1, bucketing nodes).
- [ ] **One harness.** One maintained harness instead of per-run copies with ad-hoc patches:
  - repo-root `sys.path` hacks;
  - `--gen` generators;
  - `spec_layout` for real inputs;
  - poisoning and determinism settings.

  It should also check the target properties a kernel depends on (SM count, cooperative launch).
- [ ] **Coordinator tooling as skill scripts.** The board, leases, promotion (`promote.py`, `board_accept.py`) and
  verifier hand-offs are rebuilt per run.
- [ ] **Supported overrides for generated regions** (G-031 was edited by hand), instead of editing generated
  files.
- [ ] **Keys.**
  - The l2 key for static inputs is the AOTAutograd key, so it covers the deterministic flag,
    `fill_uninitialized_memory` and Inductor config. The packs only hit in deterministic mode, and non-det
    timing needed a key override.
  - Key every specialization by its input description, as symbolic-size inputs already are.
- [ ] **Move the pack runtime into graph_trainer.** Today `lifting.py` is vendored in the pack repo; it has a TODO
  for this. `apply_pack` and `make_pack` belong next to `substitute_kernels`.
- [ ] **Validation built in:**
  - the bitwise gate against a same-session baseline;
  - tokens/s against both deterministic and non-deterministic baselines, back to back on one GPU;
  - a GPU-time profile;
  - a free-GPU picker (the host is shared; see the measurement notes in the reports).
- [ ] **Generated reports.** Today `results.json` and the README tables are built by throwaway scripts.
- [ ] **Policy for multi-GPU hazards.** Ban or flag cooperative launches and SM-count assumptions at verification
  time, not in a README.
- [ ] **Honest baselines by default.** Always report the non-deterministic recipe as well, and attribute the gain
  per kernel. For GPT-OSS, 3.26x vs the deterministic baseline is mostly G-005 escaping torch's deterministic
  `index_add`. Against the non-deterministic recipe the gain is 1.83x in GPU time (976.5 -> 534.7 ms per
  microbatch) and +67% in tokens/s.
- [ ] **Code placement:** experiments-folder rules, PR #5838 / #5839 status, and what goes upstream.
  Candidate: a faster deterministic CUDA `index_add` / `index_put` for few destinations in PyTorch. G-005 shows
  a 16x gap: 14.8 ms deterministic vs 0.9 ms atomics vs 0.17 ms for the kernel.

## 4. Verify the perf win end to end with a real process group on a cluster

Every number so far comes from the fake PG on one GPU. In those runs:

- **Collectives** are emulated by copies (about 35 ms per microbatch for GPT-OSS, 175 ms per step for DSv3), and
  no communication overlaps compute.
- **Only SPMD coordinate 0 runs**, so no other rank's graph executes, and there are no stragglers or cross-rank
  waits.
- **Host overhead** (the all-to-all dispatcher's host syncs, launch latency) is that of one process on a shared
  host.

The glue the kernels remove may be partly hidden behind communication on a real cluster, or more exposed.
Cooperative kernels may behave differently next to real NCCL kernels.

- [ ] **Order: cheapest topology first**, with the published recipe unchanged:
  1. Qwen3 30B-A3B: FSDP 8 x EP 8, one node.
  2. GPT-OSS 120B: FSDP 64 x EP 16, 64 GPUs.
  3. DSv3 671B: FSDP 256 x EP 64, 256 GPUs.

  Smaller variants from item 1 can go first, to shake out launch issues.
- [ ] **Runs per model**, back to back on the same allocation:
  - the deterministic baseline;
  - the non-deterministic recipe;
  - the recipe with the pack (deterministic).

  Use at least 50 steps, and report steady state (exclude warmup and compile).
- [ ] **Report:**
  - tokens/s and MFU, the speedup vs both baselines, and step-time variance across ranks;
  - profiler traces from several ranks (not just rank 0): compute / communication overlap, and where the
    removed glue sat relative to the collectives.
- [ ] **Correctness at scale:**
  - every rank lifts the same regions and hits every kernel (the pack raises on any call-count mismatch);
  - loss and grad_norm are bitwise identical to the real-PG deterministic baseline. First establish that two
    baseline runs are bitwise identical on the real PG, since NCCL reductions must also be reproducible.
- [ ] **Multi-GPU hazards:**
  - watch the five cooperative GPT-OSS kernels for hangs or serialization against overlapping NCCL kernels
    (item 1);
  - check the SM-count assumptions hold on the cluster's parts.
- [ ] **Compare with the fake-PG prediction:**
  - for each pack, the GPU-time saving measured under the fake PG vs the real step-time saving;
  - explain the gap (comm overlap, host overhead, rank imbalance).
- [ ] **Prerequisites:**
  - a cluster allocation with GB300s;
  - the exact pack target environment (torch 3c53c6f, CUDA 13.0, cutlass-dsl 4.8.0), or packs regenerated for
    the cluster's build;
  - a launcher for the real-PG recipe (torchrun or the cluster's scheduler).

## Carried over from the GPT-OSS audit

- [ ] **Decide** whether to update the GPT-OSS report and pack README with the non-deterministic profile, and lead
  with the non-det comparison (commit + push in both repos).
- [ ] **Exclusion experiment** (running when this was written): the pack without the `index_put` / `index_add`
  kernels, in deterministic and non-deterministic mode. Record the result here.
