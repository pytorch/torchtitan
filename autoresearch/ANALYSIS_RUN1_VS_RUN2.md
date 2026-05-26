# Run 1 vs Run 2: Comparative Analysis

## Summary table

| | Run 1 (`ar_base`) | Run 2 (`ar_ideas`) |
|---|---|---|
| **Scaffolding** | Empty IDEAS, no web, no refs | Curated IDEAS, web allowed, no refs |
| **Total experiments** | 59 (incl. 10 stability) | 42 |
| **Keeps** | 5 passes + 1 config change | 6 passes (no config change) |
| **Crashes** | 2 | 1 |
| **Best TPS (deterministic)** | **5,124 (+23.1%)** | **6,040 (+45.2%)** |
| **Best TPS (final)** | 6,442 (+54.8%, non-det) | 6,040 (+45.2%, det) |
| **Best MFU** | 37.7% (non-det) | 35.4% (det) |
| **Memory** | 47.0 GiB (-2 GiB) | 49.2 GiB (+0.2 GiB) |

## Apples-to-apples comparison (deterministic mode)

Run 1 achieved **+23.1%** under `--debug.deterministic` before the user
allowed dropping that flag. Run 2 achieved **+45.2%** while keeping the
flag — nearly **2x the improvement**. The dominant contributor was **CUDA
graphs (+16.4%)** which Run 2 implemented directly, while Run 1 identified
the opportunity (Exp 43) but concluded "requires infra outside passes.py."

## What each run discovered

### Run 1 — hand-rolled FX engineering from graph inspection

1. `remove_identity_views` (+7.2%) — 2006 DTensor identity views, single
   largest per-pass win
2. `remove_identity_ops` (+2.4%) — 421 identity slices + 450
   double-transposes
3. `elide_split_cat_for_reduce_scatter` (+1.6%) — 130 TP cats → zero-copy
   views
4. `bucket_fsdp_collectives` (+10.4%) — AG+RS coalesced with
   producer-chain hoist, biggest cumulative win
5. `cleanup_bundle` (+0.6% det / +1.7% non-det) — 354 ops (detach + CSE)
6. Config: dropped `--debug.deterministic` (+23% on top) — identified
   FillFunctor from profiling but couldn't fix from passes.py

### Run 2 — leveraged upstream + architectural passes

1. `remove_detach_nodes` (+0.4%) — 196 detach nodes, marginal
2. `apply_inductor_pattern_passes` (+9.9%) — single `joint_graph_passes`
   call did what Run 1 needed 3 hand-rolled passes for
3. Upstream bucketing + overlap scheduling (+4.6%) — similar to Run 1's
   Exp 13 but via upstream
4. `install_cuda_graph` (+16.4%) — **the game-changer Run 1 missed**
5. `disable_uninitialized_memory_fill` (+5.5%) — elegant in-graph solution
   to the same FillFunctor problem Run 1 solved via config change
6. Remove bucketing post-CUDA-graphs (+2.9%) — discovered anti-synergy
   between bucketing and CUDA graphs

## Key insights

### 1. CUDA graphs were the single biggest lever, and curated ideas helped find it

Run 1 noted "CUDA graphs exist but need runtime infra" and moved on.
Run 2's IDEAS.md listed "CUDA graphs: if CPU-bound, remove CPU overhead"
as a direction — this nudge was enough for the agent to implement it. The
+16.4% from CUDA graphs alone exceeds Run 1's entire deterministic-mode
improvement.

### 2. Upstream pattern matchers replaced multiple hand-rolled passes

Run 2's single `joint_graph_passes(gm)` call (+9.9%, -21% nodes) subsumes
Run 1's `remove_identity_views` (+7.2%), `remove_identity_ops` (+2.4%),
and most of the cleanup work. Run 1 spent ~8 experiments engineering these
peepholes; Run 2 got equal or better results from one upstream import.

### 3. Run 1 was more creative at the FX level

Run 1's `elide_split_cat_for_reduce_scatter` (zero-copy RS view rewrite)
and `bucket_fsdp_collectives` (hand-rolled coalesced collective with
producer-chain hoist) are genuinely novel FX transformations. Run 2 never
discovered these — it used upstream `bucket_reduce_scatter` instead, which
was later found counterproductive under CUDA graphs.

### 4. Run 2 found a deeper solution to the determinism overhead

Both runs identified FillFunctor as ~7% overhead from
`--debug.deterministic`. Run 1 requested a config change (drop the flag).
Run 2 audited the graph, proved all empty buffers are fully overwritten,
and selectively disabled just the fill — a passes.py-only solution that
preserves the deterministic guarantee for everything else.

### 5. Run 2 discovered important structural ceilings

The insight that FX-level reordering is cosmetic after
`schedule_overlap_bucketing` + CUDA graphs (iter-29), and that regional
Inductor compile is anti-synergistic with CUDA graphs (iter-32), are novel
findings not present in Run 1.

## Answer to the experiment plan question

> "Do human-curated ideas help the agent find better fusions?"

**Yes, dramatically.** Under identical deterministic constraints, curated
ideas nearly doubled the improvement (45.2% vs 23.1%). The main mechanism
wasn't that ideas told the agent *what* to do — the agent still had to
implement everything — but that ideas pointed toward **architectural
levers** (CUDA graphs, upstream pattern matchers) that the pure-autonomy
agent dismissed or didn't consider. The pure-autonomy agent excelled at
FX-level graph surgery but hit a lower ceiling because it focused
exclusively on node-count reduction rather than runtime-architecture
changes.
