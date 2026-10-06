# Aurora Multi-Node GRPO: Results and Fixes

Qwen3-0.6B GRPO on Aurora (Intel Data Center GPU Max 1550, 6 GPUs = 12 tiles per
node; every "GPU" below is one tile). Trainer and generator run on **disjoint**
tiles, so a per-GPU number is always divided by the tile count of its own role,
never by the job total.

Two documents, one file: **Part 1** is the measured results, **Part 2** is every
fix that got there, one by one.

Covers the exp1-exp9 parallelism sweep (20-step runs), the 200-step LoRA and
full-parameter runs of 2026-08-05, and the exp11/exp12 1-node LoRA and
full-parameter pair of 2026-09-14. Headline: **LoRA completes 200 steps and
learns; the full-parameter arm still hangs in the weight pull**, now at step 120
instead of step 3 -- see [The 200-step runs](#the-200-step-runs-8-nodes-2026-08-05)
and the verdict under Fix 11. Second headline, new with exp12: at one node the two
arms are a **controlled twin**, and full-parameter is not the expensive one -- see
[LoRA vs full parameter at one node](#lora-vs-full-parameter-at-one-node-exp11-vs-exp12).

**Update 2026-10-05, torch 2.14 stack:** both arms now complete 200/200 steps at 1
node and at 8 nodes, full-parameter included, once `CCL_OP_SYNC` is unset (Fix 12:
on oneCCL 2022 it silently turns cross-node all_reduce into a no-op) -- see
[The 200-step runs on torch 2.14](#the-200-step-runs-on-torch-214-1-and-8-nodes-2026-10-05).
The pull-hang headline above describes the torch 2.12 stack.

Source logs are archived outside the repo at `~/aurora_rl_logs/` (see
[Log archive](#log-archive)). Numbers were extracted mechanically by
`torchtitan/experiments/rl/extract_exp_metrics.py`, not copied by hand.

**exp11 and exp12 (1 node, added 2026-09-14)** are later additions, not part of
the 07-31/08-01 sweep: exp11 is LoRA, exp12 is its full-parameter twin, both
10 steps on 2 trainer + 2 generator tiles. Their rows are marked in every table
below. Both are now entries in `extract_exp_metrics.py`, so their numbers come
from the same parser and the same aggregation conventions as exp1-exp9 -- point
its `log_dir` at `torchtitan/experiments/rl/` rather than at the archive.

**Correction to the exp11 numbers first published on 2026-09-14.** They came from
a separate parser, and four of them move under the canonical one: `s/step`
**16.3 -> 19.5** (the old figure was reconstructed as `tokens_per_step /
full_step` instead of read off the log's wall clock, so it was never on the same
footing as the 20.2-21.8 band it was being compared against), `queue`
50.5 -> 46.0 ms, aggregate generation 4668 -> 4698 tok/s, and `inflight`
239 -> 240. Nothing else moved and no conclusion changes.

---

# Part 1: Results

## Common configuration

Identical across exp1-exp8 (`rl_grpo_lora_qwen3_0_6b`); exp9 is the
full-parameter twin (`rl_grpo_full_qwen3_0_6b_flex`) and differs **only** in
dropping the LoRA converter. exp11 runs the same LoRA config at one node and
differs **only** in `gpu_memory_limit=0.90`, which `run_grpo_lora_sn.sh:92`
hardcodes.

exp12 runs `rl_grpo_full_qwen3_0_6b_flex` at one node with no config overrides at
all. Against exp11 it therefore differs in four things, and only the first is the
intended variable:

1. no LoRA converter -- all 751.6M parameters are trained instead of rank-32
   adapters;
2. `gpu_memory_limit` 0.85 (the config default) vs exp11's 0.90 CLI override;
3. `checkpoint.load_only=True`, so exp12 loads the pretrained weights and never
   writes a checkpoint, while exp11 force-saves at step 10 (~2.4 s);
4. exp12 runs under the Fix 9 RMSNorm `LD_PRELOAD` interposer, which
   `run_grpo_sn.sh` requires and verifies with a precheck before launching --
   full-parameter training asks for the qk_norm weight gradient and the stock XPU
   kernel dies on it.

| knob | value |
|---|---|
| model | Qwen3-0.6B, bfloat16, flex attention (`max_autotune=False` on XPU) |
| adapters | LoRA rank 32, alpha 64, targets `wqkv`, `wo` (exp1-8, exp11); none (exp9, exp12) |
| global batch | `num_groups_per_train_step=8` x `group_size=8` = **64 sequences/step** |
| local batch | `local_batch_size=2`, `seq_len=2048` |
| optimizer | AdamW, lr 2e-6, 5 warmup steps, linear decay |
| loss | GRPO, `max_offpolicy_steps=3` (async, 3-step-lagged rollout buffer) |
| sampling | temperature 1.0, top_p 0.95, `max_tokens=512` |
| generator | vLLM, `gpu_memory_limit=0.85` (**0.90 in exp11**, 0.85 in exp12), cudagraph off, `dp=<gen tiles>`, `tp=1` |
| compile | `aot_eager` |
| steps requested | 10 |
| validation | 20 samples |

Node split is done by `multinode_launcher.py`: half the nodes trainer, half
generator. So a 2-node job is 4 trainer tiles + 4 generator tiles, and a 4-node
job is 8 + 8. At one node the split is within the node: 2 + 2 (exp9', exp11).
The trainer mesh axes are what the experiments vary.

## Configurations and outcomes

| exp | nodes | trainer mesh (2, 4 or 8 tiles) | gen mesh | steps | job | outcome |
|---|---|---|---|---|---|---|
| exp1 | 2 | `dp_replicate=2 x dp_shard=2` | `dp=4` | 10/10 | 8723859 | **pre-`--cpu-bind none`, NOT comparable** |
| exp2 | 2 | `dp_replicate=2 x tp=2` | `dp=4` | 10/10 | 8724090 | **pre-`--cpu-bind none`, NOT comparable** |
| exp3 | 2 | `dp_shard=4` | `dp=4` | 10/10 | 8724272 | healthy, **best 2-node** |
| exp4 | 2 | `tp=4` | `dp=4` | 10/10 | 8724356 | healthy |
| exp5 | 4 | `dp_shard=8` | `dp=8` | 10/10 | 8724494 | healthy, **best 4-node** |
| exp6 | 4 | `tp=8` (cross-node TP) | `dp=8` | **4/10** | 8724584 | stalled after step 4, see below |
| exp7 | 4 | `dp_replicate=2 x dp_shard=4` | `dp=8` | 10/10 | 8724424 | healthy, ties exp5 |
| exp8 | 4 | `dp_replicate=2 x tp=4` | `dp=8` | 10/10 | 8724540 | healthy (was the Bug B carrier) |
| exp9 | 2 | `dp_shard=4`, full-parameter | `dp=4` | **0/10** | 8724664 | crashed in the XPU RMSNorm backward |
| exp9' | 1 | `dp_shard=2`, full-parameter | `dp=2` | 2/2 | 8731367 | **clean with the RMSNorm fix preloaded** |
| exp10 | 8 | `dp_replicate=4 x dp_shard=4`, full-parameter | `dp=16` | **3/200** | 8731512 | 3 healthy steps, then hung in the weight pull -- see Fix 11 |
| exp10' | 8 | same, 150 steps, with Fix 11 | `dp=16` | pending | 8734008 | submitted on `capacity` 2026-08-04 |
| exp11 | 1 | `dp_shard=2` | `dp=2` | 10/10 | interactive | healthy, **1-node LoRA baseline** (2026-09-14) |
| exp12 | 1 | `dp_shard=2`, full-parameter | `dp=2` | 10/10 | interactive | healthy, **1-node full-parameter twin of exp11** (2026-09-14) |

exp11 is the 1-node LoRA baseline the sweep never had: exp1-exp8 start at 2 nodes
and exp9'/exp10 are full-parameter, so nothing here measured LoRA at 2+2 tiles. It
ran interactively via `run_grpo_lora_sn.sh` on a held allocation, so it has no PBS
job ID; its log is `torchtitan/experiments/rl/train_lora_1n.log` (23:01:36 to
23:08:28, 6m52s). It is clean: 10/10 steps, both validations, zero `RuntimeError`,
zero `ValueError`, zero `died from signal`, and the controller reached `Closing:
tearing down actors`. The 148 tracebacks in its log are all the inductor
FX-graph-cache fault (74 failed cache loads plus 74 `CompiledFxGraph.__del__`
`AttributeError`s, a 1:1 pairing), which is warning-level and non-fatal -- it costs
a full recompile of step 1, nothing else.

exp12 is the full-parameter twin of exp11 and the **first 10-step full-parameter
run on this stack** -- exp9 died at 0 steps and exp9' was a 2-step kernel-fix
validation. It ran interactively via `run_grpo_sn.sh` on the same held allocation
half an hour earlier, so it likewise has no PBS job ID; its log is
`torchtitan/experiments/rl/train_full_1n.log` (22:29:16 to 22:35:30, 6m14s). Same
clean bill as exp11: 10/10 steps, both validations, zero `RuntimeError`, zero
`ValueError`, zero `died from signal`, `Closing: tearing down actors` reached, and
its 168 tracebacks are the same FX-graph-cache fault in the same 1:1 pairing
(84 + 84). It also confirms Fix 9 at 10 steps rather than 2: the interposer
precheck passed and no `_fused_rms_norm_backward` failure appears anywhere in the
log.

exp1 and exp2 ran at 21:03 and 22:11 on 07-31; `--cpu-bind none` landed in the
launch scripts at 22:44 that night. Their own `.pbs` bodies confirm it: exp1/exp2
contain zero `--cpu-bind none` occurrences, exp3-exp9 contain it. **Do not read
exp1/exp2 against exp3-exp8 as a parallelism comparison** -- the difference is
Bug C (all ranks pinned to one core), not the mesh. Their ITL, 259 and 268 ms
against 50-54 ms everywhere else, is the tell.

exp9' is a 1-node validation run of the RMSNorm fix, not a sweep entry. It was
configured for 2 steps, so its throughput comes from a single steady-state step.

## Trainer throughput

`perf/trainer/tokens_per_second_*` is **global** (the opposite convention from
core torchtitan's per-device `throughput(tps)`). Per-GPU below is global divided
by trainer tiles.

Step 1 pays torch.compile and the first weight push and is 1-2 orders of
magnitude slower, so every figure is over steps 2..N.

- **fwd_bwd** = compute only. Arithmetic mean; this is the number that reflects
  the parallelism choice.
- **full_step** = wall-clock end to end, including waiting on the generator.
  It is **bimodal**, not noisy -- a step either has its batch ready (~13k) or
  blocks on generation (~700) -- so it is aggregated as a harmonic mean
  (equivalent to total tokens / total time), never a median.

| exp | fwd_bwd global | fwd_bwd per-GPU | full_step global | full_step per-GPU | s/step | fwd_bwd range |
|---|---|---|---|---|---|---|
| exp1 (pinned) | 6624 | 1656 | 683 | 171 | 106.9 | 5639-8206 |
| exp2 (pinned) | 1767 | 442 | 590 | 147 | 127.8 | 1590-1904 |
| exp3 | **13837** | **3459** | **2645** | **661** | 20.2 | 12038-15774 |
| exp4 | 5349 | 1337 | 2242 | 561 | 21.5 | 5018-5580 |
| exp5 | 21137 | 2642 | 2629 | 329 | 21.6 | 15349-26446 |
| exp6 | 3903 | 488 | 3901 | 488 | 15.0 | 3812-4013 |
| exp7 | **21734** | 2717 | 2587 | 323 | 21.6 | 15335-26536 |
| exp8 | 8412 | 1051 | 2388 | 299 | 21.8 | 8031-9003 |
| exp9' | 8099 | 4049 | 8097 | 4049 | n/a | single step |
| exp11 | 8928 | **4464** | 2553 | **1276** | 19.5 | 8419-10219 |
| exp12 (full) | 8364 | 4182 | 2784 | **1392** | 17.8 | 7782-9338 |

Units: tokens/s; `s/step` is wall clock over steps 2..N. exp6's full_step equals
its fwd_bwd, and its s/step is short, only because it never reached a
generator-blocked step before stalling.

**exp11 has the highest per-GPU fwd_bwd of any LoRA run, 4464, and the tightest
spread, 8419-10219 (+/-10% over nine steps).** Per-GPU fwd_bwd now falls
monotonically with scale across the whole LoRA range: 4464 (2 tiles) -> 3459 (4,
exp3) -> 2642 (8, exp5). Two trainer tiles is the most efficient point measured,
and the 76% marginal efficiency quoted below for 2->4 nodes is the tail of a curve
that was already declining.

**Both 1-node runs sit just below the 20.2-21.8 s band -- 19.5 s (exp11) and 17.8 s
(exp12) -- and that gap should not be read as a faster pipeline.** tokens/step is
*not* constant across runs; it is whatever the rollouts happened to be. exp12's own
two columns imply 2784 x 17.8 = **49,600** tokens/step against exp5's
2629 x 21.6 = **56,786**. `full_step` is the reproducible quantity, and `s/step`
inherits the rollout token count. Read the band as "no mesh in the sweep changed the
step time", not as a constant of the workload.

The two 1-node arms are not perfectly matched on this column either: exp11
force-saves a checkpoint at step 10 (`Saving a full checkpoint at last step,
step 10` at 23:07:19,632, ~2.4 s) inside its measured span, while exp12 never saves
at all (`load_only=True`). That is worth about 1% of exp11's `s/step`, and it is one
of the four differences listed under [Common configuration](#common-configuration).

Step budget, measured between consecutive trainer weight pushes -- the only
per-step timestamps these logs carry:

| | exp11 (LoRA) | exp12 (full) |
|---|---|---|
| push-to-push training span | 313.3 s | 278.8 s |
| step 1, all `torch.compile` | 152.2 s, **49%** of the span | 134.5 s, **48%** of the span |
| remaining nine gaps, total / mean | 161.1 s / 17.9 s | 144.3 s / 16.0 s |
| gaps with the batch ready | 7, at 3.1-7.2 s | 6, at 2.7-4.8 s |
| gaps blocked on generation | 2, at 60.8 and 65.2 s | 3, at 12.9, 47.7 and 60.3 s |

**Half of each 10-step run is step 1's compile**, and the inductor cache cannot be
reused (see the FX-graph-cache fault above), so that cost is paid on every launch.
The metric lines and the push timestamps can be off by one step relative to each
other (separate actors, independently buffered), so these gaps bound the per-step
time but must not be joined to a specific step's metrics. `step_time_ratio/batch`,
which *is* per step, tells the same story from the other side and tells it
identically for the two arms: **each blocks on generation in three of its nine
steady steps** -- exp11 at 0.92 / 0.91 / 0.39 (steps 5, 8, 7) and exp12 at
0.89 / 0.89 / 0.59 (steps 5, 9, 8), with everything else below 0.001.

**The `s/step` column is the punchline: 17.8-21.8 s in every unpinned 10-step run,
regardless of mesh, tile count, or LoRA-vs-full.** exp3 computes 2.6x faster than
exp4 and finishes a step 1.3 s sooner. exp5 computes 5.4x faster than exp6 (over its
4 steps) at the same tile count. Trainer parallelism is not what sets the step time
on this workload.

What this says:

- **FSDP beats TP at this model size, by 2.6x.** exp3 (`dp_shard=4`) 13837 vs
  exp4 (`tp=4`) 5349 on the same 4 tiles; exp5 (`dp_shard=8`) 21137 vs exp6
  (`tp=8`) 3903 on the same 8. A 0.6B model does not have enough work per layer
  to amortize TP's per-layer collectives.
- **`dp_replicate` costs nothing and is not the fold's problem.** exp7
  (`rep2 x dp_shard4`, 21734) vs exp5 (`dp_shard8`, 21137) is +2.8%, inside
  run-to-run noise. This retired the claim that unfolding was worth +19%; see
  Fix 6.
- **2 nodes -> 4 nodes scales 1.53x on fwd_bwd** (exp3 13837 -> exp5 21137) at
  2x the tiles, i.e. 76% efficiency, per-GPU 3459 -> 2642.
- **full_step is generator-bound, not trainer-bound.** exp5's trainer computes
  8x faster than it steps (21137 vs 2629): the pipeline spends its time waiting
  on rollouts, and adding trainer tiles cannot help. Per-GPU full_step *falls*
  from 661 (2 nodes) to 329 (4 nodes) for exactly this reason -- the generator
  side did not get faster, so the extra trainer tiles idle. Any further trainer
  scaling work on this workload is premature until generation throughput moves.

## Generation throughput

`decode_time_ms` and `inter_token_latency_ms` are **per request** -- vLLM reports
one sampled sequence, dropping the other n-1 group siblings. So ITL gives
per-sequence decode speed, and aggregate throughput has to be reconstructed from
the concurrency the engine actually held (`inflight_requests_at_completion`).

| exp | ITL (ms) | per-seq tok/s | inflight | aggregate global tok/s | per gen-GPU tok/s | decode (s) | queue (ms) |
|---|---|---|---|---|---|---|---|
| exp1 (pinned) | 259.3 | 3.9 | 245 | 999 | 250 | 112.0 | 40.1 |
| exp2 (pinned) | 267.8 | 3.7 | 237 | 927 | 232 | 115.0 | 7.2 |
| exp3 | 51.8 | 19.3 | 237 | 4575 | **1144** | 22.2 | 4.9 |
| exp4 | 52.2 | 19.2 | 235 | 4512 | 1128 | 22.6 | 2.0 |
| exp5 | 53.1 | 18.8 | 237 | 4469 | 559 | 22.7 | 9.7 |
| exp6 | 50.4 | 19.8 | 240 | 4763 | 595 | 21.2 | 1.9 |
| exp7 | 52.7 | 19.0 | 235 | 4467 | 558 | 23.0 | 9.2 |
| exp8 | 53.8 | 18.6 | 236 | 4399 | 550 | 23.0 | 3.4 |
| exp9' | 44.2 | 22.6 | 254 | 5763 | **2881** | 18.4 | 42.6 |
| exp11 | 51.2 | 19.5 | 240 | **4698** | **2349** | 22.0 | 46.0 |
| exp12 (full) | 45.6 | 21.9 | 249 | **5475** | **2738** | 19.9 | 50.1 |

**Generation does not scale with generator tiles at all.** 4 gen tiles deliver
4575 tok/s; 8 gen tiles deliver 4469. Doubling the generator halved per-GPU
throughput (1144 -> 559) for zero aggregate gain. The reason is visible in the
same table: `inflight` is pinned at 235-240 in every run regardless of tile
count, so the *controller* is supplying a fixed amount of concurrent work
(64 sequences/step x the 3-step off-policy buffer + validation), and 4 tiles
already absorb it. The generator half of a 4-node job is idle capacity.

**exp11 extends this downward and makes it stronger: 2 gen tiles deliver 4698
tok/s, more than the 4-tile 4575 and the 8-tile 4469.** Aggregate throughput is
*highest at the smallest generator measured*, and per gen-GPU collapses
monotonically 2349 (2 tiles) -> 1144 (4) -> 559 (8). exp11's `inflight` of 240 is
inside the same 235-254 band as every other run. So the correct statement is not
"4 tiles already absorb the work" but **2 tiles already absorb it, and every tile
past that is pure idle capacity.** Its ITL of 51.2 ms is squarely in the healthy
50-56 ms band.

**Both 1-node full-parameter runs decode measurably faster than any LoRA run:
exp12 at 45.6 ms ITL and exp9' at 44.2 ms, against 50-56 ms everywhere else.** The
effect reproduces across two runs a month apart, so it is not noise, but the
mechanism is **not established** -- the generator is serving the identical merged
bf16 weights in both arms, so nothing about the model it runs differs. The live
confound is `gpu_memory_limit`: the full-parameter arm runs at 0.85 and exp11 at
0.90, i.e. the *smaller* KV cache decodes faster, which is the opposite of the
intuition and needs a deliberate A/B (one LoRA run at 0.85) before anything is
concluded. Do not attribute it to LoRA.

The `queue` column is not a steady-state figure in either 1-node run: exp11 is
78.7 -> 153.6 -> 191.7 ms at steps 2-4 and then ~1.5 ms from step 6 on, and exp12
is 95.9 -> 167.9 -> 173.5 ms at the same steps before falling to ~1.5 ms. That is
the pre-filled off-policy buffer draining faster than it refills; once drained the
trainer waits instead of the requests waiting. The same shape appears in exp9'
(42.6 ms mean).

This is the single largest finding in the sweep and it is a **workload** limit,
not a hardware or fabric one: raise `num_groups_per_train_step`,
`max_offpolicy_steps`, or `group_size` before adding generator nodes.

ITL is also the fastest way to classify any log on this stack: **50-56 ms means
ranks are unpinned, 214-268 ms means Bug C is live.**

## Loss, reward, and health

The task is alphabet-sort with a 0-1 rubric reward. 10 steps at lr 2e-6 is a
plumbing check, not a convergence run -- read these as "healthy and identical
across meshes", not as learning curves.

| exp | reward first -> last (mean) | loss mean | grad_norm range | entropy | `logprob_diff/max` |
|---|---|---|---|---|---|
| exp1 | 0.31 -> 0.21 (0.230) | -0.0078 | 0.058-0.150 | 0.58 -> 0.48 | 0.77 |
| exp2 | 0.34 -> 0.24 (0.248) | -0.0079 | 0.063-0.130 | 0.56 -> 0.45 | 0.72 |
| exp3 | 0.29 -> 0.29 (0.236) | -0.0082 | 0.064-0.130 | 0.57 -> 0.45 | 1.32 |
| exp4 | 0.33 -> 0.18 (0.231) | -0.0081 | 0.065-0.120 | 0.57 -> 0.45 | 1.39 |
| exp5 | 0.34 -> 0.26 (0.248) | -0.0064 | 0.063-0.140 | 0.55 -> 0.45 | 1.34 |
| exp6 | 0.35 -> 0.32 (0.305) | -0.0028 | 0.089-0.120 | 0.57 -> 0.55 | 0.90 |
| exp7 | 0.38 -> 0.22 (0.228) | -0.0075 | 0.056-0.140 | 0.55 -> 0.46 | 0.87 |
| exp8 | 0.27 -> 0.20 (0.244) | -0.0092 | 0.056-0.120 | 0.56 -> 0.49 | 1.07 |
| exp9' | 0.34 -> 0.26 (0.300) | -0.0080 | **0.290-0.430** | 0.56 -> 0.53 | 1.11 |
| exp11 | 0.38 -> 0.19 (0.246) | -0.0057 | 0.065-0.110 | 0.56 -> 0.46 | 0.80 |
| exp12 (full) | 0.33 -> 0.20 (0.236) | -0.0083 | **0.200-0.510** | 0.58 -> 0.44 | 6.18 |

**Every LoRA run sits in grad_norm 0.056-0.150.** That band is the acceptance
test for this stack: exp8 was at **88-2464** before Fix 4 and is at 0.056-0.120
after, with no code change. exp11 at 0.065-0.110 holds the band at 2 tiles, so it
is not a property of the 4- and 8-tile meshes. Reward drifting down over 10 steps at lr 2e-6 with
entropy also falling is sampling noise on a 64-sequence batch, and it happens
identically in all eight -- it is not a mesh-dependent signal.

**The two full-parameter runs form their own band, 0.20-0.51**, and are the only
legitimate outliers here: exp9' at 0.29-0.43 and exp12 at 0.20-0.51, 3-8x the LoRA
band. Expected, not a bug -- they train all 751.6M parameters instead of rank-32
adapters, so the gradient norm is taken over a far larger parameter vector. Loss,
reward, and entropy stay in family with the LoRA runs, and exp12's ten steps
(against exp9''s two) are what make this a band rather than a single reading.

exp12's `logprob_diff/max` of **6.18 is a one-step spike, not a trend**: the metric
is a max over tokens, it hits 6.18 at step 6 alone, and the other nine steps are
0.39-0.83 -- tighter than exp3-exp8's 0.87-1.39. exp11 peaks at the same step 6
(0.80), which is the first step after the pre-filled off-policy buffer drains, so
the spike sits where the trainer and generator weights are furthest apart. Worth
watching over a longer run; not a reason to distrust these ten steps.

## Validation reward and generator memory (exp11 and exp12)

No table above has a column for either, because the sweep never recorded them.
The two 1-node runs do, and both are worth carrying forward.

**Validation reward, 20 samples, pre / post 10 steps: 0.251 -> 0.512 (exp11,
LoRA) and 0.233 -> 0.508 (exp12, full)** (`_std` 0.255 -> 0.387 and
0.255 -> 0.341; `_sum` 5.028 -> 10.250 and 4.659 -> 10.151). These are the first
positive pre/post deltas recorded on this stack. Neither is evidence of convergence
at 10 steps -- they are evidence that the validation path works and that the reward
signal moves in the right direction, which the `rollout_reward` column (0.38 -> 0.19
and 0.33 -> 0.20, both drifting *down* on sampling noise) cannot show. **The two
arms land within 0.004 of each other after ten steps**, from near-identical starting
points; at this step count that is agreement, not a ranking.

Generator memory, one FLAT tile = 63.98 GiB:

| | exp11 (LoRA 1n) | exp12 (full 1n) |
|---|---|---|
| `gpu_memory_limit` | 0.90 (`run_grpo_lora_sn.sh:92`) | 0.85 (config default) |
| vLLM budget | 57.58 GiB | 54.38 GiB |
| model weights | 2.22 GiB | 2.20 GiB |
| KV cache | 53.37 GiB / 499,456 tokens | 50.18 GiB / 469,760 tokens |
| non-KV overhead (budget - weights - KV) | 1.99 GiB | 2.00 GiB |
| tile headroom left | 6.40 GiB | 9.60 GiB |
| max concurrency @ 40,960 tok/req | 12.19x | 11.47x |
| peak / p90 / mean KV usage | 11.8% / 9.0% / 3.4% | 13.7% / 8.6% / 3.7% |

The generator side is identical in the two arms except for the memory limit, which
is what the table shows: **2.20 vs 2.22 GiB of weights** (the generator holds the
merged bf16 model either way -- LoRA changes nothing it loads) and **1.99 vs
2.00 GiB of non-KV overhead**.

**Non-KV overhead is ~2.0 GiB at 2 visible tiles, independent of the memory limit
and of LoRA-vs-full**: exp11 at 0.90 gives 1.99 GiB and exp12 at 0.85 gives
2.00 GiB. At 4 visible tiles it is 2.46 GiB, i.e. **~0.23 GiB per additional
visible tile.** That is an order of magnitude below the ~1.3 GiB/tile that a
per-visible-tile Level Zero context would need to account for the idle-tile
occupancy seen elsewhere, so that explanation is not supported.

**The KV cache is heavily oversubscribed and `gpu_memory_limit=0.90` buys nothing.**
vLLM's own `loggers.py:273` line reports a peak `GPU KV cache usage` of **11.8%**
across the run (58,935 of 499,456 tokens; p90 9.0%, mean 3.4%) even at the moment
`Running` hits the full `max_num_seqs=128`, while leaving only 6.40 GiB of tile
headroom -- the tightest of any run measured, for a cache it uses an eighth of.

The 12.19x concurrency figure above is *not* the binding limit and should not be
read as one: vLLM computes it against `max_seq_len=40960`, which comes from
Qwen3-0.6B's `max_position_embeddings`, not from anything this workload sets. Real
requests are at most `seq_len=2048` prompt + `max_tokens=512` completion, so the
same cache holds hundreds of them. A 4-node LoRA run at 0.80 reproduced exp5's throughput
to within 3%, so lowering `run_grpo_lora_sn.sh:92` to 0.80 would triple the headroom
at no measurable cost and remove the vLLM `request_memory()` startup-guard failure
as a class of job death.

## LoRA vs full parameter at one node (exp11 vs exp12)

This is the controlled twin the sweep was missing. Both runs are 10 steps on the
same held allocation, same node, same 2 trainer + 2 generator tiles, same
`dp_shard=2`, same 64-sequence batch, same lr, same sampling; they differ in the
LoRA converter and in the three incidentals listed under
[Common configuration](#common-configuration). It is one pair of runs at one scale,
not a sweep -- but every number in it comes from the same parser.

| | exp11 (LoRA) | exp12 (full) | full vs LoRA |
|---|---|---|---|
| trained parameters | rank-32 adapters on `wqkv`, `wo` | all 751.6M | -- |
| fwd_bwd, per trainer tile | 4464 tok/s | 4182 tok/s | **-6%** |
| full_step, per trainer tile | 1276 tok/s | 1392 tok/s | **+9%** |
| s/step (steps 2..N) | 19.5 s | 17.8 s | **-9%** |
| weight payload per push | 1.52 GB | 1.50 GB | **-1%** |
| grad_norm | 0.065-0.110 | 0.200-0.510 | 3-5x |
| validation reward, pre -> post | 0.251 -> 0.512 | 0.233 -> 0.508 | -0.004 |
| ITL | 51.2 ms | 45.6 ms | -11% |
| steps completed | 10/10 | 10/10 | -- |

**Full-parameter costs 6% of trainer compute and nothing end to end.** The only
column where LoRA wins is `fwd_bwd`, the compute-only one, and it wins by 6%.
On wall clock the full-parameter arm is *faster* (17.8 vs 19.5 s/step), for the
same reason the 8-node 200-step pair was (21.3 vs 25.4 s): the pipeline is
generator-bound, so a 6% trainer difference disappears into the rollout wait. Two
independent scales now agree on the direction.

**The weight payload is the same size in both arms, measured here directly:
1.50 GB (full) vs 1.52 GB (LoRA)**, from `put_state_dict`'s own
`took 1.9888s (0.76 GB/s)` and `took 1.8862s (0.80 GB/s)`. Fix 11 argued this from
the 8-node logs; this pair confirms it in a controlled setting. `put_state_dict`
pushes the whole merged state dict either way, so **"LoRA syncs less" is false**,
and the transfer cost that dominates multi-node runs is identical for the two arms.
At one node it is also cheap and not on the XCCL path at all -- Fix 3's
`unset HOSTNAME` makes trainer and volume co-resident, so the push takes
SharedMemory at 15-19 GB/s and the generator pull reads it back in ~0.19 s
(6.2-8.5 GB/s).

**What LoRA buys on this stack is therefore optimizer state and robustness, not
speed.** The one hard operational difference in these ten steps is that
full-parameter needs the Fix 9 RMSNorm patch to run at all, while LoRA never asks
for that kernel path (`LoRAConverter` freezes the norms). The remaining caveats:
10 steps is a plumbing check, not a convergence comparison; the arms differ in
`gpu_memory_limit` (0.85 vs 0.90), which is also the confound in exp12's faster
decode; and at 8 nodes the full-parameter arm still hangs in the weight pull
(Fix 11) while LoRA completes 200 steps, which remains the real reason to prefer
LoRA for long runs.

## The three runs that did not finish

**exp6 (`tp=8` on 4 nodes) stalled after step 4.** Not a crash, not a fabric
fault: the oneCCL SEND fault Fix 1 cured is gone (0 occurrences), the 4 steps it
did produce are healthy (grad_norm 0.089-0.120, ITL 50.4 ms), and `py-spy` showed
the trainer idle in `asyncio select` rather than blocked in a collective. Cross-node
`tp=8` is also by far the slowest trainer mesh measured (3903 tok/s, 488/GPU), so
this configuration is not worth pursuing for performance regardless. **Root cause
of the stall is still open.**

**exp10 (full-parameter, 8 nodes) reached 3 of 200 steps.** The three steps are
healthy (grad_norm 0.38 / 0.30 / 0.44, `tokens_per_second_full_step` 99 / 10516 /
8434), which closes the Fix 9 kernel bug end to end at scale. It then hung in the
step-4 weight pull, root-caused to the torchstore replica redundancy of Fix 11.
Its step times are **not** usable as throughput: `max_offpolicy_steps=3` had
pre-filled the rollout buffer, so those steps were draining a queue rather than
running in steady state. **Superseded by job 8734220** (below), which carried
Fix 11 and reached step 120.

**exp9 (full-parameter, 2 nodes) reached 0 steps**, dying in
`aten::_fused_rms_norm_backward` with `RuntimeError: tensor does not have a
device`. Root-caused to an XPU kernel bug, fixed, and verified -- see Fix 9. With
the fix preloaded, exp9' ran clean at the **original** `local_batch_size=2`, so
the crash is closed, and exp12 has since run the same configuration for 10 steps at
one node. The 10-step **2-node** exp9 has still not been re-submitted; exp11 vs
exp12 now answers the LoRA-vs-full-parameter cost question at 2 tiles instead (see
[LoRA vs full parameter at one node](#lora-vs-full-parameter-at-one-node-exp11-vs-exp12)),
so what exp9 would add is that comparison at 4 tiles and across a node boundary.

## The 200-step runs, 8 nodes (2026-08-05)

The first runs long enough to show learning rather than liveness. Both carried
Fix 11 and `--trainer.checkpoint.load-only`, so no checkpoint was written (the
step-50 DCP save had been OOMing inside oneCCL; `load_only` still loads the
pretrained weights, whereas `enable=False` silently trains from random init).

| job | arm | steps | s/step | reward, first 10 -> last 10 | outcome |
|---|---|---|---|---|---|
| 8734219 | LoRA | **200/200** | 25.4 | 0.222 -> 0.355 (max 0.620) | complete |
| 8734220 | full | 120/200 | 21.3 | 0.216 -> 0.346 (max 0.550) | pull hang |

**LoRA completed all 200 steps** -- the first full-length GRPO run on this stack.
25.4 s/step matches the 23.3 s/step measured over the short sweep, so nothing
degrades over 200 steps. Reward rises monotonically under smoothing and loss
drifts from -0.009 toward 0, both as expected for GRPO.

**The two arms learn at nearly the same rate**, and full-parameter is *slightly
faster per step* (21.3 vs 25.4 s) -- LoRA's advantage on this stack is not
throughput. That is consistent with Fix 11's finding that `put_state_dict`
pushes the whole merged state dict either way, so "LoRA syncs less" is false.
These two jobs are 8 nodes and the full arm stops at 120 steps, so the pair is
suggestive rather than clean; the clean twin is now
[exp11 vs exp12](#lora-vs-full-parameter-at-one-node-exp11-vs-exp12) at one node,
which reproduces the same direction (full-parameter 9% faster per step, identical
1.5 GB payload) over 10 steps with nothing else varying.

**Both jobs ended at PBS `Exit_status -29` (walltime kill) for different
reasons**, and the exit code alone is misleading:

- LoRA *finished training* at 03:05:49, printed `Closing: tearing down actors`,
  then hung in teardown for 2.5 h until the 4 h wall. All data was already in
  the log; the hang is cosmetic. This is why the operational rule is to `qdel`
  the moment training finishes.
- Full-parameter hung for real at step 120 in the weight pull -- see the verdict
  added to Fix 11.

Artifacts: `~/aurora_rl_logs/artifacts/8734219_lora_8n_200_COMPLETE/` and
`8734220_full_8n_120of200_pullhang/`, each with the log and a loss/reward plot.

## The 200-step runs on torch 2.14, 1 and 8 nodes (2026-10-05)

Both arms at 1 node and at 8 nodes, 200 steps each, on the **torch 2.14.0+xpu /
oneCCL 2022.1.1** stack: oneAPI/SYCL 2026.1, vLLM upstream `main` (083060d04),
torchtitan `c2bfcaaa0` (the rebased `torchtitan/rl` tree), no RMSNorm interposer
(the stock torch 2.14 kernel passes the 65536-row weight-grad backward). **Every run
completed 200/200 steps** -- including the full-parameter arm at 8 nodes, which on
the torch 2.12 stack stopped at 120.

The weight pull carries two workarounds for oneCCL 2022 regressions, both on in
the XPU recipe: `storage_volumes_in_own_procs` (a storage volume hosted by an FSDP
trainer process broadcasts no data from tiles 1-3, because oneCCL 2022 fixes a
process's on-node position at its first communicator) and
`sequential_initial_pull` (concurrent communicator creation in one process fails
in oneCCL's topology discovery). The launchers add `SYCL_UR_USE_LEVEL_ZERO_V2=1`
against an intermittent `ur_die` abort of the v1 Level Zero adapter, and unset
`CCL_WORKER_AFFINITY`, on which oneCCL 2022 aborts.

**The 8-node runs must also unset `CCL_OP_SYNC`** ([Fix 12](#fix-12----unset-ccl_op_sync-on-oneccl-2022)).
With `CCL_OP_SYNC=1` -- in the launchers since Fix 4, harmless on oneCCL 2021 -- oneCCL 2022 turns
every cross-node `all_reduce` into a no-op that reports success. The full arm's
8-node mesh is `dp_replicate=4 x dp_shard=4`, so its first 8-node run (job
8904891) never averaged gradients across nodes: four replicas trained
independently while loss, reward and grad_norm looked normal. **That run is
invalid** and is kept below only as the "before" row. It was re-run with
`CCL_OP_SYNC` unset (job 8905380). The LoRA arm (`dp_shard=16`, one group spanning
the nodes, no cross-node all_reduce) and both 1-node runs were not affected; LoRA
at 8 nodes was re-run unset too (job 8905381), so the recommended setting is
covered in both arms.

### Configurations and outcomes

| | Full, 1 node | LoRA, 1 node | Full, 8 nodes | LoRA, 8 nodes |
|---|---|---|---|---|
| job | 8904955 | 8904956 | **8905380** (8904891 invalid) | **8905381** (8904922) |
| trainer tiles / mesh | 2 / `dp_shard=2` | 2 / `dp_shard=2` | 16 / `dp_replicate=4 x dp_shard=4` | 16 / `dp_shard=16` |
| generators | 2 x 1 tile | 2 x 1 tile | 16 x 1 tile | 16 x 1 tile |
| vLLM `max_num_seqs` / `gpu_memory_limit` | 128 / 0.85 | 128 / 0.90 | 16 / 0.85 | 16 / 0.85 |
| `CCL_OP_SYNC` | unset (single node) | unset (single node) | unset | unset |
| steps | **200/200** | **200/200** | **200/200** | **200/200** |
| wall time, launch -> step 200 | 50 min | 54 min | 48 min | 61 min |
| reward, first 10 -> last 10 (max) | 0.248 -> 0.342 (0.650) | 0.251 -> 0.354 (0.580) | 0.263 -> 0.360 (0.560) | 0.240 -> 0.362 (0.600) |
| cross-node weight transfers | n/a (shared memory) | n/a (shared memory) | 12819 byte-exact, 0 mismatch | 51213 byte-exact, 0 mismatch |

Everything else is the [common configuration](#common-configuration): 64
sequences per step, local batch 2 x 2048, AdamW lr 2e-6, temperature 1.0, top_p
0.95, `max_tokens=512`, checkpoint `load_only`. The full arm's 8-node mesh is the
same `dp_replicate=4 x dp_shard=4` spill as in the 2026-08-05 runs. The transfer
counts come from a study-only hook that checksums both sides of every XCCL
transfer; the few transfers short of the 12864 / 51456 of a full run belong to
the final pull, cut off when the job stops itself after step 200.

### Throughput

Conventions as in [Trainer throughput](#trainer-throughput) and
[Generation throughput](#generation-throughput): steps 2..200, `fwd_bwd`
arithmetic mean, `full_step` harmonic mean, per-GPU = global / tiles of that
role, per-seq = 1000 / ITL, aggregate = per-seq x mean concurrent requests. **One
change:** this stack logs `inflight_requests_at_completion` only as a per-step
maximum per generator, so mean concurrency is measured from the generators'
`generate_start`/`generate_end` events (summed request time in the window /
window length). The two torch 2.12 rows are the 2026-10-03 reference runs (same
configs and meshes, 10 steps), re-analyzed the same way.

| run | trainer fwd_bwd global | fwd_bwd per-GPU | trainer full_step global | full_step per-GPU | s/step | gen ITL (ms) | gen per-seq | gen agg global | gen per-GPU |
|---|---|---|---|---|---|---|---|---|---|
| Full 1n | 8716 | 4358 | 3276 | 1638 | 13.3 | 43.2 | 23.1 | 3401 | 1700 |
| LoRA 1n | 11214 | 5607 | 2977 | 1489 | 14.6 | 49.6 | 20.2 | 3089 | 1544 |
| **Full 8n** (8905380) | **22063** | 1379 | 3447 | 215 | 12.6 | 42.7 | 23.4 | 3583 | 224 |
| **LoRA 8n** (8905381) | **45032** | 2814 | 2619 | 164 | 16.5 | 54.7 | 18.3 | 2721 | 170 |
| Full 8n, `CCL_OP_SYNC=1` (8904891, **invalid**) | 7336 | 459 | 3204 | 200 | 13.5 | 42.1 | 23.7 | 3280 | 205 |
| LoRA 8n, `CCL_OP_SYNC=1` (8904922) | 40344 | 2521 | 2606 | 163 | 16.6 | 55.0 | 18.2 | 2705 | 169 |
| torch 2.12 Full 8n (8901636, 10 steps) | 32411 | 2026 | 2804 | 175 | 18.7 | 50.6 | 19.8 | 2208 | 138 |
| torch 2.12 LoRA 8n (8901637, 10 steps) | 39799 | 2487 | 2282 | 143 | 22.6 | 58.6 | 17.1 | 1561 | 98 |

Mean concurrent requests 147 / 153 / 153 / 149; decode 17.2 / 19.8 / 17.1 /
21.7 s; queue 14.0 / 15.8 / 1.2 / 1.4 ms (Full 1n / LoRA 1n / Full 8n / LoRA 8n).
The 10-step torch 2.12 rows include the start-up ramp, so only their `fwd_bwd`
and ITL are steady-state comparable.

**Every run is generator-bound, and 8 nodes buy nothing over 1.** Generation is
2700-3600 tok/s and `full_step` 2600-3500 tok/s in all four, so a step takes 13-17
s at either scale, with ~150 concurrent requests set by the controller's fan-out
regardless of tile count. Per-GPU throughput of both roles is ~8x lower at 8
nodes -- the same conclusion as the exp1-exp8 sweep.

**`CCL_OP_SYNC=1` was also the 8-node `fwd_bwd` slowdown.** Unsetting it took the
full arm from 7336 to 22063 tok/s (3.0x) and LoRA from 40344 to 45032 (+12%).
The full arm is still ~30% below the 10-step torch 2.12 figure (32411); LoRA is
13% above it. Both are hidden end to end by the generator bottleneck.

Per-sequence generation is faster on the new stack: 23.4 vs 19.8 tok/s in the
full arm (+18%), 18.3 vs 17.1 in LoRA (+7%).

### Loss, reward, and health

| run | loss mean | grad_norm range | entropy first -> last | `logprob_diff/max` mean (max) |
|---|---|---|---|---|
| Full 1n | -0.0061 | 0.130-0.550 | 0.53 -> 0.40 | 0.58 (3.91) |
| LoRA 1n | -0.0053 | 0.018-0.083 | 0.48 -> 0.46 | 0.59 (2.60) |
| Full 8n (8905380) | -0.0066 | 0.120-0.590 | 0.45 -> 0.46 | 0.56 (1.41) |
| LoRA 8n (8905381) | -0.0067 | 0.014-0.071 | 0.47 -> 0.41 | 0.60 (2.03) |
| Full 8n, `CCL_OP_SYNC=1` (invalid) | -0.0074 | 0.046-0.290 | 0.47 -> 0.47 | 0.66 (2.13) |
| LoRA 8n, `CCL_OP_SYNC=1` | -0.0069 | 0.016-0.087 | 0.53 -> 0.42 | 0.60 (2.75) |

Reward rises under the 20-step moving average in every valid run (MA-20 0.24 ->
0.40, 0.23 -> 0.39, 0.25 -> 0.39, 0.22 -> 0.40) and stays non-zero throughout, so
there is no trace of the zero-weight pull the workarounds address. The invalid
full run learned visibly worse (MA-20 0.23 -> 0.34, 1407 of 3256 rollouts
completed against 1694 of 3308 once gradients were averaged), but **nothing in
its loss, reward or grad_norm flagged it** -- its grad_norm was simply lower than
the valid run's. Only an exact-value check of the collective caught the bug (Fix
12). LoRA grad_norm (0.014-0.087) sits below the 0.040-0.140 band of the earlier
runs but nowhere near the 88-2464 signature of a corrupting transport (Fix 4); the
full arm is 3-5x higher, as expected for a norm over all parameters.

![Full, 1 node](grpo_full_0_6b_200_1n_torch214.png)
![LoRA, 1 node](grpo_lora_0_6b_200_1n_torch214.png)
![Full, 8 nodes](grpo_full_0_6b_200_8n_torch214_nosync.png)
![LoRA, 8 nodes](grpo_lora_0_6b_200_8n_torch214_nosync.png)

The `CCL_OP_SYNC=1` runs are charted in `grpo_full_0_6b_200_8n_torch214.png`
(invalid) and `grpo_lora_0_6b_200_8n_torch214.png`.

Artifacts: `~/aurora_rl_logs/exp200_20261004/` -- one folder per run (the
`*_nosync` folders are the re-runs) with the PBS script and stdout, the gzipped
train log, `structured_logs`, `rollout_samples.jsonl` and the transfer check,
plus `analyze.py` (the tables above, also written to `metrics.json`) and
`plot_loss_reward.py` (the charts).

---

# Part 2: The Fixes, One by One

Eleven changes took multi-node GRPO on Aurora from "cross-node dies or silently
corrupts gradients" to "10/10 steps, healthy, and scaling". Fixes 1-5 are the
load-bearing launch fixes and belong together as a set; 9 and 11 are the two real
library bugs; the rest are smaller. Fix 12 is for the torch 2.14 / oneCCL 2022
stack only.

| # | fix | symptom it cured | verdict |
|---|---|---|---|
| 1 | PALS `mpiexec` by absolute path | Bug A: cross-node TP, 0 steps, oneCCL SEND fault | faults 8 -> 0 |
| 2 | `export TMPDIR=/tmp` | Monarch bootstrap dies, `os error 2` | 0 faults |
| 3 | `unset HOSTNAME` | `Shared memory storage not found` | 0 faults |
| 4 | the CXI/oneCCL env block | Bug B: grad_norm 88-2464 | grad_norm 0.056-0.120 |
| 5 | `--cpu-bind none` | Bug C: everything 4-6x slow, ITL 214-268 ms | ITL 50-54 ms |
| 6 | delete the cross-node `dp_replicate` fold | ran a mesh the user did not ask for | both premises falsified |
| 7 | `max_autotune=False` for flex on XPU | flex autotune exhausts XPU registers | config builds |
| 8 | unconditional checkpoint drop | disk quota killed runs | 4.1G reclaimed |
| 9 | torch-xpu-ops RMSNorm `dbeta` alias | exp9: 0 steps, `tensor does not have a device` | 0/3 -> 3/3 |
| 10 | `@concurrent_endpoint` on `generate` | serialized generate() RPCs | not measured in isolation |
| 11 | torchstore replicated-region dedup | 8-node full GRPO hung on the step-4 weight pull | 16 volumes -> 4 |
| 12 | unset `CCL_OP_SYNC` on oneCCL 2022 | cross-node all_reduce a silent no-op; full 8n `fwd_bwd` 4.4x slow | exact; `fwd_bwd` 7336 -> 22063 |

## Fix 1 -- Launch via Cray PALS by absolute path

**Symptom (Bug A).** Cross-node `tp=8` reached 0 steps, then rank 7 died:
`atl_ofi.cpp:1071 prov_ep_handle_cq_err err 5 "Device or resource busy(16)"` ->
`send_entry.hpp:109 SEND entry failed` -> `ccl::v1::exception` -> `Killed(sig=6)`.
8 fabric faults per run.

**Mechanism.** Every script called bare `mpiexec`. `~/env-3.sh` sources an Intel
oneAPI toolchain that puts Intel MPI's **Hydra** `mpiexec` ahead of Cray PALS on
`PATH`, so the whole investigation had been launching under Hydra. PALS allocates
the job's Slingshot VNI and service IDs (`SLINGSHOT_VNIS`, `SLINGSHOT_SVC_IDS`,
`SLINGSHOT_DEVICES`) and passes them to ranks; Hydra sets none. CXI endpoints
still open under Hydra (`fi_info -p cxi` returns 0 either way) but are not on the
job's provisioned VNI.

**Change.** `MPIEXEC=${MPIEXEC:-/opt/cray/pals/1.8/bin/mpiexec}`, invoked as
`"$MPIEXEC"`. The absolute path is required; bare `mpiexec` is shadowed after
`env-3.sh`. Left overridable by env so the A/B stays possible.

**Confirmed.** exp6's fault count 8 -> 0; it later reached steps.

**Do not repeat the discredited mechanism.** "No VNI means oneCCL falls back to
TCP at ~37 MB/s" was **refuted** by a native oneCCL probe with no torch and no
MPI linked: PALS 0.937 ms/op (8.95 GB/s) vs Hydra 0.813 ms/op (10.32 GB/s) --
Hydra was *faster*. A missing VNI does not slow oneCCL down. Keep the switch (the
SEND fault is real and disappears), drop the bandwidth story.

## Fix 2 -- `export TMPDIR=/tmp`

**Symptom.** Immediately after Fix 1, jobs died at 0 steps in
`monarch/_src/actor/bootstrap.py:90`: `ValueError: No such file or directory
(os error 2) at path "/var/tmp/pbs.<jobid>.../.tmpXXXXXX"`, exit 143 after 5s.

**Mechanism.** PBS exports a per-job `TMPDIR=/var/tmp/pbs.<jobid>` that exists
**only on the mother-superior node**. PALS `--envall` forwards the head node's
env verbatim, so every other rank inherits a path that is not there. Hydra masked
it because its ssh bootstrap re-ran a login shell per node, resetting `TMPDIR`.
This bug *appears* when you adopt PALS; it is not a regression of Fix 1's value.

**Change.** `export TMPDIR=/tmp`, right after the `MPIEXEC` line. Plain `/tmp`,
**not** a `mkdir`'d subdirectory -- `/tmp` is node-local, so a head-node `mkdir`
would not exist elsewhere, while `/tmp` is already `drwxrwxrwt` on every node.

## Fix 3 -- `unset HOSTNAME`

**Symptom.** After Fix 2, a cross-node run died right after a successful trainer
`put_state_dict`: `RuntimeError: Shared memory storage not found. This may
indicate the storage volume is on a different host.` from
`torchstore/transport/shared_memory.py:507`, via `generator.py:1299
_pull_model_state_dict`.

**Mechanism.** torchstore decides shared-memory locality **by env, not by
syscall**: `get_local_hostname()` returns `os.environ.get("HOSTNAME",
socket.gethostname())` and `is_local_to_volume()` compares that to the volume's
hostname. The login shell exports `HOSTNAME`; PALS `--envall` copies the head
node's value everywhere; all ranks report the same hostname;
`is_local_to_volume()` returns True for a **remote** volume and the client
attaches shared memory that physically lives on another node.

**Change.** `unset HOSTNAME`, so torchstore falls back to per-rank
`socket.gethostname()`.

**The general lesson, which paid off twice:** under PALS `--envall`, anything
that reads identity or paths from ENV rather than from a syscall will silently
misbehave. `TMPDIR` and `HOSTNAME` were two. Audit for others before concluding a
run failed for a scientific reason.

## Fix 4 -- The CXI/oneCCL scale-out env block

**Symptom (Bug B).** `rep2 x tp4` on 4 nodes trained all 10 steps but with
**grad_norm 88-2464** against a healthy band of 0.05-0.15. Loss and reward looked
completely normal, which is what made it dangerous.

**Mechanism.** The transport was corrupting cross-node FSDP payloads. The tunable
group was migrated from the working plain-torchtitan DeepSeekV3 stack on the same
hardware.

**Change.**

```bash
export CCL_OFI_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64/libfabric.so.1
export FI_PROVIDER=cxi
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_OFI_PROVIDER=cxi
# env-3.sh points FI_PROVIDER_PATH at a libfabric with NO cxi provider, so only
# oneCCL (via CCL_OFI_LIBRARY_PATH) would reach CXI and every other consumer
# would silently fall back to tcp. Force the Cray libfabric:
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:${LD_LIBRARY_PATH}
export FI_PROVIDER_PATH=/opt/cray/libfabric/1.22.0/lib64/libfabric
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_OVFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=20
export CCL_ALLREDUCE_SCALEOUT=direct
export CCL_BCAST=double_tree
export CCL_SYCL_SCALEOUT_HOST_BUF_SIZE=$((2 * 1024 * 1024 * 1024))
export CCL_OP_SYNC=1
export CCL_WORKER_AFFINITY="5,13,21,29,37,45,57,65,73,81,89,97"
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export TORCH_LLM_ALLREDUCE=1
```

Note `CCL_WORKER_AFFINITY` and **not** `CCL_WORKER_COUNT` -- setting the latter
measured 29% slower.

**Confirmed.** exp8 grad_norm 88-2464 -> 0.058-0.110, and again 0.056-0.120 in
the 4-node re-measure under `--cpu-bind none`.

**The biggest lesson of the whole investigation:** grad_norm went from 2464 to
0.05 **with zero code change**. Bug B was never a numerics, DTensor-RNG, DCP, or
stride/geometry bug. Every "which mesh axis, which collective geometry" theory was
chasing a correlate of a broken transport. Two specific hypotheses were measured
and refuted offline: DTensor RNG `_set_pre_op_offset` ignoring `Replicate` is
CORRECT and is what makes replicate peers agree; DCP read byte-ranges from
`Shard` are likewise correct.

**Caveat, still true:** the block landed as a bundle with Fixes 1-3, so **which
single variable fixed Bug B was never bisected.** If a correctness bisect is
wanted, drop one var per job and judge on grad_norm using `rep2 x tp4` (the Bug B
carrier) and `tp8` (the Bug A tripwire). Do **not** bisect it for perf -- see
Fix 5.

## Fix 5 -- `--cpu-bind none` (Bug C)

**Symptom (Bug C).** Everything 4-6x slow: generator ITL 214-268 ms instead of
~52, s/step 73-120 s instead of ~19, with **zero numerical signature** -- loss,
reward, and grad_norm all healthy.

**Mechanism, established by direct observation.** `mpiexec -n <nodes> -ppn 1
--envall` carried no `--cpu-bind`, and PALS binds each rank to its own core slice
-- at `-ppn 1` that slice is a **single core**. The launcher's mask is inherited
by every Monarch actor and vLLM worker forked from it. From `/proc/<pid>/status`
on a live compute node with `nproc=204`:

```
pid=196862 mask=[1] threads=115  multinode_launcher
pid=197354 mask=[1] threads=127  monarch actor bootstrap   (x4, all mask=[1])
```

~126 threads per process timesharing CPU 1, across 5 processes sharing that one
core. The `mpiexec` **parent** kept the full `1-51,53-103,105-155,157-207`, which
is why a login-shell check would never reveal it.

**Change.** `--cpu-bind none` on the `mpiexec` line in all four launch scripts.

**Confirmed.** ITL 51.8 / 52.2 ms at 2 nodes, 50-54 ms at 4 nodes. Pin cost
measured three independent ways: generator 5.2-6.1x, s/step 5.3-6.1x, ITL
4.1-4.9x, trainer 2.2-8.6x. In this document it is the exp1/exp2 vs exp3-exp8
gap.

**The asymmetry that cracked it:** the GENERATOR loses more (5-6x) than the
trainer (2-5x), because vLLM runs ~126 CPU threads for tokenization, scheduling,
and detokenization. No fabric or collective-library theory predicts that, and
noticing it is what killed the env hypothesis.

**Consequence: the fabric-env perf bisect is dead.** The env costs nothing once
ranks are unpinned. Do not run `CCL_OP_SYNC` / `CCL_WORKER_AFFINITY` /
`TORCH_LLM_ALLREDUCE` perf arms.

### Two wrong diagnoses on the way here

1. "The 4-node slowdown is the generator mesh spanning nodes." Confounded: only
   post-fix 4-node and pre-fix 2-node logs existed, so generator-node-count and
   the env moved together.
2. "The slowdown IS the fabric env" -- backed by a 13/13 perfect separation on
   the env and 0/13 on node count. Still wrong: pinning landed alongside the env
   in every run, so the env was a co-traveller.

**Lesson: when two candidate causes separate the data equally well, stop adding
runs and go measure the mechanism.** One read of `Cpus_allowed_list` on a live
process settled what 13 correlated runs could not. And "controlled pair" is only
controlled for what you actually varied -- the pair described as isolating the env
had also changed the launcher.

## Fix 6 -- Delete the cross-node `dp_replicate` fold

**What it was.** `multinode_launcher.py` silently folded a cross-node
`dp_replicate` axis onto `dp_shard`, on the premise that cross-node
`dp_replicate` was broken on this stack.

**Why it went.** Both premises were falsified:

- *Correctness*: with Fix 4 in place, both cross-node `dp_replicate` layouts
  train clean -- exp7 `rep2 x dp_shard4` grad_norm 0.056-0.140, exp8 `rep2 x tp4`
  0.056-0.120. The explosion the fold existed to prevent **was Bug B.**
- *Perf*: unfolded measured faster, +19% and +9% on the two pairs. Mechanism:
  `dp_replicate` all-reduces gradients once per step while `dp_shard` all-gathers
  parameters every layer, so folding cross-node DP onto `dp_shard` only ADDS
  fabric traffic.

**Change.** The fold block, its env var, and the `_replicate_crosses_node`
helper are deleted -- no knob in any form. A comment records both falsified
premises with the numbers so the fold is not reinvented.

**Correction from the re-measure in this document:** under `--cpu-bind none` the
unfolded-vs-folded gap is **+2.8%** (exp7 21734 vs exp5 21137), not +19%, i.e.
within noise. Deleting the fold cost nothing and removed a silent surprise, but
its performance justification was overstated because both original arms were
pinned.

**Kept, do not confuse with the fold:** the `max_dp_shard` cap (= LoRA rank) that
spills excess width onto `dp_replicate` to avoid zero-sized LoRA shards torchstore
cannot handle. This is also why the full-parameter config needs its own note --
with no LoRA that cap falls back to 4, which is a no-op at 2 nodes but spills onto
`dp_replicate` at 4. Read the effective mesh from the `Mesh split` log line, never
from the requested flags.

## Fix 7 -- `max_autotune=False` for flex attention on XPU

Flex attention's autotune path tries kernel configs that exceed XPU register
limits (`OUT_OF_RESOURCES`). Both the LoRA config and the full-parameter twin set
`max_autotune=False` and `coordinate_descent_tuning=False`, and must re-create
`FlexAttention._compiled_flex_attn` because `torch.compile` captures options at
definition time. Harmless on CUDA (just uses the default heuristic config).

Related gotcha: `--compile.no-enable` does **not** stop flex_attention lowering,
so `AUTOTUNE flex_attention*` blocks appear either way -- do not read them as
evidence compile is on. Autotune cost is not a discriminator either: ~750-900 s
in every experiment, healthy and broken alike.

## Fix 8 -- Unconditional checkpoint deletion

**Symptom.** A run died at 0 steps on `OSError: [Errno 122] Disk quota exceeded`
(1053 errors in `logging.flush`).

**Change.** `drop_checkpoint()` plus `trap drop_checkpoint EXIT`, installed
**before** `mpiexec` so it fires even when the job is qdel'd during teardown. A
post-`mpiexec` `rm` was skipped exactly when it was needed, because torchtitan
force-saves at the last step and the operational rule is to qdel during the
teardown hang. Reclaimed 4.1G.

## Fix 9 -- torch-xpu-ops: RMSNorm backward aliased `dgamma` into the `dbeta` slot

The only genuine kernel bug in the list, and the only thing that blocked exp9.

**Symptom.** exp9 (full-parameter) reached 0 steps:
`torch.ops.aten._fused_rms_norm_backward.default(..., [True, True])` raised
`RuntimeError: tensor does not have a device`.

**Root cause.** In `src/ATen/native/xpu/sycl/LayerNormKernels.cpp`,
`rms_norm_backward_kernel` passed `dgamma` as **both** the dgamma and dbeta
arguments of the shared `layer_norm_backward_kernel_impl`. The impl branches on
`dbeta->defined()`, so above `xe_core_count * 1024` rows it took the both-defined
arm of the two-stage column reduction, which ends in
`*dbeta = dbeta_blocks.sum(0)` on a buffer allocated only under
`if constexpr (!rms_norm)` -- never allocated for RMSNorm. Introduced by
`1a4d8352` (Fused RMSNorm, #2205).

`xe_core_count = syclGpuEuCount() / syclGpuEUCountPerSubslice()` = 56 on an
Aurora Max 1550 tile, so the bound is **57344 rows**, bisected exactly (last PASS
57344, first FAIL 57345).

Exposed region: **`normalized_shape < 2048` AND rows > 57344 AND
`output_mask == [True, True]`.**

**Why exp9 and nothing before it.** Rows = every dimension except the last, so
only norms whose `normalized_shape` is `head_dim` (applied after the head split)
carry the `x n_heads` multiplier:

```
qk_norm rows = local_batch_size * seq_len * (n_heads / tp_degree) <= 57344
```

exp9's `q_norm` saw `2 x 2048 x 16` = 65536 rows (over) while `k_norm` saw 8 KV
heads = 32768 (under) -- hence one failing and one passing call in the same
backward. It tracks head count and sequence length, **not** parameter count. The
LoRA runs never hit it because `LoRAConverter` freezes every norm, so they request
`output_mask=[True, False]`. exp9 was simply the first run to ask for a norm
weight gradient above the bound. Nothing to do with LoRA, FSDP, flex attention,
activation checkpointing, or recompilation -- five probes refuted all of those
before the kernel was suspected.

**Change.** One hunk at the call site: pass an undefined `Tensor` instead of
aliasing `dgamma` into the `dbeta` slot, so control reaches the
`dgamma->defined() && !dbeta->defined()` arm. Only two callers of the impl exist,
and the LayerNorm one instantiates `rms_norm=false` and is untouched.
torch-xpu-ops commit `60a0e260` on branch `fix-rmsnorm-dbeta-alias` (base
`22be5ddd`), plus a regression test. **Not yet pushed or filed upstream.**

Do **not** instead guard `*dbeta = dbeta_blocks.sum(0)` on
`if constexpr (!rms_norm)`: the `!dgamma && dbeta` arm has a second such write, so
guarding one is incomplete and guarding both turns those arms into silent no-ops.

**Bonus finding.** Pre-patch, the below-bound simple path was writing beta sums
into **dgamma's** storage, harmless only because of a constexpr guard. The
aliasing was a latent silent-corruption hazard, not merely a crash.

**How it was tested in 70 seconds instead of a multi-hour rebuild.** The installed
wheel exports the kernel symbol dynamically and calls it through the PLT
(`readelf -rW libtorch_xpu.so` shows `R_X86_64_JUMP_SLOT`), so it is interposable.
`build_rmsnorm_interposer.sh` compiles only the patched translation unit into a
`.so` and `LD_PRELOAD` makes the linker resolve to the patched copy. The conda env
is untouched -- drop the `LD_PRELOAD` and the stock wheel is back bit-for-bit, so
every earlier measurement in Part 1 remains valid. Mangled names are compared
against the wheel before any result is trusted, and the run asserts on
`/proc/self/maps` so a no-op preload cannot masquerade as a pass.

**Verified (jobs 8731343, 8731359, 8731367).**

| rows | stock wheel | patched |
|---|---|---|
| 1024 / 32768 / 57344 | PASS | PASS |
| 57345 / 65536 / 131072 | FAIL `tensor does not have a device` | PASS |

Over-bound passing: stock **0/3**, patched **3/3**. exp9's exact shape
(2 x 2048 x 16 x 128) via `nn.RMSNorm` autograd: stock FAIL, patched PASS.

Numerics vs a **float64 CPU reference**, fp32 inputs: rel_L2 **1.2e-7** above the
bound vs **7.4e-7** below it, cosine **1.0000000000** at every size -- the
two-stage arm the fix routes into is *more* accurate than the simple arm, as a
tree reduction should be. Below-bound results are unchanged. LayerNorm
`bias.grad` exact in both arms.

Then job 8731367 ran exp9' end to end at the original `local_batch_size=2`:
2/2 steps, **0** crashes, grad_norm 0.29-0.43, reward 0.34 -> 0.26. No workaround
needed. exp12 (2026-09-14) extends that to **10/10 steps** in the same
configuration, still as an `LD_PRELOAD` interposer, with no
`_fused_rms_norm_backward` failure anywhere in the log.

**Metric warning worth carrying forward.** An earlier probe reported "max rel
diff 3.28" above the bound, which looked like a broken kernel and was purely a
bad metric: per-element division by `clamp(|expected|, min=1e-3)` when a column
sum of 65536 random-sign terms passes arbitrarily close to zero. Use float64 +
rel_L2 + cosine for reduction correctness; never per-element relative error on a
signed sum. (bf16 rel_L2 ~2.5e-3 is input quantization -- the same magnitude
appears below the bound, where nothing changed.)

**What remains unverified.** The patch was run as an interposer, not as part of a
full PyTorch build, and the interposer replaces the whole translation unit (so it
also supplies the forward kernels from that file, same source and flags). A full
build is the real integration test. Upstream, the fix landed as
[intel/torch-xpu-ops#4790](https://github.com/intel/torch-xpu-ops/pull/4790).

## Fix 10 -- `@concurrent_endpoint` on `VLLMGenerator.generate`

**Symptom.** vLLM reported `Running: 1` with one generate call completing per
~20 s, instead of the ~235 concurrent requests the controller fans out.

**Mechanism.** A plain `@endpoint` runs each message body to completion before
dequeuing the next, so the concurrent `generate()` RPCs were serialized inside the
actor. `@concurrent_endpoint` returns after scheduling each body as a background
task, so the actor immediately handles the next queued call.

**Change.** `@endpoint` -> `@concurrent_endpoint` on `generate`, with the
import added.

**Status: not measured as its own arm.** It is in the working tree and every run
in Part 1 shows `inflight` at 235-254, so concurrency was live throughout the
sweep -- but there is no A/B pair isolating this change, so no speedup number can
be attributed to it.

## Fix 11 -- torchstore fetched every replicated shard once per replica

The second genuine library bug, and what blocked the first 8-node full-parameter
run. Found only at 8 nodes because that is where `dp_replicate` first reaches 4.

**Symptom.** Job 8731512 (8 nodes, full-parameter, 200 steps requested) trained
**3 healthy steps** -- the Fix 9 kernel bug is closed end to end -- then hung
forever and burnt the rest of its 6-hour allocation. The log localizes it
precisely: generator leaders 0, 1, and 2 each logged `get_mapping=4 get_batch=4`,
while **leader 3 logged `get_mapping=4 get_batch=3`**. One leader never returned
from `ts.get_state_dict`, and the other 15 ranks then timed out on the stagger
barrier (gloo, 1800 s).

**Root cause.** `LocalClient._build_volume_requests` /
`_expand_tensor_slices` in `torchstore/client.py` iterated every volume in
`volume_map` and requested every stored slice it offered. Under `dp_replicate`,
each shard is stored once **per replica** with identical content, so the same
region was fetched from every replica that held it. The code carried a TODO
admitting exactly this: *"This is extra inneficient in the case of DP, where we
fetch all Replicate shards unnecessarily."*

The 8-node trainer mesh is `dp_replicate=4 x dp_shard=4` (16 trainer tiles, and
the `max_dp_shard` cap of 4 with no LoRA spills the excess onto `dp_replicate` --
see Fix 6). So each generator leader's pull contacted **16 volumes instead of 4**,
opening 16 concurrent private 2-rank XCCL process groups instead of 4, up to
**64 cluster-wide** across the four leaders per pull cycle. Wire bytes per pull
were 6.0 GB instead of 1.5 GB.

**Measured, not inferred.** A direct `_build_volume_requests` call on a
`mesh_shape=(2,2)` fixture: pre-fix contacts 4 volumes / 4 sub-requests, post-fix
2 / 2. The ratio is exactly `dp_replicate`.

**Change.** Thread a `claimed_regions` set through `_expand_tensor_slices`, keyed
on `(offsets, local_shape)`; a region already claimed from an earlier volume is
skipped. Safe because replicas hold identical data and `_assemble_results` /
`assemble_tensor` key off offsets and shapes only, never mesh coordinates.

The `if sub_requests:` guard before the `extend` is **load-bearing**:
`volume_requests` is a `defaultdict`, so an unconditional `extend([])` would still
create the key and the volume would be contacted with an empty request list.

Also in `torchstore/transport/xccl.py`: the two bare `work.wait()` calls in
`_receive_tensor` / `_send_tensor` are now bounded by
`TORCHSTORE_XCCL_TRANSFER_TIMEOUT` (default 600 s, 0 = wait forever). The PG's own
timeout covers rendezvous but does not fire on a collective that never completes.
This is failure-**surfacing**, not a masking retry: it converts an infinite hang
into a `RuntimeError` naming the `store_key`.

**Why the throughput numbers hid it for the whole sweep.** `LatencyTracker`
computes GB/s from the **logical** state-dict size. Qwen3-0.6B at 751.6M params
bf16 is 1.503 GB, and the logged `0.9333 s x 1.61 GB/s = 1.503 GB` exactly -- so
the reported get throughput is logical bytes over wall time. The 4x redundant wire
traffic is **invisible** in every log; the pull merely looked slow.
**Do not read those GB/s as wire rates.**

**Why smaller runs survived.** The bug was active in every multi-node run ever
made; the amplification just stayed small. From the archived logs:

| run | trainer mesh | volumes | redundancy | wire/pull | outcome |
|---|---|---|---|---|---|
| exp3, 2 nodes | `rep1 x sh4` | 4 | x1 | 1.52 GB | 20 steps OK |
| exp1, 2 nodes | `rep2 x sh2` | 4 | x2 | 3.05 GB | 20 steps OK |
| exp5, 4 nodes | `rep1 x sh8` | 8 | x1 | 1.52 GB | 20 steps OK |
| exp7, 4 nodes | `rep2 x sh4` | 8 | x2 | 3.03 GB | 20 steps OK |
| full, 8 nodes | `rep4 x sh4` | 16 | **x4** | 6.0 GB | 3 steps then hang |

`dp_replicate=2` did occur and did carry 2x redundancy, and survived; exp7 is the
clean 4-node comparison (exp1 predates `--cpu-bind none`). `dp_replicate=4` only
arises at 8 nodes. Note the payload is 1.52 GB in the LoRA runs too --
`put_state_dict` pushes the whole merged state dict, so "LoRA syncs less" is
**false** and was not the reason LoRA held up.

**The put/get asymmetry, correctly explained.** Puts never hung, and this is *not*
evidence about XCCL. Fix 3's `unset HOSTNAME` makes `get_local_hostname()` fall
back to `gethostname()` per node, so a trainer rank and **its own** volume are on
the same host, `is_local_to_volume` is true, and puts take **SharedMemory** --
never touching XCCL at all (12.6-17.9 GB/s). Only the cross-node generator-leader
pull uses XCCL.

**Tested.** `test_replicated_regions_fetched_once` in
`torchstore/tests/test_tensor_slice.py`, pure logic with no cluster: it **fails on
the pristine tree and passes with the fix**, confirmed in both directions.

Two environment traps in that suite, neither a regression: `pytest-asyncio` was
missing (without it every async test errors with "async def functions are not
natively supported", which looks like 16 failures); and the RDMA native bindings
are unavailable on this platform, so the first RDMA-parametrized test dies and
leaves TorchStore initialized, after which every later test in the same process
fails with "TorchStore is already initialized". Base and fix give an **identical**
7 failed / 1 passed set. Run the resharding tests **one at a time** for a clean
signal -- isolated, all of `test_data_parallel_replicate_only`,
`test_data_parallel_with_sharding`, `test_2d_to_2d`, `test_1d_to_2d`, and
`test_2d_to_1d` pass with the fix.

**What is not proven.** The dedup removes the 4x amplification, but it is **not**
established that amplification alone caused the stall -- no threshold was measured
showing 64 concurrent process groups exceeds a oneCCL/CXI limit while 16 does not.
It may be a latent race that 4x concurrency merely made probable. The argument for
the fix standing on its own is that it takes the 8-node pull to 4 volumes / x1 /
1.5 GB, **below exp7**, which ran 20 clean steps. The bounded `work.wait()` is the
hedge: the next hang raises with a `store_key` in 600 s instead of blocking 1800 s.

**Verdict from job 8734220 (2026-08-05): the latent-race reading was the right
one.** With Fix 11 live, the 8-node full-parameter run went from 3 steps to
**120** -- so the amplification was real and removing it helped enormously -- but
it then hung in the **same** place with the same signature. The dedup is
therefore a **rate reduction, not a cure**; roughly 1 pull in 120 still wedges.
Two details that narrow the remaining bug:

- **The hang moved.** 8731512 hung at step 3 on host 3's leader; 8734220 hung at
  step 120 on host **0**'s leader. Neither step nor host is deterministic, which
  rules out a fixed bad rank or a size threshold and is consistent with a race.
- **The 600 s hedge did not fire, and that is the sharpest clue we have.** The
  run blocked the full 1800 s and died on the *gloo stagger barrier* at
  `generator.py:1309`, not inside XCCL. Leader host0/gpu0 logged
  `get_mapping took 0.0071s` and never logged `get_batch`, so it was inside
  `ts.get_state_dict` -- yet `_wait_with_timeout` never raised. The `.so` and the
  editable checkout were both confirmed live, so the leader is stalling
  **before** it reaches a `work.wait()`, i.e. in request construction or PG
  setup rather than in the transfer itself. Chasing the timeout for not firing
  is the wrong lead; the stall is upstream of it.

**Do not "fix" this by folding `dp_replicate` into `dp_shard`.** See Fix 6 --
that fold was deleted after both its premises measured false, and
`multinode_launcher.py` says so at the site.

---

## Fix 12 -- unset `CCL_OP_SYNC` on oneCCL 2022

**Symptom.** On the torch 2.14 / oneCCL 2022.1.1 stack, the full arm at 8 nodes
(`dp_replicate=4 x dp_shard=4`) ran `fwd_bwd` at 7336 tok/s against 32411 on
torch 2.12 with the same mesh; LoRA (`dp_shard=16`) was unaffected. Every run
logged thousands of `CCL_ERROR| atl_ofi.cpp:483 send: fi_tsendmsg`, yet loss,
reward and grad_norm looked normal.

**Mechanism.** With `CCL_OP_SYNC=1`, oneCCL 2022 returns success from a
cross-node `all_reduce` while leaving every rank's buffer as its own input. Under
FSDP2 HSDP the replicate-axis all_reduce therefore did nothing: each node's
replica trained on its own node's gradients only. In the 2022.1.0 source the
variable also switches the SYCL allreduce / reduce-scatter / allgather kernels to
their blocking variants (`allreduce_ring.hpp:151` and siblings), the likely
source of the slowdown. Reduce-scatter inside a node, and reduce-scatter +
all-gather over one group spanning nodes (the LoRA layout), are correct with it.

**How it was found.** A standalone benchmark of the full arm's HSDP gradient sync
(596M fp32 parameters, 16 ranks on 4 nodes) with one oneCCL variable changed per
arm: only removing `CCL_OP_SYNC` moved it (1.348 -> 0.222 s per step; torch 2.12
0.299 s). An exact-value version (small-integer inputs, every output element
checked) then showed 127,714,285 wrong elements with the variable set and 0
without it, and 0 on torch 2.12 either way. The smallest reproducer is 2 ranks on
2 nodes and one `all_reduce`: each rank gets its own value back.

**Change.** `unset CCL_OP_SYNC` in the torch 2.14 multinode launchers (it stays
in the torch 2.12 ones, where it is harmless).

**Confirmed.** Exact reduction; full 8n `fwd_bwd` 7336 -> 22063 tok/s and LoRA
8n 40344 -> 45032 over 200 steps; the Bug B carrier (`rep2 x tp4`, Fix 4) stays
healthy without it (grad_norm 0.022-0.089 vs 0.022-0.082 with it), so dropping it
does not bring Bug B back. **grad_norm cannot detect this bug** -- the invalid run's
was merely lower than the valid one's -- which is the second time a silent
transport failure on this stack showed no numerical signature.

## Open items

- **Full-parameter 8-node `fwd_bwd` is still ~30% below torch 2.12** after Fix 12
  (22063 vs 32411 tok/s, the latter from a 10-step run). Not investigated.
- **The oneCCL workarounds are not upstream fixes.** The bugs are filed with
  standalone reproducers: the zero-byte broadcast ([#222](https://github.com/uxlfoundation/oneCCL/issues/222)), the
  `CCL_OP_SYNC` no-op all_reduce ([#223](https://github.com/uxlfoundation/oneCCL/issues/223)), the scheduler-path SIGSEGV
  ([#224](https://github.com/uxlfoundation/oneCCL/issues/224)) and concurrent communicator creation ([#225](https://github.com/uxlfoundation/oneCCL/issues/225), also on
  oneCCL 2021.17.2). `SYCL_UR_USE_LEVEL_ZERO_V2=1` is not root-caused.
- **exp9 at 10 steps on 2 nodes has not been re-run.** It would give the
  LoRA-vs-full-parameter cost as a controlled twin of exp3, i.e. at 4 trainer tiles
  and across a node boundary. Downgraded in priority: exp11 vs exp12 now answers the
  same question at 2 tiles on one node, and full-parameter came out 6% behind on
  compute and 9% ahead end to end. The kernel fix that unblocks the 2-node run is
  verified; only the job is missing.
- **exp12's faster decode (45.6 ms ITL vs LoRA's 51.2) is unexplained.** It
  reproduces in exp9' (44.2 ms), so it is not noise, but `gpu_memory_limit` differs
  between the arms (0.85 vs 0.90) and is the obvious confound. One LoRA 1-node run
  at 0.85 settles it, and that run is worth doing anyway -- see the KV-cache
  oversubscription note above.
- **exp6's step-4 stall is unexplained.** Low priority: `tp=8` is the slowest
  mesh measured, so the configuration has no performance case.
- **Fix 9 is not upstream.** Branch `fix-rmsnorm-dbeta-alias`, commit `60a0e260`,
  unpushed; no PR filed. Also untested in a full PyTorch build.
- **Bug B was never bisected to a single env variable** (Fix 4 caveat).
- **Fix 11 is uncommitted and not upstream.** It lives in the editable
  `~/git/torchstore` checkout (`client.py`, `transport/xccl.py`,
  `tests/test_tensor_slice.py`), so any run on this machine picks it up silently --
  which also means a torchstore reinstall would revert it without warning.
- **Fix 11 is a rate reduction, not a cure -- this is now the top blocker.** Job
  8734220 reached step 120 (up from 3) and then hung with the identical
  signature, so the redundancy was real but was not the whole bug. The live lead:
  the leader stalls inside `ts.get_state_dict` *after* `get_mapping` and *before*
  any `work.wait()`, since the 600 s `_wait_with_timeout` hedge never fired
  during an 1800 s hang. Next step is a `py-spy` dump of the stuck generator
  leader, not more timeout tuning. Until then the full-parameter arm cannot
  finish 200 steps -- expect roughly a 50% chance of hanging before step 200.
- **The 600 s XCCL hedge is unproven in the field.** It has never fired, and the
  one hang since it landed went around it. Do not count it as protection.
- **Generation is the binding constraint, and the sweep never varied it.** Every
  run held `inflight` at ~237 regardless of generator tiles. Raising
  `num_groups_per_train_step`, `group_size`, or `max_offpolicy_steps` is the
  experiment that would actually move end-to-end throughput.
- **Fix 10 has no isolated measurement.**

## Log archive

The raw training logs were moved out of `torchtitan/experiments/rl/` to
`~/aurora_rl_logs/` and gzipped. What is kept:

- `train_exp{1..8}_*.log.gz`, `train_exp9_2n_dpshard4_full.log.gz`,
  `exp9_1n_interposed.log.gz` -- the ten runs tabulated above
- **exp11 and exp12 are not archived yet**: their logs are still live at
  `torchtitan/experiments/rl/train_lora_1n.log` and `train_full_1n.log`, and
  `run_grpo_lora_sn.sh` / `run_grpo_sn.sh` each `tee` over their own file on the
  next 1-node run. Move them to the archive before re-running either script.
  `extract_exp_metrics.py` reads both from the RL directory, so pass that directory
  as its `log_dir` until they are moved.
- `exp200_20261004/` -- the four torch 2.14 200-step runs, with their own
  `analyze.py` and `plot_loss_reward.py`
- `*.VERDICT.txt` -- the per-job verdicts, which carry job IDs, the `Mesh split`
  line, and the fault/step counts that provenance the numbers here

Everything else (driver logs, watcher logs, pre-sweep and superseded reruns,
compile-graph dumps, and one-off diagnostics) was deleted. `extract_exp_metrics.py`
stays in the RL directory next to the code it parses; point its `RL_DIR` at the
archive to reproduce Part 1.
