# Pass-rate data curation for SWE-R2E RL (and what msl/rl does)

This is the recipe + the tools for the "data washing" step the user asked for:
filter the raw R2E task pool down to a **learnability band** (e.g. 20-70% pass
rate) before training, so binary-reward GRPO actually gets a gradient.

It is a direct port of the msl/rl coding-agent RL recipe. Below: (1) what msl/rl
does and why, with file pointers; (2) how our two scripts implement it; (3) the
step-2 training plan on the curated data.

---

## 1. Why curate at all (the problem the band solves)

GRPO/DAPO learns from *within-group reward variance*. With a binary reward (solved
/ not), a prompt-group of K rollouts gives a gradient only if it is **mixed**
(some solve, some don't). A group that is all-fail (0/K) or all-solve (K/K) has
zero std -> zero advantage -> zero gradient, and the soft filter drops it.

On the **full** R2E pool the Qwen3-32B base policy solves only **~3%** of tasks
(measured live: 17 solved / 575 graded rollouts). Those solves cluster on a few
easy tasks; the rest are ~never solved. So the per-task pass-rate is bimodal:
mostly 0% (too hard), a few ~100% (too easy), very little in between. Training on
the full pool starves the trainer -- almost every group is all-fail, dropped,
and the trainer waits forever for a mixed group (we saw 0 train steps land).

The fix is a **curriculum**: pre-measure each task's pass-rate and keep only the
middle band (not-too-hard, not-too-easy). Every kept task tends to produce mixed
groups -> non-zero advantage -> a real gradient every step.

---

## 2. What msl/rl does (the reference recipe)

The canonical coding-agent RL stack: `genai/msl/rl/projects/swe` (rollout/grade/
rejection-sampling) + `projects/agents/experiments/coding/rl_v1.py` (trainer
presets). The end-to-end flow is **two stages**:

### Stage 1 - data washing / learnability filtering (rejection sampling, pass@k)
- Run the **same actor + grader you will train with** over the raw task pool, N
  samples per task (msl uses ~8-10), recording per-(task, sample) `resolved` (0/1).
  Launcher: `projects/swe/run_distributed.py` -- an embarrassingly-parallel MAST
  job that shards `dataset_size * num_samples` linear indices across nodes, each
  (instance, sample) a subprocess, writes per-instance `metrics.json` (+ per-node
  `stats.json` with n_list/c_list), resumes by metrics.json existence, aggregates
  pass@k with `projects/swe/utils/pass_at_k.py:estimate_pass_at_k`.
- Aggregate to a per-instance summary: `projects/.../aggregate_results.py:1677`
  groups by instance_id -> `[instance_id, resolved, num_samples]` (instance_summary.csv).
  `pass_rate = resolved / num_samples`.
- **Band filter** (THE primitive): `swe/data/adhoc_scripts/0204_calculate_pass_rate.py:232`
  `filter_instances_by_pass_rate` keeps `pass_rate_min < pr < pass_rate_max`
  (EXCLUSIVE bounds), dropping 0% (too hard / broken env) and 100% (too easy).
  The Scuba-driven twin `aai_tbr/data_prep/filter_by_rs_pass_rate.py:228` keeps
  `pass_lo < pass@1 <= pass_hi` (default **0.2 .. 0.7**) and stamps
  `metadata.rs_pass_at_1` onto the output JSONL.
- Datasets are literally **named by band**: `.../1208/aggregated/40-75.jsonl`,
  `10-50.jsonl`, `10-70.jsonl` (`rl_v1.py:1577,1612,1659`). A real funnel:
  33,920 raw -> 12,592 (correlation filter) -> 5,556 (30-70 band) -> 2,982 final
  (`registry_swe_rl.py:20`).
- **Secondary filters** (optional, after banding): tool-call<->reward correlation
  >= 0, has_repro_or_test, no gibberish / banned commands, drop broken envs.

### Stage 2 - train on the banded set
- Async GRPO-style RL with **normal group size** `num_trajectories_per_prompt=64`
  (debug 16) (`rl_v1.py:474`).
- Keep **online** zero-variance filtering even on banded data: `zvf=True` appends
  `DropSmallStd(threshold=1e-3)` -- drops any group whose reward std collapses as
  the model improves (`rl_v1.py:485,1238`). Plus `DropNan`,
  `DropOffpolicyManager(max_offpolicy_steps=16)`.
- Optional cheaper online curriculum: `buffer_size_for_on_policy_filtering` runs a
  small pilot batch and drops a prompt *before* spending the full 64-rollout
  budget if the pilot shows no reward variance (`buffer_loop_utils.py:277`).
- **Curriculum schedule**: train the easier band first (e.g. 40-75), then
  *continue from that checkpoint* on a harder band (10-50) (`octopus_5p2b_cont`).
  The band is a moving target -- re-wash with the improved policy between stages.

Key principle: **measure the band with the exact reward the trainer uses.** We
curate on binary `solved` (default reward), not the dense fraction, so the band
matches what GRPO will see.

---

## 3. Our implementation (two scripts, mirrors the msl two stages)

### `curate_passrate.py` -- stage 1 worker (per host)
A standalone, embarrassingly-parallel worker (no controller / trainer / mesh /
weight sync), modeled on `local_smoke_harness.py` + msl's `run_distributed.py`:
- one in-process vLLM `AsyncLLMEngine` serving the policy (continuous batching for
  the K-way-per-task fanout), one `AnthropicAdapter`;
- shards the pool with `--shard-id i --num-shards N` (strided, so each host gets a
  difficulty-balanced slice of the repo-sorted pool);
- per (task, sample): `boot_agent_sandbox -> run_claude_code -> git_diff ->
  evaluate_r2e` -> binary `solved` (the exact training path, minus turn capture);
- writes `<out-dir>/results/<instance_id>/sample_<k>.json`
  (`{instance_id, sample_idx, solved, reward, applied, status, error, num_turns}`),
  **resumable** (an existing result is skipped) so a preempted host re-runs only
  its missing attempts;
- bounded by `--concurrency` (per-host active rollouts) + a per-attempt wall-clock
  guard. Reuses the harness env knobs (`SWE_BOOT_CONCURRENCY`,
  `SWE_TIME_BUDGET_SEC`, `SWE_EVAL_TIMEOUT_SEC`, `SWE_MAX_CONTEXT_LEN`,
  `DAYTONA_API_KEY`).

Launch N hosts pointed at one shared `--out-dir`. K should match the training
group size (8-16) and use the SAME base checkpoint the trainer starts from
(pass-rate is policy-dependent).

### `aggregate_passrate.py` -- stage 2 filter (pure CPU)
Reads `<out-dir>/results`, rolls up per-instance `(resolved, graded)` ->
`pass_rate`, prints the full histogram, then keeps the **exclusive** band
(`--pass-min 0.2 --pass-max 0.7`, `--min-samples 8`) and joins back to the source
JSONL, stamping `metadata.pass_rate / resolved / num_samples`. Output is a drop-in
for `SWER2EDataset`. Also writes `instance_summary.csv`. ERROR/timeout attempts
are excluded from the denominator by default (infra failures, not difficulty).

```
# stage 1: on each of N hosts (run.sh SWE_CURATE branch sets shard id)
python -m ...swe_r2e.curate_passrate --model .../Qwen3-32B \
    --out-dir <bucket>/curate_out --shard-id $i --num-shards $N --k 8 \
    --tensor-parallel 8 --concurrency 24
# stage 2: once, on the devvm (pure CPU)
python torchtitan/experiments/rl/examples/swe_r2e/aggregate_passrate.py \
    --results-dir <bucket>/curate_out/results \
    --source-jsonl <bucket>/r2e_subset_4p5k.jsonl \
    --out <bucket>/r2e_band_20_70.jsonl --pass-min 0.2 --pass-max 0.7 --min-samples 8
```

---

## 4. Step-2 training on the curated band

Point the existing async config at the curated JSONL via `SWE_PROMPT_DATA` and
restore **normal** group settings (the current `num_groups_per_train_step=1 /
group_size=16` is only the sparse-full-R2E starvation workaround):
- `group_size` ~16-32, `num_groups_per_train_step` > 1 (the banded data makes ~every
  group mixed, so multi-group steps fill quickly);
- keep the soft filter on (drop zero-std groups) -- as the policy improves, banded
  tasks drift toward all-solve; the online filter catches that, like msl's `zvf`;
- keep binary/sparse reward (see `[[feedback_swe_sparse_reward]]`);
- when reward plateaus, re-wash with the improved checkpoint and re-band downward
  (curriculum), as in msl's 40-75 -> 10-50 continuation.

## 5. Cost / sizing

Full R2E = 4578 tasks. At K=8 that is ~36.6k agent rollouts (each a multi-turn
Claude Code episode, ~3-4 min, up to ~70 turns) + a grading sandbox each. With ~90
hosts x ~16-24 concurrent rollouts/host the wash is a few hours wall-clock. The
20-70% band is expected to be a *small* fraction of 4578 (msl funnels drop
~60-90%); the histogram from stage 2 tells us the real yield and lets us widen the
band if it is too thin at the 32B base's ~3% solve rate.
