# SWE-R2E coding-agent RL recipe (torchtitan, async loop)

How to train a Qwen3 coding agent (Claude Code in a Daytona sandbox) on R2E-Gym
with GRPO/DAPO in torchtitan's async RL loop, "fast and good". Written 2026-06-27
after porting swe_r2e to the async loop (PR #3642) and studying three reference
recipes: **Tmax** (arXiv 2606.23321, AllenAI, github.com/hamishivi/tmax),
Meta-internal **msl/rl** (`genai/msl/rl/projects/agents/experiments/coding`), and
**slime** (`examples/coding_agent_rl`).

## TL;DR

- **Keep the reward sparse/binary** (1.0 iff the patch makes all R2E hidden tests
  pass, else 0.0). All three references use binary outcome-only reward. Do NOT add
  dense/partial-credit or format/length reward to the *reward*; shape *data* and
  *filtering* instead.
- **The thing that makes binary reward learn is "soft filtering"**: drop GRPO
  groups with zero reward std (all-pass or all-fail -> zero advantage -> zero
  gradient). Already implemented (`TrainingSampleBuilder.drop_zero_std_reward_groups
  = True`). The whole game is maximizing the fraction of groups that survive it.
- **Two ways to raise the surviving-group fraction**: (1) a **pass-rate data
  curriculum** -- train on tasks the model solves ~10-70% of the time (the single
  biggest lever; the #1 gap in our recipe); (2) **bigger group size** (more
  siblings/task -> better advantage estimate; lower priority than data).
- **Async is the throughput story**: collection (multi-turn Claude rollouts, minutes
  each) is the wall-clock bottleneck, hidden behind training via `max_offpolicy_steps
  > 0` + multiple generators.

## What's implemented and working (2026-06-27)

The async loop trains Qwen3-32B end-to-end on MAST H100:

| Piece | Setting | Notes |
|---|---|---|
| Loop | async (#3642) | `max_offpolicy_steps=2`; collection overlaps training |
| Trainer | FSDP-16 (2 hosts), TP=1, fp32 master, FullAC | true 16-way shard, needs the fused-QKV fix below |
| Generators | 5 x vLLM TP-8 (5 hosts) | sticky-session routing for prefix reuse |
| Model | Qwen3-32B, fused QKV, **fp32 lm_head** (`LMHeadCastConverter`) | fp32 lm_head matches train/inference logprobs (Tmax does this too) |
| Loss | `ChunkedLossWrapper(DAPOLoss(clip_low=0.2, clip_high=0.28))` | chunked = no full-vocab OOM at long ctx; DAPO clip-higher per slime/Tmax |
| Reward | binary (per-test exact match -> 1.0 iff fully solved) | `SWE_REWARD_DENSE` opt-in dense exists but kept OFF |
| Soft filter | drop zero-std groups | `drop_zero_std_reward_groups=True` |
| Weight sync | CPU-staged TorchStore (e1ffd44 SHM PUT + MonarchRDMA GET) | ~21s/step after the OMP/MKL fix (was ~140s) |
| Batch | 8 groups/step x group_size 16 = 128 rollouts/step, seq 24576 | |
| Entitlement | `msl_infra_pytorch_dev`, region `pci`, 8 hosts | 96-node pool |

Run: `conda activate rlmast; export DAYTONA_API_KEY=dtn_...;
SWE_NUM_GENERATORS=5 SWE_PROMPT_DATA=mast_rl/swe_assets/r2e_train.jsonl bash
mast_rl/submit_swe.sh`.

## The two hard bugs fixed in the port

1. **FSDP-16 of fused-QKV Qwen3-32B crashed in weight init** ("Cannot unflatten
   unevenly sharded tensor: 8 not divisible by 16"). Root cause: PR #3714 made qwen3
   fused-QKV the default; the fused `wqkv` reshapes per kv-head (n_kv_heads=8) in
   init AND in the checkpoint save/load hooks, which can't shard beyond n_kv_heads.
   **Fix = cherry-pick PR #3807** (gather to Replicate before the per-kv-head
   reshape) -> fused QKV shards to any FSDP degree. (Pre-#3714 non-fused QKV also
   works and is what older FSDP-16 runs used.)

2. **~115s/step weight sync** was BLAS/OpenMP oversubscription, not the transport:
   MAST packs multiple actor processes per host, and without `OMP_NUM_THREADS=1` /
   `MKL_NUM_THREADS=1` the per-rank CPU pools fight during the CPU-staged copy.
   **Fix = set both in the launcher env** -> generator pull 115s -> 12.6s, trainer
   push 21s -> 8.6s (~7x). TorchStore was already the fast commit (e1ffd44);
   `USE_TORCHCOMMS=0` was already set.

## Where the wall-clock goes (measured, FSDP-16, 5 gen)

- **Collection dominates.** A group of 8 Claude rollouts takes ~80-230s (agents do
  real multi-turn work, ~3-4 min, usually well under the 900s budget). Daytona
  **sandbox boot is NOT the bottleneck (~2.8s)**; it's the multi-turn model
  generation. -> more generators / concurrency + async overlap.
- **Weight sync ~21s/step** after the OMP/MKL fix (was the #1 fixed cost at ~140s).
- **fwd/bwd**: FSDP-16 cross-host all-gather/reduce-scatter over **IB**
  (`NCCL_TRANSPORT=IB`, needs rdma-core 60) -- keep IB; Socket is ~9x slower.

## Recipe knobs: ours vs the references

| Knob | titan (now) | Tmax (paper) | msl/rl | slime |
|---|---|---|---|---|
| reward | binary | binary outcome-only | binary (mult. judge x len, floor 0.5) | binary |
| group size | 16 | **32** | **128-256** | 8 |
| prompts/step | 8 | 8 | 16-32 | 1 |
| soft filter (drop zero-std) | yes | yes | yes (zvf 1e-3 + PassRateFilter) | - |
| data curriculum | **none (random 4.5k)** | data axes + graded verifiers | **pass-rate band 20-70%** | none |
| loss | DAPO clip 0.2/0.28 | DAPO + **DPPO mask** | REINFORCE + logprob-corr 0.003 | GRPO+KL 0.001+clip 0.2/0.28+**TIS** |
| off-policy stability | clip only | **DPPO** (mask train/infer logprob disagreement) | sampler/trainer fwd penalty | TIS |
| KL | 0 | ~0 (rely on group norm) | - | 0.001 low-var |
| context | 24576 | **65536** | - | 32k-96k |
| max_tokens/turn | 4096 | 16384 | - | 8192-32768 |
| async off-policy steps | 2 | fully async | **16-32** | sync (colocated) |
| lm_head precision | fp32 | fp32 | - | fp32 |
| steps | 40 | 500 | - | - |
| warm start | none | **SFT on successful rollouts** | - | none |

## CRITICAL empirical finding (2026-06-27): full R2E starves the trainer

Running 32B on the **unfiltered** full R2E-Gym-Subset (4.5k) with binary reward,
group_size 16, 12 generators: in 134 collected+scored+built groups, **0 train steps
fired** -- fewer than `num_groups_per_train_step` (=4) groups were "mixed" (1..15 of
16 solved). The 32B's pass profile on random R2E is bimodal: most tasks all-fail
(0/16), the rare easy task all-solve (16/16); **both have zero reward std and are
dropped by the soft filter**, so mixed (gradient-bearing) groups are <3% of groups.
The batcher waits for `num_groups_per_train_step` *surviving* groups, so it never
fills -> the trainer starves -> no learning. (~2% overall pass rate but clustered,
not spread.) This is the concrete proof that **binary RL on unfiltered SWE data does
not train** -- exactly why Tmax/msl/rl filter data to a learnable pass-rate band.

Two fixes:
- **Quick (online): `num_groups_per_train_step=1`** -- fire a step on each mixed
  group the moment it appears. Unstarves the run on full data, but slow/noisy
  (~one mixed group per ~30 groups collected) and a 1-task batch.
- **Proper (offline): pass-rate curriculum** -- one-time rejection-sampling pass
  (run the base 32B k=8-16x per task, keep ~10-70% solve-rate tasks), then train on
  that subset where ~every group is mixed -> dense gradient, fast clean reward rise.
  The current full-data run's per-task dumps (group_size siblings = a pass-rate
  estimate per task) can be mined to build this without a separate sampling job.

## Recommendations, prioritized (fast AND good)

**P0 -- data curriculum (the biggest lever for "reward rises").** Binary reward only
learns from groups with mixed reward; at ~1% pass on random R2E tasks almost every
group is all-fail and gets dropped, so there is ~no gradient. Both msl/rl
(`0313_universal_pass_rate/20-70`) and Tmax (graded verifiers) get their signal from
*learnable* data. Build a one-time rejection-sampling pass: run the base 32B k=8-16x
on each R2E task, keep tasks with solve-rate ~10-70%, drop trivial and
currently-unsolvable. This raises the surviving-group fraction far more than any
group-size change. (Interim: `r2e_train.jsonl` (47) / `r2e_solvable.jsonl` (7) are
small hand-curated stand-ins; the 4.5k subset is unfiltered and too sparse.)

**P0 -- off-policy stability (DPPO / TIS).** We run async (`max_offpolicy_steps=2`);
the dominant async failure is train/inference (vLLM vs FSDP trainer) logprob
mismatch causing collapse. Tmax's **DPPO** masks tokens where the two logprobs
disagree (binary TV-divergence approx); slime uses **TIS** (truncated importance
sampling); msl adds a 0.003 sampler/trainer forward-logprob penalty. We have neither
(only DAPO clip). Adding DPPO-style masking to `DAPOLoss` is cheap (both logprobs are
already on hand for the ratio) and is what lets Tmax run group-32 fully-async stably.

**P1 -- bigger group size.** Tmax 32, msl 128-256, us 16. For a fixed rollout budget
the *number* of surviving groups is ~constant, but bigger groups give lower-variance
advantage baselines and stronger per-group gradients. Raise toward 32 once the data
curriculum lands (so the extra siblings aren't wasted on all-fail tasks).

**P1 -- context + per-turn tokens.** Our 24576 / 4096 is tight vs slime (32-96k /
8192-32768) and Tmax (65536 / 16384); long agent trajectories truncate into empty
turns (wasted rollouts). Raise context to 32768 and `max_tokens` to 8192 (FSDP-16 +
FullAC + chunked loss has the headroom; watch HBM). Keep the agent budget
`SWE_MAX_CONTEXT_LEN` ~2k under the model context.

**P1 -- throughput.** (a) sequence packing of variable-length trajectories
(open-instruct reports 2-10x; check the async batcher packs); (b) skip zero-grad
batches before fwd (soft filter already drops groups; ensure no empty fwd/bwd);
(c) KV reuse across agent turns (sticky routing is on; pair with vLLM prefix cache /
CPU-KV-offload so the growing transcript isn't re-prefilled each turn); (d) keep
generators saturated -- ~2x as many in-flight rollouts as generator capacity.

**P2 -- SFT warm-start.** Tmax and msl both SFT on successful rollouts before RL. If
cold-start pass-rate is ~0 (all groups zero-std), do a short rejection-sampling SFT
pass to lift the base into the regime where GRPO advantages are non-degenerate. For a
32B Claude-Code agent this matters less than for small models.

**Do NOT:** add dense/partial-credit reward as the default, or an additive
format/length reward -- under group normalization a stray format reward inflates long
wrong answers (documented in open-instruct). Shape data + filtering, not the reward.

## Diagnostics to watch (per Tmax/msl)

`rollout_reward/group_zero_std_frac` (want it falling),
`training_sample_builder/num_groups_dropped_zero_std`, solved-count per step,
the per-step `train_step` / `wait_for_training_batch` / `generator_pull` spans
(structured_logs), and -- if DPPO/TIS is added -- the train/inference logprob
disagreement / mask rate and `ratio_clipped_frac`. Collapse shows up as reward
flatlining while clipfrac / mask-rate spikes.

## References

- Tmax: arXiv 2606.23321; code github.com/hamishivi/tmax (open-instruct `grpo_fast`,
  `Vanillux2Agent/agent.py`, `rl_data/comparison/adapters/r2e_gym.py`).
- msl/rl: `genai/msl/rl/projects/agents/experiments/coding/rl_v2.py` (zvf,
  PassRateFilter, pass-rate-band data, multiplicative judge reward).
- slime: `examples/coding_agent_rl` (binary R2E reward, GRPO+KL+clip 0.2/0.28+TIS).
- Async RL loop: torchtitan PR #3642; fused-QKV FSDP fix: PR #3807 (over #3714).
