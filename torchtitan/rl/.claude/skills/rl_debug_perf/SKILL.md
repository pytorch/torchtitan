---
name: rl_debug_perf
description: Debug a slow or broken TitanRL run. Use when the trainer waits for rollouts, a run is slow, hangs or crashes, rollouts error or score 0, reward or a metric looks wrong, or when choosing target_offpolicy_steps, num_prompts_per_train_step or the trainer/generator GPU split. For the generator's kernel speed against native vLLM, use inference_perf_hillclimb. Use also when the user invokes /rl_debug_perf.
---

# TitanRL debugging and performance

Read rollouts before curves, and errored rollouts before the rest. In the runs behind this skill, most "the run is broken" reports were an env bug, a grader bug, or noise from which prompts landed in a step.

Start here:
- Rollouts error or score 0, or reward looks wrong: section 4, then section 5.
- The run hangs or crashes: section 5.
- The run is slow: section 1, then section 3.
- A reward change you don't trust: section 6.

References: [worked_examples.md](references/worked_examples.md) (real runs), [rollout_inspection.md](references/rollout_inspection.md) (scripts), [metrics.md](references/metrics.md), [common_bugs.md](references/common_bugs.md).

## Mental model

- **Slots:** `(target_offpolicy_steps + 1) * num_prompts_per_train_step` groups. A group holds a slot from admission until it is trained and the generators pull the new weights; a dropped or failed group frees it at once.
- A group finishes only when its slowest sample finishes.
- **Seats:** each generator runs up to `max_num_seqs` sequences, sized so all generators together fit every sample of every slot, capped at 512 (logged at startup).

When the trainer waits for rollouts:

```text
step_time ~= G * W / L
  G = groups consumed per step = num_prompts_per_train_step + groups dropped (zero-std, untrainable, failed)
  W = mean group latency, from admission to its slowest sample
  L = groups generating at once (rollout_buffer/num_groups_inflight) ~= slots - groups waiting to be released
```

Every lever in section 3 moves G, W or L.

## 1. Triage

```text
0. Are rollouts erroring?        status mix of rollout_samples.jsonl, "marking ERROR" in the job log
                                 (no metric counts errored rollouts)
     yes -> section 4 first
1. Does the trainer wait?        perf/trainer/step_time_ratio/batch
     near 0              -> trainer-bound: trainer/mfu_percent, memory; add trainer GPUs
     over half the step  -> rollouts are the bottleneck, go to 2
2. Are the engines full?         generator/inflight_requests_at_completion/max vs max_num_seqs x generator DP,
                                 generator/queue_time_ms/mean, generator/inter_token_latency_ms/mean, KV usage
     running ~ max_num_seqs, queue time ~ ITL or more, preemptions,
     or ITL rising with running seqs at fixed KV  -> throughput-bound: more engines, faster decode, shorter outputs
     running << max_num_seqs, KV low, queue ~ 0   -> concurrency-bound: raise target_offpolicy_steps or
                                                     num_prompts_per_train_step; go to 3 to shorten the tail
3. Where does a rollout spend its time?   per-turn model time vs env time (tools, sandbox, grader)
     model time dominates -> long tail: response or turn cap
     env time dominates   -> env-bound: faster env, more rollouts in flight
```

The default train log line already prints `perf/*` and the generator keys above (`metrics.console_log_keys_train`; `None` prints every metric). Engines idling while the trainer waits is the common case: see worked example A. For idle gaps across actors, `generate_gantt_trace` turns `<dump_folder>/structured_logs/` into a Perfetto timeline ([structured logger](../../../../observability/structured_logger/README.md#gantt-trace-from-jsonl)).

## 2. Metrics that mislead

- `rollout_buffer/available_active_slots` is normally 0: freed slots refill at once. The budget binds when it is 0 and engines have idle seats.
- No metric counts errored rollouts. `rollout/group_failures/sum` counts only groups that failed whole.
- `timing/weight_sync/*` is how long push and pull took. The trainer's wait is `perf/trainer/step_time_ratio/blocking_*`.
- `loss/mean` is not progress: long failing rollouts dominate a token-mean loss.
- `trainer/grad_norm/mean` moves with batch length under a token-mean loss (r = -0.74 in one run), so a wiggle is not a spike.
- `bit_wise/logprob_diff/mean` flat and small (-1e-4 to -5e-4 nats per token in two runs without batch invariance) is numerics. A jump or a trend is a trainer/generator mismatch.

Every other key: [references/metrics.md](references/metrics.md).

## 3. Perf levers

**If generation is under-utilized and the trainer waits for the generator, increase off-policyness or batch size. A larger batch keeps the trainer busier and gives more ready groups to pick from.**

A bigger batch raises the groups needed per step (G) as much as the groups in flight (L), so it doesn't shorten the step by itself; it helps through trainer utilization and group choice. Rule of thumb: if the trainer waits over half the step, doubling the batch still fits. Worked example A projects both levers.

- **`target_offpolicy_steps`** (L up). Costs: older samples (watch `train_batch/policy_age/*`, `bit_wise/logprob_diff/*`) and more KV per engine. Age is unbounded with `windowed_fifo_batches=None` (default); a window bounds it but can stall on the oldest group ([windowed FIFO](../../../docs/windowed_fifo.md)).
- **`num_prompts_per_train_step`** (G and L up, trainer busier). Costs: fewer optimizer steps for the same data, so recheck the learning rate, and more KV per engine. Both levers raise `max_num_seqs`, which stops at 512.
- **Rebalance GPUs.** More engines help only when throughput-bound. A trainer idle most of the step is a cost lever: fewer trainer GPUs, after a memory check.
- **Fewer wasted groups** (G down). Zero-std groups cost a full generation and train nothing. Filter prompts by the policy's own pass rate.
- **Shorter tail** (W down): a response or turn cap. Truncated samples often set a group's critical path. Truncation acts as a length penalty, so don't cut the cap for speed when the eval needs long reasoning.
- **Faster env** (env-bound). Compare per-turn model time (`generator/time_to_first_token_ms/mean` + `generator/decode_time_ms/mean`) with the env's turn time (worked example B). Start a sandbox pool at 3x the rollouts per step, a heuristic, then size it to the peak lease.
- **Weight sync.** With `step_time_ratio/blocking_*` near 0 the trainer never waits, but the pull still holds slots until it ends; watch `timing/weight_sync/generator_pull_model_state_dict`.

## 4. Inspect rollouts, errored ones first

Errors hide from the curves. An error before the first completion drops its whole group as untrainable. A later error usually scores 0 and, in a mixed group, gets a negative advantage it did not earn. In one agentic run errors went from 0.3% to 59% in two steps: a tool failed to install in the sandbox.

1. **Record every rollout.** The default `KeepExtremeRewardsFilter` (`k=1`) keeps each group's best and worst only, so most errors are never written. Set `rollout_recorder.filter.k = num_samples_per_prompt`.
2. **Count statuses per step** in `<dump_folder>/rollout_samples.jsonl`.
3. **Find each error's final exception**, not the wrapper type: the last `Type: message` line of the traceback.
   - Native rollouter: `grep -A 30 "marking ERROR" <job log>`.
   - Verifiers: the `rollout done: ... stop=<stop>` line gives only the error class.
   - If your checkout has `Rollout.logs` and `KeepExtremeRewardsFilter.Config.keep_errors`, the JSONL carries the traceback.
4. **Then the silent zeros:** groups that scored 0 in every rollout with no error status. Read their grader output: a `ModuleNotFoundError` in the tests is a dataset bug.
5. **Read 2-3 examples of each class** ([scripts](references/rollout_inspection.md)).

Example classes from a terminal-agent env, compared with its reference harness (the benchmark's original runner); counts in worked example C. Write your own from the final exceptions you see.

```text
errored, by final exception
  sandbox infra         transport drop, lost job or shell            retry once on a fresh sandbox
  agent broke its env   killed its shell or container, wiped /tmp,   score 0, as the reference harness does
                        fork exhaustion, deleted its workdir
  agent invalid input   a NUL byte in the keystrokes                 score 0, don't retry
  agent timeout         hit the wall-clock budget                    grade the partial state if the reference does
scored 0 with no error status ("silent zeros")
  dataset bug           a test dependency missing from the task image
  grading artifact      env cleanup killed the agent's services before tests
  truncation            truncated_max_turns, truncated_length, truncated_prompt_too_long
  parse failure         empty content, e.g. a stray </think> moved the answer into reasoning
  agent wrong           the tests ran and failed
```

Rules:
- **Retry only sandbox infra.** An agent-caused error is a real failure the policy should learn from.
- **Check grading parity** with the reference harness: what runs before grading, which crashes become 0, what happens at each cap. One port killed the agent's services before grading: up to 5.6% of rollouts wrongly scored 0.
- **Save the tool and grader output per rollout.** Without it, a dataset bug looks like a wrong answer.

## 5. Common bugs, by symptom

Confirmation steps and fixes: [references/common_bugs.md](references/common_bugs.md).

- **The run stops without crashing:** a reward fn runs CPU-bound code inside `async def`, and one input holds the GIL in a C call. `RewardMathVerify` does this (`examples/dapo_math/rubric.py:58-64`); its `ThreadTimeout` can't interrupt it. Grade in a killable process.
- **`engine loop crashed`, then `generator is closed`:** vLLM rejected one request, e.g. a prompt with no room for an output token. The router keeps feeding the dead generator and the next weight sync crashes the job. Keep every prompt cap below `max_model_len`.
- **`rollout/branches_per_rollout/mean` > 1** without intended history edits: a turn's prompt did not extend the previous turn. Find the first diverging token.
- **Reward about 0 from step 1:** read 10 completions; check stop tokens, the `max_tokens` budget and the renderer.
- **`RuntimeError: N consecutive untrainable batches`:** prompts all solved or all failed, with zero-std groups dropped. Fix the data; keeping zero-std groups dilutes the update.
- **Truncated rollouts score above 0:** `Rubric.Config.truncation_reward` defaults to None, so truncated rollouts are graded. Set it to 0.0 if truncation should fail.
- **Samples or groups vanish:** `batcher/num_samples_dropped_oversized/sum` and `training_sample_builder/num_groups_dropped_untrainable/sum`.
- **The logprob gap moves:** the trainer and generator compute different math (attention kernel, MoE routing, lm head dtype).
- **The attention kernel isn't the one you think:** `varlen_attn` can pick cuDNN over FA3/FA4. Log `torch.nn.attention.varlen` at INFO.
- **Slow decode:** ITL in the hundreds of ms can mean no CUDA graph replay. For kernel speed, use `inference_perf_hillclimb`.

## 6. Measure learning correctly

- **With few prompts per step, training reward is mostly task mix.** In two same-seed terminal-agent runs, per-task reward correlated at r = 0.89.
- **Compute the SE before reading a trend.** The logged step reward averages every group consumed, dropped ones included: SE = sd(per-group reward) / sqrt(groups consumed), times sqrt(2) between two steps. In that run (~15 groups, sd 0.397), a drop from 0.65 to 0.33 was about 2 SE.
- **Completion order biases batches.** With `windowed_fifo_batches=None`, short, easy groups finish first; the hard ones land later as a "reward dip".
- **Compare the same tasks.** `group_id` is the index in the dataset stream, so a same-seed relaunch gives each prompt the same `group_id`. Compare reward per group, e.g. after a speed-up.
- **Eval offline on a fixed set.** Online validation runs only at the start and end (`ValidationConfig`). Evaluate saved checkpoints, k >= 4 samples per problem, as per-problem paired deltas against step 0.
- **Eval at the training cap and with room to think.** After 50 capped steps, one 35B model gained 4.4 points at its 49K training cap but lost 10.6 on harder problems at 131K. Report finish rate and accuracy among finished separately.
