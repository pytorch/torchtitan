# Common bugs: details

Each entry expands one bullet of section 5 in [SKILL.md](../SKILL.md): symptom, cause, how to confirm, fix.

## The run stops without crashing

- **Symptom:** the trainer waits forever and a rollout worker goes quiet. Nothing raises, so the job never exits.
- **Confirm:** `py-spy dump --pid <rollout worker pid>` shows the event-loop thread inside the reward fn.
- **Cause:** a reward fn runs CPU-bound code (sympy, Math-Verify) inside `async def` on the rollout worker's event loop. One answer like `\boxed{2000^{2000^{2000}}}` makes sympy compute `2000 ** (2000 ** 2000)`, a single C call that holds the GIL. Every group on that worker freezes and holds its slot; once the slots are gone, the trainer starves.
- **Why the timeout doesn't help:** `RewardMathVerify` wraps Math-Verify in `ThreadTimeout` (`examples/dapo_math/rubric.py:58-64`). A thread timeout is delivered only between bytecodes, so it cannot interrupt a C call, and SIGALRM only works on the main thread.
- **Fix:** grade in a separate process with a hard timeout; kill and replace the process on timeout.

## A generator's engine loop crashes

- **Symptom:** one generator logs `engine loop crashed; failing all outstanding replies`, then `generator is closed; cannot call generate`, and errors jump.
- **Cause:** vLLM rejected one request, e.g. a prompt of exactly `max_model_len` tokens with no room to generate. The exception reached the engine loop's catch-all and killed it.
- **What follows:**
  1. The router keeps the dead generator as a candidate. It has zero load, so least-loaded routing sends it most new work (61 of 64 rollouts in a simulation with 1 of 4 generators dead).
  2. Those groups fail before their first completion and are dropped as untrainable, so the trainer starves.
  3. If the run reaches the next weight sync, it crashes with `generator is closed; cannot call pull_model_state_dict`. Restart from the last checkpoint.
- **Fix the trigger:** every prompt cap must leave room for one output token.
  - `TokenEnv.Config.max_rollout_tokens` (default None, no cap) rejects `prompt_len >= cap`, so it is safe at `max_model_len`.
  - The Verifiers `GenerationServer.Config.max_rollout_tokens` is inclusive, so set it to at most `max_model_len - 1`.

## `rollout/branches_per_rollout/mean` > 1

- **Expected** when the env edits history on purpose, e.g. compaction.
- **Otherwise:** a turn's prompt tokens did not extend the previous prompt plus completion, so the rollout split into several training samples. The usual causes are listed in `TrainingSampleBuilder.rollout_to_training_samples`.
- **Cost:** each completion still trains once, on the prompt it was sampled from, so the gradient is right. The cost is duplicated prompt tokens, and a history that differs from what earlier turns saw.
- **Confirm:** record token ids, find the first token where turn k+1's prompt diverges from turn k's prompt plus completion, and decode around it ([script](rollout_inspection.md#where-a-branched-rollout-diverges)).
- **One case:** the harness re-sent a turn that hit `max_tokens` without its reasoning. The chat template then re-rendered the whole history and stripped the thinking from every earlier turn.

## Reward about 0 from step 1

Read 10 completions end to end.
- Most end on the same unexpected token: a wrong stop token. One run stopped 98% of completions at the first `"`.
- Most end at `max_tokens`: the budget is too small. A Qwen3.5 thinking model needed ~20K tokens per math answer and finished 0 of 64 at 8K.
- A renderer for the wrong model family can produce answers the grader cannot parse.

## `RuntimeError: N consecutive untrainable batches`

- **Cause:** with `drop_zero_std_reward_groups=True` (the default) and prompts that are all solved or all failed, no group trains. The batcher raises after 10 untrainable batches. Before that, too-easy data wastes generation while the batcher waits (a 35B base model solved 85 of the first 90 finished DAPO-Math groups).
- **Fix the data first:** filter prompts by the policy's own pass rate.
- **Keeping zero-std groups** (`drop_zero_std_reward_groups=False`) avoids the wait. But they count toward `num_prompts_per_train_step`, and their tokens join the token-mean denominator with zero advantage, so they dilute the update like a lower learning rate. They also cost trainer compute.

## Truncated rollouts score above 0

`Rubric.Config.truncation_reward` defaults to None, so the reward fns grade truncated rollouts, and a `\boxed{}` written mid-reasoning can score 1. Set `truncation_reward=0.0` if truncation should fail; it then acts as an explicit length penalty.

## Samples or groups vanish

- `batcher/num_samples_dropped_oversized/sum`: samples longer than the trainer's `max_context_length`, dropped with a `Batcher dropped` warning.
- `training_sample_builder/num_groups_dropped_untrainable/sum`: groups where a sibling produced no completion tokens, e.g. an error before the first completion.
- `training_sample_builder/num_samples_dropped_no_valid_tokens/sum`: samples with no loss token left.

## The logprob gap moves

- **Symptom:** `bit_wise/logprob_diff/mean` leaves its baseline, or `bit_wise/logprob_diff/max` trends up.
- **Cause:** usually the trainer and generator compute different math: a different attention kernel, MoE routing, a bf16 vs fp32 lm head.
- **Not staleness:** at small learning rates staleness barely moves it. In one 35B run at lr 1e-6, each step of policy age added ~0.1% of the baseline gap.
- [Bitwise parity](../../../../docs/bitwise_parity.md) makes the gap exactly 0 for on-policy samples.

## The attention kernel is not the one you think

- On recent torch nightlies, `varlen_attn` tries cuDNN before Flash. Activating FA3/FA4 only swaps the kernel behind the Flash option, so cuDNN can still win where it is eligible. Different kernels on the trainer and generator add logprob drift.
- **Check, don't assume:** `logging.getLogger("torch.nn.attention.varlen").setLevel(logging.INFO)` logs the chosen backend per call.
- vLLM's `Using FlashAttention version N` line describes vLLM's own copy, not the kernel TitanRL's attention backend calls.

## Slow decode

Inter-token latency in the hundreds of ms can mean the generator runs eager instead of replaying CUDA graphs (one MoE generator: 234 ms per engine step eager, 22 ms with graphs, at 8 sequences). Check the generator log for CUDA graph capture. For generator kernel speed against native vLLM, use the `inference_perf_hillclimb` skill.
