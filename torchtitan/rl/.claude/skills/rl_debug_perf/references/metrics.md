# Metrics

What each metric means, grouped by where it comes from. The traps are in section 2 of [SKILL.md](../SKILL.md). The default train log line prints `perf/*` and a few generator keys (`metrics.console_log_keys_train`); W&B and TensorBoard get everything.

## Trainer

- `perf/trainer/step_time_ratio/batch`: share of the step spent waiting for a batch.
- `perf/trainer/step_time_ratio/fwd_bwd` (forward/backward plus optimizer), `/blocking_trainer_push_model_state_dict`, `/blocking_generator_pull_model_state_dict` and `/unaccounted` split the rest. Seconds are under `timing/step/*`.
- `timing/weight_sync/trainer_push_model_state_dict` and `timing/weight_sync/generator_pull_model_state_dict`: how long the background push and pull took. The pull holds `num_prompts_per_train_step` slots until it ends.
- `trainer/mfu_percent`, `perf/trainer/tokens_per_second_fwd_bwd`.
- `trainer/memory/max_reserved_percent`, `trainer/memory/num_alloc_retries`, `trainer/memory/num_ooms`.

## Rollout buffer

Sampled at the end of each step.
- `rollout_buffer/available_active_slots`: normally 0, since the data loop refills freed slots at once. Nonzero means data loading lags.
- `rollout_buffer/num_groups_inflight`: groups generating, L in `step_time ~= G * W / L`.
- `rollout_buffer/num_groups_finalized`: finished, waiting for the batcher.
- `rollout_buffer/num_groups_waiting`: admitted, not yet claimed by a rollout loop.

## Generator

- `generator/inflight_requests_at_completion/max`: outstanding requests per generator actor when a request finished, across its DP ranks and including requests not yet admitted. Compare it with `max_num_seqs` times the generator DP degree.
- `generator/queue_time_ms/mean`: time a request waited for a seat. A few ms is nothing; compare it with the inter-token latency.
- `generator/inter_token_latency_ms/mean`, `generator/time_to_first_token_ms/mean`, `generator/decode_time_ms/{mean,max}`. A `decode_time_ms/max` near `max_tokens * ITL` means truncated samples set the group tail.
- `generator/num_cached_tokens/mean`: prompt tokens served from the prefix cache.
- KV usage, running and waiting requests, preemptions: vLLM's periodic `Running: N reqs, Waiting: N reqs, GPU KV cache usage: X%` log lines, or `generator.vllm_stat_logger = VllmOtelStatLogger.Config()` with `OTEL_METRICS_EXPORTER=jsonl` ([vLLM engine metrics](../../../../observability/metrics/README.md#vllm-engine-metrics)).

## Rollouts and data

The `rollout/*` and `rollout_reward/*` keys average every group the batcher consumed that step, dropped groups included.
- `rollout_reward/_mean`, and `rollout_reward/component/<RewardFn>/mean` per reward fn.
- `rollout_reward/group_zero_std_frac/mean`: share of groups whose rewards are all equal, dropped or not. `training_sample_builder/num_groups_dropped_zero_std/sum` counts the dropped ones (default `drop_zero_std_reward_groups=True`).
- `rollout/truncation_rate/mean`: share of rollouts whose status is truncated.
- `rollout/response_length/{mean,max}` and `rollout/output_tokens/*` are per turn. `rollout/total_length/{mean,max}` (last prompt plus completion) and `rollout/num_turns/{mean,max}` are per rollout.
- `rollout/group_failures/sum`: groups whose whole run raised. No metric counts errored rollouts.
- `rollout/branches_per_rollout/{mean,max}`: training samples per rollout; 1.0 unless the history changed between turns.
- `batcher/num_samples_dropped_oversized/sum`, `training_sample_builder/num_samples_dropped_no_valid_tokens/sum`, `training_sample_builder/num_groups_dropped_untrainable/sum`: data dropped with only a warning.

## Off-policy and numerics

- `train_batch/policy_age/mean`, `train_batch/policy_age_max`, `train_batch/pct_samples_over_target_age`.
- `bit_wise/logprob_diff/mean`: mean trainer-minus-generator log-ratio of the sampled tokens, a k1 estimate of -KL(generator || trainer). `bit_wise/logprob_diff/max` is noisy; watch its trend. `bit_wise/ratio_tokens_different/mean` was 0.86-0.90 in one run without batch invariance.
- `loss/ratio_clipped_frac`, `loss/ratio_mean`, `loss/generator_logprob_nan_frac` (should be 0).
- `trainer/entropy/mean`: mean entropy over loss tokens. Rising while reward is flat and truncation rises can be an early sign of degeneration; check it against length first.
- `trainer/grad_norm/mean`, `loss/mean`: see section 2 of [SKILL.md](../SKILL.md).
