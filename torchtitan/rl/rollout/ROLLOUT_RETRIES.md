# Rollout retries

`Rollouter.Config` owns retry policy. Existing recipes inherit bounded retries
without changing their configuration:

```python
config.rollouter.max_retries_per_prompt = 2
config.rollouter.max_infra_failures_per_run = 1000
config.rollouter.retry_backoff_s = 0.5
```

Two retries mean at most three attempts per sibling. The optional linear delay
is 0.5 seconds before retry 1 and 1.0 second before retry 2 in this example.
Zero retries disables retrying, but still counts failures and enforces the run
budget. A zero run budget aborts on the first infrastructure failure. Negative
budgets and negative or nonfinite delays are rejected at config construction.

## Classification

| Failure | Result | Retry | Score/train |
| --- | --- | --- | --- |
| Generation or worker RPC `TimeoutError` / `ConnectionError` | `ERROR_INFRA` | Within both budgets | No |
| Invalid input (`ValueError` / `TypeError`) | `ERROR_DATA` | No | No |
| Environment construction/execution failure, including its own timeout | `ERROR_ENV` | No | No |
| Unclassified exception | `ERROR_ENV` | No | No |
| Explicit `RolloutError(status, message)` | The supplied failure class | Only `ERROR_INFRA` | No |

The Monarch adapter unwraps `ActorError.exception`, including nested wrappers,
before classification. It never classifies by exception-message substrings.
An environment's own timeout must not be confused with a generator RPC timeout.
`RolloutError` carries a classification across a worker RPC when operation
context matters, such as an environment constructor raising `TimeoutError`.
Existing policy outcome statuses, such as `ERROR_PARSE` and `ERROR_TIMEOUT`,
retain their scoring behavior and are never retried.

## Ownership and failure paths

```text
loader -> WAITING -> CLAIMED ---------------------------------> FINALIZED
                       |                                           |
                       +-> worker attempt                          +-> batcher
                       |      |                                         |
                       |      +-> completed/data/env: retain             +-> optimizer
                       |      +-> infra: count failure                         |
                       |                    |                                 +-> commit
                       +<--- backoff + retry failed siblings only
```

1. `Controller._rollout_loop` claims one `RolloutGroupWork`. It calls
   `Rollouter.run_group_rollouts` with the original sample and group ID.
2. `MessageEnvRolloutWorker._run_single_rollout` classifies generator failures
   separately from environment failures. It preserves the partial trajectory
   for diagnostics and releases the attempt's session in `finally`.
3. Each worker waits for every sibling before returning. Failed rollouts receive
   no reward or advantage. Successful siblings are scored and their advantages
   are centered using only scored, trainable siblings. The Verifiers worker
   follows the same scoring and retention contract.
4. The controller-side rollouter counts failures from only the siblings executed
   in this attempt. If infrastructure failures remain and the prompt budget
   permits, it chooses a worker again and passes the previous rollouts. The
   worker reconstructs environments only for infrastructure-failed siblings;
   completed and nonretryable siblings are retained. Sibling IDs and sampling
   seeds are unchanged. MessageEnv retry request/session IDs include the retry
   number, isolating late responses and stale routing affinity. Verifiers creates
   fresh trace/session IDs for each invocation.
5. A worker RPC that raises is classified after unwrapping. Its unresolved
   siblings become failed rollouts; previously completed siblings remain intact.
   A failed group RPC counts one failure per unresolved sibling, not one per RPC.
6. Only the terminal group is finalized. `TrainingSampleBuilder` excludes
   infrastructure/data/environment failures, including partial trajectories,
   while allowing valid siblings to train. An all-failed group contributes no
   training samples. The batcher's existing consumed-group accounting carries
   its ID to the optimizer checkpoint transaction. No retry commits a sample,
   consumes a new dataset item, or acquires another active buffer slot.

Keeping the original slot preserves window ordering and backpressure. Requeuing
as a new work item would risk duplicate IDs and early commits. Retries restart
whole failed trajectories because their environment and generator state cannot
be assumed reusable; successful trajectories are retained.

## Budgets, cancellation, and metrics

The prompt budget is local to the pending group call; each sibling gets at most
`1 + max_retries_per_prompt` attempts. A shared rollouter counter covers all
training and validation calls in one controller lifetime. Every failed attempt
counts, including the initial attempt, failures followed by recovery, and the
final exhausted attempt. Counter updates and the budget check contain no await,
so concurrent group coroutines cannot lose increments on the controller's event
loop. The run aborts when the count **exceeds** the configured allowance.
Already-dispatched RPCs may finish during shutdown; no new attempt starts after
observing exhaustion.

`RolloutInfraBudgetExceeded` bypasses the controller's catch-and-drop handler and
validation's exception collection. The existing `Controller.run` shutdown path
closes the buffer and cancels/joins sibling tasks. Cancellation is never counted
as a failure or retried. Shielded worker RPCs remain tracked so `Rollouter.close`
can drain them before stopping worker processes.

The existing step-metric backends receive cumulative `NoReduce` values with
these exact keys, without a reduction suffix:

- `rollout_infra_retries_total`: additional sibling attempts dispatched.
- `rollout_infra_failures_total`: infrastructure-failed sibling attempts.
- `rollout_data_failures_total`: data-failed sibling attempts.
- `rollout_env_failures_total`: environment-failed sibling attempts.

The controller emits these on training steps, validation completion, and before
propagating a budget exception, including when no training batch was produced.
They are controller-lifetime counters; restarting the controller starts a new
budget and counter series. Retry state is not part of the dataloader checkpoint;
uncommitted prompts replay according to the existing checkpoint contract.
Actor-system fatal supervision and generator engine-loop failures still use the
existing hard-failure path; this policy retries recoverable rollout/RPC failures
and does not respawn dead actor meshes.

## Regression tests

```bash
pytest tests/rl/test_rollout_retry.py
```

The tests cover recovery, per-prompt exhaustion, data/environment fail-fast,
concurrent run-budget exhaustion, error unwrapping, cancellation during backoff,
fresh environments and sessions, retained sibling seeds, reward centering,
training exclusion, and propagation through training and validation. The RL CPU
workflow includes this standalone file.
