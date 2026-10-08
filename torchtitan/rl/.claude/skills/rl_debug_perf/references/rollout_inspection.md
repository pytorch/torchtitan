# Rollout inspection scripts

Snippets for `<dump_folder>/rollout_samples.jsonl` (default `outputs/rl/`) and the job log. Each JSONL line is one rollout written by `RolloutSampleRecorder`. Record every rollout while debugging:

```python
config.rollout_recorder.filter.k = config.async_loop.num_samples_per_prompt  # default 1: each group's best and worst only
config.rollout_recorder.log_tensors = True  # token ids, needed only for the branch check below
```

## Status mix

```python
import collections
import json

rows = [json.loads(line) for line in open("outputs/rl/rollout_samples.jsonl")]
train = [row for row in rows if not row["is_validation"]]

print(collections.Counter(row["status"] for row in train))
# Counter({'completed': 2421, 'truncated_max_turns': 160, 'truncated_length': 138, 'error': 33})

rewards_by_status = collections.defaultdict(list)
for row in train:
    rewards_by_status[row["status"]].append(row["reward"])
for status, rewards in sorted(rewards_by_status.items()):
    print(f"{status:28s} n={len(rewards):5d} mean reward={sum(rewards) / len(rewards):.3f}")
# completed                    n= 2421 mean reward=0.488
# error                        n=   33 mean reward=0.000
# truncated_length             n=  138 mean reward=0.014
# truncated_max_turns          n=  160 mean reward=0.019
```

## Errored rollouts: the final exception

The JSONL has each rollout's status but not its exception; the job log has it. Read the last `Type: message` line of each traceback: wrapper errors often share one type, and the last line (usually with the tool's stderr) names the cause.

```bash
# Native rollouter: one traceback per errored rollout.
grep -A 30 "marking ERROR" <job log> | less

# Verifiers: one "rollout done" line per attempt, with the error class (not the message) in stop=.
grep "rollout done:" <job log> | sed -E 's/.* stop=//' | sort | uniq -c | sort -rn
#    2589 agent_completed
#     188 max_turns
#     156 context_length
#      39 HarnessError
#      24 SandboxError
#       1 TaskError
grep "rollout done:" <job log> | grep -v "stop=\(agent_completed\|max_turns\|context_length\)"

# Whole groups lost (rollout/group_failures), and data dropped with only a warning.
grep "rollout group .* failed; dropping" <job log>
grep -E "Consecutive untrainable batches|Batcher dropped|older than target_offpolicy_steps" <job log>
```

Retried attempts get their own `rollout done` line, so retried sandbox errors show up here but not in the JSONL.

### Once `Rollout.logs` lands

Checkouts with `Rollout.logs` save each failure in the JSONL as `logs["errors"]`, a list of `{type, message, traceback}`, and `KeepExtremeRewardsFilter.Config.keep_errors=True` records every errored rollout. The Verifiers rollouter there also logs one line per failed rollout, joinable on `logs["verifiers_trace_id"]`:

```text
Verifiers rollout failed: trace=<id> group=<g> rollout=<r> task=<task> stop=<stop> error=<Type>: <message> | setup=<s>s agent=<s>s (model=<s>s harness=<s>s) scoring=<s>s | model_calls=<n> failed_calls=<n> slowest_call=<s>s | <traceback tail>
```

The timing separates a slow generator from a slow env: `agent=7200s model=6900s` on a timeout is generation, not the task. With `logs`, classify in Python:

```python
import re

EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception)\b.*")


def final_exception(row: dict) -> str:
    """The last `Type: message` line of the rollout's last error: the cause behind a wrapper error."""
    errors = (row.get("logs") or {}).get("errors") or []
    if not errors:
        return "no error record"
    text = "\n".join(filter(None, [errors[-1].get("message"), errors[-1].get("traceback")]))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    exception_lines = [line for line in lines if EXCEPTION_LINE.match(line)]
    return (exception_lines or lines or ["empty error"])[-1]


# Patterns from one terminal-agent env; write your own from the final exceptions you see.
ERROR_CLASSES = [
    ("sandbox infra", r"transport failure|HTTP 50\d|background job was lost|shell was lost"),
    ("agent timeout", r"agent timeout"),
    ("agent invalid input", r"null byte|contains NUL"),
    (
        "agent broke its env",
        r"no server running|error connecting to|return_code=137|running containers"
        r"|fork: (retry: )?Resource temporarily unavailable",
    ),
]


def error_class(row: dict) -> str:
    exception = final_exception(row)
    for name, pattern in ERROR_CLASSES:
        if re.search(pattern, exception):
            return name
    return "other: " + exception[:100]


errored = [row for row in train if row["status"].startswith("error")]
print(collections.Counter(error_class(row) for row in errored))
# Counter({'agent broke its env': 32, 'agent timeout': 1})
```

## Silent zeros and failure classes over time

Reward-0 rollouts with no error status need the same reading. Start with groups that scored 0 in every rollout without an error, and read their last turns and grader output: a missing test dependency or a service killed before grading looks exactly like a wrong answer.

```python
import collections
import json

train = [json.loads(line) for line in open("outputs/rl/rollout_samples.jsonl")]
train = [row for row in train if not row["is_validation"]]

groups = collections.defaultdict(list)
for row in train:
    groups[row["group_id"]].append(row)
silent_zero_groups = [
    group_id
    for group_id, group in groups.items()
    if all(row["reward"] <= 0 and not row["status"].startswith("error") for row in group)
]
print(f"{len(silent_zero_groups)} of {len(groups)} groups scored 0 in every rollout without an error")
# 94 of 344 groups scored 0 in every rollout without an error


def failure_class(row: dict) -> str:
    if row["status"].startswith("error"):
        return "error"  # with Rollout.logs: error_class(row)
    if row["status"].startswith("truncated"):
        return row["status"]
    return "reward_0" if row["reward"] <= 0 else "success"


# Classes by the policy version a rollout started under.
by_version = collections.defaultdict(collections.Counter)
for row in train:
    if row["turns"]:
        by_version[row["turns"][0]["min_policy_version"]][failure_class(row)] += 1
for version, counts in sorted(by_version.items()):
    total = sum(counts.values())
    print(version, {name: f"{count / total:.1%}" for name, count in counts.most_common()})
# 0 {'success': '47.7%', 'reward_0': '43.7%', 'truncated_max_turns': '4.6%', 'truncated_length': '3.7%', 'error': '0.2%'}
```

## Where a branched rollout diverges

`rollout/branches_per_rollout/mean > 1` in an env that does not edit history on purpose means a turn's prompt did not extend the previous prompt plus completion. Find the first token that differs and decode around it. This needs `log_tensors=True`.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("<hf_assets_path>")


def first_divergence(prev_turn: dict, turn: dict) -> int | None:
    """Index of the first prompt token that does not extend the previous prompt + completion."""
    expected = prev_turn["prompt_token_ids"] + prev_turn["completion_token_ids"]
    prompt = turn["prompt_token_ids"]
    for index, (want, got) in enumerate(zip(expected, prompt)):
        if want != got:
            return index
    return None if len(prompt) >= len(expected) else len(prompt)


for row in train:
    for prev_turn, turn in zip(row["turns"], row["turns"][1:]):
        index = first_divergence(prev_turn, turn)
        if index is None:
            continue
        expected = prev_turn["prompt_token_ids"] + prev_turn["completion_token_ids"]
        start = max(index - 20, 0)
        print(row["group_id"], row["rollout_id"], "turn", turn["turn_id"], "diverges at token", index)
        print("  expected:", repr(tokenizer.decode(expected[start : index + 20])))
        print("  got:     ", repr(tokenizer.decode(turn["prompt_token_ids"][start : index + 20])))
```

If the recorder never kept a branched rollout, reproduce it on CPU: run the env with its real libraries and a scripted generate function that returns the suspicious completion, e.g. one that hits `max_tokens` right after closing its thinking block.

## Same tasks across two runs

A training group's `group_id` is its index in the dataset stream. With a seeded dataset order, the same `group_id` is the same prompt in both runs (198 of 198 shared groups in two same-seed runs), so compare reward per group instead of per step. This needs every rollout recorded. A resumed run restarts the stream at 0 and appends to the same JSONL, so split it at the restart first.

```python
import collections
import json
import statistics


def mean_reward_by_group(path: str) -> dict[int, float]:
    rewards = collections.defaultdict(list)
    for line in open(path):
        row = json.loads(line)
        if not row["is_validation"]:
            rewards[row["group_id"]].append(row["reward"])
    return {group_id: statistics.mean(values) for group_id, values in rewards.items()}


run_a = mean_reward_by_group("run_a/rollout_samples.jsonl")
run_b = mean_reward_by_group("run_b/rollout_samples.jsonl")
shared = sorted(run_a.keys() & run_b.keys())
deltas = [run_b[group_id] - run_a[group_id] for group_id in shared]
print(f"{len(shared)} shared groups, delta {statistics.mean(deltas):+.3f} +- {statistics.stdev(deltas) / len(shared) ** 0.5:.3f} (SE)")
print("r =", statistics.correlation([run_a[g] for g in shared], [run_b[g] for g in shared]))
# 198 shared groups, delta +0.028 +- 0.013 (SE)
# r = 0.889
```
