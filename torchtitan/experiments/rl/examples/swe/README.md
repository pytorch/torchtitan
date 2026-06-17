# SWE coding-agent RL (R2E-Gym)

Multi-turn RL for a software-engineering agent: the model drives a `bash` /
`submit` tool loop inside a per-instance container sandbox, edits the repo to fix
a real GitHub issue, and is rewarded by running that instance's hidden tests.

This is a torchtitan-native port of slime's `coding_agent_rl` R2E-Gym setup. The
key difference: **the env is the agent loop**. There is no embedded coding-agent
CLI -- the policy model itself emits tool calls (parsed by the renderer), and the
env executes them. That maps cleanly onto torchtitan's `MessageEnv` / `TokenEnv`
multi-turn machinery and keeps the whole rollout on-policy.

## Layout

The sandbox layer is task-agnostic and lives outside this example, in
`torchtitan/experiments/rl/sandbox/` (`Sandbox` protocol + `ExecResult` in
`base.py`; abstract `SandboxFactory` backend seam; `DockerSandbox` /
`DockerSandboxFactory` in `docker.py`). Any "run commands in an isolated env" RL
task reuses it. This example is the task-specific layer on top:

| file | role |
| --- | --- |
| `data.py` | `R2EGymSample` + `R2EGymDataset` (jsonl stream) |
| `env.py` | `SweEnv(MessageEnv)`: `bash`/`submit` agent loop; grades in the live sandbox on the terminal step |
| `grading.py` | R2E-Gym grading: inject hidden tests, run pytest (junit), compare to `expected_output_json` |
| `rubric.py` | `RewardR2EGym(RewardFn)`: thin reader of the env's grade |
| `rollouter.py` | `SweRollouter(Rollouter)`: wires it; injects one shared `SandboxFactory` |
| `config_registry.py` | `rl_swe_r2e_qwen3_8b`, `rl_swe_r2e_qwen3_1_7b` |

## Why grading happens inside the env

The rollouter tears every env (and its sandbox) down *before* scoring, so a live
sandbox only exists during a rollout's steps. Grading therefore runs on the
terminal env step and is published through `MessageEnvStepOutput.env_rewards`;
`RewardR2EGym` only reads it back (it never re-runs tests).

## Anti-cheat

- **Hidden tests injected fresh at grade time**, overwriting anything the agent
  wrote at those paths -- the agent cannot tamper with the grading tests, only
  influence the score via its source edits.
- **No egress during the rollout**: containers run `--network none` (R2E-Gym
  images have all deps baked in, so the tests need no network).
- **No answer leak**: `expected_output_json` and the hidden test code live in the
  dataset, never in the image and never shown to the agent.

## Prerequisites

1. **podman** (or docker) on the box.
2. **R2E-Gym images** cached locally. The bundled smoke set
   (`data/r2e_smoke.jsonl`) uses two `orange3_final` instances; pull on first use
   happens automatically, or pre-pull:
   ```bash
   podman pull docker.io/namanjain12/orange3_final:2d9617bd0cb1f0ba61771258410ab8fae8e7e24d
   ```
3. The Qwen3 checkpoint under `torchtitan/experiments/rl/example_checkpoint/`.

### Building a larger dataset

The smoke jsonl is produced by slime's converter (R2E-Gym HF rows -> one row per
line). Point `train_dataset.data_path` at a bigger jsonl with the same schema:
```
{"prompt", "label", "metadata": {"instance_id", "image", "workdir",
 "problem_statement", "r2e": {"test_file_names", "test_file_codes",
 "expected_output_json"}}}
```

## Run

```bash
source /home/yichuan/torchtitan/.venv_rl_env.sh
unset HTTPS_PROXY HTTP_PROXY   # keep NCCL/monarch off the proxy

python -m torchtitan.experiments.rl.train \
    --module torchtitan.experiments.rl.examples.swe \
    --config rl_swe_r2e_qwen3_8b --metrics.no-enable-wandb
```

8 GPUs: 2 generator (TP=2) + 4 trainer (TP=4); sandboxes run on CPU via podman.
Use `rl_swe_r2e_qwen3_1_7b` for faster pipeline iteration.

## Validate without GPUs

```bash
# unit tests (fake in-process sandbox; no podman, no GPU)
pytest torchtitan/experiments/rl/tests/test_swe.py -x

# real sandbox + grading smoke against an R2E-Gym container (podman, no GPU)
python -m torchtitan.experiments.rl.examples.swe.smoke_check
```

## Cleanup

Sandboxes are removed when each rollout ends. To reap containers leaked by a
hard-killed run:
```bash
podman rm -f $(podman ps -aq --filter label=ttrl-sandbox)
```

## What to expect

Small models (1.7B/8B) score ~0 reward on one-shot R2E-Gym (so advantage and
gradient are ~0). That is expected -- this example proves the **pipeline** runs
end to end on a local sandbox; meaningful reward curves need a larger model, many
instances, and long training. See slime's `RUNBOOK_R2E.md` for the same caveat.

## Scaling knobs

| knob | where | note |
| --- | --- | --- |
| `max_turns` | `SweEnv.Config` | agent turn cap; the env grades on the last turn |
| `eval_timeout_s` | `SweEnv.Config` | hidden-test timeout; keep below `token_env.step_timeout_s` |
| `max_rollout_tokens` | `token_env` | must stay below the trainer `seq_len` or the rollout truncates without grading |
| `max_concurrent_provision` | `SandboxFactory.Config` | global cap on simultaneous container boots |
| `network` | `DockerSandboxFactory.Config` | `none` (default) enforces egress lock; `host` only if an eval truly needs network |
```
