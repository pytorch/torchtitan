# Terminal-Bench Pattern 2

This example keeps the multi-turn model/tool loop in TorchTitan. `TerminalBenchEnv`
owns the work sandbox and terminal tool calls. After the rollout, it exports the
task artifacts and closes the work sandbox. `TerminalBenchVerifier` then creates
a fresh verifier sandbox, imports those artifacts, and runs the task tests. The
rubric only converts the returned verifier signal into the final rollout reward.

The first version supports CPU, single-container tasks with separate verifiers.
It follows Terminal-Bench's public-network default for both environments and does
not expose a separate network-policy option. Docker is the local backend; Daytona
is the remote backend. Both implement the same `SandboxClient` contract.

The Daytona extra is pinned to the last public OSS release, `v0.190.0`. That
repository is no longer maintained, so this backend is experimental rather
than the default TorchTitan sandbox.

Install the backend selected for the run:

```bash
pip install -e '.[sandbox-docker]'
# or
pip install -e '.[sandbox-daytona]'
```

Build the two OCI images for a task before starting TorchTitan:

```bash
python -m torchtitan.experiments.rl.examples.terminal_bench.build_images \
  --tasks-dir /path/to/terminal-bench/tasks \
  cli-2ph-simplex
```

For Daytona, use a registry prefix that the remote service can pull:

```bash
python -m torchtitan.experiments.rl.examples.terminal_bench.build_images \
  --tasks-dir /path/to/terminal-bench/tasks \
  --image-prefix ghcr.io/example/torchtitan-terminal-bench \
  cli-2ph-simplex
```

Push those images, then configure `TerminalBenchDataset.Config.image_prefix`
with the same prefix and set `TerminalBenchRollouter.Config.sandbox_client` to
`DaytonaSandboxClient.Config()`.

## End-to-end smoke run

Download the Qwen3-0.6B assets, build the task images, and run the one-step
two-GPU recipe:

```bash
python scripts/download_hf_assets.py \
  --repo_id Qwen/Qwen3-0.6B \
  --local_dir torchtitan/experiments/rl/example_checkpoint \
  --assets tokenizer safetensors index config

python -m torchtitan.experiments.rl.train \
  --module terminal_bench \
  --config rl_grpo_qwen3_0_6b_terminal_bench_smoke \
  --rollouter.train-dataset.tasks-dir /path/to/terminal-bench/tasks \
  --rollouter.validation-dataset.tasks-dir /path/to/terminal-bench/tasks \
  --metrics.no-enable-wandb
```

The smoke recipe intentionally uses one rollout, so its group-relative
advantage is zero. It verifies the complete controller, generator, work
sandbox, verifier sandbox, rubric, trainer, checkpoint, and cleanup path; it is
not a useful training recipe.
