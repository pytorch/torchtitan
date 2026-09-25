# Terminal-Bench with Verifiers and TitanRL

This experiment trains a Qwen3.5-9B terminal agent on a frozen Harbor-format task
tree, and evaluates it on all 89 Terminal-Bench 2.1 tasks. TitanRL schedules
rollout groups, generates tokens, and trains the model; Verifiers 0.3.1 runs the
Terminus-2 agent and the Harbor task verifier. It is an experiment, not a new
TitanRL rollout or sandbox backend.

This experiment uses the existing TitanRL and Verifiers bridge in
`torchtitan/rl/examples/verifiers/` on `main`. Only the task-specific adapter,
recipe, and tests live here; TitanRL's core is unchanged.

## Prepare immutable task trees

Install the TorchTitan RL dependencies and this example's `requirements.txt` in
the same Python 3.12 environment. Verifiers 0.3.1 requires MCP 1.x; use a
separate environment if another project has installed MCP 2.x. Install Docker
on the controller host and make sure the user can run Docker containers. The
agent runs **inside** the task's container; `SubprocessConfig` must not be used
for untrusted terminal tasks. Verifiers pulls the task's declared image and
uploads grading files only after the agent finishes. The selected image must
have `tmux` installed.
Each task must either declare a pullable `[environment].docker_image` or have
an entry in a complete image-override manifest. A task containing only a
Dockerfile is rejected without an override rather than silently evaluated in
the wrong environment.

Use a pinned copy of the published Terminal-Bench 2.1 Harbor task tree, not a
generated JSONL or a directory of task evolution outputs. It should contain
89 subdirectories with `task.toml`, `instruction.md`, and `tests/test.sh`.
The task loader accepts either a repository root with a `tasks/` subdirectory
or the `tasks/` directory itself. Record the revision of the task tree alongside
the model checkpoint; the 2.0 and 2.1 benchmarks are not interchangeable.

For training, point at a separate, frozen Harbor-format corpus such as a
published TerminalWorld/SWE-Smith selection. It must likewise contain task
directories with verifiers and prebuilt, pullable images (declared directly
or through the image manifest). This example does not fetch, filter, evolve,
or publish training examples. A task-tree path is a
local filesystem path on the TitanRL controller host (also visible to the
Verifiers environment-server processes).

The colleague's benchmark uses derived images that install and verify `tmux`
in every task environment; the published TB2.1 images do not guarantee that
prerequisite. Build and pin equivalent images before running this recipe.
The optional `TERMINAL_BENCH_EVAL_IMAGES` and `TERMINAL_BENCH_TRAIN_IMAGES`
files are JSON objects mapping every selected task-directory name to its
pullable, `tmux`-ready image, for example:

```json
{"fix-git": "registry.example.com/terminal-bench/fix-git:verified"}
```

The mapping leaves the public task tree untouched. When provided, every task
must have an entry; missing entries fail at dataset load. A prepared image
must retain the published task filesystem and working directory, adding only
the agent runtime prerequisites. If an image lacks `tmux`, task setup reports
it before attempting an agent rollout. Record image digests and both task
tree revisions for numerical comparisons.

```bash
export TERMINAL_BENCH_TRAIN_TASKS_ROOT=/path/to/frozen/train/tasks
export TERMINAL_BENCH_EVAL_TASKS_ROOT=/path/to/terminal-bench-2-1/tasks
export TERMINAL_BENCH_TRAIN_IMAGES=/path/to/train-images.json
export TERMINAL_BENCH_EVAL_IMAGES=/path/to/terminal-bench-2.1-images.json

python -m torchtitan.rl.train \
  --module torchtitan.rl.experiments.verifiers.terminal_bench \
  --config rl_grpo_qwen35_9b_terminal_bench \
  --hf_assets_path /path/to/Qwen3.5-9B
```

The training and evaluation paths must be different. The recipe uses 8 trainer
GPUs (FSDP) and 8 one-GPU generator replicas by default (16 GPUs total);
adjust GPU counts for the available hosts. TitanRL's CLI defaults to a single
host: placing these meshes across hosts requires a caller-provided `HostMeshes`
launcher. Mainline TitanRL cannot configure eight independent vLLM DP replicas
inside one generator without expert parallelism, so the generator layout is not
identical to the colleague branch's single 8-DP generator. Keep the selected
task IDs disjoint as well. It uses a
65,536-token model/trainer context, up to 16,384
generated tokens per turn, 120 agent turns, 32 siblings per group, 8 groups
per training step, 4 target off-policy steps, 100 training steps, and a 1e-6
constant learning rate. It keeps fp32 master weights/Adam states with bf16
FSDP compute, full activation checkpointing, and 20-step DCP saves. Historical
thinking is retained across turns. Verifiers uses the
Terminus-2 scaffold with Harbor 0.22.0 and stages `tests/` for grading in
the agent's own container. The binary task reward flows through TitanRL's
ordinary advantage and GRPO training path. Its XML parser, disabled context
summarization, and 120-turn limit match the colleague's Terminus-2 setup.
The example configures these three options on Verifiers' upstream program;
Terminus-2 itself is not forked. Timeouts are 7,200 seconds for the
agent and 12,000 seconds for scoring; the task's authored timeouts are ignored
to avoid prematurely cutting off slow inference.
The task adapter extracts the image's final `WORKDIR` from its Dockerfile when
Harbor metadata does not specify one; 3 of the 89 TB2.1 tasks use paths other
than `/app`. Test fixtures are unpacked without restoring the host UID so
rootless Docker/Podman can run the same in-container grader.

## Evaluate a checkpoint

```bash
export TERMINAL_BENCH_EVAL_TASKS_ROOT=/path/to/terminal-bench-2-1/tasks
export TERMINAL_BENCH_EVAL_IMAGES=/path/to/terminal-bench-2.1-images.json
export TERMINAL_BENCH_CHECKPOINT=/path/to/dcp/checkpoint

python -m torchtitan.rl.train \
  --module torchtitan.rl.experiments.verifiers.terminal_bench \
  --config rl_grpo_qwen35_9b_terminal_bench_eval \
  --hf_assets_path /path/to/Qwen3.5-9B
```

This sets `num_training_steps=0`. The experiment's eval-only controller
executes one greedy pass@1 over the 89 tasks without starting training rollout
producers. `TERMINAL_BENCH_CHECKPOINT` selects a TorchTitan DCP
model checkpoint; without it the recipe evaluates the initial HF weights.
Run with a fresh output directory for each benchmark attempt.

## Fidelity and open dependencies

The example preserves the colleague branch's Qwen3.5-9B model choice,
Terminus-2 v0.22 scaffold, 120-turn budget, Harbor task instructions, isolated
per-task environment, in-place verifier, and 0/1 reward. Verifiers reads
binary grading fixtures directly from the task tree; it does not need the
branch's base64-encoded JSONL conversion.

These parts do **not** currently reproduce that branch's measured run:

- Verifiers 0.3.1 supports Docker and Prime runtimes but not Daytona. A
  Daytona backend is a TODO for the Verifiers repository; integrate and test it
  separately before comparing sandbox resource limits or rollout throughput.
- Dockerfile-only tasks need prebuilt images and a complete image manifest;
  Verifiers does not build task Dockerfiles. The public TB2.1 images also need
  a verified `tmux` runtime before the XML agent can run.
- Verifiers' built-in Terminus-2 config does not expose the XML parser and
  summary policy. `harness.py` configures the upstream program for this
  experiment; upstreaming those knobs to Verifiers would remove the adapter.
- Verifiers' Harbor fixture staging restores host file ownership, which fails
  in a rootless container with unmapped host IDs. `data.py` keeps the upstream
  grading contract but extracts test files with `--no-same-owner`. This should
  eventually be fixed in Verifiers' shared Harbor task implementation.
- Mainline TitanRL supports GRPO, while the colleague's production recipe uses
  DPPO with a different trust-region loss and dedicated async scheduling.
  Mainline also lacks its per-group KV cache salting on weight sync, independent
  vLLM DP for this dense model, dedicated async validation generators, and
  mid-training periodic validation. Do not add experiment-only branches to
  TitanRL core to emulate these features.
- TitanRL's current validation reports greedy pass@1. The colleague's
  Terminal-Bench runs also report pass@5; these metrics cannot be compared
  directly without a validated multi-sample evaluation path.
- Container runtime details, agent transcripts, and model/token-level
  numerics still need a live Docker + GPU comparison against a pinned
  colleague-branch checkpoint and the same frozen 2.1 task revision. This
  example does not claim identical reward curves or loss without that run.

Only the recipe, task adapter, and CPU checks belong to this upstream change.
The colleague branch's evolution loop, curated task outputs, runbooks, logs,
and training-data artifacts are intentionally out of scope.
