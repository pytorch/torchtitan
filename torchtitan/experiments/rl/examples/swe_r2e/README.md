# SWE R2E coding-agent RL with a pluggable harness (Claude Code)

Post-train a Qwen model on R2E-Gym SWE tasks where the rollout is driven by an
**unmodified agentic CLI harness** (Claude Code first) running inside a cloud
sandbox (Daytona). Every model turn the agent makes is captured as on-policy
training data, graded by the R2E hidden tests, and fed to GRPO.

This is the TorchTitan analogue of THUDM/slime's `examples/coding_agent_rl` and
Meta msl/rl's "virtual actor + reverse-proxy" pattern.

## How a rollout works

```
RLTrainer (controller, one asyncio loop)
  SWER2ERollouter.run_group_rollouts(generate_fn, sample, group_size=K)
    AnthropicAdapter  <- one HTTP server (127.0.0.1:SHIM_PORT) backed by generate_fn
    per sibling (K):
      boot_agent_sandbox(image)         Daytona sandbox + install node22 + claude
      run_claude_code(... adapter_url)  claude -p, ANTHROPIC_BASE_URL -> adapter
         claude (in sandbox) --HTTP--> DaytonaBridge (fs file-relay) --> AnthropicAdapter
            adapter: render_ids / bridge_to_next_turn (TITO) -> generate_fn -> Completion
            adapter records CapturedTurn(prompt_ids, completion_ids, logprobs) per turn
      git_diff(tracked_only)            capture the agent's patch
      evaluate_r2e(image, diff, r2e)    fresh sandbox: apply diff, run hidden tests
      finish_session -> CapturedTurns -> RolloutTurns (reward on last turn env_rewards)
  score (RewardR2E) -> advantage (GRPO) -> rollout_to_episodes -> Batcher -> backward
```

The key seam is `harness/anthropic_adapter.py`: it speaks the agent's Anthropic
Messages wire format, but renders/generates with TorchTitan's own
`renderers.Renderer` + `generate_fn`. Because it reuses prior turns' exact sampled
tokens via `bridge_to_next_turn` (Token-In-Token-Out), each turn's prompt exactly
extends `prev_prompt + prev_completion`, so a whole multi-turn trajectory packs
into ONE training episode (assistant tokens trained, prompt/tool-result tokens
masked). A Claude Code auto-compaction breaks the prefix and opens a new episode
branch, exactly as `rollout_to_episodes` expects.

Swapping Claude Code for another CLI agent (Codex/OpenCode/...) is a new run
command (point its provider base URL at the adapter); the adapter, sandbox,
grading, and training path are agent-agnostic.

## Layout

`harness/` (shared, agent-agnostic) is split along three orthogonal axes -- WHERE
code runs, HOW the model is served, WHICH agent runs -- mirroring Meta msl/rl's
container-resource / wire-proxy / per-agent split:

- `harness/sandbox/`: `base.py` (the `Sandbox` contract + `make_sandbox` factory),
  `daytona.py` / `docker.py` (backends), `bridge.py` (Daytona fs file-relay).
- `harness/adapters/`: `anthropic.py` (token-capturing Anthropic Messages
  endpoint). Add an `openai.py` for Codex/OpenCode -- the token capture is shared.
- `harness/agents/`: `claude_code.py` (install toolchain + run `claude -p` +
  capture diff). Add a sibling per new CLI agent.

So a new CLI agent = a new `agents/` runner (+ reuse/extend an `adapters/` wire
module); a new sandbox provider = a new `sandbox/` backend.

- `examples/swe_r2e/` (this folder): `data.py` (R2E JSONL dataset), `grading.py`
  (R2E test-pass scoring), `rubric.py`, `rollouter.py`, `config_registry.py`,
  `env.py` (placeholder), `run_swe_r2e_daytona.sh`, `local_smoke_harness.py`.

## Run

Prereqs: `DAYTONA_API_KEY` exported; `daytona` installed (`uv pip install daytona`);
the TorchTitan RL dev env active (`TT_VENV_ENV` -> your venv setup script); an R2E
JSONL (`PROMPT_DATA`, built with `local_smoke/r2e_to_slime.py`); and the model's HF
weights (`HF_ASSETS_PATH`). The Claude Code binary is downloaded inside the sandbox
from its CDN (override via `SWE_CLAUDE_CDN`), so no host toolchain tarball is needed.

Fastest full-pipeline smoke (2 GPUs, 1 task x 2 samples -> backward):

```bash
DAYTONA_API_KEY=dtn_... \
  CONFIG=rl_grpo_qwen3_1_7b_swe_r2e \
  PROMPT_DATA=/path/to/r2e.jsonl \
  HF_ASSETS_PATH=/path/to/Qwen3-1.7B \
  bash torchtitan/experiments/rl/examples/swe_r2e/run_swe_r2e_daytona.sh
```

Target model (6 GPUs): `CONFIG=rl_grpo_qwen3_8b_swe_r2e`, `HF_ASSETS_PATH=/path/to/Qwen3-8B`.

The launcher clears orphaned Daytona sandboxes at startup and (via a trap) on exit;
every sandbox also gets a cloud-side auto-delete TTL so orphans self-reap even if
the job is SIGKILL'd (e.g. MAST preemption). See `daytona_cleanup.py`.

Isolated harness debug (one sandbox, plain vLLM, no trainer):

```bash
PROMPT_DATA=.../easy_one.jsonl DAYTONA_API_KEY=dtn_... VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python -m torchtitan.experiments.rl.examples.swe_r2e.local_smoke_harness
```

## Knobs that matter

| Var | Why |
| --- | --- |
| model `max_seq_len` (config) | == vLLM `max_model_len`. Claude Code's system + tool-schema prompt is 15k+ tokens; the qwen3 spec default 4096 truncates it to an empty generation. The configs raise it to 24576 via `_set_max_seq_len`. |
| `SWE_MAX_CONTEXT_LEN` | Adapter per-turn budget; must stay under `max_seq_len` with room for one generation. |
| `seq_len` (batcher) | Must exceed a full episode (`SWE_MAX_CONTEXT_LEN` + a turn's gen) or the episode is dropped during packing. |
| `global_batch_size` | `num_tokens_target = global_batch_size * seq_len`; the controller boots rollout groups until met. Keep small for a smoke (1) so it runs ONE group. |
| `SWE_CLAUDE_EXTRA_ARGS` | Disallows `Task`/`Agent` sub-agents (they run in separate git worktrees whose edits `git_diff` can't see -> reward-always-0) and useless headless tools. |

## What to expect (honest)

Small models on a tight context score ~reward 0 on real R2E one-shot (zero
advantage -> zero gradient), so a smoke proves the **pipeline** runs end to end,
not a high score. Verified 2026-06-21: `rl_grpo_qwen3_1_7b_swe_r2e` ran 2 Claude
Code rollouts in Daytona (26 / 5 turns), graded them, and completed GRPO Train
Step 1 (`timing/step ~57s`). Meaningful reward needs a bigger model
(8B / 30B-A3B), a larger context (~48k like slime), and many steps.

## Models

`rl_grpo_qwen3_8b_swe_r2e` (target) and `rl_grpo_qwen3_1_7b_swe_r2e` (fast smoke)
are TorchTitan-registered with HF weights on disk. `rl_grpo_qwen3_30b_a3b_swe_r2e`
(MoE) is the scale-up and needs the Qwen3-30B-A3B HF weights downloaded first.
NOTE the slime recipe's Qwen3.6-35B-A3B is a different (Megatron-only) model and
is not in TorchTitan's qwen3 registry; 30B-A3B is the closest TorchTitan MoE.
