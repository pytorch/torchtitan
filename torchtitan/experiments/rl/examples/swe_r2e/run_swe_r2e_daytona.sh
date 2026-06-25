#!/usr/bin/env bash
# Single-node SWE R2E coding-agent (Claude Code) RL on Daytona cloud sandboxes.
#
# Mirrors slime's run_qwen3_*_swe_r2e_daytona.sh but drives TorchTitan's RL loop:
# the policy is served by the vLLM generator behind an on-box Anthropic adapter;
# Claude Code runs in a Daytona sandbox per rollout, reaches the adapter via the
# file-relay bridge, and every turn is captured as on-policy training tokens.
#
#   CONFIG=rl_grpo_qwen3_1_7b_swe_r2e  bash run_swe_r2e_daytona.sh   # fast smoke (2 GPUs)
#   CONFIG=rl_grpo_qwen3_8b_swe_r2e    bash run_swe_r2e_daytona.sh   # target  (6 GPUs)
#
# Prereqs: DAYTONA_API_KEY exported; HF weights on disk at the config's
# hf_assets_path. The Claude Code binary is downloaded inside the sandbox from its
# CDN (override via SWE_CLAUDE_CDN), so no host toolchain tarballs are needed.
# NOTE: no `set -u` -- .venv_rl_env.sh appends to possibly-unset LD_LIBRARY_PATH.
set -ex
export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:-rl_grpo_qwen3_1_7b_swe_r2e}"

if [[ -z "${DAYTONA_API_KEY:-}" ]]; then
  echo "ERROR: DAYTONA_API_KEY is not set (app.daytona.io key, dtn_...)." >&2
  exit 1
fi

# ---- TorchTitan RL dev env (venv + CUDA libs); machine-specific, set TT_VENV_ENV ----
TT_VENV_ENV="${TT_VENV_ENV:-${HOME}/torchtitan/.venv_rl_env.sh}"
[[ -f "${TT_VENV_ENV}" ]] && source "${TT_VENV_ENV}"

# Repo root = the dir 5 levels up from this script
# (<repo>/torchtitan/experiments/rl/examples/swe_r2e/run_swe_r2e_daytona.sh).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
WORKTREE="${WORKTREE:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
export PYTHONPATH="${WORKTREE}:${PYTHONPATH:-}"
cd "${WORKTREE}"

# ---- Daytona sandbox backend (small per-sandbox quota) ----
export TT_SANDBOX_BACKEND="daytona"
export DAYTONA_API_KEY
export TT_DAYTONA_CPU="${TT_DAYTONA_CPU:-2}"
export TT_DAYTONA_MEM_GB="${TT_DAYTONA_MEM_GB:-2}"
export TT_DAYTONA_DISK_GB="${TT_DAYTONA_DISK_GB:-5}"
export TT_DAYTONA_CREATE_TIMEOUT="${TT_DAYTONA_CREATE_TIMEOUT:-900}"

# ---- R2E dataset (JSONL); read by config_registry as SWE_PROMPT_DATA ----
export SWE_PROMPT_DATA="${PROMPT_DATA:-${SWE_PROMPT_DATA:-}}"
if [[ -z "${SWE_PROMPT_DATA}" ]]; then
  echo "ERROR: set PROMPT_DATA to your R2E JSONL (build one with prepare_r2e_data.py)." >&2
  exit 1
fi

# ---- agent + adapter knobs ----
# SWE_MAX_CONTEXT_LEN must stay under the config's model max_seq_len (24576 for the
# smokes) with room for a turn's generation; the adapter caps per-turn max_tokens
# at (SWE_MAX_CONTEXT_LEN - prompt_len).
MAX_CONTEXT_LEN="${SWE_MAX_CONTEXT_LEN:-20480}"
AUTO_COMPACT_WINDOW="${AUTO_COMPACT_WINDOW:-16000}"
export SWE_MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN}"
export SHIM_BIND_HOST="${SHIM_BIND_HOST:-127.0.0.1}"
export SHIM_PORT="${SHIM_PORT:-18031}"
export SWE_TIME_BUDGET_SEC="${SWE_TIME_BUDGET_SEC:-900}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-400}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-2}"
# Claude Code's own stream-json agent trace (every Read/Edit/Bash/tool turn).
export SWE_TRAJECTORY_DUMP_DIR="${SWE_TRAJECTORY_DUMP_DIR:-${WORKTREE}/torchtitan/experiments/rl/examples/swe_r2e/trajectories}"
# Training-view trace: per-rollout decoded model completions + token lengths +
# reward + diff (what the model actually generated and trains on each turn).
export SWE_ROLLOUT_DUMP_DIR="${SWE_ROLLOUT_DUMP_DIR:-${WORKTREE}/torchtitan/experiments/rl/examples/swe_r2e/rollout_dumps}"
# Disallow sub-agents (Task/Agent): they run in separate git worktrees whose edits
# are invisible to git_diff (reward-always-0 bug). WebFetch/WebSearch/AskUserQuestion
# are useless in headless -p mode. See slime LESSONS.md.
SETTINGS_JSON="{\"permissions\":{\"defaultMode\":\"bypassPermissions\"},\"autoCompactEnabled\":true,\"autoCompactWindow\":${AUTO_COMPACT_WINDOW}}"
export SWE_CLAUDE_EXTRA_ARGS="${SWE_CLAUDE_EXTRA_ARGS:---settings '${SETTINGS_JSON}' --disable-slash-commands --disallowedTools WebFetch WebSearch Task Agent AskUserQuestion}"

# ---- proxy: daytona SDK (HTTP) needs fwdproxy; local NCCL/monarch loopback must not ----
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},$(hostname)"
export NO_PROXY="${no_proxy}"

mkdir -p "${SWE_TRAJECTORY_DUMP_DIR}"

# Orphaned sandboxes self-reap via each sandbox's cloud-side auto-delete TTL
# (TT_DAYTONA_AUTO_STOP_MIN / TT_DAYTONA_AUTO_DELETE_MIN), so no explicit cleanup is needed.

# hf_assets_path defaults to example_checkpoint/Qwen3-<size>; override with the HF
# weights on disk via HF_ASSETS_PATH (passed only when set).
python3 -m torchtitan.experiments.rl.train \
  --module swe_r2e --config "${CONFIG}" \
  --metrics.no-enable-wandb \
  ${HF_ASSETS_PATH:+--hf_assets_path "${HF_ASSETS_PATH}"} \
  "$@"
