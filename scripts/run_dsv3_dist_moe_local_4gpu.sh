#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace_dir=$(cd -- "$repo_dir/.." && pwd)
runtime_root="${CODA_LOCAL_RUNTIME_ROOT:-$workspace_dir/pytorch-env}"
dist_moe_root="${CODA_DIST_MOE_ROOT:-$workspace_dir/dist_moe}"
binary_runtime_root="${CODA_BINARY_RUNTIME_ROOT:-$HOME/tmp/coda-gt-local-runtime-gb300/conda}"
binary_overlay="${CODA_BINARY_OVERLAY:-$HOME/tmp/coda-gt-local-runtime-gb300/github-binary-overlay}"
stack_overlay="${CODA_STACK_OVERLAY:-$HOME/tmp/coda-gt-local-runtime-gb300/github-stack-overlay}"
torchao_root="${CODA_TORCHAO_ROOT:-$HOME/tmp/torchao-main}"
python_source_root="${CODA_PYTORCH_SOURCE_ROOT:-$workspace_dir/pytorch}"
runs_root="${CODA_LOCAL_RUNS_ROOT:-$HOME/tmp/coda-gt-local-runs}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-13.0}"
cutlass_python="${CODA_CUTLASS_PYTHON_ROOT:-$binary_runtime_root/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages}"
steps=15
precision=mxfp8
trainer_mode=graph
run_name=""
preflight_only=0
profile=0
deterministic=0
grad_accum_steps=16
local_batch_size=2

require_option_value() {
  if [ "$#" -lt 2 ]; then
    echo "$1 requires a value" >&2
    exit 2
  fi
}

require_positive_integer() {
  case "$2" in
    ''|*[!0-9]*|0)
      echo "$1 must be a positive integer, got '$2'" >&2
      exit 2
      ;;
  esac
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --steps)
      require_option_value "$@"
      steps="$2"
      shift 2
      ;;
    --bf16)
      precision=bf16
      shift
      ;;
    --eager)
      trainer_mode=eager
      shift
      ;;
    --run-name)
      require_option_value "$@"
      run_name="$2"
      shift 2
      ;;
    --grad-accum-steps)
      require_option_value "$@"
      grad_accum_steps="$2"
      shift 2
      ;;
    --local-batch-size)
      require_option_value "$@"
      local_batch_size="$2"
      shift 2
      ;;
    --preflight-only)
      preflight_only=1
      shift
      ;;
    --profile)
      profile=1
      shift
      ;;
    --deterministic)
      deterministic=1
      shift
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 2
      ;;
  esac
done

require_positive_integer --steps "$steps"
require_positive_integer --grad-accum-steps "$grad_accum_steps"
require_positive_integer --local-batch-size "$local_batch_size"
if [ "$grad_accum_steps" -ne 16 ]; then
  echo "the checked-in comparison recipe requires 16 gradient accumulation steps" >&2
  exit 2
fi
if [ "$local_batch_size" -ne 2 ]; then
  echo "the checked-in comparison recipe requires local batch size 2" >&2
  exit 2
fi
if [ "$steps" -lt 15 ]; then
  echo "performance runs require five warmup and ten measured steps" >&2
  exit 2
fi
if [ "$profile" -eq 1 ] && [ "$steps" -lt 16 ]; then
  echo "the step-16 profile requires at least 16 steps" >&2
  exit 2
fi
if [ "$trainer_mode" = eager ]; then
  module_name=deepseek_v3
  config_name="deepseek_v3_16b_dist_moe_${precision}_local_4gpu"
else
  module_name=graph_trainer.deepseek_v3
  config_name="graph_trainer_deepseek_v3_16b_dist_moe_${precision}_local_4gpu"
fi
if [ -z "$run_name" ]; then
  run_name="github-stack-dsv3-16b-$precision-4gpu-$(date +%Y%m%d%H%M%S)"
fi
if [ -n "$(git -C "$repo_dir" ls-files --others --exclude-standard)" ]; then
  echo "refusing to benchmark with untracked TorchTitan source files" >&2
  exit 1
fi
if ! git -C "$python_source_root" diff --quiet HEAD -- || [ -n "$(git -C "$python_source_root" ls-files --others --exclude-standard)" ]; then
  echo "refusing to benchmark with a dirty PyTorch source tree" >&2
  exit 1
fi
run_root="$runs_root/$run_name"
cache_root="${CODA_LOCAL_CACHE_ROOT:-$runs_root/../coda-gt-local-cache/$run_name}"
if [ -e "$run_root/run.log" ]; then
  echo "refusing to overwrite existing run: $run_root/run.log" >&2
  exit 1
fi
mkdir -p "$run_root/output" "$cache_root/triton" "$cache_root/torchinductor"
mkdir -p "$cache_root/hf/datasets" "$cache_root/xdg" "$cache_root/nv"

export VIRTUAL_ENV="$runtime_root"
export PATH="$runtime_root/bin:$cuda_home/bin:/usr/local/bin:/usr/bin"
export PYTHONPATH="$repo_dir/scripts/local_runtime:$repo_dir:$dist_moe_root:$torchao_root:$binary_overlay:$stack_overlay:$cutlass_python"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_HOME="$cuda_home"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$cache_root/hf"
export HF_DATASETS_CACHE="$cache_root/hf/datasets"
export XDG_CACHE_HOME="$cache_root/xdg"
export TRITON_CACHE_DIR="$cache_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
export TORCHINDUCTOR_COMPILE_THREADS=1
export CUTE_DSL_CACHE_DIR="$cache_root/cutlass"
export CUDA_CACHE_PATH="$cache_root/nv"
export NCCL_RAS_ENABLE=0
export NCCL_DEBUG=WARN
export TORCHAO_MXFP8_EXTENSION="$binary_runtime_root/lib/python3.12/site-packages/torchao/_C_mxfp8.cpython-312-aarch64-linux-gnu.so"
export CODA_PYTORCH_SOURCE_ROOT="$python_source_root"
export CODA_TRAINER_MODE="$trainer_mode"
export CODA_PRECISION="$precision"
export CODA_LOCAL_BATCH_SIZE="$local_batch_size"
export CODA_GRAD_ACCUM_STEPS="$grad_accum_steps"

export LD_LIBRARY_PATH="$binary_runtime_root/lib:$cuda_home/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python=("$runtime_root/bin/python")

train=(
  "${python[@]}" -u -m torchtitan.train
  --dump-folder "$run_root/output"
  --module "$module_name"
  --config "$config_name"
  --training.steps "$steps"
  --hf-assets-path "$repo_dir/tests/assets/tokenizer"
  --metrics.log-freq 1
  --metrics.enable-tensorboard
  --comm.trace-buf-size 0
  --debug.seed 42
  --debug.no-print-config
)
if [ "$deterministic" -eq 1 ]; then
  train+=(--debug.deterministic)
fi
profiler=(--profiler.no-enable-profiling)
if [ "$profile" -eq 1 ]; then
  profiler=(
    --profiler.enable-profiling
    --profiler.profile-freq 16
    --profiler.profiler-warmup 0
    --profiler.profiler-active 1
    --profiler.profiler-repeat 1
  )
fi
train+=("${profiler[@]}")
train+=(
  --training.num-tokens-per-train-step "$((4096 * local_batch_size * 4 * grad_accum_steps))"
)
command=(
  "${python[@]}" -m torch.distributed.run
  --no-python
  --standalone
  --nproc-per-node=4
  "--local-ranks-filter=0,1,2,3"
  --role rank
  --tee 3
  "${train[@]}"
)

measurement_protocol=warmup_steps_1-5,measured_steps_6-15,profile=off
if [ "$profile" -eq 1 ]; then
  measurement_protocol=warmup_steps_1-5,measured_steps_6-15,profile_step_16
fi
{
  printf 'source_commit='
  git -C "$repo_dir" rev-parse HEAD
  printf 'source_diff_sha256='
  git -C "$repo_dir" diff HEAD --binary --no-ext-diff | sha256sum | cut -d ' ' -f 1
  printf 'runtime_root=%s\n' "$runtime_root"
  printf 'pytorch_source_commit=%s\n' "$(git -C "$python_source_root" rev-parse HEAD)"
  printf 'dist_moe_source=D120941775@3863026abff6\n'
  printf 'dist_moe_root=%s\n' "$dist_moe_root"
  printf 'cutlass_python_root=%s\n' "$cutlass_python"
  printf 'torchao_commit=%s\n' "$(git -C "$torchao_root" rev-parse HEAD)"
  printf 'cache_root=%s\n' "$cache_root"
  printf 'local_batch_size=%s\n' "$local_batch_size"
  printf 'gradient_accumulation_steps=%s\n' "$grad_accum_steps"
  printf 'deterministic=%s\n' "$deterministic"
  printf 'tokens_per_microbatch_per_dp_rank=%s\n' "$((4096 * local_batch_size))"
  printf 'tokens_per_train_step=%s\n' "$((4096 * local_batch_size * 4 * grad_accum_steps))"
  printf 'fsdp_symmetric_memory=off\n'
  printf 'measurement_protocol=%s\n' "$measurement_protocol"
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} | tee "$run_root/manifest.txt"

"${python[@]}" - <<'PY' | tee -a "$run_root/manifest.txt"
import torch

print(f"torch_version={torch.__version__}")
print(f"torch_git_version={torch.version.git_version}")
print(f"torch_path={torch.__file__}")
PY

"${python[@]}" - <<'PY'
import os

import dist_moe
import flash_attn.cute.interface
import grain.python
import renderers
import torch
import torchao
from torchtitan.components.data import ConcatThenSplitPackingConfig
from torchtitan.components.dist_moe import DistMoeRoutedExperts
from torchtitan.experiments.graph_trainer.deepseek_v3.config_registry import (
    graph_trainer_deepseek_v3_16b_dist_moe_bf16_local_4gpu,
    graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_local_4gpu,
)
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_16b_dist_moe_bf16_local_4gpu,
    deepseek_v3_16b_dist_moe_mxfp8_local_4gpu,
)
from torchtitan.models.common.attention import VarlenInnerAttention

if torch.cuda.device_count() != 4:
    raise RuntimeError(f"expected 4 GPUs, found {torch.cuda.device_count()}")
for device_index in range(4):
    capability = torch.cuda.get_device_capability(device_index)
    if capability != (10, 3):
        raise RuntimeError(f"expected sm_103 GPU {device_index}, found {capability}")
if os.environ["CODA_TRAINER_MODE"] == "graph":
    config = (
        graph_trainer_deepseek_v3_16b_dist_moe_bf16_local_4gpu()
        if os.environ["CODA_PRECISION"] == "bf16"
        else graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_local_4gpu()
    )
    if config.compile.inductor_compilation != "regional":
        raise RuntimeError("the comparison recipe must use regional Inductor")
    if config.compile.memory_policy != "none":
        raise RuntimeError("the comparison recipe must disable graph rematerialization")
    if config.compile.fsdp_param_unshard_mode != "extracted_in_schedule_stage":
        raise RuntimeError("the comparison recipe must extract FSDP unshard")
    if config.compile.fsdp_gradient_sync_mode != "deferred_as_schedule_stage":
        raise RuntimeError("the comparison recipe must defer FSDP gradient sync")
    if config.compile.gradient_accumulation_mode != "in_graph":
        raise RuntimeError("the comparison recipe must accumulate gradients in graph")
else:
    config = (
        deepseek_v3_16b_dist_moe_bf16_local_4gpu()
        if os.environ["CODA_PRECISION"] == "bf16"
        else deepseek_v3_16b_dist_moe_mxfp8_local_4gpu()
    )
if config.activation_checkpoint is not None:
    raise RuntimeError("the comparison recipe must disable activation checkpointing")
if config.parallelism.fsdp_symm_mem_scope is not None:
    raise RuntimeError("the comparison recipe must disable FSDP symmetric memory")
if config.parallelism.fsdp_reshard_after_forward != "never":
    raise RuntimeError("the comparison recipe must not reshard after forward")
expected_parallelism = {
    "data_parallel_replicate_degree": 1,
    "data_parallel_shard_degree": 4,
    "tensor_parallel_degree": 1,
    "context_parallel_degree": 1,
    "pipeline_parallel_degree": 1,
    "expert_parallel_degree": 4,
}
for field, expected in expected_parallelism.items():
    actual = getattr(config.parallelism, field)
    if actual != expected:
        raise RuntimeError(
            f"the comparison recipe requires {field}={expected}, got {actual}"
        )
if config.training.disable_cuda_graphs:
    raise RuntimeError("the comparison recipe must enable CUDA graphs")
if not isinstance(config.dataloader.dataset, ConcatThenSplitPackingConfig):
    raise RuntimeError("the comparison recipe must use concat packing")
if config.dataloader.dataset.mask_document_boundaries:
    raise RuntimeError("the comparison recipe must use complete fixed rows")
if config.dataloader.max_num_documents != 2:
    raise RuntimeError("the comparison recipe must expose two fixed rows")
inner_attentions = list(config.model.traverse(VarlenInnerAttention.Config))
if not inner_attentions or any(
    not inner_attention.fixed_length_rows
    for _, inner_attention, _, _ in inner_attentions
):
    raise RuntimeError("the comparison recipe must use fixed-row dense attention")
experts = list(config.model.traverse(DistMoeRoutedExperts.Config))
if not experts or any(
    expert.backend.device_scratch_capacity_factor != 1.0
    for _, expert, _, _ in experts
):
    raise RuntimeError("the comparison recipe must use DistMoE scratch factor 1")
expected_microbatch_tokens = 4096 * int(os.environ["CODA_LOCAL_BATCH_SIZE"])
expected_train_step_tokens = (
    expected_microbatch_tokens * 4 * int(os.environ["CODA_GRAD_ACCUM_STEPS"])
)
if config.training.num_tokens_per_microbatch_per_dp_rank != expected_microbatch_tokens:
    raise RuntimeError("the checked-in recipe and requested local batch size disagree")
if config.training.num_tokens_per_train_step != expected_microbatch_tokens * 4 * 16:
    raise RuntimeError("the checked-in comparison recipe must default to GA16")
print(
    f"effective_microbatch_tokens={expected_microbatch_tokens}",
    f"effective_train_step_tokens={expected_train_step_tokens}",
)
print(torch.__version__, torchao.__file__, dist_moe.__file__, "local runtime OK")
PY

if [ "$preflight_only" -eq 1 ]; then
  exit 0
fi

cd "$repo_dir"
"${command[@]}" 2>&1 | tee "$run_root/run.log"
