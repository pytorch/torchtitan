#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)

runtime_root="${CODA_LOCAL_RUNTIME_ROOT:-$HOME/tmp/coda-gt-local-runtime-gb300/conda}"
overlay_root="${CODA_LOCAL_OVERLAY_ROOT:-$HOME/tmp/coda-gt-local-runtime-gb300/github-overlay-min}"
platform_lib="${CODA_LOCAL_PLATFORM_LIB:-/usr/local/fbcode/platform010-aarch64/lib}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-13.0}"
runtime_id=593b9310487595c6
runtime_manifest_sha256=bd6172f58eb609f2f6b77064c07faf35e711e1aa1272d7162e46bae6b7f54c8d
steps=10
deterministic=0
tensorboard=0
profile=0
dry_run=0
preflight_only=0
run_name=""

usage() {
  echo "usage: $0 [--runtime-root PATH] [--overlay-root PATH] [--steps N]" >&2
  echo "          [--run-name NAME] [--deterministic] [--tensorboard] [--profile]" >&2
  echo "          [--preflight-only] [--dry-run]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --runtime-root)
      runtime_root="$2"
      shift 2
      ;;
    --overlay-root)
      overlay_root="$2"
      shift 2
      ;;
    --steps)
      steps="$2"
      shift 2
      ;;
    --run-name)
      run_name="$2"
      shift 2
      ;;
    --deterministic)
      deterministic=1
      shift
      ;;
    --tensorboard)
      tensorboard=1
      shift
      ;;
    --profile)
      profile=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --preflight-only)
      preflight_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [ ! -x "$runtime_root/bin/python" ]; then
  echo "runtime Python is missing: $runtime_root/bin/python" >&2
  exit 1
fi
manifest_path="$(dirname "$runtime_root")/RUNTIME_MANIFEST.json"
if [ ! -f "$manifest_path" ]; then
  echo "runtime manifest is missing: $manifest_path" >&2
  exit 1
fi
actual_manifest_sha256=$(sha256sum "$manifest_path" | cut -d ' ' -f 1)
if [ "$actual_manifest_sha256" != "$runtime_manifest_sha256" ]; then
  echo "runtime manifest hash does not match $runtime_id" >&2
  echo "expected: $runtime_manifest_sha256" >&2
  echo "actual:   $actual_manifest_sha256" >&2
  exit 1
fi
if [ ! -d "$overlay_root/grain" ]; then
  echo "Grain runtime overlay is missing: $overlay_root/grain" >&2
  exit 1
fi

loader="$platform_lib/ld-linux-aarch64.so.1"
if [ ! -x "$loader" ]; then
  echo "glibc loader is missing: $loader" >&2
  exit 1
fi

if [ -z "$run_name" ]; then
  suffix=performance
  if [ "$deterministic" -eq 1 ]; then
    suffix=deterministic
  fi
  if [ "$tensorboard" -eq 1 ]; then
    suffix="$suffix-tensorboard"
  fi
  if [ "$profile" -eq 1 ]; then
    suffix="$suffix-profile"
  fi
  run_name="github-dsv3-16b-fsdp4-ep4-ga16-$suffix-$(date +%Y%m%d%H%M%S)"
fi

run_root="$HOME/tmp/coda-gt-local-runs/$run_name"
cache_root="$HOME/tmp/coda-gt-local-cache/$run_name"
if [ -e "$run_root/run.log" ]; then
  echo "refusing to overwrite existing run: $run_root/run.log" >&2
  exit 1
fi

mkdir -p "$run_root/output" "$cache_root/triton" "$cache_root/torchinductor"
mkdir -p "$cache_root/hf/datasets" "$cache_root/xdg" "$cache_root/nv"

export VIRTUAL_ENV="$runtime_root"
export PATH="$runtime_root/bin:$cuda_home/bin:/usr/local/bin:/usr/bin"
export PYTHONPATH="$repo_dir:$overlay_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_HOME="$cuda_home"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$cache_root/hf"
export HF_DATASETS_CACHE="$cache_root/hf/datasets"
export XDG_CACHE_HOME="$cache_root/xdg"
export TRITON_CACHE_DIR="$cache_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
export CUDA_CACHE_PATH="$cache_root/nv"
export NCCL_RAS_ENABLE=0
export NCCL_DEBUG=WARN

torch_lib="$runtime_root/lib/python3.12/site-packages/torch/lib"
compat_lib="$runtime_root/lib/cuda-compat-13-1"
library_path="$runtime_root/lib:$torch_lib:$platform_lib:/usr/lib64:/lib64"
export LD_PRELOAD="$compat_lib/libcuda.so.1:$compat_lib/libnvidia-ptxjitcompiler.so.1"

python=("$loader" --library-path "$library_path" "$runtime_root/bin/python")
metrics=(--metrics.no-enable-tensorboard)
if [ "$tensorboard" -eq 1 ]; then
  metrics=(--metrics.enable-tensorboard)
fi
debug=(--debug.seed 42)
if [ "$deterministic" -eq 1 ]; then
  debug+=(--debug.deterministic)
fi
profiler=(--profiler.no-enable-profiling)
if [ "$profile" -eq 1 ]; then
  profiler=(
    --profiler.enable-profiling
    --profiler.profile-freq 10
    --profiler.profiler-warmup 3
    --profiler.profiler-active 1
    --profiler.profiler-repeat 1
  )
fi

train=(
  "${python[@]}" -u -m torchtitan.train
  --dump-folder "$run_root/output"
  --module graph_trainer.deepseek_v3
  --config graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf_16gpu
  --training.num-tokens-per-microbatch-per-dp-rank 4096
  --training.num-tokens-per-train-step 262144
  --training.max-context-length 4096
  --training.steps "$steps"
  --parallelism.data-parallel-replicate-degree 1
  --parallelism.data-parallel-shard-degree 4
  --parallelism.tensor-parallel-degree 1
  --parallelism.context-parallel-degree 1
  --parallelism.pipeline-parallel-degree 1
  --parallelism.expert-parallel-degree 4
  --parallelism.fsdp-reshard-after-forward never
  --parallelism.enable-fsdp-symm-mem
  --parallelism.fsdp-symm-mem-policy widest
  --compile.mode aot_fx_trace
  --compile.inductor-compilation none
  --compile.memory-policy none
  --compile.enable-graph-gradient-accumulation
  --compile.enable-deferred-fsdp-gradient-sync
  --compile.enable-fsdp-ag-rs-overlap
  --compile.enable-fsdp-dense-region-overlap
  --training.no-disable-cuda-graphs
  --compile.require-cudagraph
  --optimizer.implementation fused_opt_states_bf16
  --hf-assets-path "$repo_dir/tests/assets/tokenizer"
  --metrics.log-freq 1
  "${metrics[@]}"
  --checkpoint.no-enable
  --comm.trace-buf-size 0
  "${debug[@]}"
  --debug.no-print-config
  "${profiler[@]}"
  activation-checkpoint:none
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

{
  printf 'source_commit='
  git -C "$repo_dir" rev-parse HEAD
  printf 'runtime_root=%s\n' "$runtime_root"
  printf 'runtime_id=%s\n' "$runtime_id"
  printf 'runtime_manifest_sha256=%s\n' "$actual_manifest_sha256"
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} | tee "$run_root/manifest.txt"

if [ "$dry_run" -eq 1 ]; then
  exit 0
fi

"${python[@]}" - <<'PY'
import torch
import dist_moe
import flash_attn.cute.interface
import grain.python
import torchao
from torch._inductor.fx_passes.bucketing import (
    enable_symmetric_memory_for_fsdp_buckets,
)
from torchao.prototype.mx_formats.kernels import mxfp8_quantize_cuda
from torchtitan.experiments.graph_trainer.deepseek_v3.config_registry import (
    graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf_16gpu,
)

if torch.cuda.device_count() != 4:
    raise RuntimeError(f"expected 4 GPUs, found {torch.cuda.device_count()}")
for device_index in range(4):
    capability = torch.cuda.get_device_capability(device_index)
    if capability != (10, 3):
        raise RuntimeError(f"expected sm_103 GPU {device_index}, found {capability}")
if not hasattr(torch.Tensor, "_scaled_addmm_"):
    raise RuntimeError("Tensor._scaled_addmm_ is missing")
if not hasattr(torch, "_mm_with_compute_mode"):
    raise RuntimeError("torch._mm_with_compute_mode is missing")
if not callable(enable_symmetric_memory_for_fsdp_buckets):
    raise RuntimeError("symmetric-memory FSDP bucket lowering is missing")

x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
if len(mxfp8_quantize_cuda(x, rowwise=True, colwise=True)) != 4:
    raise RuntimeError("MXFP8 CUDA quantization returned an invalid result")

config = graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf_16gpu()
if config.compile.inductor_compilation != "none":
    raise RuntimeError("the reference recipe must disable Inductor")
if config.activation_checkpoint is not None:
    raise RuntimeError("the reference recipe must disable activation checkpointing")
if config.dataloader.shuffle or not config.dataloader.repeat:
    raise RuntimeError("the reference c4_test data order is not configured")
print(torch.__version__, torch.cuda.get_arch_list(), "local runtime OK")
PY

if [ "$preflight_only" -eq 1 ]; then
  exit 0
fi

cd "$repo_dir"
"${command[@]}" 2>&1 | tee "$run_root/run.log"
