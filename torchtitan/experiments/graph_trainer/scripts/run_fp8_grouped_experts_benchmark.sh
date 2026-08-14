#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

NGPU=${NGPU:-2}
STEPS=${STEPS:-30}
OUTPUT_DIR=${OUTPUT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tt_fp8_grouped_bench.XXXXXX")}
RUN_PROFILE=${RUN_PROFILE:-0}
RUN_PRECOMPILE=${RUN_PRECOMPILE:-0}

if (( NGPU != 2 )); then
    echo "This benchmark is calibrated for exactly 2 GPUs; got NGPU=${NGPU}."
    exit 2
fi
if (( STEPS < 10 )); then
    echo "Performance tests require STEPS>=10; got STEPS=${STEPS}."
    exit 2
fi

read -r CUDA_MAJOR CUDA_MINOR HAS_TORCHAO <<EOF
$(python3 - <<'PY'
import importlib.util
import torch

major, minor = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
print(major, minor, int(importlib.util.find_spec("torchao") is not None))
PY
)
EOF

if (( CUDA_MAJOR < 9 || HAS_TORCHAO == 0 )); then
    echo "SKIP: benchmark requires SM90+ and TorchAO."
    exit 0
fi

COMMON_ARGS=(
    --training.steps "${STEPS}"
    --parallelism.data_parallel_shard_degree 2
    --parallelism.tensor_parallel_degree 1
    --parallelism.expert_parallel_degree 2
    --metrics.log_freq 1
    --metrics.no-enable_tensorboard
    --profiler.no-enable_profiling
    --comm.trace_buf_size 0
    --checkpoint.no-enable
)

run_case() {
    local name=$1
    local module=$2
    local config=$3
    shift 3

    local case_dir="${OUTPUT_DIR}/${name}"
    mkdir -p "${case_dir}"
    echo "===== ${name} ====="
    NGPU="${NGPU}" MODULE="${module}" CONFIG="${config}" ./run_train.sh \
        --dump_folder "${case_dir}" \
        "${COMMON_ARGS[@]}" \
        "$@" \
        2>&1 | tee "${case_dir}/run.log"
}

echo "Writing benchmark artifacts to ${OUTPUT_DIR}"
echo "The debug model validates relative compiler overhead, not production MoE peak throughput."

run_case \
    trainer_float8 \
    deepseek_v3 \
    deepseek_v3_debugmodel_float8

run_case \
    graph_full_float8 \
    graph_trainer.deepseek_v3 \
    graph_trainer_deepseek_v3_debugmodel_float8 \
    --compile.inductor_compilation full \
    --compile.disable_passes cudagraph_pass

run_case \
    graph_regional_float8 \
    graph_trainer.deepseek_v3 \
    graph_trainer_deepseek_v3_debugmodel_float8 \
    --compile.inductor_compilation regional \
    --compile.disable_passes cudagraph_pass

if (( RUN_PRECOMPILE == 1 )); then
    PRECOMPILE_DIR="${OUTPUT_DIR}/precompile_artifact"
    python3 -m torchtitan.experiments.graph_trainer.precompile_main \
        --module graph_trainer.deepseek_v3 \
        --config graph_trainer_deepseek_v3_debugmodel_float8_sdpa \
        --compile.mode aot_fx_trace \
        --compile.inductor_compilation regional \
        --compile.precompile_artifact_dir "${PRECOMPILE_DIR}" \
        --parallelism.data_parallel_shard_degree 2 \
        --parallelism.tensor_parallel_degree 1 \
        --parallelism.expert_parallel_degree 2 \
        2>&1 | tee "${OUTPUT_DIR}/precompile.log"

    mkdir -p "${OUTPUT_DIR}/graph_regional_precompiled_float8"
    NGPU="${NGPU}" MODULE=graph_trainer.deepseek_v3 \
        CONFIG=graph_trainer_deepseek_v3_debugmodel_float8_sdpa \
        ./torchtitan/experiments/graph_trainer/run_train_precompile.sh \
        --dump_folder "${OUTPUT_DIR}/graph_regional_precompiled_float8" \
        "${COMMON_ARGS[@]}" \
        --compile.inductor_compilation regional \
        --compile.precompile_artifact_dir "${PRECOMPILE_DIR}" \
        2>&1 | tee "${OUTPUT_DIR}/graph_regional_precompiled_float8/run.log"
else
    echo "SKIP: grouped MoE CooR runtime precompile still has an unbacked-SymInt limitation."
    echo "Set RUN_PRECOMPILE=1 to exercise the retained experimental case."
fi

if (( RUN_PROFILE == 1 )); then
    run_case \
        graph_regional_float8_profile \
        graph_trainer.deepseek_v3 \
        graph_trainer_deepseek_v3_debugmodel_float8 \
        --compile.inductor_compilation regional \
        --compile.disable_passes cudagraph_pass \
        --profiler.enable_profiling \
        --profiler.profile_freq 10 \
        --profiler.profiler_warmup 3 \
        --profiler.profiler_active 1
fi

echo "===== Final logged steps ====="
for log_path in "${OUTPUT_DIR}"/*/run.log; do
    echo "--- ${log_path}"
    rg "step:.*tps:" "${log_path}" | tail -n 5 || true
done

if (( CUDA_MAJOR < 10 )); then
    echo "SKIP: MXFP8 performance execution requires SM100+; H20 is SM90."
fi
echo "Benchmark complete. Artifacts: ${OUTPUT_DIR}"
