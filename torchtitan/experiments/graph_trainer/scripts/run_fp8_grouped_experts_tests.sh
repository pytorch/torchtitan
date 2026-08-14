#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

NGPU=${NGPU:-2}
OUTPUT_DIR=${OUTPUT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tt_fp8_grouped_tests.XXXXXX")}

if (( NGPU < 2 )); then
    echo "This test matrix requires NGPU>=2; got NGPU=${NGPU}."
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
    echo "SKIP: Float8 grouped-expert GPU tests require SM90+ and TorchAO."
    echo "Detected capability ${CUDA_MAJOR}.${CUDA_MINOR}, torchao=${HAS_TORCHAO}."
    exit 0
fi

echo "Writing test artifacts to ${OUTPUT_DIR}"

python3 -m pytest -q \
    tests/unit_tests/test_quantization.py::test_quantized_grouped_experts \
    torchtitan/experiments/graph_trainer/tests/test_fp8.py::TestFP8ValidationPass::test_grouped_compute_requires_scaled_grouped_mm \
    torchtitan/experiments/graph_trainer/tests/test_fp8.py::TestFP8RegionalAnnotation::test_identifies_and_tags_grouped_experts_component \
    torchtitan/experiments/graph_trainer/tests/test_fp8.py::TestFP8PrecompileFingerprint::test_grouped_expert_signature_includes_padding_contract \
    torchtitan/experiments/graph_trainer/tests/test_fp8.py::TestFP8PrecompileFingerprint::test_grouped_expert_padding_changes_fingerprint \
    torchtitan/experiments/graph_trainer/tests/test_fp8.py::TestFP8PrecompileFingerprint::test_torchao_version_changes_fingerprint \
    torchtitan/experiments/graph_trainer/tests/test_fp8.py::TestFP8RegionalCompilation::test_float8_grouped_experts_regional_inductor_runs_forward_and_backward

python3 -m torchtitan.experiments.graph_trainer.tests.integration_tests \
    "${OUTPUT_DIR}/integration_1gpu" \
    --test_suite graph_trainer_h100 \
    --test_name aot_fx_trace_dsv3_fp8_grouped_experts_regional \
    --ngpu "${NGPU}"

python3 -m torchtitan.experiments.graph_trainer.tests.integration_tests \
    "${OUTPUT_DIR}/integration_2gpu" \
    --test_suite graph_trainer_h100 \
    --test_name aot_fx_trace_dsv3_fp8_grouped_experts_fsdp_ep \
    --ngpu "${NGPU}"

python3 -m pytest -q \
    torchtitan/experiments/graph_trainer/tests/test_numerics.py::TestGraphTrainerFP8Numerics::test_deepseek_v3_fp8_grouped_experts_regional_vs_trainer

python3 -m torchtitan.experiments.graph_trainer.tests.run_precompile_tests \
    "${OUTPUT_DIR}/precompile" \
    --test_name aot_fx_trace_dsv3_fp8_grouped_experts_precompile \
    --ngpu "${NGPU}"

if (( CUDA_MAJOR >= 10 )); then
    NGPU=1 MODULE=graph_trainer.deepseek_v3 \
        CONFIG=graph_trainer_deepseek_v3_debugmodel_mxfp8 \
        ./run_train.sh \
        --training.steps 10 \
        --parallelism.data_parallel_shard_degree 1 \
        --parallelism.expert_parallel_degree 1 \
        --compile.inductor_compilation regional \
        --compile.disable_passes cudagraph_pass \
        --dump_folder "${OUTPUT_DIR}/mxfp8_smoke"
else
    echo "SKIP: MXFP8 execution requires SM100+; H20 is SM90."
fi

echo "All supported tests completed. Artifacts: ${OUTPUT_DIR}"
