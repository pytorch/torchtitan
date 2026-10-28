#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

required_env_vars=(
  VALIDATION_SUITE
  TORCH_VERSION
  TORCHVISION_VERSION
  PYTORCH_INDEX_URL
)
if [[ "${VALIDATION_SUITE:-}" != rl* ]]; then
  required_env_vars+=(TORCHAO_INDEX_URL)
fi
for var in "${required_env_vars[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "Missing required environment variable: ${var}"
    exit 1
  fi
done

case "${VALIDATION_SUITE}" in
  core|graph-trainer|graph-trainer-h100|h100|b200|torchft|transformers-modeling-backend) ;;
  rl-unit|rl|rl-b200)
    : "${VLLM_VERSION:?Set VLLM_VERSION to an exact version compatible with the release torch}"
    export VLLM_VERSION
    ;;
  *)
    echo "Unknown validation suite: ${VALIDATION_SUITE}" >&2
    exit 1
    ;;
esac

# Resolve from this script so B200 archive checkouts work without a .git directory.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RC_VERSION="$(tr -d '[:space:]' < "${REPO_ROOT}/assets/version.txt")"
if [[ "${RC_VERSION}" != *rc* ]]; then
  echo "Expected an RC version, got: ${RC_VERSION}"
  exit 1
fi
export RC_VERSION VALIDATION_SUITE TORCH_VERSION TORCHVISION_VERSION

if [[ "${VALIDATION_SUITE}" == "b200" || "${VALIDATION_SUITE}" == "rl-b200" ]]; then
  # The B200 CUDA container has no Conda environment.
  uv venv --seed --python 3.12 "${RUNNER_TEMP}/validate-release-b200-venv"
  source "${RUNNER_TEMP}/validate-release-b200-venv/bin/activate"
else
  eval "$(conda shell.bash hook)"
  conda activate "$(conda env list --json | jq -r '.envs[-1]')"
fi

export HF_HOME="${RUNNER_TEMP}/hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONUNBUFFERED=1

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 || true)
echo "CUDA driver version: ${DRIVER_VERSION}"
nvidia-smi

python -m pip uninstall -y torch torchvision torchao torchtitan

PIP_EXTRA_INDEX_URL='' python -m pip install --pre \
  --index-url "${PYTORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}"
# RL CI does not explicitly install torchao; quantization CI installs it separately.
if [[ "${VALIDATION_SUITE}" != rl* ]]; then
  USE_CPP=0 PIP_EXTRA_INDEX_URL='' python -m pip install --pre \
    --index-url "${TORCHAO_INDEX_URL}" torchao
fi

RC_PACKAGE="torchtitan"
if [[ "${VALIDATION_SUITE}" == rl* ]]; then
  RC_PACKAGE="torchtitan[rl]"
  # RL dependencies must not replace the release wheels with their own torch.
  export PIP_CONSTRAINT="${RUNNER_TEMP}/release-rl-constraints.txt"
  python -m pip list --format=freeze | grep -E '^(torch|torchvision|triton)==' > "${PIP_CONSTRAINT}"
  printf 'torchtitan==%s\nvllm==%s\n' "${RC_VERSION}" "${VLLM_VERSION}" >> "${PIP_CONSTRAINT}"
fi

install_release_candidate() {
  for attempt in {1..5}; do
    echo "Installing torchtitan==${RC_VERSION} from TestPyPI (attempt ${attempt}/5)"
    if python -m pip install \
      --index-url https://test.pypi.org/simple/ \
      --extra-index-url https://pypi.org/simple/ \
      "${RC_PACKAGE}==${RC_VERSION}"; then
      return 0
    fi
    if [[ "${attempt}" -lt 5 ]]; then
      sleep 15
    fi
  done

  echo "Failed to install torchtitan==${RC_VERSION} after 5 attempts" >&2
  return 1
}
install_release_candidate

if [[ "${VALIDATION_SUITE}" == rl* ]]; then
  python -m pip install "pytest==7.3.2" "expecttest>=0.2.0" uv
  vllm_requirement="vllm==${VLLM_VERSION}"
  if [[ "${TORCH_VERSION}" == "2.15.0" && \
        "${VLLM_VERSION}" == "1.0.0.dev20261005+cu132" && \
        "${PYTORCH_INDEX_URL}" == "https://download.pytorch.org/whl/test/cu132" ]]; then
    # Use the RC in place of this wheel's torch nightly dependency.
    export UV_OVERRIDE="${RUNNER_TEMP}/release-vllm-torch-override.txt"
    grep '^torch==' "${PIP_CONSTRAINT}" > "${UV_OVERRIDE}"
    vllm_requirement="vllm @ https://download.pytorch.org/whl/nightly/cu132/vllm-1.0.0.dev20261005%2Bcu132-cp38-abi3-manylinux_2_28_x86_64.whl"
  fi
  uv pip install --constraint "${PIP_CONSTRAINT}" \
    --extra-index-url "${PYTORCH_INDEX_URL}" \
    --index-strategy unsafe-best-match "${vllm_requirement}"
  python -m pip install --no-deps \
    "git+https://github.com/meta-pytorch/torchstore.git@main" \
    "git+https://github.com/thinking-machines-lab/batch_invariant_ops.git@main"
  # Match RL CI's Attention Gym kernels instead of the optional tilelang path.
  python -m pip uninstall -y tilelang
  if [[ "${VALIDATION_SUITE}" != "rl-b200" ]]; then
    python -m pip install "nvidia-cuda-nvcc==13.2.*"
  fi
  if [[ "${VALIDATION_SUITE}" == "rl-b200" ]]; then
    python -m pip install "flash-attn-4[cu13]>=4.0.0b34"
  fi
  if [[ "${VALIDATION_SUITE}" == "rl-b200" ]]; then
    python -m pip install "git+https://github.com/meta-pytorch/dist_moe.git@main"
  fi
fi

if [[ "${VALIDATION_SUITE}" == "b200" ]]; then
  python -m pip install \
    "pytest==7.3.2" "expecttest>=0.2.0" \
    "flash-attn-4[cu13]>=4.0.0b31" \
    "git+https://github.com/meta-pytorch/dist_moe.git@main"
fi

ARTIFACTS="${RUNNER_TEMP}/artifacts-to-be-uploaded"
VALIDATION_ROOT="${RUNNER_TEMP}/validate-release-gpu"
mkdir -p \
  "${ARTIFACTS}" \
  "${VALIDATION_ROOT}/scripts" \
  "${VALIDATION_ROOT}/torchtitan/experiments/graph_trainer" \
  "${VALIDATION_ROOT}/torchtitan/models/flux/inference"

# Copy test support files without copying TorchTitan Python packages. Training
# subprocesses must import the installed TestPyPI RC rather than the checkout.
cp "${REPO_ROOT}/run_train.sh" "${VALIDATION_ROOT}/run_train.sh"
cp "${REPO_ROOT}/scripts/loss_compare.py" "${VALIDATION_ROOT}/scripts/loss_compare.py"
cp "${REPO_ROOT}/scripts/_checkpoint_test_config.py" "${VALIDATION_ROOT}/scripts/_checkpoint_test_config.py"
if [[ "${VALIDATION_SUITE}" == rl* ]]; then
  mkdir -p "${VALIDATION_ROOT}/scripts/rl"
  cp "${REPO_ROOT}/scripts/rl/loss_compare.py" "${VALIDATION_ROOT}/scripts/rl/loss_compare.py"
fi
cp -R "${REPO_ROOT}/tests" "${VALIDATION_ROOT}/tests"
cp "${REPO_ROOT}/pyproject.toml" "${VALIDATION_ROOT}/pyproject.toml"
cp \
  "${REPO_ROOT}/torchtitan/models/flux/inference/prompts.txt" \
  "${VALIDATION_ROOT}/torchtitan/models/flux/inference/prompts.txt"
cp \
  "${REPO_ROOT}/torchtitan/experiments/graph_trainer/run_train_precompile.sh" \
  "${VALIDATION_ROOT}/torchtitan/experiments/graph_trainer/run_train_precompile.sh"

# Installed test recipes resolve repository-local assets relative to
# site-packages. Link that location to the staged source-test assets.
SITE_PACKAGES="$(python - <<'PY'
import sysconfig

print(sysconfig.get_path("purelib"))
PY
)"
mkdir -p "${SITE_PACKAGES}/tests/assets"
ln -s \
  "${VALIDATION_ROOT}/tests/assets/synthetic_multimodal" \
  "${SITE_PACKAGES}/tests/assets/synthetic_multimodal"

cd "${VALIDATION_ROOT}"
unset PYTHONPATH

python - <<'PY'
import os

import torch
import torchtitan
import torchvision
import triton

assert torchtitan.__version__ == os.environ["RC_VERSION"]
assert "site-packages" in torchtitan.__file__
assert torch.__version__.split("+")[0] == os.environ["TORCH_VERSION"]
assert torchvision.__version__.split("+")[0] == os.environ["TORCHVISION_VERSION"]
expected_cuda = "13.2" if os.environ["VALIDATION_SUITE"].startswith("rl") else "13.0"
assert torch.version.cuda == expected_cuda
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 8
if os.environ["VALIDATION_SUITE"] in ("b200", "rl-b200"):
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        raise ValueError(
            f"B200 validation requires SM100 or SM103, got SM{capability[0]}{capability[1]}"
        )
if os.environ["VALIDATION_SUITE"].startswith("rl"):
    from importlib.metadata import version

    import batch_invariant_ops
    import monarch
    import torchstore
    import vllm
    from packaging.version import Version

    import torchtitan_recipes

    assert Version(version("vllm")) == Version(os.environ["VLLM_VERSION"])
    assert "site-packages" in torchtitan_recipes.__file__
    for name in ("vllm", "torchmonarch", "torchstore", "batch_invariant_ops"):
        print(f"{name}={version(name)}")
else:
    import torchao

    assert ".dev" in torchao.__version__.split("+")[0]
    print(f"torchao={torchao.__version__}")
print(f"torchtitan={torchtitan.__version__} ({torchtitan.__file__})")
print(f"torch={torch.__version__} ({torch.version.git_version})")
print(f"torchvision={torchvision.__version__}")
print(f"triton={triton.__version__}")
print(f"num_cuda_devices={torch.cuda.device_count()}")
PY

TORCHTITAN_PACKAGE_ROOT="$(python - <<'PY'
from pathlib import Path

import torchtitan

print(Path(torchtitan.__file__).parent)
PY
)"

run_gpu_training_smoke_test() {
  CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
import torch.nn.functional as F

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.llama3 import build_model_config

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.compiler.set_stance("force_eager")
device = torch.device("cuda")
model_config = build_model_config(
    "debugmodel",
    parallelism=ParallelismConfig(),
    seq_len=16,
    attn_backend="flex",
)
model = model_config.build()
model.init_states(buffer_device=device)
model.to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
tokens_T = torch.randint(0, 2048, (16,), device=device)
positions_T = torch.arange(16, device=device)
attention_metadata = model._get_attention_metadata(positions_T)
losses = []

for _ in range(3):
    optimizer.zero_grad(set_to_none=True)
    logits_TV = model(tokens_T, positions_T, attention_metadata)
    loss = F.cross_entropy(logits_TV[:-1], tokens_T[1:])
    assert torch.isfinite(loss), loss
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

assert losses[-1] < losses[0], losses
assert all(parameter.grad is not None for parameter in model.parameters())
print(f"losses={losses}")
PY
}

run_core_tests() {
  run_gpu_training_smoke_test

  CUDA_VISIBLE_DEVICES=0 python -m pytest tests/unit_tests/gpu \
    -m "not multi_gpu" \
    --strict-markers \
    --durations=20 \
    -vv

  python -m pytest tests/unit_tests/gpu \
    -m multi_gpu \
    --strict-markers \
    --durations=20 \
    -vv

  python -m tests.integration_tests.run_tests \
    --gpu_arch_type cuda \
    --test_suite features,models \
    --execution_mode real_pg \
    --ngpu 8 \
    "${ARTIFACTS}/integration_tests"

  python -m tests.integration_tests.run_tests \
    --gpu_arch_type cuda \
    --test_suite flux \
    --execution_mode real_pg \
    --ngpu 8 \
    "${ARTIFACTS}/flux_tests"
}

run_torchft_tests() {
  local lighthouse_pid
  local status=0

  python -m pip install torchft==0.2.0

  RUST_BACKTRACE=1 torchft_lighthouse \
    --min_replicas 1 \
    --quorum_tick_ms 100 \
    --join_timeout_ms 10000 > /dev/null 2>&1 &
  lighthouse_pid=$!

  python -m torchtitan.experiments.torchft.tests.integration_tests \
    "${ARTIFACTS}/torchft" \
    --ngpu 8 || status=$?

  kill "${lighthouse_pid}" 2>/dev/null || true
  wait "${lighthouse_pid}" 2>/dev/null || true
  return "${status}"
}

run_transformers_modeling_backend_tests() {
  python -m pip install transformers==5.9.0
  python -m torchtitan.experiments.transformers_modeling_backend.tests.integration_tests \
    "${ARTIFACTS}/transformers_modeling_backend" \
    --ngpu 8
}

run_graph_trainer_tests() {
  local tests_root="${TORCHTITAN_PACKAGE_ROOT}/experiments/graph_trainer/tests"

  python -m torchtitan.experiments.graph_trainer.tests.integration_tests \
    --test_suite graph_trainer_default \
    --gpu_arch_type cuda \
    "${ARTIFACTS}/graph_trainer" \
    --ngpu 8
  python -m pytest "${tests_root}/test_numerics.py::TestSimpleFSDP" -v
  python -m pytest "${tests_root}/test_numerics.py::TestGraphTrainerNumerics" \
    -v -k dense
  python -m pytest "${tests_root}/test_passes.py" -v
  python -m pytest "${tests_root}/test_profiler.py" -v
  python -m pytest "${tests_root}/test_trace_module.py" -v
  python -m pytest "${tests_root}/test_precompile.py" -v
  python -m torchtitan.experiments.graph_trainer.tests.run_precompile_tests \
    "${ARTIFACTS}/graph_trainer_precompile" \
    --ngpu 8 \
    --test_name aot_fx_trace_llama3_precompile_fsdp_tp
  python -m pytest "${tests_root}/test_bitwise_deterministic.py" -v
  python -m pytest "${tests_root}/test_sac_peak_memory.py" -v
}

install_hybrid_ep() {
  export CUDA_HOME=/usr/local/cuda
  export NCCL_NVLS_ENABLE=0
  export TORCH_SHOW_CPP_STACKTRACES=1
  bash /install_deepep.sh
}

run_h100_tests() {
  local status=0

  python -m pip install flash-attn-3 \
    --extra-index-url "${PYTORCH_INDEX_URL}"

  if ! CUDA_HOME=/usr/local/cuda NCCL_NVLS_ENABLE=0 TORCH_SHOW_CPP_STACKTRACES=1 \
    python -m tests.integration_tests.run_tests \
    --test_suite h100 \
    --execution_mode real_pg \
    --exclude qwen3_fsdp+deepep,deepseek_v3_fsdp+hybridep \
    --gpu_arch_type cuda \
    --gpu_arch h100 \
    --ngpu 8 \
    "${ARTIFACTS}/h100/base"; then
    status=1
  fi

  bash "${REPO_ROOT}/.github/scripts/install_deepep_v2.sh"
  if ! CUDA_HOME=/usr/local/cuda NCCL_NVLS_ENABLE=0 EP_DISABLE_GIN=1 \
    python -m pytest \
    tests/unit_tests/gpu/distributed/test_deepep_activation_checkpoint.py -v; then
    status=1
  fi
  if ! CUDA_HOME=/usr/local/cuda NCCL_NVLS_ENABLE=0 EP_DISABLE_GIN=1 \
    TORCH_SHOW_CPP_STACKTRACES=1 python -m tests.integration_tests.run_tests \
    --test_suite h100 \
    --execution_mode real_pg \
    --test_name qwen3_fsdp+deepep \
    --gpu_arch_type cuda \
    --gpu_arch h100 \
    --ngpu 8 \
    "${ARTIFACTS}/h100/deepep_v2"; then
    status=1
  fi

  install_hybrid_ep
  if ! python -m tests.integration_tests.run_tests \
    --test_suite h100 \
    --execution_mode real_pg \
    --test_name deepseek_v3_fsdp+hybridep \
    --gpu_arch_type cuda \
    --gpu_arch h100 \
    --ngpu 8 \
    "${ARTIFACTS}/h100/hybrid_ep"; then
    status=1
  fi

  return "${status}"
}

run_graph_trainer_h100_tests() {
  local tests_root="${TORCHTITAN_PACKAGE_ROOT}/experiments/graph_trainer/tests"

  python -m pip install flash-attn-3 \
    --extra-index-url "${PYTORCH_INDEX_URL}"

  install_hybrid_ep
  python -m torchtitan.experiments.graph_trainer.tests.integration_tests \
    --test_suite graph_trainer_h100 \
    --gpu_arch_type cuda \
    "${ARTIFACTS}/graph_trainer_h100" \
    --ngpu 8
  python -m pytest "${tests_root}/test_numerics.py::TestGraphTrainerNumerics" \
    -v -k moe
  python -m torchtitan.experiments.graph_trainer.tests.run_precompile_tests \
    "${ARTIFACTS}/graph_trainer_h100_precompile" \
    --ngpu 8 \
    --test_name aot_fx_trace_deepseek_v3_precompile_fsdp_tp_ep
  python -m pytest "${tests_root}/test_bitwise_deterministic.py" -v
}

run_b200_tests() {
  NCCL_NVLS_ENABLE=0 python -m pytest \
    tests/unit_tests/cpu/models/common/test_dist_moe.py \
    tests/unit_tests/gpu/models/common/test_dist_moe.py \
    tests/unit_tests/gpu/quantization/test_mxfp8_fsdp.py \
    -v

  CUDA_HOME=/usr/local/cuda TORCH_SHOW_CPP_STACKTRACES=1 \
    python -m tests.integration_tests.run_tests \
    "${ARTIFACTS}/b200" \
    --test_suite b200 \
    --execution_mode real_pg \
    --gpu_arch_type cuda \
    --gpu_arch b200 \
    --ngpu 8
}

download_rl_model() {
  python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download("Qwen/Qwen3-0.6B"))
PY
}

run_rl_unit_tests() {
  local tokenizer_path gdn_model_path model_path
  tokenizer_path="$(python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download("meta-models/Muse-Glimmer-30B", allow_patterns=[
    "config.json", "tokenizer*.json", "chat_template.jinja",
]))
PY
)"
  export MUSE_GLIMMER_TOKENIZER="${tokenizer_path}"
  python -m pip install -r "${REPO_ROOT}/torchtitan/rl/examples/dapo_math/requirements.txt"
  CUDA_VISIBLE_DEVICES='' python -m pytest tests/rl/unit_tests/cpu \
    --strict-markers --durations=20 -vv

  gdn_model_path="$(python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download("Qwen/Qwen3.5-0.8B"))
PY
)"
  export TORCHTITAN_QWEN3_5_0_8B_HF_PATH="${gdn_model_path}"
  CUDA_VISIBLE_DEVICES=0 python -m pytest tests/rl/unit_tests/gpu \
    -m "not multi_gpu" --strict-markers --durations=20 -vv

  model_path="$(download_rl_model)"
  export RL_TEST_DUMP_FOLDER="${ARTIFACTS}/rl_unit"
  local test_class
  for test_class in TestBitwiseParityVarlen TestBitwiseParityFlex; do
    HF_ASSETS_PATH="${model_path}" python -m torch.distributed.run \
      --standalone --nproc-per-node=2 -m pytest \
      "tests/rl/unit_tests/gpu/test_bitwise_parity.py::${test_class}" -v
  done
  python -m torch.distributed.run --standalone --nproc-per-node=2 -m pytest \
    tests/rl/unit_tests/gpu/test_bitwise_parity.py::TestBitwiseParityQwen35DebugVarlen \
    -v
  python -m torch.distributed.run --standalone --nproc-per-node=4 -m pytest \
    tests/rl/unit_tests/gpu/test_generator_uneven_decode.py -v

  # As in RL CPU CI, test Verifiers last: it constrains mcp differently from
  # vLLM. Keep the release wheels fixed while resolving the optional extra.
  uv pip install --constraint "${PIP_CONSTRAINT}" \
    --extra-index-url "${PYTORCH_INDEX_URL}" \
    --index-strategy unsafe-best-match \
    -r "${REPO_ROOT}/torchtitan/rl/examples/verifiers/dapo_math/requirements.txt"
  CUDA_VISIBLE_DEVICES='' python -m pytest \
    tests/rl/unit_tests/cpu/test_verifiers.py \
    tests/rl/unit_tests/cpu/test_verifiers_example.py \
    "tests/rl/unit_tests/cpu/test_verifiers_terminal_bench.py::test_env_server_workers_skip_the_trainer_imports" \
    "tests/rl/unit_tests/cpu/test_model_defaults.py::test_rl_recipes_get_model_defaults" \
    --strict-markers --durations=20 -vv
}

run_rl_tests() {
  local model_path
  model_path="$(download_rl_model)"
  # Run the A10G golden first, before other RL tests can leave live processes.
  python -m scripts.rl.loss_compare \
    --hf-assets-path "${model_path}" \
    --dump-folder "${ARTIFACTS}/rl_loss_guard" \
    --module torchtitan_recipes.rl.alphabet_sort \
    --config rl_grpo_qwen3_0_6b_varlen_batch_invariant \
    --import-result tests/assets/losses/rl_grpo_cuda.txt \
    --assert-equal
  python -m tests.integration_tests.run_tests "${ARTIFACTS}/rl/integration" \
    --test_suite rl --execution_mode real_pg --gpu_arch_type cuda \
    --ngpu 8 --hf_assets_path "${model_path}"
}

run_rl_b200_tests() {
  local status=0
  if ! RL_TEST_DUMP_FOLDER="${ARTIFACTS}/rl_b200" \
    python -m torch.distributed.run --standalone --nproc-per-node=1 -m pytest \
    tests/rl/unit_tests/gpu/test_bitwise_parity.py::TestBitwiseParityKimiK3DebugVarlen \
    tests/rl/unit_tests/cpu/test_rl_dist_moe_recipe.py -v; then
    status=1
  fi
  if ! python -m tests.integration_tests.run_tests "${ARTIFACTS}/rl_b200/integration" \
    --test_suite rl_b200 --execution_mode real_pg --gpu_arch_type cuda \
    --gpu_arch b200 --ngpu 8; then
    status=1
  fi
  return "${status}"
}

# Run the selected suite without letting `set -e` abort the script on
# failure: we still need to prune checkpoint directories from the artifacts
# below and report the correct exit code afterwards. The subshell re-enables
# `set -e` so a failing step inside the chosen run_*_tests function still
# fails fast exactly as before; only the outer script's early exit is
# deferred until after cleanup.
suite_status=0
set +e
(
  set -e
  case "${VALIDATION_SUITE}" in
    core) run_core_tests ;;
    graph-trainer) run_graph_trainer_tests ;;
    graph-trainer-h100) run_graph_trainer_h100_tests ;;
    h100) run_h100_tests ;;
    b200) run_b200_tests ;;
    torchft) run_torchft_tests ;;
    transformers-modeling-backend) run_transformers_modeling_backend_tests ;;
    rl-unit) run_rl_unit_tests ;;
    rl) run_rl_tests ;;
    rl-b200) run_rl_b200_tests ;;
  esac
)
suite_status=$?
set -e

find "${ARTIFACTS}" -type d -name checkpoint -prune -exec rm -rf {} +

exit "${suite_status}"
