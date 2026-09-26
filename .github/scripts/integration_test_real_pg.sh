#!/usr/bin/env bash
# Real PG (8 GPU) integration suite.
#
# Shared by the CUDA job in integration_test.yaml and the ROCm job in
# integration_test_rocm.yaml. They cannot be one job because `uses:` is not
# templatable and ROCm has to stay on linux_job_v2 -- v3 pulls from the private
# ECR, which the ROCm runners cannot authenticate to.
#
# Inputs (environment):
#   GPU_ARCH_TYPE   cuda | rocm
#   TORCH_VERSION   pinned torch version, or empty for the channel default
#   INDEX_URL       pip index to install torch/torchvision/torchao from
#   TEST_SUITE      features | models
#   EXPORT_RESULTS  true to export numerics instead of comparing with goldens
#   TEST_NAME       a single test name, or 'all'
#   TEST_SCOPE      real_pg_required to narrow the suite, or empty for all.
#                   A ciflow/fake-pg tag runs only the tests that cannot use
#                   Fake PG; main pushes, ciflow/real-pg tags, schedules and
#                   manual runs leave it empty and run the full suite.

set -eux

# Running as a script rather than inline means this is not a login shell, so
# conda is not initialised. Same as .github/scripts/validate_release_gpu.sh.
eval "$(conda shell.bash hook)"
CONDA_ENV=$(conda env list --json | jq -r ".envs | .[-1]")
conda activate "${CONDA_ENV}"
export HF_HOME="$RUNNER_TEMP/hf_home"
export HF_DATASETS_CACHE="$RUNNER_TEMP/hf_home/datasets"

pip config --user set global.progress_bar off
TORCH_SPEC="torch"
if [ -n "${TORCH_VERSION}" ]; then
  TORCH_SPEC="torch==${TORCH_VERSION}"
fi
python -m pip install --force-reinstall --pre \
  "${TORCH_SPEC}" torchvision --index-url "${INDEX_URL}"
USE_CPP=0 python -m pip install --pre torchao --index-url "${INDEX_URL}"

GPU_ARCH="a10g"
if [[ "${GPU_ARCH_TYPE}" == "rocm" ]]; then
  GPU_ARCH="mi350x"
  HIPBLASLT_LIB_DIR="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib", "hipblaslt", "library"))')"
  if [ -d "${HIPBLASLT_LIB_DIR}" ]; then
    export HIPBLASLT_TENSILE_LIBPATH="${HIPBLASLT_LIB_DIR}"
  fi
fi

sudo mkdir -p "$RUNNER_ARTIFACT_DIR"
sudo mkdir -p "$HF_HOME"
sudo chown -R "$(id -u):$(id -g)" "$RUNNER_ARTIFACT_DIR"
sudo chown -R "$(id -u):$(id -g)" "$HF_HOME"
cleanup_artifacts() {
  find "$RUNNER_ARTIFACT_DIR" -type d \( -name checkpoint -o -name inference_results \) -prune -exec rm -rf {} +
  chmod -R a+rX "$RUNNER_ARTIFACT_DIR"
}
trap cleanup_artifacts EXIT

EXPORT_ARG=""
if [[ "${EXPORT_RESULTS}" == "true" ]]; then
  EXPORT_ARG="--export-numerics"
fi
TEST_NAME_ARG=""
if [[ "${TEST_NAME}" != "all" ]]; then
  TEST_NAME_ARG="--test_name=${TEST_NAME}"
fi
TEST_SCOPE_ARG=""
if [[ -n "${TEST_SCOPE}" ]]; then
  TEST_SCOPE_ARG="--test_scope=${TEST_SCOPE}"
fi

python -m tests.integration_tests.run_tests \
  --gpu_arch_type "${GPU_ARCH_TYPE}" \
  --gpu_arch "$GPU_ARCH" \
  --test_suite "${TEST_SUITE}" --execution_mode real_pg --ngpu 8 \
  $EXPORT_ARG $TEST_NAME_ARG $TEST_SCOPE_ARG \
  "$RUNNER_ARTIFACT_DIR"
if [[ "${TEST_SUITE}" == "models" ]]; then
  python -m tests.integration_tests.flux \
    --execution_mode real_pg --ngpu 8 \
    $TEST_NAME_ARG $TEST_SCOPE_ARG \
    "$RUNNER_ARTIFACT_DIR/flux"
fi
