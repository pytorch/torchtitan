#!/usr/bin/bash
# Convenience launcher for TorchTitan OLMo3 training.

set -ex

cd "$(dirname "${BASH_SOURCE[0]}")"

export NGPU=${NGPU:-2}
export MODULE=${MODULE:-"olmo3"}
export CONFIG=${CONFIG:-"olmo3_7b"}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}

exec ./run_train.sh "$@"
