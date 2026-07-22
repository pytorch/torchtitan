#!/usr/bin/bash
# Convenience launcher for TorchTitan OLMo3 training.

set -ex

cd "$(dirname "${BASH_SOURCE[0]}")"

export NGPU=${NGPU:-2}
export MODULE=${MODULE:-"olmo3"}
export CONFIG=${CONFIG:-"olmo3_7b"}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}

# Optional downstream eval hook. Enable with EXTERNAL_EVAL_ENABLE=1.
export EXTERNAL_EVAL_ENABLE=${EXTERNAL_EVAL_ENABLE:-0}
export EXTERNAL_EVAL_FREQ=${EXTERNAL_EVAL_FREQ:-100}
export EXTERNAL_EVAL_PATH=${EXTERNAL_EVAL_PATH:-"/home/ruisizhang123/scaling-ladder/eval/run_eval.py"}
export EXTERNAL_EVAL_TASKS=${EXTERNAL_EVAL_TASKS:-"mmlu,wikitext2"}
export EXTERNAL_EVAL_CUDA_VISIBLE_DEVICES=${EXTERNAL_EVAL_CUDA_VISIBLE_DEVICES:-""}
export EXTERNAL_EVAL_EXPORT_DTYPE=${EXTERNAL_EVAL_EXPORT_DTYPE:-"bfloat16"}
export EXTERNAL_EVAL_EXTRA_ARGS=${EXTERNAL_EVAL_EXTRA_ARGS:-""}

EXTRA_ARGS=()
if [[ "${EXTERNAL_EVAL_ENABLE}" == "1" || "${EXTERNAL_EVAL_ENABLE}" == "true" ]]; then
  EXTRA_ARGS+=(
    --checkpoint.enable
    --external_eval.enable
    --external_eval.freq "${EXTERNAL_EVAL_FREQ}"
    --external_eval.path "${EXTERNAL_EVAL_PATH}"
    --external_eval.tasks "${EXTERNAL_EVAL_TASKS}"
    --external_eval.export_dtype "${EXTERNAL_EVAL_EXPORT_DTYPE}"
  )

  if [[ -n "${EXTERNAL_EVAL_CUDA_VISIBLE_DEVICES}" ]]; then
    EXTRA_ARGS+=(
      --external_eval.eval_cuda_visible_devices "${EXTERNAL_EVAL_CUDA_VISIBLE_DEVICES}"
    )
  fi

  if [[ -n "${EXTERNAL_EVAL_EXTRA_ARGS}" ]]; then
    EXTRA_ARGS+=(--external_eval.extra_args "${EXTERNAL_EVAL_EXTRA_ARGS}")
  fi
fi

exec ./run_train.sh "${EXTRA_ARGS[@]}" "$@"
