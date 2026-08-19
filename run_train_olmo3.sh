#!/usr/bin/bash
# Convenience launcher for TorchTitan OLMo3 training.

set -ex

cd "$(dirname "${BASH_SOURCE[0]}")"

export NGPU=${NGPU:-6}
export TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5"}
export MODULE=${MODULE:-"olmo3"}
export CONFIG=${CONFIG:-"olmo3_7b"}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
export DATA_PARALLEL_SHARD_DEGREE=${DATA_PARALLEL_SHARD_DEGREE:-${NGPU}}
export LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-1}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-6}
export TRAINING_STEPS=${TRAINING_STEPS:-1192092}
export DATASET_PATH=${DATASET_PATH:-"/home/ruisizhang123/ruisizhang123_data/dolma3_mix-6T-1025-7B/pre-tokenize-data"}
export DATASET_SHUFFLE_BLOCK_SIZE=${DATASET_SHUFFLE_BLOCK_SIZE:-1024}
# Must stay >= the number of token files in the mix (906 for dolma3_mix-6T),
# or the global shuffle thrashes the dataloader's LRU fd cache.
export DATASET_MAX_OPEN_FILES=${DATASET_MAX_OPEN_FILES:-1024}
export DATASET_CHUNK_SIZE=${DATASET_CHUNK_SIZE:-1}
export DATASET_NUM_THREADS=${DATASET_NUM_THREADS:-4}
export DATASET_NUM_WORKERS=${DATASET_NUM_WORKERS:-8}
export DATASET_PREFETCH_FACTOR=${DATASET_PREFETCH_FACTOR:-8}
export DATASET_READ_AHEAD=${DATASET_READ_AHEAD:-32}
export PROFILE_FREQ=${PROFILE_FREQ:-1000}
export COMM_TRAIN_TIMEOUT_SECONDS=${COMM_TRAIN_TIMEOUT_SECONDS:-1800}

EXTRA_ARGS=(
  --profiler.profile_freq "${PROFILE_FREQ}"
  --parallelism.data_parallel_replicate_degree 1
  --parallelism.data_parallel_shard_degree "${DATA_PARALLEL_SHARD_DEGREE}"
  --training.local_batch_size "${LOCAL_BATCH_SIZE}"
  --training.global_batch_size "${GLOBAL_BATCH_SIZE}"
  --training.steps "${TRAINING_STEPS}"
  --dataloader.dataset_path "${DATASET_PATH}"
  --dataloader.shuffle_block_size "${DATASET_SHUFFLE_BLOCK_SIZE}"
  --dataloader.max_open_files "${DATASET_MAX_OPEN_FILES}"
  --dataloader.chunk_size "${DATASET_CHUNK_SIZE}"
  --dataloader.num_threads "${DATASET_NUM_THREADS}"
  --dataloader.read_ahead "${DATASET_READ_AHEAD}"
  --dataloader.num_workers "${DATASET_NUM_WORKERS}"
  --dataloader.prefetch_factor "${DATASET_PREFETCH_FACTOR}"
  --dataloader.persistent_workers
  --dataloader.pin_memory
  --comm.train_timeout_seconds "${COMM_TRAIN_TIMEOUT_SECONDS}"
)

# Optional checkpoint averaging and downstream eval hooks.
export EXTERNAL_EVAL_ENABLE=${EXTERNAL_EVAL_ENABLE:-1}
export CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-10}
export CHECKPOINT_KEEP_LATEST_K=${CHECKPOINT_KEEP_LATEST_K:-10}
export EXTERNAL_EVAL_FREQ=${EXTERNAL_EVAL_FREQ:-10}
export EXTERNAL_EVAL_PATH=${EXTERNAL_EVAL_PATH:-"$(pwd)/scripts/run_external_eval.py"}
export EXTERNAL_EVAL_TASKS=${EXTERNAL_EVAL_TASKS:-"olmo3_arc_challenge_5shot,olmo3_arc_easy_5shot,olmo3_hellaswag_5shot,olmo3_mmlu_humanities_5shot,olmo3_mmlu_other_5shot,olmo3_mmlu_social_sciences_5shot,olmo3_mmlu_stem_5shot,olmo3_humaneval_gold_bpb_3shot,olmo3_mbpp_gold_bpb_3shot,olmo3_math500_gold_bpb_0shot"}
export EXTERNAL_EVAL_EVAL_RAW=${EXTERNAL_EVAL_EVAL_RAW:-0}
export EXTERNAL_EVAL_CUDA_VISIBLE_DEVICES=${EXTERNAL_EVAL_CUDA_VISIBLE_DEVICES:-"6,7"}
export EXTERNAL_EVAL_GPUS=${EXTERNAL_EVAL_GPUS:-2}
export EXTERNAL_EVAL_EXPORT_DTYPE=${EXTERNAL_EVAL_EXPORT_DTYPE:-"bfloat16"}
export EXTERNAL_EVAL_EXTRA_ARGS=${EXTERNAL_EVAL_EXTRA_ARGS:-"--eval-gpus ${EXTERNAL_EVAL_GPUS} --batch-size 1 --max-sequence-length 8192"}
export EMA_ENABLE=${EMA_ENABLE:-1}
export EMA_FREQ=${EMA_FREQ:-${EXTERNAL_EVAL_FREQ}}
export EMA_CHECKPOINT_COUNT=${EMA_CHECKPOINT_COUNT:-4}
export EMA_CHECKPOINT_INTERVAL=${EMA_CHECKPOINT_INTERVAL:-${CHECKPOINT_INTERVAL}}
export EMA_START_STEP=${EMA_START_STEP:--1}
export EMA_DECAY=${EMA_DECAY:-1.0}
export EMA_STATEFUL_DECAY=${EMA_STATEFUL_DECAY:-0.0}

if [[ "${EXTERNAL_EVAL_ENABLE}" == "1" || "${EXTERNAL_EVAL_ENABLE}" == "true" || "${EMA_ENABLE}" == "1" || "${EMA_ENABLE}" == "true" ]]; then
  EXTRA_ARGS+=(
    --checkpoint.enable
    --checkpoint.no_last_save_model_only
    --checkpoint.interval "${CHECKPOINT_INTERVAL}"
    --checkpoint.keep_latest_k "${CHECKPOINT_KEEP_LATEST_K}"
  )
fi

if [[ "${EMA_ENABLE}" == "1" || "${EMA_ENABLE}" == "true" ]]; then
  EXTRA_ARGS+=(
    --ema.enable
    --ema.freq "${EMA_FREQ}"
    --ema.checkpoint_count "${EMA_CHECKPOINT_COUNT}"
    --ema.checkpoint_interval "${EMA_CHECKPOINT_INTERVAL}"
    --ema.start_step "${EMA_START_STEP}"
    --ema.decay "${EMA_DECAY}"
    --ema.stateful_decay "${EMA_STATEFUL_DECAY}"
  )
fi

if [[ "${EXTERNAL_EVAL_ENABLE}" == "1" || "${EXTERNAL_EVAL_ENABLE}" == "true" ]]; then
  EXTRA_ARGS+=(
    --external_eval.enable
    --external_eval.freq "${EXTERNAL_EVAL_FREQ}"
    --external_eval.path "${EXTERNAL_EVAL_PATH}"
    --external_eval.tasks "${EXTERNAL_EVAL_TASKS}"
    --external_eval.export_dtype "${EXTERNAL_EVAL_EXPORT_DTYPE}"
  )

  if [[ "${EXTERNAL_EVAL_EVAL_RAW}" != "1" && "${EXTERNAL_EVAL_EVAL_RAW}" != "true" ]]; then
    EXTRA_ARGS+=(--external_eval.no_eval_raw)
  fi

  if [[ -n "${EXTERNAL_EVAL_CUDA_VISIBLE_DEVICES}" ]]; then
    EXTRA_ARGS+=(
      --external_eval.eval_cuda_visible_devices "${EXTERNAL_EVAL_CUDA_VISIBLE_DEVICES}"
    )
  fi

  if [[ -n "${EXTERNAL_EVAL_EXTRA_ARGS}" ]]; then
    EXTRA_ARGS+=("--external_eval.extra_args=${EXTERNAL_EVAL_EXTRA_ARGS}")
  fi
fi

export CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES}
exec ./run_train.sh "${EXTRA_ARGS[@]}" "$@"
