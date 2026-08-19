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
export TRAINING_STEPS=${TRAINING_STEPS:--1}
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

export CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES}
exec ./run_train.sh "${EXTRA_ARGS[@]}" "$@"
