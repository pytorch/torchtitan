#!/bin/bash

TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC=0 TORCH_NCCL_DUMP_ON_TIMEOUT=0 TORCH_NCCL_TRACE_BUFFER_SIZE=0 TORCH_SHARE_RDZV_TCP_STORE=1 LOGLEVEL=INFO NCCL_DEBUG_SUBSYS=ALL NCCL_DEBUG=INFO TORCH_CPP_LOG_LEVEL=INFO CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1 NGPU=2 ./run_train.sh --training.local_batch_size=2 --parallelism.data_parallel_shard_degree=2 --profiling.enable_profiling --profiling.profile_freq=1 --profiling.profiler_active=1 --profiling.profiler_warmup=0 --training.steps=1000 --comm.train_timeout_seconds=1 --comm.trace_buf_size=0
