# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# DeepSeek V2 Lite

[job]
dump_folder = "./outputs"
description = "DeepSeek v2 training"

[profiling]
enable_profiling = false
save_traces_folder = "profile_trace"
profile_freq = 100

[metrics]
log_freq = 10
enable_tensorboard = false
save_tb_folder = "tb"

[model]
name = "DeepSeek"
flavor = "v2"
norm_type = "rmsnorm"  # layernorm / np_layernorm / rmsnorm
tokenizer_path = "./assets/tokenizer/tokenizer.model"
# converters = "float8"

[optimizer]
name = "AdamW"
lr = 4e-3  # 4.2e-4
eps = 1e-15
# B1 = 0.9 / B2 = 0.95
[lr_scheduler]
warmup_steps = 600
lr_min = 0.1

[training]
batch_size = 2
seq_len = 4096
max_norm = 1.0  # grad norm clipping
steps = 20
compile = false
dataset = "c4"

[parallelism]
data_parallel_replicate_degree = 1
data_parallel_shard_degree = -1
tensor_parallel_degree = 8
enable_async_tensor_parallel = false
pipeline_parallel_degree = 1
context_parallel_degree = 1

[checkpoint]
enable_checkpoint = false
folder = "checkpoint"
interval = 500
model_weights_only = false
export_dtype = "float32"
async_mode = "disabled"  # ["disabled", "async", "async_with_pinned_mem"]

[activation_checkpoint]
mode = "full"  # ['none', 'selective', 'full']
