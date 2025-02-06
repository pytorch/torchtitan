#!/bin/bash

# read -p "write world_size : " _ENV_WORLD_SIZE
# read -p "write master addr : " _ENV_MASTER_ADDR
# export ENV_WORLD_SIZE="$_ENV_WORLD_SIZE"
# export ENV_MASTER_ADDR="$_ENV_MASTER_ADDR"


if [ -z "${ENV_WORLD_SIZE}" ]; then
      echo "write world_size : "
      read  _ENV_WORLD_SIZE
      export ENV_WORLD_SIZE="$_ENV_WORLD_SIZE"
fi


if [ -z "${ENV_MASTER_ADDR}" ]; then
      echo "write master addr : " 
      read _ENV_MASTER_ADDR
      export ENV_MASTER_ADDR="$_ENV_MASTER_ADDR"
fi

if [ -z "${ENV_NODE_RANK}" ]; then
      echo "write node rank : "
      read  _ENV_NODE_RANK
      export ENV_NODE_RANK="$_ENV_NODE_RANK"
fi

TOKENIZER_DIR_PATH="./torchtitan/datasets/tokenizer/original"
if [ -d "$TOKENIZER_DIR_PATH" ]; then
      echo ""
else
      echo "HF_TOKEN: " 
      read HF_TOKEN
      python3 torchtitan/datasets/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original" --hf_token=$HF_TOKEN
fi


if [ -z "${CONFIG_FILE}" ]; then
      echo "write CONFIG_FILE path : "
      read  _ENV_CONFIG_FILE
      export CONFIG_FILE="$_ENV_CONFIG_FILE"
fi


DEBUG_TOML_PATH="./train_configs/debug_model.toml"
LLAMA_8B_TOML_PATH="./train_configs/llama3_8b.toml"


cat <<EOL > $DEBUG_TOML_PATH
# torchtitan Config.toml

[job]
dump_folder = "./outputs"
description = "Llama 3 debug training"
print_args = false
use_for_integration_test = true

[profiling]
enable_profiling = false
save_traces_folder = "profile_trace"
profile_freq = 10
enable_memory_snapshot = false
save_memory_snapshot_folder = "memory_snapshot"

[metrics]
log_freq = 1
disable_color_printing = false
enable_tensorboard = false
save_tb_folder = "tb"
enable_wandb = false

[model]
name = "llama3"
flavor = "debugmodel"
norm_type = "rmsnorm"  # layernorm / np_layernorm / rmsnorm / fused_rmsnorm
# test tokenizer.model, for debug purpose only
tokenizer_path = "./tests/assets/test_tiktoken.model"

[optimizer]
name = "AdamW"
lr = 8e-4

[training]
batch_size = $ENV_WORLD_SIZE
seq_len = 2048
warmup_steps = 2  # lr scheduler warm up, normally 20% of the train steps
max_norm = 1.0  # grad norm clipping
steps = 10
data_parallel_replicate_degree = 1
data_parallel_shard_degree = -1
tensor_parallel_degree = 1
compile = false
dataset = "c4_test"  # supported datasets: c4_test (2K), c4 (177M)

[experimental]
context_parallel_degree = 1
pipeline_parallel_degree = $ENV_WORLD_SIZE
enable_async_tensor_parallel = false

[checkpoint]
enable_checkpoint = false
folder = "checkpoint"
interval_type = "steps"
interval = 10
model_weights_only = false
export_dtype = "float32"
async_mode = "disabled"  # ["disabled", "async", "async_with_pinned_mem"]

[activation_checkpoint]
mode = 'selective'  # ['none', 'selective', 'full']
selective_ac_option = '2'  # 'int' = ac every positive int layer or 'op', ac based on ops policy

[float8]
enable_float8_linear = false
EOL

cat <<EOL > $LLAMA_8B_TOML_PATH
# torchtitan Config.toml
# NOTE: this toml config is a preset for 64 A100 GPUs.

[job]
dump_folder = "./outputs"
description = "Llama 3 8B training"

[profiling]
enable_profiling = true
save_traces_folder = "profile_trace"
profile_freq = 100

[metrics]
log_freq = 10
enable_tensorboard = true
save_tb_folder = "tb"

[model]
name = "llama3"
flavor = "8B"
norm_type = "rmsnorm"  # layernorm / np_layernorm / rmsnorm / fused_rmsnorm
tokenizer_path = "./torchtitan/datasets/tokenizer/original/tokenizer.model"

[optimizer]
name = "AdamW"
lr = 3e-4

[training]
batch_size = $ENV_WORLD_SIZE
seq_len = 8192
warmup_steps = 200  # lr scheduler warm up
max_norm = 1.0  # grad norm clipping
steps = 1000
data_parallel_replicate_degree = 1
data_parallel_shard_degree = -1
tensor_parallel_degree = 1
compile = false
dataset = "c4"

[experimental]
context_parallel_degree = 1
pipeline_parallel_degree = $ENV_WORLD_SIZE

[checkpoint]
enable_checkpoint = false
folder = "checkpoint"
interval_type = "steps"
interval = 500
model_weights_only = false
export_dtype = "float32"
async_mode = "disabled" # ["disabled", "async", "async_with_pinned_mem"]

[activation_checkpoint]
mode = 'selective'  # ['none', 'selective', 'full']
selective_ac_option = 'op'  # 'int' = ac every positive int layer or 'op', ac based on ops policy

[float8]
enable_float8_linear = false
EOL



