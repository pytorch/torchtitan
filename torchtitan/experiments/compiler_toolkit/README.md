# Compiler Toolkit

Exploring toolkit-style use of the compiler stack for authoring parallel models.

Joint Graph based Training Prototype:
- DTensor based model authoring
- Trace joint graph
- Apply optimizations to the joint/fw/bw graphs
  - regional inductor compilation
  - fsdp bucketing/prefetching for comm/compute overlap
- Run using the aot_compile_joint_with_descriptors API

## DeepSeek v3

**SimpleFSDP + TP + EP**
```shell
NGPU=4 CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/debug_model.toml ./run_train.sh --model.name compiler_toolkit.deepseek_v3 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --parallelism.expert_parallel_degree=2 --activation_checkpoint.mode none
```

**SimpleFSDP + TP + EP + FlexAttention**
```shell
NGPU=4 CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/debug_model.toml ./run_train.sh --model.name compiler_toolkit.deepseek_v3 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --parallelism.expert_parallel_degree=2 --activation_checkpoint.mode none --model.flavor=debugmodel_flex_attn
```

## llama3

**SimpleFSDP + TP**
```shell
NGPU=8 CONFIG_FILE=./torchtitan/models/llama3/train_configs/debug_model.toml ./run_train.sh --model.name compiler_toolkit.llama3 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=4
```

**SimpleFSDP + TP + FlexAttention**
```shell
NGPU=8 CONFIG_FILE=./torchtitan/models/llama3/train_configs/debug_model.toml ./run_train.sh --model.name compiler_toolkit.llama3 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=4 --model.flavor=debugmodel_flex_attn
```
