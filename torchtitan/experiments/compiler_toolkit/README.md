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
NGPU=4 TRAIN_FILE=torchtitan.experiments.compiler_toolkit.train MODEL=compiler_toolkit.deepseek_v3 CONFIG=compiler_toolkit_deepseek_v3_debugmodel ./run_train.sh --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --parallelism.expert_parallel_degree=2 --activation_checkpoint.mode none
```

**SimpleFSDP + TP + EP + FlexAttention**
```shell
NGPU=4 TRAIN_FILE=torchtitan.experiments.compiler_toolkit.train MODEL=compiler_toolkit.deepseek_v3 CONFIG=compiler_toolkit_deepseek_v3_debugmodel_flex_attn ./run_train.sh --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --parallelism.expert_parallel_degree=2 --activation_checkpoint.mode none
```

## llama3

**SimpleFSDP + TP**
```shell
NGPU=8 TRAIN_FILE=torchtitan.experiments.compiler_toolkit.train MODEL=compiler_toolkit.llama3 CONFIG=compiler_toolkit_llama3_debugmodel ./run_train.sh --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=4
```

**SimpleFSDP + TP + auto-bucketing**
```shell
NGPU=8 TRAIN_FILE=torchtitan.experiments.compiler_toolkit.train MODEL=compiler_toolkit.llama3 CONFIG=compiler_toolkit_llama3_debugmodel ./run_train.sh --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=4 --compile.passes autobucketing_reordering
```

**SimpleFSDP + TP + transformer-block-bucketing**
```shell
NGPU=8 TRAIN_FILE=torchtitan.experiments.compiler_toolkit.train MODEL=compiler_toolkit.llama3 CONFIG=compiler_toolkit_llama3_debugmodel ./run_train.sh --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=4 --compile.passes transformer_block_bucketing
```

**SimpleFSDP + TP + FlexAttention**
```shell
NGPU=8 TRAIN_FILE=torchtitan.experiments.compiler_toolkit.train MODEL=compiler_toolkit.llama3 CONFIG=compiler_toolkit_llama3_debugmodel_flex_attn ./run_train.sh --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=4
```

**SimpleFSDP + TP + FlexAttention + auto-bucketing + regional-inductor**

```shell
NGPU=8 TRAIN_FILE=torchtitan.experiments.compiler_toolkit.train MODEL=compiler_toolkit.llama3 CONFIG=compiler_toolkit_llama3_debugmodel_flex_attn ./run_train.sh --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=4 --compile.passes autobucketing_reordering,regional_inductor
```

**SimpleFSDP + TP + FlexAttention + transformer-block-bucketing + regional-inductor**

```shell
NGPU=8 TRAIN_FILE=torchtitan.experiments.compiler_toolkit.train MODEL=compiler_toolkit.llama3 CONFIG=compiler_toolkit_llama3_debugmodel_flex_attn ./run_train.sh --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=4 --compile.passes transformer_block_bucketing,regional_inductor
```

**SimpleFSDP + TP + FlexAttention + transformer-block-bucketing + regional-inductor + cudagraph**

```shell
NCCL_GRAPH_REGISTER=0 NGPU=8 TRAIN_FILE=torchtitan.experiments.compiler_toolkit.train MODEL=compiler_toolkit.llama3 CONFIG=compiler_toolkit_llama3_debugmodel_flex_attn ./run_train.sh --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=4 --compile.passes transformer_block_bucketing,regional_inductor,cudagraph
```

**SimpleFSDP + TP + Full Inductor compilation**

```shell
NGPU=8 TRAIN_FILE=torchtitan.experiments.compiler_toolkit.train MODEL=compiler_toolkit.llama3 CONFIG=compiler_toolkit_llama3_debugmodel ./run_train.sh --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=4 --compile.joint_passes inductor_decomposition --compile.passes full_inductor_compilation
```
