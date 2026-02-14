# Graph-Based Training

Unified experiment merging [SimpleFSDP](../simple_fsdp/) and [Compiler Toolkit](../compiler_toolkit/) into a single framework with two compilation modes:

- **JIT mode** (`--compile.mode jit`): Uses `torch.compile` with a custom backend. Graph passes are registered to the backend and applied during just-in-time compilation.
- **AOT mode** (`--compile.mode aot`): Captures the joint forward-backward graph ahead of time and applies optimization passes directly to the FX graph modules before execution.

Both modes share the same DTensor-based SimpleFSDP model authoring and the same unified pass registry.

## Configuration

Compilation is configured via `--job.custom_config_module=torchtitan.experiments.graph_based_training.job_config`:

- `--compile.mode`: `"jit"` or `"aot"` (omit to disable compilation)
- `--compile.passes`: Comma-separated list of pass names

### Available Passes

| Pass | Modes | Description |
|------|-------|-------------|
| `auto_bucketing` | jit, aot | Automatic comm/compute overlap bucketing |
| `transformer_block_bucketing` | jit, aot | Manual per-transformer-block bucketing |
| `regional_inductor` | aot | Regional Inductor compilation |
| `cudagraph` | aot | CUDA graph capture and replay (must be last) |
| `full_inductor_compilation` | aot | Full Inductor code generation (requires `inductor_decomposition`) |
| `inductor_decomposition` | aot | Inductor decompositions on joint graph |

Constraints:
- `auto_bucketing` and `transformer_block_bucketing` are mutually exclusive
- `full_inductor_compilation` requires `inductor_decomposition`
- `cudagraph` must be the last pass in the list

## Llama3

**JIT mode (no passes)**
```shell
NGPU=4 TRAIN_FILE=torchtitan.experiments.graph_based_training.train CONFIG_FILE=./torchtitan/models/llama3/train_configs/debug_model.toml ./run_train.sh --model.name graph_based_training.llama3 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --job.custom_config_module=torchtitan.experiments.graph_based_training.job_config --compile.mode jit
```

**JIT mode + auto-bucketing**
```shell
NGPU=4 TRAIN_FILE=torchtitan.experiments.graph_based_training.train CONFIG_FILE=./torchtitan/models/llama3/train_configs/debug_model.toml ./run_train.sh --model.name graph_based_training.llama3 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --job.custom_config_module=torchtitan.experiments.graph_based_training.job_config --compile.mode jit --compile.passes auto_bucketing
```

**JIT mode + transformer-block-bucketing**
```shell
NGPU=4 TRAIN_FILE=torchtitan.experiments.graph_based_training.train CONFIG_FILE=./torchtitan/models/llama3/train_configs/debug_model.toml ./run_train.sh --model.name graph_based_training.llama3 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --job.custom_config_module=torchtitan.experiments.graph_based_training.job_config --compile.mode jit --compile.passes transformer_block_bucketing
```

**AOT mode (no passes)**
```shell
NGPU=4 TRAIN_FILE=torchtitan.experiments.graph_based_training.train CONFIG_FILE=./torchtitan/models/llama3/train_configs/debug_model.toml ./run_train.sh --model.name graph_based_training.llama3 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --job.custom_config_module=torchtitan.experiments.graph_based_training.job_config --compile.mode aot
```

**AOT mode + auto-bucketing**
```shell
NGPU=4 TRAIN_FILE=torchtitan.experiments.graph_based_training.train CONFIG_FILE=./torchtitan/models/llama3/train_configs/debug_model.toml ./run_train.sh --model.name graph_based_training.llama3 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --job.custom_config_module=torchtitan.experiments.graph_based_training.job_config --compile.mode aot --compile.passes auto_bucketing
```

**AOT mode + transformer-block-bucketing + regional-inductor**
```shell
NGPU=4 TRAIN_FILE=torchtitan.experiments.graph_based_training.train CONFIG_FILE=./torchtitan/models/llama3/train_configs/debug_model.toml ./run_train.sh --model.name graph_based_training.llama3 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --job.custom_config_module=torchtitan.experiments.graph_based_training.job_config --compile.mode aot --compile.passes transformer_block_bucketing,regional_inductor
```

**AOT mode + transformer-block-bucketing + regional-inductor + cudagraph**
```shell
NCCL_GRAPH_REGISTER=0 NGPU=4 TRAIN_FILE=torchtitan.experiments.graph_based_training.train CONFIG_FILE=./torchtitan/models/llama3/train_configs/debug_model.toml ./run_train.sh --model.name graph_based_training.llama3 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --job.custom_config_module=torchtitan.experiments.graph_based_training.job_config --compile.mode aot --compile.passes transformer_block_bucketing,regional_inductor,cudagraph
```

**AOT mode + full Inductor compilation**
```shell
NGPU=4 TRAIN_FILE=torchtitan.experiments.graph_based_training.train CONFIG_FILE=./torchtitan/models/llama3/train_configs/debug_model.toml ./run_train.sh --model.name graph_based_training.llama3 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --job.custom_config_module=torchtitan.experiments.graph_based_training.job_config --compile.mode aot --compile.passes inductor_decomposition,full_inductor_compilation
```

## DeepSeek v3

**JIT mode (SimpleFSDP + TP + EP)**
```shell
NGPU=4 TRAIN_FILE=torchtitan.experiments.graph_based_training.train CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/debug_model.toml ./run_train.sh --model.name graph_based_training.deepseek_v3 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --parallelism.expert_parallel_degree=2 --activation_checkpoint.mode none --job.custom_config_module=torchtitan.experiments.graph_based_training.job_config --compile.mode jit
```

**AOT mode (SimpleFSDP + TP + EP)**
```shell
NGPU=4 TRAIN_FILE=torchtitan.experiments.graph_based_training.train CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/debug_model.toml ./run_train.sh --model.name graph_based_training.deepseek_v3 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --parallelism.expert_parallel_degree=2 --activation_checkpoint.mode none --job.custom_config_module=torchtitan.experiments.graph_based_training.job_config --compile.mode aot
```

**AOT mode (SimpleFSDP + TP + EP + auto-bucketing)**
```shell
NGPU=4 TRAIN_FILE=torchtitan.experiments.graph_based_training.train CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/debug_model.toml ./run_train.sh --model.name graph_based_training.deepseek_v3 --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --parallelism.expert_parallel_degree=2 --activation_checkpoint.mode none --job.custom_config_module=torchtitan.experiments.graph_based_training.job_config --compile.mode aot --compile.passes auto_bucketing
```
