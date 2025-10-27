## Compiler Toolkit

Exploring toolkit-style use of the compiler stack for authoring parallel models.

Joint Graph based Training Prototype:

DeepSeek v3
- DTensor based model authoring
- Trace joint graph
- Apply optimizations to the joint/fw/bw graphs
- Run using the aot_compile_joint_with_descriptors API

Run with: NGPU=4 CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/debug_model.toml ./run_train.sh --model.name compiler_toolkit.deepseek_v3 --compile.enable --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --parallelism.expert_parallel_degree=2 --activation_checkpoint.mode none
