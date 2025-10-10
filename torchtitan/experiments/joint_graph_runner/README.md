## Joint Graph Runner

Exploring toolkit-style use of the compiler stack for authoring parallel models.

Joint Graph based Training Prototype:

Llama3
- User code: SimpleFSDP + TP
- Trace joint
- Apply passes to the joint
- Run using the Joint Graph Runner

Run with: NGPU=8 CONFIG_FILE="./torchtitan/models/llama3/train_configs/debug_model.toml" with-proxy ./run_train.sh --model.name joint_graph_runner.llama3 --compile.enable --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=4
