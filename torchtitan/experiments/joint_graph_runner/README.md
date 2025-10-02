## Joint Graph Runner

Exploring toolkit-style use of the compiler stack for authoring parallel models.

Joint Graph based Training Prototype:

Llama3
- User code: SimpleFSDP + TP
- Trace joint
- Apply passes to the joint
- Run using the Joint Graph Runner

Run with: NGPU=1 CONFIG_FILE="./torchtitan/models/llama3/train_configs/debug_model.toml" wp ./run_train.sh --model.name joint_graph_runner.llama3 --compile.enable
