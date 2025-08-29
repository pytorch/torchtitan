## Auto Parallel

requires installing git@github.com:pytorch-labs/autoparallel.git

`CONFIG_FILE="./torchtitan/models/llama3/train_configs/debug_model.toml" ./run_train.sh --model.name llama3_auto_parallel --parallelism.tensor_parallel_degree 4`

Use simplefsdp's autobucketing pass:

`CONFIG_FILE="./torchtitan/models/llama3/train_configs/debug_model.toml" ./run_train.sh --model.name llama3_auto_parallel --parallelism.tensor_parallel_degree 4 --experimental.enable_simplefsdp_passes --compile.enable`

(or llama3-8b.toml)
