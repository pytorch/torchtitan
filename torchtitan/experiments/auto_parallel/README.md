## Auto Parallel

requires installing git@github.com:pytorch-labs/autoparallel.git

`CONFIG_FILE="./torchtitan/models/llama3/train_configs/debug_model.toml" ./run_train.sh --model.name llama3_auto_parallel --parallelism.tensor_parallel_degree 4`

Use autobucketing pass:

`CONFIG_FILE="./torchtitan/models/llama3/train_configs/debug_model.toml" ./run_train.sh --model.name llama3_auto_parallel --parallelism.tensor_parallel_degree 4 --experimental.enable_autobucketing_passes "aten" --compile.enable`

Set `experimental.enable_autobucketing_passes` to `aten` to enable aten level bucketing, and to `inductor` to enable simplefsdp inductro bucketing pass.

(or llama3-8b.toml)
