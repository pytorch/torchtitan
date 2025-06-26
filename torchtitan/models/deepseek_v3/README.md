Download tokenizer:

```
# DeepSeek tokenizer (automatically downloads tokenizer.json and tokenizer_config.json)
python scripts/download_tokenizer.py --repo_id deepseek-ai/DeepSeek-V3
```

Run:

Single GPU - debug_model
```
NGPU=1 LOG_RANK=0 CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/debug_model.toml" ./run_train.sh
```

FSDP:

```
NGPU=8 LOG_RANK=0 CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/debug_model.toml" ./run_train.sh --parallelism.data_parallel_shard_degree 8

# OOM
NGPU=8 LOG_RANK=0 CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml" ./run_train.sh --parallelism.data_parallel_shard_degree 8
```

PP:

for additional logging use: TORCH_LOGS=+pp

```
NGPU=2 LOG_RANK=0,1 CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/debug_model.toml" ./run_train.sh --parallelism.pipeline_parallel_degree 2

NGPU=4 LOG_RANK=0,1,2,3 CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/debug_model.toml" ./run_train.sh --parallelism.pipeline_parallel_degree 4

# works with AC=none, but why doesn't this work with AC=full?
NGPU=8 LOG_RANK=0,1,2,3,4,5,6,7 CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml" ./run_train.sh --parallelism.pipeline_parallel_degree 8 --parallelism.pipeline_parallel_schedule Interleaved1F1B
```
