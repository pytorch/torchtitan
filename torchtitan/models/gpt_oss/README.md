# gpt-oss Model in torchtitan

## Quick Start
```bash
CONFIG_FILE="./torchtitan/models/gpt_oss/train_configs/debug_model.py" ./run_train.sh
```

## Supported Features
- FSDP/HSDP, TP, EP, ETP
- Grouped matrix multiplication for efficient computation


## TODO
1. More parallelism support: CP, PP
