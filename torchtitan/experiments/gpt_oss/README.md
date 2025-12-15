# gpt-oss Model in torchtitan

## Quick Start
```bash
CONFIG_FILE="./torchtitan/experiments/gpt_oss/train_configs/debug_model.toml" ./run_train.sh
```

## Supported Features
- FSDP/HSDP, TP, EP, ETP
- Grouped matrix multiplication for efficient computation


## TODO
1. More parallelism support: CP, PP
2. Conversion between HF weights (StateDictAdapter)
3. Forward parity verification
4. CI support
