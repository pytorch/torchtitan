# gpt-oss Model in torchtitan

## Quick Start
```bash
MODULE=gpt_oss CONFIG=gpt_oss_debugmodel ./run_train.sh
```

## Supported Features
- FSDP/HSDP, TP, EP, ETP
- Grouped matrix multiplication for efficient computation


## TODO
1. More parallelism support: CP, PP
