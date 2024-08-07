## Enable Float8 Training on H100s

Please install latest [TorchAO](https://github.com/pytorch/ao/tree/main/torchao/float8) to support float8 dtype
```
USE_CPP=0 python -m pip install git+https://github.com/pytorch/ao.git
```

Launch training job with the following command (or alternatively set configs in toml files)
```
CONFIG_FILE="./train_configs/llama3_8b.toml" ./run_llama_train.sh --float8.enable_float8_linear --float8.enable_fsdp_float8_all_gather --float8.precompute_float8_dynamic_scale_for_fsdp
```
* `--float8.enable_float8_linear`: swap `nn.Linear` with `Float8Linear` to perform float8 matmul.
* `--float8.enable_fsdp_float8_all_gather`: cast `Float8Linear.weight` from high precision to float8 before FSDP all-gather so we can communicate in float8 to save bandwidth.
* `--float8.precompute_float8_dynamic_scale_for_fsdp` (optional): communicate AMAX/scales efficiently in a single all-reduce for all parameters instead of doing many small all-reduce for each parameter.

For parallelisms, we support optional float8 all-gather for FSDP. We are actively working on float8 all-gather in TP.

For scaling strategy, we currently support tensor-wise scaling with dynamic scales, and are actively working on tensor-wise scaling with delayed scales. Row-wise scaling is under exploration.
