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

For parallelisms, we support float8 all-gather for FSDP (optional) and for TP (by default for `Float8Linear`).

For scaling strategy, we currently support tensor-wise scaling with dynamic scales, and are actively working on tensor-wise scaling with delayed scales. Row-wise scaling is under exploration.

## Why Composing Float8 with `torch.distributed`
As shown below, for float8 for matmul, `torch._scaled_mm` requires both float8 tensor and scales. Scales are calculated from `max(abs)` of a high precision tensor.
```
# float32/bfloat16 matmul, `torch.mm(input, weight)`, does not require scales
# float8 matmul requires scales to ensure values can
# fit within the representable range
torch._scaled_mm(input_fp8, weight_fp8, scale_a=scale_input, scale_b=scale_weight)
```

Without considering distributed, we cast input and weight into float8 inside forward before calling torch._scaled_mm.

Considering FSDP, we can cast high precision weights (1/N on each rank) into float8, and perform float8 all-gather to save bandwidth. At the beginning of the forward, we already have the unsharded float8 weights. Similarly, we can do float8 all-gather in TP to save bandwidth. The overhead is communicating `max(abs)` across ranks but for llama herd models, we have net wins in training speed.
