This was performed by Trainy team on WhiteFiber in June 2025, to get a baseline of performance
of the Trainy platform on H200s platform over multiple hosts.

### Models

Llama 3.1 8B

### Hardware

Each host has

- 8 NVIDIA H200 GPUs connected via NVLink.
- Hosts are inter-connected with a backend RDMA fabric with 400Gb/s (Mellanox CX-7) per GPU.

### Configuration

Runs were invoked with the following, where `NUM_NODES` was `4` and `8`.

**Warning**: the command below reflects the original invocation at the time of this benchmark. The torchtitan CLI has since changed to use `--module` and `--config` flags instead of `--job.config-file`. See the current [README](/README.md) for up-to-date usage.
```
  torchrun \
    --nnodes $NUM_NODES  \
    --nproc_per_node 8 \
    --rdzv_id 101 \
    --rdzv_backend c10d \
    --rdzv_endpoint "$MASTER_ADDR:29500" \
    torchtitan/train.py \
    --job.config-file torchtitan/models/llama3/train_configs/llama3_8b.toml \
    --metrics.enable_wandb \
    --training.local_batch_size=2 \
    --training.compile \
    --model.converters="quantize.linear.float8" \
    --quantize.linear.float8.enable_fsdp_float8_all_gather \
    --quantize.linear.float8.precompute_float8_dynamic_scale_for_fsdp \
    --quantize.linear.float8.force_recompute_fp8_weight_in_bwd \
    --profiling.profile_freq 1000000
    --training.steps 2000
```

### Results

Detailed performance results and training configurations can be found in the tables below along and can visualized in [this WandB report](https://api.wandb.ai/links/asaiacai/w4c46stp). `TPS` and `Memory(GiB)` are arbitrarily sampled at the 100th iteration:

| NUM_NODES | TPS/GPU | Memory(GiB) |
| ----- | ----: | ----: |
| 4 | 10938 | 47.96 |
| 8 | 10753 | 46.97 |


### Versions and Dates

| repo | commit | date |
| --- | --- | --- |
| torch | [2.8.0a0+5228986c39](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-05.html) | 2025/05/29 |
| torchao | [0afa4c1](https://github.com/pytorch/ao/commit/0afa4c1bd28c82921e360ddbd1b27c9d6da5b947) | 2025/06/13 |
| torchtitan | [e7c0cae](https://github.com/pytorch/torchtitan/commit/e7c0cae934df78d6e9c2835f42ff1f757dc3fddc) | 2025/06/13 |
