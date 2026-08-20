# DeepSeek V4

This folder contains the TorchTitan implementation of DeepSeek V4.

The model entry point is `torchtitan.models.deepseek_v4.model_registry`, and the
training configs are exposed from `torchtitan.models.deepseek_v4.config_registry`.
The currently registered configs are:

- `deepseek_v4_debugmodel`
- `deepseek_v4_mtp_debugmodel`
- `deepseek_v4_flash`
- `deepseek_v4_pro`

## Components

- `model.py`: decoder model and transformer block definitions.
- `attention.py`: DeepSeek V4 sparse attention variants, including sliding
  window attention, heavily compressed attention, and compressed sparse
  attention.
- `compressor.py`: KV compression and sparse-index selection helpers.
- `mhc.py`: multi-token/head-coupled branch mixing modules.
- `moe.py`: DeepSeek V4 MoE router and expert wrapper.
- `sharding.py`: declarative sharding config for TP, SP, EP, FSDP, DTensor, and
  `spmd_types`.
- `state_dict_adapter.py`: checkpoint key mapping for DeepSeek V4.

## Smoke Test

Run the debug model on 4 GPUs with FSDP2, TP2, and EP2:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 MODULE=deepseek_v4 CONFIG=deepseek_v4_debugmodel ./run_train.sh \
  --training.steps 1 \
  --metrics.log_freq 1 \
  --parallelism.data_parallel_shard_degree 2 \
  --parallelism.tensor_parallel_degree 2 \
  --parallelism.expert_parallel_degree 2
```

To exercise the `spmd_types` backend, add:

```bash
--parallelism.spmd_backend spmd_types
```

## Status

The debug model has been smoke-tested with 4 GPUs using FSDP2, TP2, EP2, and the
`spmd_types` backend. The optional MTP path is available through
`deepseek_v4_mtp_debugmodel`. Checkpoint compatibility and larger-scale convergence
validation should be verified before using the larger configs for production
training.
