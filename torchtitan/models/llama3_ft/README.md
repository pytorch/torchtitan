# Llama 3 with Fault Tolerance (TorchFT)

This model extends the Llama 3 implementation with fault tolerance capabilities via [TorchFT](https://github.com/pytorch/torchft). It reuses the same model definition and parallelization code as `llama3`, adding resilience for long-running training jobs.

## Supported Features

All features from [Llama 3](../llama3/README.md) are inherited, plus:

| Feature | Status | Details |
|---|---|---|
| **All Llama3 features** | ✅ Inherited | FSDP, HSDP, TP, PP, CP, DDP, AC, compile, Float8, MXFP8 |
| **TorchFT integration** | ✅ Supported | Fault-tolerant training with replica management |
| **DiLoCo-style fragmentation** | ✅ Supported | `fragment_llm` for distributed optimization |
| **Semi-sync training** | ✅ Supported | Configurable via `fault_tolerance.semi_sync_method` |
| **Per-replica checkpointing** | ✅ Supported | Independent checkpoint per fault-tolerant replica |

## Prerequisites

```bash
pip install torchft
```

## Training

```bash
# Start the TorchFT lighthouse (coordinator)
# Then run training:
TORCHFT_LIGHTHOUSE=http://<lighthouse_ip>:<port> \
CONFIG_FILE="./torchtitan/models/llama3_ft/train_configs/debug_model.toml" ./run_train.sh
```

## Parity Checks

Same as Llama 3 — uses identical model weights and `Llama3StateDictAdapter`. Fault tolerance does not affect model numerics; the forward/backward pass is identical.

## Performance

Fault tolerance introduces minimal overhead during normal operation. The primary cost is per-replica checkpoint saving and the TorchFT coordinator communication. No separate performance benchmarks are published; throughput is expected to match Llama 3 within measurement noise during failure-free training.

## TODO

- See [Llama 3 TODO](../llama3/README.md#todo) for inherited items
