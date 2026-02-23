# Qwen3 in torchtitan

**The Qwen3 model is still under development.**

Qwen3 supports both dense and MoE architectures. Dense models use weight tying for smaller variants. MoE models use token-choice routing with auxiliary-loss-free load balancing.

## Supported Features

| Feature | Status | Details |
|---|---|---|
| **FSDP** | ✅ Supported | EP-aware sharding for MoE variants |
| **HSDP** | ✅ Supported | Hybrid sharding for both dense and MoE layers |
| **Tensor Parallel** | ✅ Supported | Custom TP with QK-norm parallelization (`SequenceParallel(sequence_dim=2)`) |
| **Async Tensor Parallel** | ✅ Supported | Requires `torch.compile` |
| **Pipeline Parallel** | ❌ Not supported | `pipelining_fn=None` in TrainSpec |
| **Context Parallel** | ⚠️ Partial | Only `sdpa` type supported; flex/varlen raise `NotImplementedError` |
| **Expert Parallel** | ✅ Supported | For MoE flavors via `apply_moe_ep_tp` |
| **DDP** | ✅ Supported | Fallback when FSDP not enabled |
| **Activation Checkpoint** | ✅ Supported | Modes: `full`, `selective`, `memory_budget` |
| **torch.compile** | ✅ Supported | MoE-aware compilation |
| **Float8** | ✅ Supported | Tensorwise Float8 TP via `Float8ColwiseParallel` / `Float8RowwiseParallel` |
| **MXFP8** | ⚠️ Untested | Available via generic `model.converters=["mx"]` but not yet validated for Qwen3 |
| **Gradient Accumulation** | ✅ Supported | Via `training.global_batch_size` config |
| **HF Checkpoint Interop** | ✅ Supported | `Qwen3StateDictAdapter` for HF ↔ DCP conversion |
| **Weight Tying** | ✅ Supported | Enabled for smaller dense models (0.6B, 1.7B, 4B) |
| **DualPipeV** | ❌ Not active | Code scaffolded, but `get_dual_pipe_v_flag` requires PP enabled — unreachable without PP support |
| **Validation** | ✅ Supported | `build_validator` wired in TrainSpec |

## Model Variants

### Dense Models

| Flavor | Parameters | Dimensions | Layers | Heads | Weight Tying |
|---|---|---|---|---|---|
| `debugmodel` | ~1M | 256 | 8 | 16 | Yes |
| `0.6B` | 0.6B | 1024 | 28 | — | Yes |
| `1.7B` | 1.7B | 2048 | 28 | — | Yes |
| `4B` | 4B | 2560 | 36 | — | Yes |
| `8B` | 8B | 4096 | 36 | — | No |
| `14B` | 14B | 5120 | 40 | — | No |
| `32B` | 32B | 5120 | 64 | — | No |

### MoE Models

| Flavor | Parameters | Experts | Top-k | Max Seq Len |
|---|---|---|---|---|
| `debugmodel_moe` | ~1M | 64 | 8 | — |
| `30B-A3B` | 30B (3B active) | 128 | 8 | 262,144 |
| `235B-A22B` | 235B (22B active) | 128 | 8 | 4,096 |

## Download Tokenizer

```bash
# Example for Qwen3 0.6B
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-0.6B --assets tokenizer

# Example for Qwen3 1.7B
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-1.7B --assets tokenizer
```

## Training

```bash
# Quick 0.6B run
CONFIG_FILE="./torchtitan/models/qwen3/train_configs/qwen3_0.6b.toml" ./run_train.sh

# MoE debug
CONFIG_FILE="./torchtitan/models/qwen3/train_configs/qwen3_moe_debug.toml" ./run_train.sh
```

## Parity Checks (HF Baseline Comparison)

Qwen3 includes a `Qwen3StateDictAdapter` for checkpoint conversion with HuggingFace. The parity check methodology is the same as Llama 3:

1. Convert HF checkpoint → DCP using `scripts/checkpoint_conversion/convert_from_hf.py`.
2. Load identical weights into torchtitan and HF `AutoModelForCausalLM`.
3. Compare forward-pass logits via KL divergence (expected: ~1e-13).

See [`scripts/checkpoint_conversion/README.md`](/scripts/checkpoint_conversion/README.md) for the full methodology.

> **Qwen3-specific note:** The QK-norm implementation uses `SequenceParallel(sequence_dim=2)` for TP, which differs from Llama's standard approach. Ensure the state dict adapter handles the QK-norm weight mapping correctly.

## Performance

No dedicated Qwen3 performance benchmarks have been published yet. Community benchmarks are welcome — see [`benchmarks/README.md`](/benchmarks/README.md) for submission guidelines.

## TODO

- [ ] CP: Only `sdpa` attention type supported; add flex/varlen CP support (blocked by RoPE embedding implementation)
- [ ] Add Pipeline Parallel support
- [ ] Learning rate verification: verify LR and schedule with real training jobs (e.g., 3k steps), or find official references
- [ ] The model should be tested against established performance benchmarks
- [ ] CI integration tests
- [ ] Add toml configs for remaining model sizes (8B, 14B, 30B-A3B, 235B-A22B)
