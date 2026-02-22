# Llama 3 in torchtitan

Llama 3.1 is the primary and most thoroughly tested model in torchtitan. It serves as the reference implementation for all parallelism techniques and training optimizations.

## Supported Features

| Feature | Status | Details |
|---|---|---|
| **FSDP** | ✅ Supported | Per-parameter sharding via `fully_shard` with configurable `reshard_after_forward` |
| **HSDP** | ✅ Supported | Hybrid sharding via `dp_replicate` + `fsdp` mesh dims |
| **Tensor Parallel** | ✅ Supported | Full sequence parallel on attention and FFN (ColwiseParallel / RowwiseParallel) |
| **Async Tensor Parallel** | ✅ Supported | Overlaps TP communication with computation; requires `torch.compile` |
| **Pipeline Parallel** | ✅ Supported | All schedules: 1F1B, Interleaved 1F1B, ZBV (Zero Bubble), DualPipeV, custom CSV |
| **Context Parallel** | ✅ Supported | Types: `sdpa` (default), `flex`, `varlen` — selected via model flavor |
| **DDP** | ✅ Supported | Fallback when FSDP not enabled |
| **Activation Checkpoint** | ✅ Supported | Modes: `full`, `selective` (op-level SAC), `memory_budget` |
| **torch.compile** | ✅ Supported | Per-TransformerBlock compilation |
| **Float8** | ✅ Supported | Both tensorwise and rowwise recipes; composable with TP |
| **MXFP8** | ✅ Supported | Via `model.converters=["mx"]` — requires SM100+ (Blackwell: B100/B200/GB200) or ROCm gfx950+ |
| **Gradient Accumulation** | ✅ Supported | Via `training.global_batch_size` config |
| **HF Checkpoint Interop** | ✅ Supported | `Llama3StateDictAdapter` for bidirectional HF ↔ DCP conversion |
| **Distributed Checkpointing** | ✅ Supported | Sync and async modes, including async with pinned memory |

## Model Variants

| Flavor | Parameters | Dimensions | Layers | Heads | KV Heads | Notes |
|---|---|---|---|---|---|---|
| `debugmodel` | ~1M | 256 | 6 | 16 | 16 | For development and CI |
| `debugmodel_flex_attn` | ~1M | 256 | 6 | 16 | 16 | FlexAttention + block causal mask |
| `debugmodel_varlen_attn` | ~1M | 256 | 6 | 16 | 16 | Variable-length attention |
| `8B` | 8B | 4096 | 32 | 32 | 8 | Standard Llama 3.1 8B |
| `8B_flex` | 8B | 4096 | 32 | 32 | 8 | 8B + FlexAttention |
| `8B_varlen` | 8B | 4096 | 32 | 32 | 8 | 8B + variable-length attention |
| `70B` | 70B | 8192 | 80 | 64 | 8 | Llama 3.1 70B |
| `405B` | 405B | 16384 | 126 | 128 | 8 | Llama 3.1 405B |

## Download Tokenizer

```bash
# Get your HF token from https://huggingface.co/settings/tokens
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...
```

## Training

```bash
# Quick debug run (no GPU required with fake backend)
COMM_MODE=fake_backend CONFIG_FILE="./torchtitan/models/llama3/train_configs/debug_model.toml" ./run_train.sh

# 8B model on 8 GPUs
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh

# 70B model on 256 GPUs (FSDP 32 × TP 8)
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_70b.toml" ./run_train.sh

# 405B model on 512 GPUs (FSDP 8 × TP 8 × PP 8)
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_405b.toml" ./run_train.sh
```

## HF ↔ DCP Checkpoint Conversion

```bash
# HuggingFace → DCP
python scripts/checkpoint_conversion/convert_from_hf.py <hf_dir> <dcp_dir> --model_name llama3 --model_flavor 8B

# DCP → HuggingFace
python scripts/checkpoint_conversion/convert_to_hf.py <dcp_dir> <hf_dir> --model_name llama3 --model_flavor 8B
```

## Parity Checks (HF Baseline Comparison)

Llama 3 has been verified for numerical parity against HuggingFace Transformers. The methodology:

1. **Forward-pass parity (KL Divergence):** Load the same weights into both torchtitan and HF `AutoModelForCausalLM`, run a forward pass on identical inputs, and compute KL divergence between output logit distributions. The expected KL divergence with correct state dict conversion (including RoPE permutation) is on the order of **1e-13** (essentially zero). See [`scripts/checkpoint_conversion/numerical_tests_example.py`](/scripts/checkpoint_conversion/numerical_tests_example.py) for the full script.

   ```bash
   python scripts/checkpoint_conversion/numerical_tests_example.py
   # Expected output:
   # Average loss of test from_hf is -1.45e-13
   ```

2. **Greedy decode sanity check:** Run greedy decoding on both the torchtitan model (via [`scripts/generate/test_generate.py`](/scripts/generate/test_generate.py)) and the HF model, and confirm identical output tokens. See the [generation README](/scripts/generate/README.md).

3. **Loss convergence:** Validated across 1D to 4D parallelism configurations — see the [convergence documentation](/docs/converging.md) and the loss curves below.

> **How to run a parity check for your own checkpoint:**
> 1. Convert your torchtitan checkpoint to HF format (see above).
> 2. Run `numerical_tests_example.py` with your checkpoint path.
> 3. Verify KL divergence ≈ 0 (order of 1e-13 or smaller).

## Performance Benchmarks

All benchmarks below were obtained on NVIDIA H100 SXM GPUs (96 GB HBM2e, NVLink) by the torchtitan team. For full details, see [`benchmarks/llama3_h100_202412_torchtitan.md`](/benchmarks/llama3_h100_202412_torchtitan.md).

### 8B Model

**Table 1 — 1D FSDP, 8 GPUs.** Local batch size 2, global batch size 16. Selective AC.

| Techniques | TPS/GPU | Memory (GiB) |
|---|---:|---:|
| FSDP | 5,762 | 82.4 |
| FSDP + torch.compile | 6,667 | 77.0 |
| FSDP + torch.compile + Float8 | 8,532 | 76.8 |

**Table 2 — FSDP + CP + compile + Float8, 8 GPUs.** Local batch size 1. Full AC.

| Parallelism | Seq Length | TPS/GPU | Memory (GiB) |
|---|---:|---:|---:|
| FSDP 8, CP 1 | 32,768 | 3,890 | 83.9 |
| FSDP 4, CP 2 | 65,536 | 2,540 | 84.2 |
| FSDP 2, CP 4 | 131,072 | 1,071 | 84.0 |
| FSDP 1, CP 8 | 262,144 | 548 | 84.5 |

**Table 3 — 1D FSDP, 128 GPUs.** Local batch size 2, global batch size 256. Selective AC.

| Techniques | TPS/GPU | Memory (GiB) |
|---|---:|---:|
| FSDP | 5,605 | 67.0 |
| FSDP + torch.compile | 6,514 | 62.0 |
| FSDP + torch.compile + Float8 | 8,380 | 61.8 |

**MFU note:** 1D Llama 3.1 8B on 8 or 128 H100s without Float8 achieves **33–39% MFU** (with or without torch.compile). MFU is not reported for Float8 runs because mixed Tensor Core usage (BF16 + FP8) makes the definition ambiguous.

### 70B Model

**Table 4 — 2D (FSDP + TP) + compile + Float8, 256 GPUs.** FSDP 32 × TP 8. Local batch size 16. Full AC.

| Techniques | TPS/GPU | Memory (GiB) |
|---|---:|---:|
| 2D | 829 | 71.9 |
| 2D + Async TP | 876 | 67.6 |

### 405B Model

**Table 5 — 3D (FSDP + TP + PP) + compile + Float8 + Async TP, 512 GPUs.** FSDP 8 × TP 8 × PP 8. Local batch size 32. Full AC.

| PP Schedule | TPS/GPU | Memory (GiB) |
|---|---:|---:|
| 1F1B | 100 | 82.5 |
| Interleaved 1F1B | 128 | 72.7 |

**Table 6 — 4D (FSDP + TP + PP + CP) + compile + Float8 + Async TP + 1F1B, 512 GPUs.** TP 8 × PP 8. Local batch size 8. Full AC.

| Parallelism | Seq Length | TPS/GPU | Memory (GiB) |
|---|---:|---:|---:|
| FSDP 8, CP 1 | 32,768 | 76 | 75.3 |
| FSDP 4, CP 2 | 65,536 | 47 | 75.9 |
| FSDP 2, CP 4 | 131,072 | 31 | 77.1 |
| FSDP 1, CP 8 | 262,144 | 16 | 84.9 |

### Async TP Speedup (June 2025)

See [`benchmarks/asyncTP_llama3_h100_2025-06_torchtitan.md`](/benchmarks/asyncTP_llama3_h100_2025-06_torchtitan.md).

| Model | Quantization | Vanilla TP | Async TP | Speedup |
|---|---|---:|---:|---:|
| 70B (256 H100s) | BF16 | 597 | 652 | 1.09× |
| 70B (256 H100s) | Float8 tensorwise | 810 | 942 | 1.16× |
| 8B (64 H100s) | BF16 | 4,378 | 4,809 | 1.10× |
| 8B (64 H100s) | Float8 tensorwise | 5,078 | 5,570 | 1.10× |

## Loss Convergence

Validated with the Llama 3.1 8B model on the C4 dataset using controlled settings (fixed seed, fixed local batch size 4, FSDP 8, 3000 steps, warmup 600 steps). The convergence guidelines are documented in [`docs/converging.md`](/docs/converging.md).

| Configuration | Parallelism | Techniques |
|---|---|---|
| 1D baseline | FSDP 8 | Default |
| 3D test | FSDP 8 + TP 2 + PP 2 | compile, Float8, Async TP, Interleaved 1F1B |
| 4D test | FSDP 8 + TP 2 + CP 2 + PP 2 | compile, Float8, Async TP, Interleaved 1F1B |
| CP stress test | FSDP 8 + CP 8 | Default |

All configurations converge to the same loss trajectory. See the [loss curves image](/assets/images/loss_curves.png).

## TODO

- [ ] Uneven `seq_len` handling in TP (current limitation across all models)
