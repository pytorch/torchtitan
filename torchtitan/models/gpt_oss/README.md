# GPT-OSS Model in torchtitan

## Overview

GPT-OSS is a MoE (Mixture of Experts) transformer model with FlexAttention and sliding window support, implemented natively in torchtitan.

## Supported Features

| Feature | Status | Details |
|---|---|---|
| **FSDP** | ✅ Supported | Reuses llama4's `apply_fsdp` with MoE-aware expert FSDP sharding (`edp_mesh`) |
| **HSDP** | ✅ Supported | Hybrid sharding via `dp_replicate` + `fsdp` mesh |
| **Tensor Parallel** | ✅ Supported | Custom `apply_non_moe_tp` — embedding, attention (wq/wk/wv/wo), norm, output parallelized |
| **Pipeline Parallel** | ❌ Not supported | `pipelining_fn=None` in TrainSpec |
| **Context Parallel** | ❌ Not supported | Raises `NotImplementedError("CP support for gpt-oss model is still in progress.")` |
| **Expert Parallel (EP)** | ✅ Supported | `apply_moe_ep_tp` with `ExpertParallel` for expert sharding |
| **Expert Tensor Parallel (ETP)** | ✅ Supported | `GptossExpertTensorParallel` for combined expert + tensor parallel |
| **DDP** | ✅ Supported | Reuses llama3's `apply_ddp` |
| **DualPipeV** | ❌ Not active | Code scaffolded (`DualPipeExpertParallel`), but `get_dual_pipe_v_flag` requires PP enabled — unreachable without PP support |
| **Activation Checkpoint** | ✅ Supported | Selective op-level AC with custom `_op_sac_save_list` (mm, sdp attention, flex_attention, reduce_scatter, all_to_all) |
| **torch.compile** | ❌ Not applied | `apply_compile()` not called in `parallelize_gptoss`; infrastructure exists but not wired up |
| **Float8** | ✅ Partial | Float8 linear available via generic converter (`model.converters=["float8"]`), but Float8 tensorwise TP all-gather is disabled (`enable_float8_tensorwise_tp` hardcoded to `False`) |
| **MXFP8** | ❌ Not listed | No explicit MXFP8 support |
| **Gradient Accumulation** | ✅ Supported | Standard gradient accumulation via training config |
| **Async TP** | ❌ Disabled | Infrastructure exists in `apply_non_moe_tp` but `enable_async_tp` hardcoded to `False` in `parallelize_gptoss` |
| **FlexAttention** | ✅ Required | Only attention type supported (`attn_type: str = "flex"`) |
| **Sliding Window Attention** | ✅ Supported | Configurable `sliding_window_size` (default: 128) with `BlockMask` |
| **YaRN RoPE** | ✅ Supported | Extended context via `rope_factor`, `ntk_alpha`, `ntk_beta` |
| **Grouped MM** | ✅ Supported | For MoE expert computation; requires SM90+ (H100/H200) |
| **MoE Load Balancing** | ✅ Supported | `build_optimizers_with_moe_load_balancing`; configurable `load_balance_coeff` |
| **Loss Parallel** | ✅ Supported | Can be disabled via `disable_loss_parallel` config |
| **HF Checkpoint Interop** | ✅ Supported | `GptOssStateDictAdapter` for HF ↔ DCP conversion |

## Model Variants

| Flavor | Dim | Layers | Heads | KV Heads | Experts | Top-K | Notes |
|---|---|---|---|---|---|---|---|
| `debugmodel` | 256 | 4 | 64 | 8 | 8 | 4 | Tiny model for development/CI |
| `20b` | 2880 | 24 | 64 | 8 | 32 | 4 | ~20B parameter MoE |
| `120b` | 2880 | 36 | 64 | 8 | 128 | 4 | ~120B parameter MoE |

All variants use: `softmax` scoring, `gate_bias=True`, `route_norm=True`, `use_grouped_mm=True`, `vocab_size=201088`, `sliding_window_size=128`.

## Quick Start

```bash
CONFIG_FILE="./torchtitan/models/gpt_oss/train_configs/debug_model.toml" ./run_train.sh
```

## Parity Checks (HF Baseline Comparison)

GPT-OSS includes a `GptOssStateDictAdapter` for checkpoint conversion between HF and DCP formats.

**Methodology:**
1. Convert weights using `scripts/checkpoint_conversion/convert_from_hf.py` / `convert_to_hf.py`
2. Run forward-pass comparison on identical inputs, comparing logit outputs
3. For MoE models, verify expert routing consistency (same top-k selection, same load balancing)

> **Note:** GPT-OSS uses FlexAttention exclusively, so parity checks should account for potential minor numerical differences compared to SDPA-based implementations.

## Performance

No dedicated GPT-OSS performance benchmarks have been published yet. Community benchmarks are welcome — see [`benchmarks/README.md`](/benchmarks/README.md) for submission guidelines.

**Known performance considerations:**
- Grouped matrix multiplication (SM90+ required) is the primary performance feature for MoE experts
- FlexAttention with sliding window provides efficient attention for long sequences
- EP+ETP allows scaling across many GPUs for large expert counts (e.g., 128 experts in the 120b variant)
- `load_balance_coeff` (default: 1e-3) controls the auxiliary load-balancing loss

## TODO
- [ ] Context Parallel support (currently raises `NotImplementedError`)
- [ ] Pipeline Parallel support (also needed to activate DualPipeV EP-PP overlap)
- [ ] Enable `torch.compile` (add `apply_compile()` in `parallelize_gptoss`)
- [ ] Enable Async TP (currently `enable_async_tp` hardcoded to `False`)
- [ ] Enable Float8 tensorwise TP all-gather (currently `enable_float8_tensorwise_tp` hardcoded to `False`)
- [ ] Publish performance benchmarks (MFU, throughput)
