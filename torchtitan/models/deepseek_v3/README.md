# DeepSeek-V3 in torchtitan

DeepSeek-V3 is a Mixture-of-Experts (MoE) model with Multi-head Latent Attention (MLA). It features a distinct architecture from other MoE models with compressed KV projections and sigmoid routing.

## Supported Features

| Feature | Status | Details |
|---|---|---|
| **FSDP** | ✅ Supported | EP-aware sharding (experts on `edp_mesh`, dense layers on `dp_mesh`) |
| **HSDP** | ✅ Supported | Hybrid sharding for both dense and MoE layers |
| **Tensor Parallel** | ✅ Supported | Custom MLA-aware TP (handles `wkv_a`, `wkv_b`, `kv_norm`, `q_lora_rank` variants) |
| **Async Tensor Parallel** | ✅ Supported | Via `maybe_enable_async_tp`; requires `torch.compile` |
| **Pipeline Parallel** | ✅ Supported | All schedules: 1F1B, Interleaved 1F1B, ZBV, DualPipeV, custom CSV |
| **Context Parallel** | ⚠️ Partial | Only `sdpa` attention type supported with CP (flex/varlen raise `NotImplementedError`) |
| **Expert Parallel** | ✅ Supported | Full EP via `ExpertParallel`, `ExpertTensorParallel` |
| **DeepEP Backend** | ✅ Supported | Optional `expert_parallel_comm_backend="deepep"` for optimized expert dispatch |
| **DualPipeV** | ✅ Supported | EP-PP overlap via `DualPipeExpertParallel` |
| **DDP** | ✅ Supported | Fallback when FSDP not enabled |
| **Activation Checkpoint** | ✅ Supported | Modes: `full`, `selective`, `memory_budget` |
| **torch.compile** | ✅ Supported | MoE-aware compilation (per-submodule for MoE, whole-block for dense) |
| **Float8** | ⚠️ Partial | **Rowwise only** — tensorwise Float8 TP raises `NotImplementedError` (not yet tested) |
| **MXFP8** | ⚠️ Untested | Available via generic `model.converters=["mx"]` but not yet validated for DeepSeek-V3 |
| **Gradient Accumulation** | ✅ Supported | Via `training.global_batch_size` config |
| **HF Checkpoint Interop** | ✅ Supported | `DeepSeekV3StateDictAdapter` for HF ↔ DCP conversion |
| **Validation** | ❌ Not yet | `build_validator_fn` not wired in TrainSpec |

## Model Variants

| Flavor | Parameters | Dimensions | Layers | Experts | MLA | Attention |
|---|---|---|---|---|---|---|
| `debugmodel` | ~1M | 256 | 6 | 8 | No (q_lora_rank=0) | sdpa |
| `debugmodel_flex_attn` | ~1M | 256 | 6 | 8 | No | flex |
| `16B` | 16B | 2048 | 27 | 64 | Yes (q_lora_rank=0) | flex |
| `236B` | 236B | 5120 | 60 | 160 | Yes (q_lora_rank=1536) | flex |
| `671B` | 671B | 7168 | 61 | 256 | Yes (q_lora_rank=1536) | flex, sigmoid scoring |

## Download Tokenizer

```bash
# DeepSeek 671B tokenizer (automatically downloads tokenizer.json and tokenizer_config.json)
python scripts/download_hf_assets.py --repo_id deepseek-ai/DeepSeek-V3.1-Base --assets tokenizer
```

```bash
# DeepSeek 16B tokenizer:
python scripts/download_hf_assets.py --repo_id deepseek-ai/deepseek-moe-16b-base --assets tokenizer
```

> **Note:** We are reusing the tokenizer from deepseek-ai/deepseek-moe-16b-base to help users test and run the 16B model. This is not the official tokenizer for the DeepSeek-V3-16B model. The DeepSeek-V3 model has a different architecture from the deepseek-moe models (different attention implementation, MoE router implementation, etc.), making it not feasible to load deepseek-moe-16b model weights into DeepSeek-V3-16B.

## Training

```bash
# Quick debug run with small model
CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/debug_model.toml" ./run_train.sh
```

```bash
# 16B parameter model: adapted from older 16B parameter model from https://huggingface.co/deepseek-ai/deepseek-moe-16b-base
CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml" ./run_train.sh
```

```bash
# 671B parameter model
CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_671b.toml" ./run_train.sh
```

## HF → DCP Checkpoint Conversion

We implemented StateDictAdapter to perform HuggingFace safetensor to DCP format conversion. Currently, we only support conversion from HF checkpoints to DCP checkpoints offline (using CPU plain tensor).

```bash
python scripts/checkpoint_conversion/convert_from_hf.py <hf_checkpoints_dir> <dcp_output_dir> --model_name deepseek_v3 --model_flavor 671B
```

## Parity Checks (HF Baseline Comparison)

DeepSeek-V3 includes a `DeepSeekV3StateDictAdapter` for bidirectional checkpoint conversion with HuggingFace. To verify numerical parity:

1. Convert a HuggingFace checkpoint to DCP format (see above).
2. Load the same weights into both torchtitan and HF `AutoModelForCausalLM`.
3. Compare forward-pass output logits — the methodology is the same as described in [`scripts/checkpoint_conversion/README.md`](/scripts/checkpoint_conversion/README.md). KL divergence should be near-zero (order of 1e-13) for a correct conversion.

> **Note:** The MLA (Multi-head Latent Attention) implementation differs from standard multi-head attention. The state dict adapter handles the necessary weight transformations for `wkv_a`, `wkv_b`, and compressed KV projections. Ensure your `config.json` matches the DeepSeek-V3 architecture when loading in HuggingFace.

## Performance

No dedicated DeepSeek-V3 performance benchmarks have been published by the torchtitan team yet. Community benchmarks are welcome — see [`benchmarks/README.md`](/benchmarks/README.md) for submission guidelines.

**Known performance considerations:**
- Tensorwise Float8 TP is not yet tested and raises `NotImplementedError` — rowwise Float8 is the supported path.
- MoE models benefit significantly from Expert Parallel; EP degree should scale with expert count.
- DeepEP backend can improve expert dispatch performance for large expert counts.

## TODO

- [ ] Add tensorwise Float8 TP support (currently raises `NotImplementedError`)
- [ ] Add validation support (`build_validator_fn`)
- [ ] Publish performance benchmarks
- [ ] Add flex/varlen attention support with Context Parallel
