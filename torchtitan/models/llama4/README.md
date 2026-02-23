# Llama 4 in torchtitan

Llama 4 is a Mixture-of-Experts (MoE) transformer model. It supports both standard and iRoPE (interleaved RoPE with NoPE layers) variants, and features token-choice routing with auxiliary-loss-free load balancing.

## Supported Features

| Feature | Status | Details |
|---|---|---|
| **FSDP** | ✅ Supported | EP-aware sharding (experts on `edp_mesh`, dense layers on `dp_mesh`) |
| **HSDP** | ✅ Supported | Hybrid sharding for both dense and MoE layers |
| **Tensor Parallel** | ✅ Supported | `apply_non_moe_tp` for attention/dense FFN; `apply_moe_ep_tp` for MoE layers |
| **Async Tensor Parallel** | ✅ Supported | Via `maybe_enable_async_tp`; requires `torch.compile` |
| **Pipeline Parallel** | ✅ Supported | All schedules: 1F1B, Interleaved 1F1B, ZBV, DualPipeV, custom CSV |
| **Context Parallel** | ✅ Supported | `sdpa` (default); `flex` (for iRoPE variants) |
| **Expert Parallel** | ✅ Supported | Full EP: `ExpertParallel`, `ExpertTensorParallel`, `DeepEPExpertParallel` |
| **DeepEP Backend** | ✅ Supported | Optional `expert_parallel_comm_backend="deepep"` |
| **DualPipeV** | ✅ Supported | EP-PP overlap via `DualPipeExpertParallel` |
| **DDP** | ✅ Supported | Fallback when FSDP not enabled |
| **Activation Checkpoint** | ✅ Supported | Modes: `full`, `selective` (extended save list with `all_to_all_single`), `memory_budget` |
| **torch.compile** | ✅ Supported | MoE-aware compilation (per-submodule for MoE, whole-block for dense) |
| **Float8** | ✅ Supported | Tensorwise Float8 TP on attention and dense FFN |
| **MXFP8** | ⚠️ Untested | Available via generic `model.converters=["mx"]` but not yet validated for Llama 4 |
| **Gradient Accumulation** | ✅ Supported | Via `training.global_batch_size` config |
| **HF Checkpoint Interop** | ✅ Supported | `Llama4StateDictAdapter` for HF ↔ DCP conversion |

## Model Variants

| Flavor | Architecture | Experts | Attention | Notes |
|---|---|---|---|---|
| `debugmodel` | MoE | — | sdpa | dim=256, 6 layers, for dev/CI |
| `17bx16e` | MoE | 16 | sdpa | dim=5120, 48 layers, interleave_step=1 |
| `17bx128e` | MoE | 128 | sdpa | dim=5120, 48 layers |
| `debugmodel_irope` | MoE + iRoPE | — | flex | NoPE every 4 layers, block causal |
| `17bx16e_irope` | MoE + iRoPE | 16 | flex | iRoPE + MoE(16 experts) |
| `17bx128e_irope` | MoE + iRoPE | 128 | flex | iRoPE + MoE(128 experts) |

## Download Tokenizer

```bash
# Llama 4 tokenizer
python scripts/download_hf_assets.py --assets tokenizer --repo_id meta-llama/Llama-4-Scout-17B-16E --hf_token=...
```

## Training

```bash
# Quick debug run
CONFIG_FILE="./torchtitan/models/llama4/train_configs/debug_model.toml" ./run_train.sh

# 17B×16E model
CONFIG_FILE="./torchtitan/models/llama4/train_configs/llama4_17bx16e.toml" ./run_train.sh

# 17B×128E model
CONFIG_FILE="./torchtitan/models/llama4/train_configs/llama4_17bx128e.toml" ./run_train.sh
```

## Parity Checks (HF Baseline Comparison)

Llama 4 includes a `Llama4StateDictAdapter` for checkpoint conversion with HuggingFace. To verify numerical parity:

1. Convert a HuggingFace checkpoint to DCP format using `scripts/checkpoint_conversion/convert_from_hf.py`.
2. Load the same weights into both torchtitan and HF `AutoModelForCausalLM`.
3. Compare forward-pass output logits — methodology described in [`scripts/checkpoint_conversion/README.md`](/scripts/checkpoint_conversion/README.md).

> **MoE-specific note:** Parity checks for MoE models should verify that expert routing produces the same top-k expert selection and that the load balancing auxiliary loss (if any) matches. When using token-choice routing with auxiliary-loss-free load balancing, the expert bias updates are applied as a pre-step optimizer hook and should not affect forward-pass parity.

## Performance

No dedicated Llama 4 performance benchmarks have been published by the torchtitan team yet. Community benchmarks are welcome — see [`benchmarks/README.md`](/benchmarks/README.md) for submission guidelines.

**Expected performance considerations:**
- MoE models benefit significantly from Expert Parallel to distribute expert computation.
- DeepEP backend can improve all-to-all dispatch performance for large expert counts.
- DualPipeV provides EP-PP communication overlap for combined pipeline + expert parallelism.
- iRoPE variants with FlexAttention may have different throughput characteristics than sdpa variants.

## TODO

- [ ] Publish performance benchmarks
- [ ] Add varlen attention support with Context Parallel
- [ ] Uneven `seq_len` handling in TP
