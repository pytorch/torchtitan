# FLUX Model in torchtitan

## Overview
This directory contains the implementation of the [FLUX](https://github.com/black-forest-labs/flux/tree/main) model in torchtitan. In torchtitan, we showcase the pre-training process of text-to-image part of the FLUX model.

## Supported Features

| Feature | Status | Details |
|---|---|---|
| **FSDP** | ✅ Supported | Custom `apply_fsdp` for Flux architecture (img_in, time_in, vector_in, txt_in, double_blocks, single_blocks, final_layer) |
| **HSDP** | ✅ Supported | Hybrid sharding via `dp_replicate` + `fsdp` mesh |
| **Tensor Parallel** | ❌ Not supported | No TP implementation in `parallelize_flux` |
| **Pipeline Parallel** | ❌ Not supported | `pipelining_fn=None` in TrainSpec |
| **Context Parallel** | ✅ Supported | Custom `apply_cp` for Flux: applies to `img_attn`, `txt_attn`, and `inner_attention` in double_blocks + single_blocks; `sdpa` type only |
| **DDP** | ❌ Not supported | Not implemented in `parallelize_flux` |
| **Activation Checkpoint** | ✅ Supported | Custom AC using `ptd_checkpoint_wrapper` on double_blocks and single_blocks |
| **torch.compile** | ❌ Not supported | No compile logic in `parallelize_flux` |
| **Float8** | ❌ Not supported | No Float8 implementation |
| **MXFP8** | ❌ Not supported | No MXFP8 implementation |
| **Gradient Accumulation** | ❌ Not supported | Raises: `"FLUX doesn't support gradient accumulation for now."` |
| **HF Checkpoint Interop** | ✅ Supported | `FluxStateDictAdapter` for HF ↔ DCP conversion |
| **Distributed Checkpointing** | ✅ Supported | With special handling: re-shards FSDP modules before evaluation |
| **Custom Trainer** | ✅ Required | `FluxTrainer` — handles autoencoder, T5/CLIP encoders, custom forward/backward |
| **Loss Function** | MSE | Image generation uses `build_mse_loss`, not cross-entropy |
| **Tokenizer** | N/A | Uses T5 + CLIP encoders instead of text tokenizer |
| **Encoder Parallelism** | ✅ Supported | FSDP applied to T5 encoder via `parallelize_encoders` |

## Model Variants

| Flavor | Hidden Dim | Heads | Double Blocks | Single Blocks | Notes |
|---|---|---|---|---|---|
| `flux-dev` | 3072 | 24 | 19 | 38 | Full FLUX.1-dev |
| `flux-schnell` | 3072 | 24 | 19 | 38 | Same architecture as flux-dev |
| `flux-debug` | 1536 | 12 | 2 | 2 | Tiny model for development/CI |

## Prerequisites
Install the required dependencies:
```bash
pip install -r requirements-flux.txt
```

## Usage
First, download the autoencoder model from HuggingFace with your own access token:
```bash
python scripts/download_hf_assets.py --repo_id black-forest-labs/FLUX.1-dev --additional_patterns ae.safetensors --hf_token <your_access_token>
```

This step will download the autoencoder model from HuggingFace and save it to the `assets/hf/FLUX.1-dev/ae.safetensors` file.

Run the following command to train the model on a single GPU:
```bash
./torchtitan/models/flux/run_train.sh
```

If you want to train with other model args, run the following command:
```bash
CONFIG_FILE="./torchtitan/models/flux/train_configs/flux_schnell_model.toml" ./torchtitan/models/flux/run_train.sh
```

## Parity Checks (HF Baseline Comparison)

Flux includes a `FluxStateDictAdapter` for checkpoint conversion with HuggingFace. The parity check approach for diffusion models differs from language models:

1. **Forward-pass comparison:** Load the same weights into torchtitan and the reference [black-forest-labs/flux](https://github.com/black-forest-labs/flux) implementation. Run a forward pass with identical noise input and conditioning, then compare MSE between output predictions.
2. **Visual quality assessment:** Generate images from both implementations using the same random seed and prompt, and visually compare outputs.

> **Note:** The Flux `FluxTrainer` uses a custom `forward_backward_step` and evaluator. Forward-pass parity should be checked against the original Flux implementation rather than HF `AutoModelForCausalLM`.

## Performance

No dedicated Flux performance benchmarks have been published yet. Community benchmarks are welcome — see [`benchmarks/README.md`](/benchmarks/README.md) for submission guidelines.

**Known performance considerations:**
- Flux uses a custom trainer (`FluxTrainer`) with T5/CLIP encoder overhead on the forward pass.
- CP is the primary multi-GPU scaling strategy since TP and PP are not yet available.
- The autoencoder is loaded separately and is not parallelized.

## CI

Supported with periodically running integration tests on 8 GPUs, and unit tests.

## TODO
- [ ] More parallelism support (Tensor Parallelism, Pipeline Parallelism, etc.)
- [ ] Implement the `num_flops_per_token` calculation in `get_nparams_and_flops()` function
- [ ] Add `torch.compile` support
- [ ] Add gradient accumulation support
- [ ] Add Float8 / MXFP8 support
- [ ] Publish performance benchmarks
