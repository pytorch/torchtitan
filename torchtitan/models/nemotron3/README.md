# Nemotron-3 Model Family

Nemotron-3 is a family of hybrid Mamba-Transformer Mixture-of-Experts (MoE) models combining Mamba-2 SSM and Grouped-Query Attention (GQA) layers with granular routed experts.

## Supported Flavors

- `nemotron_debugmodel`: CI/local testing flavor
- `nemotron_4b`: 3.97B dense hybrid model, 42 layers of Mamba-2 / MLP / attention (no MoE)
- `nemotron_31b`: 31.6B total / ~3.2B active Nano MoE model, 52 layers (23 Mamba / 23 MoE / 6 attention, 128 experts, top-6 routing)

## Download Tokenizer

```bash
python scripts/download_hf_assets.py --repo_id nvidia/nemotron-3-nano --assets tokenizer
```

## Training

```bash
# Debug model (used for CI and testing)
MODULE=nemotron3 CONFIG=nemotron_debugmodel ./run_train.sh

# Nemotron-3 4B
MODULE=nemotron3 CONFIG=nemotron_4b ./run_train.sh

# Nemotron-3 Nano 31B
MODULE=nemotron3 CONFIG=nemotron_31b ./run_train.sh
```

## Fused Mamba-2 scan

The Mamba-2 chunked scan uses `mamba-ssm`'s Triton kernel when it is installed
and the model is on GPU, falling back to a pure-PyTorch reference otherwise
(CPU, or no `mamba-ssm`). The kernel is Triton-only, so it runs on both CUDA
and ROCm -- measured on 8x MI355X (gfx950), 31B at 8192 context:

| | PyTorch reference | Fused Triton |
|---|---|---|
| MFU | 0.63% | **2.65%** |
| tokens/sec | 724 | **3,042** |
| peak memory | 114.3 GiB | **67.1 GiB** |

Install it without the Mamba-1 CUDA extension (which is unrelated and does not
build on ROCm):

```bash
MAMBA_FORCE_BUILD=FALSE MAMBA_SKIP_CUDA_BUILD=TRUE pip install --no-build-isolation mamba-ssm
```

Set `NEMOTRON_DISABLE_FUSED_MAMBA=1` to force the reference path (for A/B
timing or bisecting numerics).

See [`config_registry.py`](./config_registry.py) for available configuration options.

## Checkpoint Conversion

`NemotronStateDictAdapter` supports bidirectional checkpoint conversion between Hugging Face format and TorchTitan DCP via `scripts/checkpoint_conversion/`.
