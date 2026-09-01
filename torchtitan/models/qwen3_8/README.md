# Qwen3.8: Hybrid Attention Models

This directory implements the Qwen3.8 checkpoints whose Hugging Face model
types remain `qwen3_5` and `qwen3_5_moe_text`. Qwen3.8-Flash-Next is excluded:
it uses the separate `qwen4_exp` architecture.

## Overview

The supported Qwen3.8 models combine:
- **Hybrid Decoder**: 75% GatedDeltaNet (linear attention) + 25% full attention with output gating and partial RoPE.
- **Optional Vision Encoder**: The 27B model includes a Vision Transformer (ViT) with 2D RoPE and bilinear-interpolated learned position embeddings; 2.4T-A95B is text-only.
- **Patch Merger**: Reduces vision sequence length by merging spatial patches (e.g., 2x2 patches -> 1 token).
- **MRoPE**: Interleaves RoPE from temporal, height, and width position IDs in decoder layers.
- **MoE variant**: Routed experts + shared expert with sigmoid gate.

## Vision Scatter

- `tok_embeddings` produces text token embeddings of shape `B×S`.
- `vision_encoder` produces visual token embeddings of shape `N×L`.
- Valid visual tokens (excluding padding) are scattered into the placeholder positions in the text sequence, as illustrated below (credit: [@lkhphuc](https://github.com/lkhphuc)).

<img width="1398" height="840" alt="VLM Architecture" src="https://github.com/user-attachments/assets/63fcbbc1-c587-4a63-8246-411cb72f5789" />

Note: the diagram shows each patch mapping to one vision token. In practice, the Patch Merger groups `merge_size²` patches into one token (e.g., `merge_size=2` → 4 patches per token), reducing the vision sequence length by `merge_size²`.

## Prerequisites

Install the additional dependencies:

```bash
pip install av torchvision flash-linear-attention
```

## Model Variants

### Qwen3.8

| Variant | Type | LLM dim | Layers | Heads | KV Heads | Experts | Top-k | Vision |
|---------|------|---------|--------|-------|----------|---------|-------|--------|
| 27B | Dense | 5120 | 64 | 24 | 4 | - | - | Yes |
| 2.4T-A95B | MoE | 8192 | 92 | 64 | 4 | 512 | 10 | No |

Qwen3.8-27B is architecturally identical to Qwen3.5-27B. The new 2.4T-A95B
checkpoint scales the existing MoE decoder and stores its language-model
weights under `model.*` instead of the multimodal `model.language_model.*`
prefix.

This package's model registry exposes only Qwen3.8 variants. The sibling
`torchtitan/models/qwen3_5/` package keeps the Qwen3.5 model registry and
released training recipes while reusing the implementation in this directory.
Hugging Face's retained `qwen3_5` architecture identifiers for Qwen3.8 are an
implementation detail.

The Hugging Face configs add `output_gate_type: "swish"`; this matches the
existing SiLU/Swish gate in TorchTitan's Gated DeltaNet. Full-attention output
gating remains sigmoid. Released MTP weights are not imported or exported
because this model path currently trains the primary next-token decoder only.

The pre-quantized FP8 repositories are not supported by the checkpoint adapter;
use the non-FP8 checkpoints as conversion inputs.

## Datasets

| Dataset | Type | Format |
|---------|------|--------|
| `cc12m` | Image-text pairs | WebDataset (streaming) |
| `cc12m-test` | Image-text pairs | Local WebDataset (for testing) |

## Supported Parallelisms

| Feature | Notes |
|---------|-------|
| FSDP / HSDP | Decoder sharded per-layer; vision encoder sharded as a single unit (one AllGather) |
| Tensor Parallelism (TP) | With Sequence Parallel; head-sharded TP on GatedDeltaNet projections |
| Expert Parallelism (EP) | For MoE variants |
| Pipeline Parallel (PP) | Vision encoder assigned to first stage; 1F1B and Interleaved1F1B schedules |
| Sample Packing | Opt-in via `MMSamplePackingConfig` |

## Numerical Parity

Qwen3.8-27B was validated against Hugging Face Transformers 5.15 on an H100
using the released checkpoint and three deterministic image-text samples:

- FP16: average KL divergence **1.64e-6**, cosine similarity **~0.99997**, and
  **100% top-1/top-5 match**.
- BF16: average KL divergence **1.93e-4** with **100% top-1/top-5 match**.
- Reconstructed image inputs matched for every pixel with maximum difference
  **1.19e-7**.

A text-only FP16 run produced KL divergence **4.36e-7** with **100% top-1/top-5
match**, validating the decoder and checkpoint conversion independently of the
vision encoder.

Parallelism correctness: bit-identical logits (max diff `0.0`) across no-parallel, FSDP, FSDP+EP, and FSDP+EP+TP configs.

Test scripts:
- `scripts/checkpoint_conversion/numerical_tests_qwen3_8.py` - HF vs TT comparison
- `scripts/checkpoint_conversion/numerical_tests_qwen3_8_shard.py` - parallelism correctness

## TODO

- Add video dataset training configs
- Add Context Parallel (CP) support

## References

- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)
