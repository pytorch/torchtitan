# Qwen3.8

This is a lightweight version-specific package for Qwen3.8 checkpoints. The
model architecture and implementation remain in `torchtitan/models/qwen3_5/`
because Qwen3.5 and Qwen3.8 use the same Qwen3.5 Hugging Face architecture.

This directory owns only:

- the Qwen3.8 model registry;
- Qwen3.8 training recipes and Hugging Face asset paths;
- documentation for Qwen3.8 checkpoint-specific behavior.

The shared model, Gated DeltaNet, RoPE, vision encoder, sharding,
parallelization, and state-dict adapter keep their existing Qwen3.5 names.

## Model variants

| Variant | Type | LLM dim | Layers | Heads | KV heads | Experts | Top-k | Vision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `27B` | Dense | 5120 | 64 | 24 | 4 | - | - | Yes |
| `2.4T-A95B` | MoE | 8192 | 92 | 64 | 4 | 512 | 10 | No |

Qwen3.8-27B is architecturally identical to Qwen3.5-27B. The 2.4T-A95B
checkpoint scales the same MoE decoder and stores its text-only language-model
weights under `model.*` instead of the multimodal `model.language_model.*`
prefix.

The Hugging Face configs add `output_gate_type: "swish"`; this matches the
existing SiLU/Swish gate in TorchTitan's Gated DeltaNet. Full-attention output
gating remains sigmoid. Released MTP weights are not imported or exported
because this model path currently trains only the primary next-token decoder.

Pre-quantized FP8 repositories are not supported by the checkpoint adapter;
use the non-FP8 checkpoints as conversion inputs.

## Usage

```bash
MODULE=qwen3_8 CONFIG=qwen38_27b ./run_train.sh
```

Qwen3.5 configurations remain available separately:

```bash
MODULE=qwen3_5 CONFIG=qwen35_0_8b ./run_train.sh
```
