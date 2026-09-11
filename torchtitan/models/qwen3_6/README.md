# Qwen3.6

This is a lightweight version-specific package for Qwen3.6 checkpoints. The
model architecture and implementation remain in `torchtitan/models/qwen3_5/`
because Qwen3.6 uses the same Qwen3.5 Hugging Face architecture.

This directory owns only:

- the Qwen3.6 model registry;
- Qwen3.6 training recipes and Hugging Face asset paths;
- documentation for Qwen3.6 checkpoint-specific behavior.

The shared model, Gated DeltaNet, RoPE, vision encoder, sharding,
parallelization, and state-dict adapter keep their existing Qwen3.5 names.

## Model variants

| Variant | Type | LLM dim | Layers | Heads | KV heads | Experts | Top-k |
|---|---|---:|---:|---:|---:|---:|---:|
| `27B` | Dense | 5120 | 64 | 24 | 4 | - | - |
| `35B-A3B` | MoE | 2048 | 40 | 16 | 2 | 256 | 8 |

Both released variants include the Qwen3.5 vision encoder. Released MTP
weights are not imported or exported because this model path currently trains
only the primary next-token decoder.

Pre-quantized FP8 repositories are not supported by the checkpoint adapter;
use the non-FP8 checkpoints as conversion inputs.

## Usage

```bash
MODULE=qwen3_6 CONFIG=qwen36_27b ./run_train.sh
```
