# Qwen3.6

This is a lightweight version-specific package for Qwen3.6 checkpoints. The
model architecture and implementation remain in `torchtitan/models/qwen3_5/`
because Qwen3.6 uses the same Qwen3.5 Hugging Face architecture.

This directory owns only:

- the Qwen3.6 model flavors;
- documentation for Qwen3.6 checkpoint-specific behavior.

The shared model, Gated DeltaNet, RoPE, vision encoder, sharding,
parallelization, and state-dict adapter keep their existing Qwen3.5 names.

## Supported Parallelisms

Qwen3.6 reuses the Qwen3.5 model implementation and parallelization (see the
note above), so it supports the same parallelisms documented in
[Qwen3.5's Supported Parallelisms](../qwen3_5/README.md#supported-parallelisms):

| Feature | Notes |
|---------|-------|
| FSDP / HSDP | Decoder sharded per-layer; vision encoder sharded as a single unit (one AllGather) |
| Tensor Parallelism (TP) | With Sequence Parallel; head-sharded TP on GatedDeltaNet projections |
| Expert Parallelism (EP) | For the `35B-A3B` MoE variant |
| Pipeline Parallel (PP) | Vision encoder assigned to first stage; 1F1B and Interleaved1F1B schedules |
| Context Parallelism (CP) | Attention Gym native CP for GatedDeltaNet; full attention supports K/V all-gather with contiguous, head-tail, or PTRR partitions and Ulysses with contiguous partitions |
| Sample Packing | Opt-in via `MMSamplePackingConfig` |

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

## Hugging Face assets

The former training configurations expected assets at
`assets/hf/Qwen3.6-27B` and `assets/hf/Qwen3.6-35B-A3B`, respectively. These
paths remain useful starting points for user-owned recipes, but TorchTitan does
not ship an unverified training recipe for either flavor.
