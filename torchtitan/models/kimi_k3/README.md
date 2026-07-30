# Kimi K3

This directory contains the first TorchTitan implementation of Kimi K3. The
initial scope is a topology-complete, reduced model for single-device training
and numerical comparison with the
[released HuggingFace implementation](https://huggingface.co/moonshotai/Kimi-K3).

The implementation is device-neutral. It uses PyTorch operators and does not
import accelerator-specific packages. The reference kernels prioritize
inspectable model math over throughput.

## Quick start

```bash
NGPU=1 MODULE=kimi_k3 CONFIG=kimi_k3_debugmodel ./run_train.sh
```

The multimodal data path requires `torchvision`.

## Reduced model

The `debugmodel` flavor preserves each distinct Kimi K3 forward path while
reducing widths, expert count, and depth.

| Component | Released Kimi K3 | `debugmodel` |
|---|---:|---:|
| Decoder dimension | 7168 | 256 |
| Vocabulary size | 163840 | 2048 |
| Decoder layers | 93 | 13 |
| Full MLA layers (1-based) | 4, 8, ..., 92, 93 | 4, 8, 12 |
| KDA layers | 69 | 10 |
| Dense FFN layers | 1 | 1 |
| Attention residual block size | 12 | 12 |
| Routed experts / top-k | 896 / 16 | 8 / 2 |
| Shared experts | 2 | 2 |
| Vision dimension | 1024 | 128 |
| Vision layers | 27 | 2 |
| Vision QKV dimension / heads | 1536 / 12 | 192 / 3 |

Thirteen decoder layers are intentional. They exercise two attention-residual
blocks and preserve the released model's 1-based full-attention cadence.

## Forward structure

The reference path mirrors the released implementation in these areas:

- FP32-reduction RMSNorm and SiTU activation.
- Gated MLA with low-rank query and KV projections. Kimi K3 sets
  `mla_use_nope=True`, so the RoPE-sized query/key slices are not rotated.
- KDA short causal convolutions, safe decay gate, query/key L2 normalization,
  sigmoid beta, recurrent delta-rule update, and gated output RMSNorm.
- Block-level attention residuals, including the final output residual.
- Stable LatentMoE with sigmoid top-k routing, correction bias, latent
  down/up projections, routed experts, and shared experts.
- MoonViT3d patch embedding, learned spatial positions, 2D RoPE, non-causal
  per-image attention, temporal pooling, 2x2 spatial merge, and
  PatchMergerMLPV2.
- Vision features scattered into runs of media placeholder tokens.

`KimiKDAKernel` is the optimization boundary. A future accelerated backend
should preserve its input/output contract and checkpoint schema.

## Checkpoint conversion

`KimiK3StateDictAdapter` converts between TorchTitan and an unquantized
HuggingFace state dict. It covers:

- dense, MLA, KDA, and LatentMoE decoder layers;
- attention-residual parameters;
- vision patch embedding, fused HuggingFace QKV, transformer blocks, and
  projector;
- the MoE correction bias.

The released checkpoint stores routed expert weights in MXFP4. Loading those
compressed tensors is outside this first change. Numerical comparison should
therefore instantiate the same reduced, unquantized model on both sides and
copy one state dict through the adapter.

## Tests

The CPU unit tests cover:

- the reduced layer topology;
- the explicit exact GELU against PyTorch's CPU reference;
- the KDA kernel against a direct recurrent formulation, including backward;
- a small text+image model forward and backward;
- exhaustive state-dict round-trip for that small model.

```bash
pytest -q tests/unit_tests/test_kimi_k3.py
```

## First-version limitations

- Single device only; DP, FSDP/HSDP, TP, PP, CP, and EP are rejected.
- No packed documents, activation checkpointing, `torch.compile`, or CPU
  offload.
- Image inputs are supported; video inputs are rejected.
- No generation cache.
- No optimized KDA backend.
- No MXFP4 checkpoint loading.
- No full 2.8T flavor.

These restrictions are explicit so unsupported runtime settings fail instead
of being silently ignored. Optimized kernels and distributed parallelism can be
added in follow-up changes after the reference forward is numerically locked.
