# Muse Glimmer

This is a Torchtitan implementation of Muse Glimmer.
It ports both the **text** decoder and an optional
 **vision/multimodal** path (a vision encoder + adapter that
`MuseGlimmerModel` owns and runs inside `forward`).

## Quick start

```bash
# 8 GPUs
NGPU=8 MODULE="muse_glimmer" CONFIG="muse_glimmer_debugmodel" ./run_train.sh
```

The debug config uses the `c4_test` dataset, so make sure the test tokenizer
assets exist (`./tests/assets/tokenizer`).

## Config flavors

Config functions live in [`config_registry.py`](./config_registry.py); model
configs are built in [`__init__.py`](./__init__.py).

| config | flavor | dim | layers | heads (Q/KV) | head_dim | vocab | vision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| `muse_glimmer_debugmodel` | `debugmodel` | 256 | 8 | 4 / 2 | 64 | 2048 | — |
| `muse_glimmer_30b` | `30B` | 6656 | 52 | 32 / 2 | 128 | 202048 | — |
| `muse_glimmer_debugmodel_mm` | `debugmodel_mm` | 256 | 8 | 4 / 2 | 64 | 2048 | ✓ |
| `muse_glimmer_30b_mm` | `30B_mm` | 6656 | 52 | 32 / 2 | 128 | 202048 | ✓ |

## Architecture highlights

- **Gain-centered RMSNorm** (`RMSGainCenterNorm`): effective scale is
  `weight + gain_center`, `weight` initialized to 0.
- **Scaleless RMSNorm** for QK-norm and the token-embedding norm.
- **GQA attention** with QK-norm before RoPE, a tuned query scale, and a sigmoid
  output gate applied before `wo`.
- **iRoPE**: RoPE skipped on NoPE layers (`every_n_layers_nope = 4`), complex
  RoPE backend with `theta = 500_000`.
- **Per-layer sliding-window vs global attention** from a cyclic pattern;
  implemented with the **Flex** attention backend currently.
- **Post-norm residuals** and a SwiGLU FFN.
- **`SoftCappedLinear` lm_head** applying the output multiplier + tanh soft-cap
  inside the head (so it stays correct under `ChunkedLossWrapper`).

## Multimodal (vision)

The `*_mm` flavors make `MuseGlimmerModel` own a vision encoder + adapter
([`vision_encoder.py`](./vision_encoder.py)) and run them inside `forward`,
scattering the projected features into token embeddings at `vision_mask`
positions.

## Parallelism support

Parallelism is applied in [`parallelize.py`](./parallelize.py) (FSDP, HSDP, TP,
SP, CP, `torch.compile`, and PP). Sharding for Muse Glimmer-specific modules is defined
in [`sharding.py`](./sharding.py). CP and PP are not yet supported for the
multimodal path.
