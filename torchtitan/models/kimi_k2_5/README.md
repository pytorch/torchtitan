# Kimi K2.5

Kimi K2.5 is a Vision-Language Model: the **DeepSeek-V3** language tower
(Multi-head Latent Attention + Mixture-of-Experts) with a **MoonViT3d** vision
encoder. It is implemented by extending `DeepSeekV3Model`, so the text tower
reuses DeepSeek-V3's attention, MoE, sharding, and checkpoint code — only the
vision tower and the multimodal forward are new.

## Architecture

- **Language tower** — DeepSeek-V3 (MLA + MoE), standard 1D RoPE.
- **Vision tower** — MoonViT3d: Linear patch embedding, learnable 2D spatial +
  fixed sinusoidal temporal position embeddings, interleaved 2D RoPE, pre-norm
  transformer blocks, temporal mean-pool + 2×2 spatial merge, then a 2-layer MLP
  projector to the LLM hidden size.
- **Multimodal forward** — vision features are scattered into the text
  embeddings at a single media placeholder (`<|media_pad|>`, id 163605); images
  and videos share it. Video is temporally pooled inside the encoder, so the LLM
  uses plain 1D RoPE (no MRoPE).

## Model variants

| Flavor | Language | Vision | Notes |
|---|---|---|---|
| `debugmodel` | 6 layers, dim 256, 8 experts | 4 layers, dim 256 | tiny, for tests |
| `debugmodel_mm` | same as `debugmodel` | + block→raster patch reorder | tiny, image-text |
| `1T-A32B` | 61 layers, dim 7168, 384 experts (top-8) | 27 layers, dim 1152, 16 heads | full Kimi K2.5 (~1T total / ~32B active) |

## Trainer configs (`config_registry.py`)

- `kimi_k2_5_debugmodel` — text-only debug (c4).
- `kimi_k2_5_debugmodel_mm` — image-text debug (cc12m).
- `kimi_k2_5_1t_a32b` — full-model template; set the tokenizer / dataset /
  parallelism degrees for the target cluster (needs many GPUs).

## Quick start

```bash
# text-only debug
torchrun --nproc-per-node=2 -m torchtitan.train \
    --module kimi_k2_5 --config kimi_k2_5_debugmodel

# image-text debug
torchrun --nproc-per-node=1 -m torchtitan.train \
    --module kimi_k2_5 --config kimi_k2_5_debugmodel_mm
```

## Parallelism

| FSDP / HSDP | TP | EP | PP | SP | CP |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |

TP/EP are config-based (`model.parallelize`); the vision encoder is FSDP-wrapped
as a single unit (folded into the root group under PP). PP places the vision
encoder on the first stage. SP and CP are disabled because the multimodal
forward needs the full sequence to scatter vision features. Example overrides:
`--parallelism.tensor_parallel_degree 2`,
`--parallelism.expert_parallel_degree 8`,
`--parallelism.pipeline_parallel_degree 2`.

## HuggingFace checkpoint conversion

`KimiK25StateDictAdapter` maps between the released HF layout and torchtitan for
both towers: MLA / MoE language weights, the vision Conv2d→Linear patch-embed
reshape, and the fused→separate vision-QKV split. It also handles the
`language_model.` prefix used by some releases.

## Known gaps / TODOs

- **Video**: Kimi's video-chunk pipeline (4-frame chunks, temporal pooling,
  timestamp prompts) is not wired in the data loader — image-text only for now.
- **Image resize**: the shared `MMDataLoader` uses `smart_resize(min/max_pixels)`,
  not Kimi's `navit_resize` (`in_patch_limit` / `patch_limit_on_one_side`), so
  resolutions / patch counts differ from the real processor (fine for
  from-scratch training, not for matching the release).
- **Quantized release**: the released checkpoint is int4 compressed-tensors; the
  adapter's quantized reader currently targets fp8 block-128, so loading the
  released (quantized) weights needs an int4 path.
