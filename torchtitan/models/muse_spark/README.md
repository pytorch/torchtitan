# Muse Spark Open

This directory contains the TorchTitan-native text and model-level multimodal
implementation of Muse Spark Open.

The released checkpoint is
[`meta-models/Muse-Spark-Open-816B-A42B`](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B).
Stage its assets locally at `./assets/hf/Muse-Spark-Open-816B-A42B` for the
production and SFT recipes.

## Quick start

```bash
NGPU=4 MODULE=torchtitan_recipes.tests.models.muse_spark CONFIG=muse_spark_debugmodel ./run_train.sh
```

For a 10-step debug run mixing local CC12M test images with synthetic audio
and video datasets at equal Grain weights:

```bash
NGPU=8 MODULE=torchtitan_recipes.tests.models.muse_spark CONFIG=muse_spark_debugmodel_mm ./run_train.sh
```

The recipe packs mixed examples into 512-token microbatches and uses
DP2/CP2/TP2/EP2 with sequence parallelism.

## Model flavors

| flavor | modalities | dim | layers | heads (Q/KV) | default context | experts / top-k |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `debugmodel` | text | 256 | 8 | 8 / 4 | 512 | 8 / 2 |
| `debugmodel_mm` | text, vision, audio | 256 | 8 | 8 / 4 | 512 | 8 / 2 |
| `816B-A42B` | text | 8192 | 62 | 128 / 16 | 32768 | 256 / 8 |
| `816B-A42B_mm` | text, vision, audio | 8192 | 62 | 128 / 16 | 32768 | 256 / 8 |

The production flavors default to 32,768-token training sequences and accept an
explicit sequence length up to the released checkpoint limit of 1,048,576.

## Modalities and checkpoints

All flavors build the vision encoder and audio encoder. Text-only flavors use empty
`model.dataset_modalities`. Set `model.dataset_modalities` to the
modalities the current dataset can supply across DP ranks, with the same value
on every rank:

- `("vision",)`: images and videos, using one shared vision forward.
- `("audio",)`: audio.
- `("vision", "audio")`: both; the multimodal flavor default.
- `()`: text-only training.

CC12M activates vision; the debug Grain mix activates both. Listed modalities
require their encoders, and undeclared media inputs raise `ValueError`.

Listed encoders run on every DP rank, with dummy inputs where media is absent,
including steps with no media on any rank. This static choice avoids a
media-presence collective and keeps FSDP all-gathers and reduce-scatters aligned;
rank-dependent skips can hang. Dummy outputs add a zero-valued dependency to token
embeddings so backward collectives run without changing text activations.

Unlisted encoders are skipped but remain in checkpoints. Encoders unused
throughout the training stage keep their parameter gradients at `None`,
so AdamW leaves their weights and optimizer state unchanged.
Keep encoder configs and optimizer parameter assignments consistent across stages.
If a custom model config omits one or both encoders, model-only native/HF loading
loads weights for the remaining modules. Subsequent checkpoints exclude the
omitted encoders.

## Architecture highlights

- GQA with per-head Q/K RMSNorm. Layers 2, 6, 10, ..., 62 use global
  attention without RoPE (counting from 1). All other layers use
  sliding-window attention with complex RoPE.
  Attention supports FlexAttention and varlen attention, with an output RMSNorm
  and a learned sigmoid output gate before the output projection.
- Gain-centered RMSNorm and learned norm-preserving residual gates.
  Attention and expert computation use BF16 inputs and weights. Routing
  scores and the residual stream through the decoder blocks use FP32.
- Compressed MoE with 256 SwiGLU experts. The router reads the input with
  8192 features. For expert computation, the input is projected to 4096
  features and normalized. Tokens are grouped by selected expert.
  Each expert output is normalized, then the outputs are weighted and
  summed. A final projection restores 8192 features.
- The router computes FP32 sigmoid scores and selects 8 experts per token
  using score + expert bias. Combination weights use the selected scores
  without bias, normalized across the 8 selected experts. The production
  scaling factor is 1.0. Expert bias is updated from token counts to balance
  expert usage without an auxiliary loss. Each update has its mean removed
  and uses coefficient `1e-3`.
- RMS-normalized embeddings and a scaled, tanh-soft-capped language-model head.

Both production and debug flavors use MoE in every layer.

## Parallelism support

| feature | status |
| --- | --- |
| Single device | Supported |
| FSDP | Supported alone or together with EP/TP+EP; active encoders use dummy inputs to keep participation aligned across DP ranks |
| Expert parallelism | Supported with the standard all-to-all dispatcher or Dist-MoE |
| Tensor parallelism | Supported with `EP >= TP`, including unequal degrees |
| Async tensor parallelism | Supported with BF16 projections |
| Sequence parallelism | Supported when both TP and EP are enabled |
| Pipeline parallelism | Supported |
| Context parallelism | FlexAttention with Ulysses or K/V all-gather; varlen with Ulysses |
| CP load balancing | Head-tail and PTRR with K/V all-gather FlexAttention; Ulysses requires no load balancer |
| Activation checkpointing | Full and region policies through `torch_remat` |
| Loss compilation | Supported |
| Model compilation | Regional compilation |
| CUDA graphs | Text-only with Dist-MoE; multimodal support is a TorchTitan gap; standard all-to-all EP requires graphs to be disabled |
| TP without EP | Not supported |
| DeepEP, HybridEP, and MinimalAsyncEP | Not supported |

Production TP must divide the 16 KV heads. EP must be at least TP and divide
both the 256 experts and `dp_shard * CP * TP` ranks.

Ulysses CP requires `TP * CP` to divide the 16 KV heads and
`context_parallel_load_balancer=None`.

K/V all-gather CP requires tokens per microbatch to be divisible by the CP
degree times the FlexAttention query block size. Media inputs remain replicated
across CP; token-aligned bank indices select each rank's local embeddings.

Dist-MoE requires Blackwell GPUs and the optional
`dist_moe` package. Dist-MoE owns dispatch, expert computation, and combine.

## TODO

- Add non-synthetic audio and video training recipes.
- Benchmark CP load balancing and PP/CP communication at longer contexts and
  production scale. Add native and Hugging Face checkpoint save/resume tests
  across PP stage layouts.
- Benchmark tensor parallelism for the audio block MLP with production waveform
  lengths and modality mixes. Use colwise `linear1` and rowwise `linear2` while
  initially keeping the convolution, norm, and residual stream TP-invariant;
  verify that sharded expanded activations outweigh the per-block reduction.
