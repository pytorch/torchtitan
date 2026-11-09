# Qwen3.8-Flash-Next

## Overview

Flash-Next reuses Qwen3.5's Gated DeltaNet (GDN), vision encoder and model lifecycle.

- **Gated residuals:** four hidden streams; each attention/MoE sublayer reads one
  activation and writes a gated update to all four streams.
- **Hybrid attention:** three GDN layers followed by one Qwen Sparse Attention
  (QSA) layer. QSA selects complete four-token blocks plus the causal tail.
- **MoE:** routed experts and a shared expert with a sigmoid gate.
- **PLE:** hashed bigram/trigram embeddings inject lexical features in layer 2.
- **Vision:** image/video features replace placeholder embeddings before the
  decoder; MRoPE uses temporal, height and width coordinates.

## Prerequisites

For multimodal recipes, install the additional dependencies:

```bash
pip install -r .ci/docker/requirements-vlm.txt
```

## Model Variants

| Variant | Decoder width | Layers | QSA heads / KV heads | QSA head width | Experts / top-k |
| --- | ---: | ---: | --- | ---: | --- |
| `debugmodel` | 256 | 8 | 4 / 2 | 64 | 8 / 2 |
| `debugmodel_mm` | 256 | 8 | 4 / 2 | 64 | 8 / 2 |
| `125B-A6B` | 2,560 | 48 | 24 / 2 | 256 | 512 / 10 |

The released vision encoder has 27 layers, width 1,152 and 16-pixel patches.
Its merger produces 2,560-wide decoder embeddings. The text vocabulary has
248,320 entries. Released-model training capacity and throughput are unqualified;
the training recipes use reduced models.

## PLE: Lexical Features and Sharding

PLE addresses its table with the original token IDs, independently of the
hidden activations and multimodal rotary coordinates. Exactly one decoder
layer owns PLE. Eight bigram and eight trigram hash heads each retrieve a
160-wide vector:

```text
token IDs [T]
  -> document-local n-grams
  -> hash IDs [T,16]
  -> table lookup [T,16,160]
  -> lexical features [T,2560]
```

The table is `[320001536,160]`: 51.2B parameters. Each hash head uses its own
row range. The 128 numbered HF parts are checkpoint partitions; they do not
determine the number of training owners.

PLE projects the lexical features into keys `[T,4,2560]` and values
`[T,2560]`. Each hidden stream supplies a query; normalized query/key scores
produce gates `[T,4]`. The gated values form `[T,4,2560]`, receive a dilated
depthwise convolution, and are added to the residual streams.

The weight is a `Shard(0)` DTensor on the FSDP storage mesh, excluded from FSDP
all-gathers. Each DP owner keeps contiguous rows. Preprocessing deduplicates
hash IDs and prepares the all-to-all request plan; forward exchanges only
selected vectors, and backward returns their gradients to the owners.

PLE needs fixed row ownership and individual vector lookups. All-to-all routing
and local `F.embedding` meet these needs while reusing TorchTitan's DTensor,
optimizer, clipping and checkpointing, with less integration work than TorchRec's
broader embedding stack.

```text
8-row table, 2 DP owners:
  rank 0 owns rows 0..3; rank 1 owns rows 4..7
  requests [6,1,6] -> unique requests [1,6]
  row 1 goes to rank 0; row 6 goes to rank 1
  returned vectors are restored as [E6,E1,E6]
```

Adam, clipping and DCP use native DTensor support; HF conversion avoids
gathering the full table. FP32 parameters, gradients and Adam state cost 95.37 GiB
per owner at DP=8, before the backbone and activations.
See [ROW_SHARDED_EMBEDDING.md](../../distributed/ROW_SHARDED_EMBEDDING.md)
for details.

Hash history resets after EOS. Document boundaries at position zero reset
hashing, PLE convolution and QSA visibility.

## Optimizer

The [optimizer recipe](../../../torchtitan_recipes/models/qwen3_8_flash_next.py)
implements the parameter assignment in
[the technical report, section 3.1](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/4f58f4ddd855bedaf7eadcd53bbbb1b3362cecff/tech_report.pdf#page=16):

| Optimizer | Parameters |
| --- | --- |
| DistMuon | Attention Q/K/V and output projections, GDN Q/K/V and output projections, routed/shared expert gate/up/down matrices, PLE key/value projections |
| AdamW | Token embeddings, output head, MoE router, GR low-rank projections, attention/GDN output gates, GDN decay/beta, norms and convolutions |
| Adam, no weight decay | PLE lookup table |

DistMuon uses Nesterov momentum 0.95, eight
[Polar Express iterations](https://github.com/liangyuwang/FSDP-Canzona/blob/426e8c50363cc3401e64eeb80c9e6e2451ea2ed4/matrix_based_optimizer/optimizers/muon.py),
epsilon `1e-14` and `match_rms_adamw` scaling. QKV and gates use separate
`Linear` parameters; `BlockShard` orthogonalizes Q/K/V per head.
The multimodal recipe keeps vision on AdamW.

## Supported Parallelisms

| Feature | Notes |
| --- | --- |
| FSDP | Per-layer backbone sharding; persistent row ownership for PLE |
| EP | Existing routed-expert placements and all-to-all dispatcher |

EP shards experts independently of PLE: at FSDP8/EP4 each rank owns one eighth
of the PLE rows and one fourth of the experts, with FSDP2 within each expert
shard. CPU offload and deferred FSDP reduction for PLE are unsupported.

## Numerical Parity and Tests

Results for reduced models using identical weights, measured on GB300 with the
[HF parity script](../../../scripts/checkpoint_conversion/numerical_tests_qwen3_8_flash_next.py):

| Comparison | Max output error | Parameter gradient cosine |
| --- | ---: | ---: |
| Text, BF16 | `3.97e-3` | `0.999923` |
| Vision, FP32 | `3.35e-7` | `>0.999999999999` |
| Image/video, BF16 decoder + FP32 vision | `3.91e-3` | `0.999946` |

Offline GB300 comparisons also verified unsharded versus `FSDP=2, EP=1/2`
forward/backward parity for packed text and image/video inputs, with QSA chunk
size 1 for text.

CPU tests cover indexer gradients with no complete blocks; the multimodal
integration does not explicitly force that imbalance across ranks.

A full-checkpoint save/load check on two GB300s restored model and optimizer
tensors bitwise. Resumed training's maximum loss/gradient-norm differences are
`1.80e-4` / `5.27e-4`; the first observed difference is in vision gradients.

## TODO

- Replace QSA's chunked reference attention and teacher with shared kernels/loss
  after [Attention Gym's grouped training path](https://github.com/meta-pytorch/attention-gym/pull/702)
  and [the shared teacher](https://github.com/pytorch/torchtitan/pull/5863) land.
  Current checkpoints bound saved activations per chunk, but indexer ranking
  still performs quadratic work; released-length throughput is unqualified.
- Qualify released-model capacity and throughput, including table offload and
  longer-context attention.
