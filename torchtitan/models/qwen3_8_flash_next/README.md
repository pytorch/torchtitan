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

`debugmodel` builds the text-only `Qwen38FlashNextModel`; `debugmodel_mm` and
`125B-A6B` build `Qwen38FlashNextMultimodalModel`, which adds the vision
encoder.

The released vision encoder has 27 layers, width 1,152 and 16-pixel patches.
Its merger produces 2,560-wide decoder embeddings. The text vocabulary has
248,320 entries. The
[`qwen38_flash_next_125b_dist_moe`](../../../torchtitan_recipes/models/qwen3_8_flash_next.py)
recipe fine-tunes the released decoder on 16 GB300s (8 x 16K-token
microbatches per GPU per step) with Dist-MoE and accelerator graphs at 379
TFLOP/s per GPU with natural routing, and 416 TFLOP/s per GPU (16.7% MFU)
when expert routing is forced to round-robin (perfect load balance); longer
contexts and the multimodal model at this scale are unqualified.

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

The weight is a DTensor on the dense SPMD mesh (`dp`, `cp`, `tp`), excluded
from FSDP all-gathers. Each DP owner keeps contiguous rows (`Shard(0)`); under
TP, its TP ranks split those rows' features (`Shard(1)`). TP ranks hold the
same token IDs, so only DP ranks exchange IDs and vectors, at the local feature
width, and TP then gathers the looked-up features.
Preprocessing deduplicates hash IDs and packs them into one fixed-size slot of
`C` IDs per owner; forward exchanges the requested vectors, and backward
returns their gradients to the owners. Fixed slots keep the all-to-all shapes
static, so `C` must cover the worst case for any data.

Each token reads one row from each of the 16 hash heads, so a rank requests at
most `16*T` IDs in total. That total is the ideal: one rank may need all
`16*T`, so no fixed-size scheme can send less.

The simple choice, a `16*T` slot per owner, is always safe but sends
`P*16*T`, `P` times the ideal (`P` = DP degree). We use a tighter slot size `C`
instead. Each head owns a contiguous row range, so an owner receives at most
`min(T, rows of the head it owns)` IDs from each head. `C` is that sum over
heads, maximized over owners. It depends only on `T` and the table layout, so
every rank computes the same `C` without communication.

Overhead is `P*C / (16*T)`, traffic relative to the ideal: 1.0 is ideal, and
the simple `16*T` slot would give `P`.

| DP | `C` at `T=4096` | Overhead | `C` at `T=16384` | Overhead |
| --- | ---: | ---: | ---: | ---: |
| 8 | 8,604 | 1.05 | 33,180 | 1.01 |
| 16 | 4,508 | 1.10 | 16,796 | 1.03 |
| 64 | 4,508 | 4.40 | 16,796 | 4.10 |

Each head covers about 1/16 of the table. Up to DP16 an owner holds one or two
heads, so slots stay within 10% of the ideal. Beyond DP16 several owners
split one head and each still reserves `T` slots for it, so the overhead grows
with `P`.

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
See [ROW_SHARDED_EMBEDDING.md](ROW_SHARDED_EMBEDDING.md)
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
| TP / SP | QSA heads and PLE feature columns; degree must divide the 2 QSA KV heads |
| EP | Existing routed-expert placements; all-to-all dispatcher by default, Dist-MoE with accelerator graphs in the 125B recipe and the B200 graph test |

EP shards experts independently of PLE: at FSDP8/EP4 each rank owns one eighth
of the PLE rows and one fourth of the experts, with FSDP2 within each expert
shard. CPU offload and deferred FSDP reduction for PLE are unsupported. PP and
CP are rejected at config time.

## Numerical Parity and Tests

QSA attention uses Attention Gym's fused `gather_attn` kernel on CUDA. With
separate values and grouped KV heads it supports only SM100/SM103 (Blackwell)
in BF16; other GPUs raise in Attention Gym. The GPU integration tests therefore
run in the B200 suite (including an FSDP2 + EP2 Dist-MoE test that captures and
replays forward/backward as one accelerator graph), and CPU tests use Attention
Gym's reference kernels.

Results for reduced models using identical weights, measured on GB300 with the
[HF parity script](../../../scripts/checkpoint_conversion/numerical_tests_qwen3_8_flash_next.py):

| Comparison | Max output error | Parameter gradient cosine |
| --- | ---: | ---: |
| Text, BF16 | `4.64e-3` | `0.999926` |
| Vision, FP32 | `3.35e-7` | `>0.999999999999` |
| Image/video, BF16 decoder + FP32 vision | `3.17e-3` | `0.999953` |

Both BF16 models round differently from the exact result, so the script gates
logits against an FP32 copy of the HF model: torchtitan's max error must stay
within twice the BF16 HF reference's. Measured: text `4.03e-3` versus
`4.28e-3`, image/video `4.34e-3` versus `4.59e-3`. The table's output errors
are against the BF16 HF reference and are informational.

PLE, like the GDN layers, does not skip padding inside a document; the data
pipeline starts trailing padding at document position 0. QSA excludes padding
from its blocks.

Uninterrupted training is bitwise reproducible. Resumed training is not, because
the shared compiled `flex_attention` in the vision encoder picks static or
dynamic shapes from the process's history
([#6093](https://github.com/pytorch/torchtitan/issues/6093)).

## TODO

- Replace the model-local indexer teacher with
  [the shared teacher](https://github.com/pytorch/torchtitan/pull/5863) once it
  lands.
- Qualify the released model beyond the 16K-token, 16-GB300 fine-tuning
  recipe: table offload, longer-context attention, and the multimodal model.
