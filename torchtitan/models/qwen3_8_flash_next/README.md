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
hash IDs into fixed per-owner request slots; forward exchanges the requested
vectors, and backward returns their gradients to the owners. Each of the 16
hash heads owns a contiguous row range, so an owner receives at most one ID
per token from each head meeting its rows: two heads at DP16 and DP64, which
bounds the slots at `2T` per owner without depending on the data.

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
| EP | Existing routed-expert placements and all-to-all dispatcher |
| PP | Real process groups; stages exchange all four residual streams |

EP shards experts independently of PLE: at FSDP8/EP4 each rank owns one eighth
of the PLE rows and one fourth of the experts, with FSDP2 within each expert
shard. CPU offload and deferred FSDP reduction for PLE are unsupported. CP is
rejected at config time.

TP keeps residual and PLE activations replicated, shards QSA heads and PLE
feature columns in addition to DP row ownership, and all-reduces the indexer
teacher's head probabilities before max pooling each block. The TP-replicated
indexer contributes its KL gradient once across the TP ranks.

```bash
NGPU=2 MODULE=torchtitan_recipes.tests.models.qwen3_8_flash_next \
  CONFIG=qwen38_flash_next_debugmodel_tp2 ./run_train.sh
```

Non-final PP stages send `[T, 4 * D]` and the final stage owns the stream mixer
and LM head. Every stage receives the original token IDs and document positions,
so PLE can live on any stage. The engine preprocesses inputs on a rank's first
local model part, which reaches the PLE table through a weak reference when the
table belongs to a later local part (interleaved schedules); only the owning
part registers its parameters and checkpoint keys. The DistMuon policy is not
supported under PP: only one stage owns the PLE table, and the optimizer
container requires every pattern to match on every model part. Fake process
groups are rejected because their static stage metadata assumes `[T, D]`.

```
PP rank 0: [tok_embeddings] [layers.1-3, incl. PLE]
PP rank 1: [layers.0]       [layers.4-7, mixer, norm, lm_head]
            local part 0     local part 1
```

```bash
NGPU=8 MODULE=torchtitan_recipes.tests.models.qwen3_8_flash_next \
  CONFIG=qwen38_flash_next_debugmodel_fsdp2_tp2_ep4_pp2 ./run_train.sh
```

## Numerical Parity and Tests

QSA attention uses Attention Gym's fused `gather_attn` kernel on CUDA. With
separate values and grouped KV heads it supports only SM100/SM103 (Blackwell)
in BF16; other GPUs raise in Attention Gym. The GPU integration test therefore
runs in the B200 suite, and CPU tests use Attention Gym's reference kernels.

Results for reduced models using identical weights, measured on GB300 with the
[HF parity script](../../../scripts/checkpoint_conversion/numerical_tests_qwen3_8_flash_next.py):

| Comparison | Max output error | Parameter gradient cosine |
| --- | ---: | ---: |
| Text, BF16 | `4.15e-3` | `0.999929` |
| Vision, FP32 | `3.35e-7` | `>0.999999999999` |
| Image/video, BF16 decoder + FP32 vision | `3.91e-3` | `0.999952` |

One of 2,496 text logits exceeds the script's forward bound (`4.15e-3` versus
`4.03e-3`); against an FP32 copy of the HF model, torchtitan's error on it is
smaller than the BF16 HF reference's.

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
- Qualify released-model capacity and throughput, including table offload and
  longer-context attention.
