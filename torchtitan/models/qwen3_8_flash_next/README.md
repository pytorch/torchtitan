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
hidden activations and multimodal rotary coordinates. Eight bigram and eight
trigram hash heads each retrieve a 160-wide vector:

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

```text
8-row table, 2 DP owners:
  rank 0 owns rows 0..3; rank 1 owns rows 4..7
  requests [6,1,6] -> unique requests [1,6]
  row 1 goes to rank 0; row 6 goes to rank 1
  returned vectors are restored as [E6,E1,E6]
```

Local Adam updates, clipping and native DCP use the existing DTensor paths.
HF conversion intersects each owner's rows with numbered checkpoint parts
without gathering the table. Fresh initialization depends on owner count;
checkpoint resume can reshard to a different count. The model requires the
table's row count to divide the DP degree.

The FP32 table, gradients and two Adam moments total 762.94 GiB globally,
or 95.37 GiB per owner at DP=8, before the backbone and activations.
See [ROW_SHARDED_EMBEDDING.md](../../distributed/ROW_SHARDED_EMBEDDING.md)
for routing, gradient accumulation and checkpoint details.

Hash history resets after EOS. Position zero resets hashing, PLE convolution
and QSA visibility for packed documents. EOS alone does not reset convolution
or attention, matching the reference. The debug tokenizer uses EOS 2003;
the released flavor uses 248044.

## Optimizer

Both debug recipes follow the parameter assignment in
[the technical report, section 3.1](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/4f58f4ddd855bedaf7eadcd53bbbb1b3362cecff/tech_report.pdf#page=16):

| Optimizer | Parameters |
| --- | --- |
| DistMuon | Attention Q/K/V and output projections, GDN Q/K/V and output projections, routed/shared expert gate/up/down matrices, PLE key/value projections |
| AdamW | Token embeddings, output head, MoE router, GR low-rank projections, attention/GDN output gates, GDN decay/beta, norms and convolutions |
| Adam, no weight decay | PLE lookup table |

The multimodal fixture keeps the vision encoder on AdamW; the report does not
specify a separate vision policy. Debug Adam/AdamW use betas `(0.9,0.95)`.
The debug learning rates are `8e-4` for text and `5e-3` for multimodal;
these small fixtures do not reproduce the pretraining batch size or schedule.

Muon uses Nesterov momentum 0.95, eight Polar Express iterations,
normalization epsilon `1e-14`, and `0.2 * sqrt(max(rows, columns))` update
scaling. The recipe supplies the published
[Polar Express coefficient schedule](https://github.com/liangyuwang/FSDP-Canzona/blob/426e8c50363cc3401e64eeb80c9e6e2451ea2ed4/matrix_based_optimizer/optimizers/muon.py)
to the existing DistMuon implementation.

QKV and gate parameters are stored separately but computed with one fused GEMM.
`BlockShard` orthogonalizes Q/K/V per head; native expert weight dimensions
already separate gate/up matrices. GR and router weights remain on AdamW.
The split projection's backward preserves FP32 WGRAD and uses functional
accumulation. Changing from AdamW to this mixed policy changes the training
trajectory.

## Datasets and Training

| Recipe | Dataset | Tokens per sequence | Steps |
| --- | --- | ---: | ---: |
| `qwen38_flash_next_debugmodel` | `c4_test` | 64 | 10 |
| `qwen38_flash_next_debugmodel_mm` | `cc12m-test` | 1,024 | 10 |

```bash
NGPU=2 MODULE=torchtitan_recipes.tests.models.qwen3_8_flash_next \
  CONFIG=qwen38_flash_next_debugmodel ./run_train.sh

NGPU=2 MODULE=tests.integration_tests.models \
  CONFIG=qwen38_flash_next_debugmodel_ep2 ./run_train.sh
```

The recipes enable selected-block indexer KL loss. `indexer_loss=None` freezes
the indexer for a checkpoint's fixed selector; empty/short documents retain
finite zero gradients when the loss is enabled.

## Supported Parallelisms

| Feature | Notes |
| --- | --- |
| FSDP | Per-layer backbone sharding; persistent row ownership for PLE |
| EP | Existing routed-expert placements and all-to-all dispatcher |
| TP / SP, PP, CP | Rejected by this model configuration |

EP shards experts independently of PLE: at FSDP8/EP4 each rank owns one eighth
of the PLE rows and one fourth of the experts, with FSDP2 within each expert
shard. CPU offload and deferred FSDP reduction for PLE are unsupported.

## Numerical Parity and Tests

The HF reference is
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540),
using Transformers revision `494470d677d05d44153d235d795e0d01040307ab`.
The adapter covers 1,627 decoder/vision keys, including all 128 PLE parts;
name/shape digests guard the released inventory. MTP is outside this training model.

CPU tests cover feature equations, boundaries, sparse selection/backward,
auxiliary losses, fused projection equivalence and HF roundtrips. Two-GPU tests
exercise FSDP+EP, accumulation, remat, untouched PLE rows without weight decay,
FP32 projection gradients and asynchronous model/optimizer resume.

```bash
pytest tests/unit_tests/cpu/models/qwen3_8_flash_next -q
pytest tests/unit_tests/gpu/models/qwen3_8_flash_next -q
python -m scripts.checkpoint_conversion.numerical_tests_qwen3_8_flash_next \
  --device cuda --output outputs/flash_next_hf.json
```

## TODO

- Replace QSA's chunked reference attention and teacher with shared kernels/loss
  after [Attention Gym's grouped training path](https://github.com/meta-pytorch/attention-gym/pull/702)
  and [the shared teacher](https://github.com/pytorch/torchtitan/pull/5863) land.
  Current checkpoints bound saved activations per chunk, but indexer ranking
  still performs quadratic work; released-length throughput is unqualified.
- Qualify released-model capacity and throughput, including table offload and
  longer-context attention.
