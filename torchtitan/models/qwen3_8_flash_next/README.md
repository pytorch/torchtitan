# Qwen3.8-Flash-Next

Flash-Next reuses Qwen3.5 Gated DeltaNet, vision, FSDP and optimizer lifecycle.
Its model package implements four-stream gated residuals, Qwen Sparse Attention
(QSA), lexical embeddings (PLE), and the checkpoint's router rounding.

## Reference and checkpoint

- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
  checkpoint revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- Transformers `Qwen4ExpForCausalLM`, `Qwen4ExpVisionModel`, and
  `Qwen4ExpForConditionalGeneration` at revision
  `494470d677d05d44153d235d795e0d01040307ab`.
- The adapter maps 1,627 released decoder/vision keys, including 128 numeric
  lexical shards. Pinned name/shape digests verify the complete meta inventory.
  The 31 MTP keys are excluded from offline conversion; DCP requests only mapped
  model keys. MTP is outside this training model.

## Architecture

| Path | Released shape or configuration |
| --- | --- |
| Decoder | 48 layers, model width 2,560, vocabulary 248,320 |
| Gated residual | Four streams, grouped offset RMSNorm, gate rank 320 |
| Attention | Three GDN layers followed by one QSA layer |
| QSA | 24 query / 2 KV heads, head width 256, partial MRoPE |
| Indexer | Four heads of width 128, four-token blocks, 2,048-token budget plus causal tail |
| MoE | 512 experts, top 10, routed/shared width 640, sigmoid shared gate |
| PLE | Layer 2, 16 bigram/trigram heads of width 160, gated lexical injection |
| Lexical table | 320,001,536 rows by 160 columns, 51.2B parameters |
| Vision | 27 layers, width 1,152, 16-pixel patches, merger output width 2,560 |

```text
original IDs ----> PLE hashes/lookup ----> lexical residual
                        |
embeddings [T,D] -> repeat [T,4*D] -> gated read [T,D]
                                         |
                                    GDN/QSA or MoE
                                         |
                                gated write [T,4*D]
                                         |
                            final stream mixer -> [T,D] -> LM head
```

PLE uses original token IDs and one-dimensional document positions, independently
of multimodal rotary coordinates. The hash history resets after EOS, as in
the pinned reference; EOS does not reset PLE convolution or QSA visibility.
Position zero additionally resets all three paths for packed documents, matching
separate reference executions of each document. Debug flavors use the test
tokenizer's EOS ID 2003; the released flavor uses 248044.

QSA caches complete-block/tail metadata once per input. Equal ReLU scores select
earlier blocks first, so masked future tokens cannot change past selections.
HF has an unspecified cutoff tie policy; the HF fixture requires selection
margins and separately checks natural block/expert choices.

`indexer_loss=None` freezes the indexer before FSDP and runs selection without an
autograd graph. This supports a checkpoint's fixed selector. Training from
scratch requires an explicit KL objective, as in the recipes below. Its detached
teacher and backbone inputs train only the indexer. Empty/short documents keep
finite zero indexer gradients so all DP ranks participate in FSDP reductions.
Zero gradients allow AdamW state/decay updates on entirely empty steps.

## Build and train

Supported parallelism is FSDP, TP with or without sequence parallelism, EP and
real pipeline parallelism. CP and fake pipeline process groups are rejected.
TP degree must divide the two QSA KV heads, so degrees 1 and 2 are supported.
Flavors are `debugmodel`, `debugmodel_mm`, and `125B-A6B`.

```python
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.attention import QSAIndexerLoss

config = build_model_config(
    "debugmodel",
    parallelism=ParallelismConfig(
        data_parallel_shard_degree=2, expert_parallel_degree=2
    ),
    seq_len=64,
    indexer_loss=QSAIndexerLoss.Config(coeff=1.0, reduce_mesh="loss"),
)
```

```bash
NGPU=2 MODULE=torchtitan_recipes.tests.models.qwen3_8_flash_next \
  CONFIG=qwen38_flash_next_debugmodel_ep2 ./run_train.sh

python -m tests.integration_tests.run_tests outputs/flash_next_mm \
  --test_suite models --execution_mode real_pg --ngpu 2 \
  --test_name qwen3_8_flash_next_mm_fsdp
```

The multimodal integration entry consumes `qwen38_flash_next_debugmodel_mm` with
image fusion, PLE and QSA at 1,024 tokens. The text fixture uses 64 tokens and ten
training steps. Reduced fixtures retain both attention types and routed/shared
experts.

EP uses the shared all-to-all dispatcher and expert sharding. Attention, shared
experts and PLE remain on the dense mesh; PLE rows belong to DP owners. Routed
experts use the sparse mesh's EP axis and any remaining expert data parallelism.
For FSDP8/EP4/TP1, each rank owns one eighth of the PLE rows and one fourth of
the experts, with FSDP2 within each expert shard.

## PLE ownership

PLE uses the shared [row-sharded lookup](../../distributed/ROW_SHARDED_EMBEDDING.md).
DP ranks retain only their contiguous table rows. Preprocessing routes unique
hash IDs once per microbatch; forward exchanges selected BF16 vectors, and
backward returns gradients to the FP32 row owners. The table is excluded from
FSDP gathers. Duplicate gradients accumulate on requesters before owner summation,
and ordinary AdamW updates local rows after microbatch accumulation.

The PLE weight is a DTensor parameter on the backbone's storage mesh, with DP
storage axes using `Shard(0)`. The lookup computes through its local view;
the gradient and ordinary AdamW moments retain the same DTensor placements.
The existing clipper and native DCP handle this layout directly. The recipes
apply existing optimizer configurations for `r"\.ngram_embedding\.weight$"`
and `r".*"`, with identical hyperparameters. Native checkpoints can reshard on
resume. The HF adapter maps local slices to global numbered parts without
gathering the table.

Fresh initialization uses owner-seeded normal streams and depends on owner count.
Same-checkpoint comparisons qualify training numerics. Table rows must divide
the DP degree because SPMD meta allocation pads uneven shards. This is checked
when the owner mesh is known. CPU offload, deferred
FSDP gradient reduction, and a reduction dtype different from the master dtype
are rejected.

Real PP uses PyTorch's group-level FSDP finalization. Backbone gradients
accumulate in FP32 on each requester before DP reduction. PLE sums each
microbatch's BF16 requester gradients in FP32 on row owners, then accumulates
those owner gradients. Both compute the same globally token-normalized sum with
different floating-point association. PP coverage qualifies native determinism
and C4 training; bitwise parity with the former full-table FSDP path is not
claimed for PP.

The FP32 table parameters, gradients and two AdamW moments total 762.94 GiB
globally, or 95.37 GiB per owner with DP=8, before the backbone and activations.
Removing the gather does not make the complete default FP32 AdamW state fit on
eight GB300s; released-model capacity and throughput remain unqualified.

## Tensor parallelism

TP keeps residual/PLE activations replicated, shards QSA heads and lexical
feature columns in addition to DP row ownership, and reduces detached head
probabilities before pooling each block. The replicated indexer contributes its
KL gradient once across the TP ranks.

```bash
NGPU=2 MODULE=torchtitan_recipes.tests.models.qwen3_8_flash_next \
  CONFIG=qwen38_flash_next_debugmodel_tp2 ./run_train.sh
```

The eight-GPU recipe combines FSDP2, TP2, EP4 and PP2 with four virtual stages,
Interleaved1F1B and full activation remat. PLE belongs to virtual stage 2, the
second local part of PP rank 0. Only that part owns the table. The first local
part prepares lookup routing through a weak reference to the owning embedding;
it registers no additional parameters. Optimizer patterns are validated across
PP ranks, and stages build optimizers only for their matching local parameters.
Stages exchange `[T, 4 * D]` and receive original token IDs and
document positions independently of multimodal rotary coordinates. The final
stage owns the stream mixer and LM head. The shared engine rejects SPMD
typechecking under PP; the TP/EP recipe checks types separately.

```bash
NGPU=8 MODULE=torchtitan_recipes.tests.models.qwen3_8_flash_next \
  CONFIG=qwen38_flash_next_debugmodel_all_parallelism ./run_train.sh
```

## Numerical coverage

The CPU tests cover residual/norm equations, hash and convolution boundaries,
selected-attention backward, teacher KL, frozen/empty indexer gradients, router
rounding, padding exclusion and auxiliary lifecycle under full/selective remat.
The two-GPU FSDP+EP test composes differing document lengths, padding, accumulation
and remat. The adapter roundtrip uses 16 numbered shards to detect lexical-sort
mistakes, and verifies all released names and shapes from pinned headers.
Owned-table tests additionally cover mixed-precision duplicate accumulation,
active global clipping, async native checkpoints and model/optimizer resume on
fewer owners. Real DTensor HF fragments cover row/feature intersections, empty
owners, BF16 conversion and distributed safetensors save, consolidation and load.
HF conversion performs zero collectives.

The actual eight-layer model exercises fused AdamW and full remat. Two-GB300
ten-step same-checkpoint comparisons of native DTensor storage with plain owned
rows match first-step logits and 190 gradient shards, loss/norm and all 763 final
checkpoint tensors bitwise. Full remat and resume after five updates, including
loading a plain-row checkpoint, preserve those results. Both sources use the same
committed deterministic GDN policy. The owned lookup also has reduced-model
parity with the former full-table FSDP embedding.

Eight-GB300 C4 runs compare native DTensor and plain owned-row storage with full
remat. DP2/TP2/EP4/PP2 uses four virtual stages and Interleaved1F1B; ten-step
loss/norm and all 763 final checkpoint tensors match bitwise. A repeated native
run and native resume from a plain-row checkpoint at step 5 reach the identical
final state. HSDP2x2/TP2/EP4 with SPMD type checking also matches ten-step metrics
and all 763 tensors. These checks use native asynchronous checkpoints and the
same committed deterministic GDN policy. The numerical provenance records the
pinned sources, runtime and local postprocessing of the saved GPU artifacts.
