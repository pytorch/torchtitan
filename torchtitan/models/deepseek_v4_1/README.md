# DeepSeek V4.1 Flash

Native TorchTitan model definitions for the text backbone, vision encoder,
Engram memory, and DSpark drafter. Architecture and checkpoint mappings are
pinned to [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6)
and its [technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/DeepSeek_V41_Tech_Report.pdf).

## Architecture

| Component | Released Flash flavor |
| --- | --- |
| Backbone | 40 layers, width 5120, four mHC branches |
| Attention | 64 heads, head width 512, query rank 1280 |
| Encoder / decoder | 20 / 20 layers; compression ratios 2 / 1 |
| Local attention | Causal sliding window of 128 tokens in every layer |
| Shared global KV producers | Layers 2, 8, 14, 20 (zero based) |
| Index producers | Layers 2, 8, 14, 20, 24, 28, 32, 36 |
| CSA2 | 512 selected keys; 2048 candidate blocks of eight keys |
| MoE | 384 routed experts, top six, one shared expert, intermediate width 2304 |
| Engram | Layers 1 and 14; orders 2 through 4, eight heads per order |
| Vision | 32 layers, width 1024, patch size 14, downsample ratio 3 |
| DSpark | Three draft layers, five draft positions, rank-256 Markov head |
| Vocabulary / maximum context | 129280 / 1048576 |

CSA2 builds a candidate pool at the first decoder layer, periodically reindexes
within that pool, and reuses the selected keys between index producers. Shared
KV tensors retain their gradient paths to the producer. Indexer inputs are
detached from the backbone, with the existing sparse-indexer auxiliary loss
training the indexer parameters. Every layer still computes its own local KV.

The causal encoder-decoder design preserves the hidden-state chain through all
40 layers. Teacher-forced training processes the complete sequence in each
layer. The report's decoder prompt warmup and reduced prefill work require a
serving engine's incremental caches and scheduling.

## Run the debug model

The debug flavor retains encoder and decoder layers, all three CSA2 layer
roles, Engram, and mHC. It uses the bundled test tokenizer and C4 fixture.
Recipes run ten steps with seed 42 and strict deterministic execution.

```bash
NGPU=1 MODULE=torchtitan_recipes.tests.models.deepseek_v4_1 \
  CONFIG=deepseek_v4_1_debugmodel ./run_train.sh
```

Two-GPU recipes are `deepseek_v4_1_debugmodel_tp2_ep2` and
`deepseek_v4_1_debugmodel_pp2`. The combined eight-GPU recipe uses real FSDP2,
TP2, PP2, and EP2, with four pipeline microbatches and four virtual stages:

```bash
NGPU=8 MODULE=torchtitan_recipes.tests.models.deepseek_v4_1 \
  CONFIG=deepseek_v4_1_debugmodel_fsdp2_tp2_pp2_ep2 ./run_train.sh
```

Pipeline edges explicitly carry the residual branches, next input mixing
coefficients, shared KV, index keys, selected indices, selected scores, and
candidate pool. These tensors belong to the current microbatch; no mutable
cross-microbatch cache is used. The split places KV producers before pipeline
boundaries and consumers after them. The model supplies static `PipelineStage`
metadata for its seven tensors; metadata inference cannot safely execute with
uninitialized integer selections. Custom splits must preserve layer order and
put complete transformer blocks in every stage, the token embedding in the
first stage, and the output norm and head in the last stage.
Pipeline preprocessing requires `dataloader.max_num_documents` to keep selected
key widths fixed across packed microbatches.

## Released checkpoint and tokenizer

Download a pinned tokenizer and model card first:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --revision 314113f8a7dadb388215396ba5fbee2c610a04d6 \
  --include 'tokenizer*' 'config.json' 'model.safetensors.index.json' \
  --local-dir ./assets/deepseek-v4.1-flash
```

Downloading the complete checkpoint additionally requires all safetensors
shards. Build the released architecture with the matching tokenizer:

```python
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.deepseek_v4_1 import build_model_config

config = build_model_config(
    "deepseek_v4_1_flash",
    parallelism=ParallelismConfig(),
    seq_len=4096,
    tokenizer_path="./assets/deepseek-v4.1-flash",
)
```

Engram's normalized token map and hash buffers are derived from the tokenizer
and rebuilt by `init_states()`. A metadata-only full configuration can omit
the tokenizer; a forward pass then requires explicitly supplied Engram hashes.
Hash history is isolated at packed document boundaries, padding, and images.

`DeepSeekV41StateDictAdapter` maps native and HF weights, including grouped
experts, vision, and DSpark. Its quantized HF storage reader exposes logical
floating-point tensors and dequantizes only requested slices of FP8 or packed
FP4 weights, including slices crossing quantization blocks. Loading through
`CheckpointManager` uses `initial_load_in_hf=True` and
`initial_load_in_hf_quantized=True`. Export produces ordinary floating-point HF
weights. The two released Engram tables have 384006168 and 384016682 rows; a
dequantized training model requires substantial distributed storage.

## Vision and draft APIs

Enable `vision_enabled=True` and `num_draft_layers=3` in a debug model to exercise
these components. Image forward inputs are flattened patch vectors,
`grid_hw=((height, width), ...)`, and per-token `token_types`; preprocessing
also accepts `grid_thw` with time equal to one. Image-marker embeddings and the
channel-major 3-by-3 aligner follow the reference checkpoint. The vision
encoder uses bidirectional attention separately within each image.

`model(tokens, return_draft_context=True)` returns logits and concatenated
backbone features from DSpark's three target layers. `model.forward_draft()`
accepts those features, frozen draft-token embeddings, teacher-forced previous
tokens, block anchors, context positions, and the frozen output weight. Draft
attention sees its complete five-position block plus preceding local context.
The Markov and confidence heads are trainable; the backbone and shared
embedding/output projections are detached at the draft API.

DSpark currently requires tensor-parallel degree one; explicitly set
`num_draft_layers=0` when building a tensor-parallel released backbone.
Draft-context collection currently requires an unsplit backbone. A caller
using sharded shared projection weights must gather or unshard them before
passing them to the draft API. The frozen output projection must remain owned
through auxiliary backward; a retained FSDP all-gather view needs a copy.
Input projections and final heads execute within their owning drafter blocks.
Default rematerialization checkpoints the drafter blocks, and its separate
optimizer registers the drafter expert-balancing update and counter reset. Incremental generation, acceptance/rejection,
and serving throughput measurement require a serving engine.

## Validation and current limits

See [VALIDATION.md](VALIDATION.md) for pinned sources, reference comparisons,
hardware, distributed runs, and reproducibility evidence.

CPU execution uses `attention_backend="reference"`. The standard CUDA debug
recipe uses the public attention-gym fused kernels shared with V4. The combined
numerical recipe explicitly selects the reference backend; its `_fused` variant
selects the fused backend. Both use CSA2's new selection and state flow.
Training models FP8 local KV/Engram and FP4 global
KV/indexer values with straight-through gradients while retaining ordinary
floating-point tensors. This is not a packed incremental KV-cache allocator.

Context parallelism currently requires degree one, and expert parallel degree
must be at least tensor parallel degree. The combined pipeline recipe disables
SPMD type checking because TorchTitan's training engine does not support that
checker with PP; the TP/EP recipe enables it. Full released weight loading,
full-scale convergence, and H100 execution remain to be validated before
publishing a full-scale Trainer recipe.
