# GLM-5.2 and GLM-5.3

This directory implements the shared GLM-MoE-DSA decoder architecture used by
GLM-5.2 and GLM-5.3. The GLM-5.3 model card states that it uses the same base
model as GLM-5.2, and the released training configuration retains the
same decoder, DSA index-sharing schedule, MoE dimensions, and one-layer MTP
configuration as GLM-5.2. Both releases use the same model configurations and
checkpoint adapter without duplicate version aliases.
The implementation follows TorchTitan's config-first model contract and uses
shared PyTorch-native components for distributed training.

## Model support

- Multi-head latent attention (MLA)
- DeepSeek sparse attention (DSA), with frozen, warmup, and sparse indexer stages
- Dense-to-MoE layer transition and shared experts
- Cross-layer reuse of DSA index selections
- One full NextN multi-token-prediction (MTP) block for the released model
- Data, tensor, context, pipeline, and expert parallel configuration through
  TorchTitan's common model lifecycle
- Hugging Face checkpoint conversion

Available flavors are defined in `flavors.py`:

- `debugmodel`: small development model
- `glm5_2_debugmodel`: small model preserving the released layer schedule
- `glm5_2_debugmodel_mtp`: small model with one trainable NextN/MTP objective
- `glm5_2_scaled_debugmodel`: larger validation model
- `744B-A40B`: released GLM-5.2 architecture

GLM-5.2 and GLM-5.3 use these same architecture configurations; separate
version aliases are not required for loading their checkpoints.

Runnable development recipes live in
`torchtitan_recipes/tests/models/glm5.py`. The released flavor describes the
full architecture and is not intended for single-host smoke tests.

## Training

Run the small GLM-5.2 development model with the repository test assets:

```bash
MODULE=torchtitan_recipes.tests.models.glm5 \
CONFIG=glm5_debugmodel \
./run_train.sh
```

The recipe defaults to 10 steps and uses TorchTitan's standard configuration
overrides. An 8-GPU run combining FSDP, TP, CP, and EP is registered as the
`glm5_mtp_fsdp+tp+cp+ep` integration test, including the auxiliary objective:

```bash
python -m tests.integration_tests.run_tests \
  ./outputs/glm5-integration \
  --test_suite models \
  --test_name glm5_mtp_fsdp+tp+cp+ep \
  --ngpu 8
```

## Supported parallelisms

| Feature | Notes |
|---------|-------|
| FSDP2 / HSDP | Transformer blocks are sharded through the common decoder lifecycle. |
| Tensor / sequence parallelism | Attention, dense projections, routed experts, shared experts, and the DSA indexer use declarative sharding configs. |
| Context parallelism | DSA gathers indexer keys before Top-K selection and gathers compact MLA KV before expansion. |
| Expert parallelism | Routed experts use TorchTitan's common MoE implementation; shared experts remain replicated across the EP axis. |
| Pipeline parallelism | Decoder layers are split through the common pipeline lifecycle; shared DSA selections are forwarded when their consumer is on a later stage. |

Index-sharing groups must be contiguous: a layer may use its own producer or
continue the preceding layer's group, but cannot return to an older group.
Pipeline boundaries carry one live producer's indices. Empty embedding-only or
output-only stages still prepare CP metadata from the global input.

DSA gathers the selected keys and values through Attention Gym. It does
not use the packed-sequence `varlen` attention contract because DSA selects an
arbitrary learned Top-K key set for each query.

The indexer delegates discrete weighted-ReLU Top-K selection to Attention Gym's
`lightning_indexer`, using the repository's existing `attn-gym==0.0.17`
dependency. Like DeepSeek V4, it selects the fused implementation for FP16/BF16
query features on NVIDIA SM90 or newer; other inputs use the model-local
FP32 tiled reference selector without a full query/key score tensor.
Mixing-weight parameters and projection remain FP32; weights
are cast to the query dtype at the indexer kernel boundary, as required by
Attention Gym. Valid tokens are packed by document
before selection, and the returned
document-local indices are mapped back to global key indices. Under CP, the
indexer keeps queries and mixing weights local and gathers only indexer keys.
Device-resident candidate bounds describe each local query's valid document
prefix in the packed global key pool; only local queries are ranked.

CP uses a model-local streaming selector: CUDA FP16/BF16 scores key tiles and
merges a bounded running top-k in one kernel per local query. The PyTorch
reference path bounds query/key tiles rather than materializing full scores.
This requires neither modified Attention Gym nor DeepSelect/cuDNN dependencies;
the attention still uses the existing `gather_attn` interface. The selector's
new CPU/GPU and distributed validation is pending; no performance claim is made.

The default continued-training recipe keeps the pretrained indexer frozen.
Explicit `indexer_stage` settings also provide dense-attention `warmup`, which
freezes the backbone and trains indexers, and `sparse`, which jointly trains
the backbone and indexers. Indexer inputs and attention teacher targets are
detached; the KL objective reuses DeepSeek-V4's `SparseIndexerLoss` and the
shared auxiliary-loss normalization and optimizer hooks. Dense warmup is
processed in query chunks rather than constructing a full sequence mask.

```text
warmup: frozen backbone + dense attention -> train indexer
                         model weights
sparse: train backbone + top-k attention -> train indexer
```

Select a stage when constructing the existing Python recipe:

```python
config = glm5_debugmodel(seq_len=128, indexer_stage="warmup")
# Construct a separate sparse-stage recipe after warmup.
config = glm5_debugmodel(seq_len=128, indexer_stage="sparse")
```

The native CLI loads a named, zero-argument recipe factory, not a Python call
expression. Named launchable stage recipes are provided:

```bash
MODULE=torchtitan_recipes.tests.models.glm5 CONFIG=glm5_debugmodel_warmup ./run_train.sh
MODULE=torchtitan_recipes.tests.models.glm5 CONFIG=glm5_debugmodel_sparse ./run_train.sh
```

These are separate runs, not an automatic handoff. Stage changes rebuild the
optimizer; load the warmup model weights for sparse training using the normal
checkpoint path. There is no automatic step-based stage switch.

The model checks model/indexer/attention/loss stage consistency again at build
time, so a partial override fails rather than silently freezing all training.
Warmup projects features but does not run top-k. Producer query chunks use the
KL loss chunk budget; shared consumers cap dense support at 128 query rows.
The current shared AuxLoss carrier still needs activation backward through the
frozen backbone. Warmup is not a backward-free teacher pass; its step time and
saved activations must be measured against sparse training. Indexer training
convergence has not been established by the short numerical tests.

Attention specializes the shared `MLAAttention` backend in absorbed form.
CP gathers normalized compressed KV and the shared rotary key, never per-head
expanded K/V. Q/KV projections, RoPE,
TP column/row projections, dense/MoE layer construction, routing, and MTP
blocks reuse DeepSeek-V3 implementations. GLM retains its indexer
and cross-layer selection schedule. The K part of `wkv_b` is contracted into
each query; gathered attention uses a single-head `[kv_latent, k_rope]` pool.
The V part of `wkv_b` projects the weighted latent after attention. This keeps
the original query-key scale and the original checkpoint parameter layout,
without `[q, 0]` padding or per-head K/V materialization. The CUDA backend
uses Attention Gym's fused kernel; the reference implementation materializes
selected slots and is intended for correctness checks rather than large-model
training. Device-specific implementations belong in out-of-tree integrations.

Native KV-up-projection parameters now live under
`attention.inner_attention.wkv_b`; Hugging Face checkpoint names are unchanged.
The adapter inherits DeepSeek parameter mappings and reshapes fused routed
W13 weights directly, without model-specific gate/up all-to-all conversion.

CP load balancing with token reordering and PP with MTP are rejected at config
construction; the current DSA indexer requires contiguous document order.

## Download tokenizer and checkpoint assets

Download tokenizer and model metadata from the released GLM-5.2 repository:

```bash
python scripts/download_hf_assets.py \
  --repo_id zai-org/GLM-5.2 \
  --assets tokenizer config index
```

For GLM-5.3 training and checkpoint conversion, use the BF16 release rather
than the quantized inference checkpoint:

```bash
python scripts/download_hf_assets.py \
  --repo_id zai-org/GLM-5.3-BF16 \
  --assets tokenizer config index
```

To load the released weights for continued training or checkpoint conversion,
also request `safetensors`. The checkpoint is very large, so the debug recipes
use the repository test tokenizer and randomly initialized weights by default.

Validate the model definition and Hugging Face checkpoint mapping with the
reduced deterministic parity test:

```bash
python -m scripts.checkpoint_conversion.numerical_tests_glm5
```

The same development flavor validates the shared GLM-5.2/5.3 architecture:

```bash
python -m scripts.checkpoint_conversion.numerical_tests_glm5 \
  --model_flavor glm5_2_debugmodel
```

## Multi-token prediction

The released GLM-5.2 checkpoint declares 78 target decoder layers and one
``num_nextn_predict_layers`` block.  The extra block is stored as
``model.layers.78`` and contains the complete DSA attention and MoE sublayers,
plus ``enorm``, ``hnorm``, ``eh_proj``, and ``shared_head.norm``.  It reuses the
model's token embedding and top-level language-model head rather than storing a
second vocabulary projection.

TorchTitan therefore models training-time MTP as a real auxiliary decoder
depth, not as an inference-only flag.  The dataloader prepares a shifted token
and label sequence, the GLM block builds the corresponding DSA metadata, and
``MTPLoss`` adds the weighted auxiliary cross-entropy objective.  See
[`MTP.md`](MTP.md) for the GLM-specific checkpoint, DSA, and support
contracts.

## Validation

### FLOPs and MFU

Attention follows TorchTitan's model-FLOPs convention: forward and backward
contractions count, while causal triangular savings and rematerialization do
not. Frozen/sparse stages cap the attended key count at
`min(seq_len, index_topk)` in every layer, including shared-index consumers;
warmup counts full attention. The indexer estimate separately includes
projection and full-document scoring, plus distillation work in trainable
stages, once per actual producer, including MTP indexers.

Attention contractions use widths `kv_lora_rank + qk_rope_head_dim` and
`kv_lora_rank`; projection parameters are counted separately as before.
These are model FLOPs, not measured hardware instructions. Temporary tensors,
the unused weighted RoPE output channels and communication can increase
execution cost without increasing useful model FLOPs.
Use measured step time, throughput and peak memory alongside MFU; no speedup
is implied by the FLOPs estimate alone.

Run the CPU model tests with:

```bash
pytest tests/unit_tests/cpu/models/glm5 -q
```

Run the reduced Hugging Face parity check with the pinned Transformers
reference documented by the script:

```bash
python -m scripts.checkpoint_conversion.numerical_tests_glm5 --device cuda --seq_len 128 --seed 61
```

The parity check builds equivalent reduced TorchTitan and Hugging Face models,
uses sequence length 128 and seed 61 by default, strictly loads the converted
state dict, compares effective causal indexer selections and MoE routing choices, and
checks full-sequence logits with KL-divergence, cosine-similarity, and top-1
agreement thresholds. The sequence length must exceed the indexer's top-k
to exercise sparse selection. This forward-only script is distinct from
external paired forward/backward parity reports.

The integration suite registers an MTP training case, which also exercises the
base decoder, using an 8-GPU FSDP2+TP2+CP2+EP2 configuration:

```text
glm5_mtp_fsdp+tp+cp+ep
```

The MTP case exercises shifted targets, the auxiliary DSA/MoE block, the
combined loss, backward, and the optimizer step. Pipeline parallelism with MTP
is not yet supported; base GLM-5.2 pipeline parallelism is supported.

## References

- [GLM-5.2 model and checkpoint](https://huggingface.co/zai-org/GLM-5.2)
- [GLM-5.3 model and checkpoint](https://huggingface.co/zai-org/GLM-5.3)
- [GLM-5.3 BF16 checkpoint](https://huggingface.co/zai-org/GLM-5.3-BF16)
- [Hugging Face GLM-MoE-DSA reference implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm_moe_dsa)
