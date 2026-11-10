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
- DeepSeek sparse attention (DSA) with a frozen learned indexer
- Dense-to-MoE layer transition and shared experts
- Cross-layer reuse of DSA index selections
- One full NextN multi-token-prediction (MTP) block for the released model
- Data, tensor, context, pipeline, and expert parallel configuration through
  TorchTitan's common model lifecycle
- Hugging Face checkpoint conversion

Available flavors are defined in `flavors.py`:

- `debugmodel`: small development model
- `shared_dsa_debugmodel`: development model exercising index sharing
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

DSA gathers the selected keys and values through Attention Gym. It does
not use the packed-sequence `varlen` attention contract because DSA selects an
arbitrary learned Top-K key set for each query.

The frozen indexer delegates weighted-ReLU Top-K selection to Attention Gym's
`lightning_indexer`, using the repository's existing `attn-gym==0.0.17`
dependency. Like DeepSeek V4, it selects the fused implementation for FP16/BF16
query features on NVIDIA SM90 or newer; other inputs use the FP32 reference
implementation. Mixing weights remain FP32. Valid tokens are packed by document
before selection, and the returned
document-local indices are mapped back to global key indices. Under CP, the
indexer gathers query features and weights in addition to keys to satisfy
Attention Gym's complete-document input contract, then returns local queries'
selections. This currently repeats indexer selection across CP ranks.
The learned indexer is loaded from the checkpoint and remains frozen during
continued training; this implementation does not train a new indexer with an
auxiliary distillation loss from random initialization.
Attention uses the shared `MLAAttention` backend, or
the GLM compact-KV adapter under CP. CP gathers normalized compressed KV
and the shared rotary key before KV expansion. Q/KV projections, RoPE,
TP column/row projections, dense/MoE layer construction, routing, and MTP
blocks reuse DeepSeek-V3 implementations. GLM retains its indexer,
selection filtering, and cross-layer selection schedule. Gathered attention
encodes queries as `[q, 0]` and the shared key/value pool as `[k, v]`, retaining
the original query-key scale and returning the value tail. The CUDA backend
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
[`MTP.md`](MTP.md) for the data, model, checkpoint, parallelism, and validation
contracts.

## Validation

Run the CPU model tests with:

```bash
pytest tests/unit_tests/cpu/models/glm5 -q
```

Run the reduced Hugging Face parity check with the pinned Transformers
reference documented by the script:

```bash
python -m scripts.checkpoint_conversion.numerical_tests_glm5 --device cuda
```

The parity check builds equivalent reduced TorchTitan and Hugging Face models,
strictly loads the converted state dict, compares MoE routing choices, and
checks full-sequence logits with KL-divergence, cosine-similarity, and top-1
agreement thresholds.

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
