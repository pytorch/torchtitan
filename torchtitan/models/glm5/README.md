# GLM-5.2 and GLM-5.3

This directory implements the shared GLM-MoE-DSA decoder architecture used by
GLM-5.2 and GLM-5.3. The GLM-5.3 model card states that it uses the same base
model as GLM-5.2, and the released training configuration retains the
same decoder, DSA index-sharing schedule, MoE dimensions, and one-layer MTP
configuration as GLM-5.2. Versioned flavors keep recipes and checkpoint
validation explicit without duplicating model code.
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
- `dsa_debugmodel`: alias for the DSA development model
- `shared_dsa_debugmodel`: development model exercising index sharing
- `glm5_2_debugmodel`: small model preserving the released layer schedule
- `glm5_2_debugmodel_mtp`: small model with one trainable NextN/MTP objective
- `glm5_2_scaled_debugmodel`: larger validation model
- `744B-A40B`: released GLM-5.2 architecture
- `glm5_3_debugmodel`: GLM-5.3 development model
- `glm5_3_debugmodel_mtp`: GLM-5.3 development model with one MTP layer
- `glm5_3_scaled_debugmodel`: larger GLM-5.3 validation model
- `glm5_3_744b_a40b`: released GLM-5.3 training architecture

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
`glm5_fsdp+tp+cp+ep` integration test:

```bash
python -m tests.integration_tests.run_tests \
  ./outputs/glm5-integration \
  --test_suite models \
  --test_name glm5_fsdp+tp+cp+ep \
  --ngpu 8
```

## Supported parallelisms

| Feature | Notes |
|---------|-------|
| FSDP2 / HSDP | Transformer blocks are sharded through the common decoder lifecycle. |
| Tensor / sequence parallelism | Attention, dense projections, routed experts, shared experts, and the DSA indexer use declarative sharding configs. |
| Context parallelism | DSA gathers indexer keys before Top-K selection and uses the common KV all-gather FlexAttention path. |
| Expert parallelism | Routed experts use TorchTitan's common MoE implementation; shared experts remain replicated across the EP axis. |
| Pipeline parallelism | Decoder layers are split through the common pipeline lifecycle; shared DSA selections are forwarded when their consumer is on a later stage. |

GLM-5.2 DSA is represented with a PyTorch FlexAttention `BlockMask`. It does
not use the packed-sequence `varlen` attention contract because DSA selects an
arbitrary learned Top-K key set for each query.

The frozen indexer delegates weighted-ReLU Top-K selection to Attention Gym's
`lightning_indexer`, using the repository's existing `attn-gym==0.0.17`
dependency. Its reference implementation preserves FP32 ranking and mixing
weights. Valid tokens are packed by document before selection, and the returned
document-local indices are mapped back to global key indices. Under CP, the
indexer gathers query features and weights in addition to keys to satisfy
Attention Gym's complete-document input contract, then returns local queries'
selections. This currently repeats indexer selection across CP ranks.
Attention continues to use the common KV all-gather FlexAttention backend.

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

Select the versioned GLM-5.3 development flavor with:

```bash
python -m scripts.checkpoint_conversion.numerical_tests_glm5 \
  --model_flavor glm5_3_debugmodel
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
and label sequence, the GLM block builds the corresponding DSA mask, and
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

The integration suite registers separate base and MTP training cases. Both use
an 8-GPU FSDP2+TP2+CP2+EP2 configuration with a real process group:

```text
glm5_fsdp+tp+cp+ep
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
