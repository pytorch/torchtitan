# GLM-5.2 multi-token prediction

This note records the training implementation and its boundary with
serving-time speculative decoding.  It is an implementation contract, not a
claim that every distributed combination has already passed accelerator
acceptance.

## Released checkpoint contract

The official GLM-5.2 configuration contains:

- `num_hidden_layers = 78`
- `num_nextn_predict_layers = 1`
- `index_share_for_mtp_iteration = true`

The safetensors index stores the NextN block under `model.layers.78`.  It is a
full decoder block: MLA/DSA projections and indexer, routed and shared MoE
weights, input and post-attention norms, plus the four MTP-only tensors
`enorm`, `hnorm`, `eh_proj`, and `shared_head.norm`.  It does not store another
vocabulary projection; the top-level `lm_head` is shared.

Primary references:

- [GLM-5.2 config](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)
- [GLM-5.2 checkpoint index](https://huggingface.co/zai-org/GLM-5.2/blob/main/model.safetensors.index.json)

## Training data path

`torchtitan_recipes/tests/models/glm5.py` derives the effective MTP depth from
`len(model_config.mtp_layers)`, passes it to the dataloader, and selects
`MTPLoss` only for an MTP flavor.  This keeps the ordinary GLM recipe unchanged.

`Glm5Model.preprocess_inputs()` performs the following before CP sharding:

1. left-shift token ids and labels by each prediction depth;
2. reject shifts that cross packed-document or padding boundaries;
3. use `IGNORE_INDEX` for invalid auxiliary labels;
4. build one DSA additive mask per shifted sequence;
5. declare token, label, validity-mask, and DSA-mask SPMD placements;
6. return aligned tuples for the main and auxiliary objectives.

The generic rolling and loss semantics are shared with DeepSeek-V3 through
`torchtitan.models.deepseek_v3.mtp`; GLM only specializes the DSA attention
interface and checkpoint schema.

The recipe exposes `mtp_scale` explicitly.  Its default value of `0.3` follows
the TorchTitan DeepSeek-V3 training convention; it is not claimed as a
published GLM-5.2 pretraining hyperparameter because the released GLM config
does not contain that value.  Experiments may override it through the recipe
or the nested loss configuration, and reports must record the effective value.

## Model path

For depth `d`, `Glm5MTPTransformerBlock` computes:

```text
shifted embedding -- enorm --+
                              +-- concat -- eh_proj -- DSA -- MoE -- mtp_norm
previous hidden  -- hnorm ----+
```

The block receives the pre-final-norm hidden state from the target decoder,
masks invalid shifted positions, executes the full GLM DSA attention and MoE
block, and returns a normalized auxiliary hidden state.  The normal top-level
`lm_head` projects both the target and auxiliary hidden states.  `MTPLoss`
computes normal next-token cross entropy plus the configured weighted MTP term.

The pre-final-norm input follows TorchTitan's DeepSeek-V3 training MTP
contract.  Serving runtimes may instead discuss post-final-norm state recycling
inside an iterative speculative-decoding loop.  That serving fix does not by
itself redefine the one-pass auxiliary training objective.  A future claim of
exact GLM forward parity must still compare first-depth logits against an
authoritative checkpoint runtime; the Transformers GLM implementation cannot
serve as that oracle because it currently omits the MTP layer.

## Checkpoint conversion

`Glm5StateDictAdapter` maps native `mtp_layers.0.*` keys to HF
`model.layers.78.*` for the released flavor.  It preserves the existing
individual-expert/grouped-expert conversion and maps the MTP-only tensors
explicitly.  Tests assert both a bitwise native-to-HF-to-native round trip and
the real released key shape, including the absence of a duplicated
`shared_head.head.weight`.

## Parallelism and acceptance boundary

The implementation reuses the common MTP FSDP lifecycle and MTP TP/SP sharding
configs.  GLM additionally assigns the DSA mask's query-sharded CP placement.
These are implementation paths, not hardware acceptance results.

Parameter and FLOP accounting includes both the 78 target decoder layers and
the configured MTP layers.  In particular, the frozen DSA indexer is charged
for forward work but not nonexistent backward work for every executed target
or MTP block.  This keeps MTP on/off MFU comparisons internally consistent.

Required validation order is:

1. CPU config construction and checkpoint round trip;
2. single-device forward, backward, finite loss, and finite gradient norm;
3. TP and SP numerical comparison;
4. FSDP and EP smoke plus checkpoint save/resume;
5. CP smoke and numerical comparison of the auxiliary DSA mask;
6. graph-mode eager-versus-compiled comparison on each supported backend;
7. profiler-off MTP off/on performance and memory A/B;
8. only then broaden to combined topologies.

### TODO: pipeline parallel MTP

Pipeline parallel MTP currently requires the token embedding, target decoder
tail, MTP block, final norm, and shared lm_head to be represented coherently at
the pipeline boundary.  `Glm5Model.forward()` rejects a partition missing the
embedding or lm_head instead of silently dropping the auxiliary objective.
Until a dedicated PP contract and test exist, PP+MTP is unsupported rather
than a fallback to non-MTP training.

## Training versus speculative decoding

`index_share_for_mtp_iteration` is a serving optimization. A speculative
decoder can apply the same single MTP block repeatedly: iteration zero computes
the lightning-indexer Top-K and later iterations reuse that selection.

TorchTitan's current implementation is the training objective.  It runs one
NextN block once for each configured objective and does not contain a serving
scheduler, KV cache, acceptance/rejection loop, or multi-stream proposer.
Consequently:

- the full MTP weights and loss are implemented here;
- iterative Top-K reuse belongs to a future inference/runtime integration;
- setting an environment variable must never pretend that the training path
  implements speculative serving;
- accelerator-specific fused kernels or communication overlap belong in the
  platform extension, with explicit capability checks and A/B evidence.
