# Numerical Parity Against a Reference Model

Compare the same checkpoint and inputs. Find the cause of differences before
changing the model. Record the reference code and checkpoint revisions,
model flavor, tokenizer, dtype, attention backend, seed, and input preparation.

## Choose a reference and checkpoint

Choose Hugging Face `transformers`, the model author's implementation, SGLang,
or vLLM based on support for the exact variant and features being tested. For
SGLang/vLLM, record dtype and quantization settings, whether the test uses prefill
or decode, and the KV-cache state.

Prefer the model's official open weights on Hugging Face Hub when available;
record the repository ID and pin the checkpoint revision. Use the matching
model config, tokenizer, and preprocessing assets.

Use a reference that supports training or independent component references to
check gradients, auxiliary objectives, and balancing updates as well.

## Reuse checkpoint converters and comparison scripts

The model package exports `build_model_config`, and its model class defines
`state_dict_adapter_cls`. The converters in `scripts/checkpoint_conversion/`
use these APIs. Reuse `convert_from_hf.py` and `convert_to_hf.py`; put changes to
checkpoint keys and layouts in the adapter and its shared helpers.

For example, after downloading the matching HF assets:

```bash
python3 -m scripts.checkpoint_conversion.convert_from_hf \
  assets/hf/Llama-3.1-8B outputs/parity/llama3-8b-dcp \
  --model_name llama3 --model_flavor 8B
```

Use the comparison script's supported loading path. Convert to native DCP when
that path requires it. Start from tracked examples such as
`numerical_tests_qwen3_5.py` and `numerical_tests_kimi_k3.py`. Check their model
builders and CLI options before adapting them; older or untracked scripts may
use a different API.

Keep large checkpoints and activation dumps in generated outputs. Commit small
test fixtures with reproducible inputs and instructions or a generator to recreate
them. Comparison scripts belong in `scripts/checkpoint_conversion/`; CPU tests
belong in `tests/unit_tests/cpu/models/`.

## Match inputs, weights, and settings

1. Build the matching config with `ParallelismConfig()`. Match the reference's
   head layout, context length, norm eps, RoPE settings, routing, auxiliary
   objectives, and weight tying.
2. Use the same tokens, images, positions, masks, sequence boundaries, and
   padding. Check tokenizer and preprocessing separately when necessary.
3. Materialize and initialize model states before loading the checkpoint. Check
   that every required weight loads with the right shape and layout, and that
   the adapter maps the expected keys.
4. Use evaluation mode and matching dtypes and optimization settings. Set the
   seed, normally 42. Use deterministic execution when claiming exact equality;
   keep `deterministic_warn_only` disabled.
5. Compare components (RoPE, norms, projections, router), then one block, any
   vision encoder or projector, and full-model logits. Start with higher
   precision where supported.

## Compare tensors and set tolerances

For intermediate tensors, check shape, finite values, maximum absolute error,
and relative error. Use `torch.testing.assert_close` with explicit `atol` and
`rtol`. Justify both for the operation, value magnitude, dtype, and reference.
Absolute tolerance matters near zero, where relative error is unstable.

For logits, also report probability KL divergence, cosine similarity, and top-k
agreement. Compute KL from normalized distributions in sufficient precision;
record its direction and reduction method. Compare the actual logits too:
adding a constant leaves softmax probabilities unchanged. For example,
`[1, 2, 3]` and `[101, 102, 103]` have identical probabilities but differ by 100.
Use tensor errors to validate intermediate activations.

Choose tolerances using a higher-precision reference and an eager baseline in
the test's dtype where possible. Accumulation order and sequence length affect
low-precision errors, so calibrate tolerances for each model and operation.
Record tolerances and measured errors. Investigate failures before changing
thresholds; do not widen them just to accept the result.

For new operations that require gradients, compare input and parameter gradients
using the same scalar loss or output gradient. Include custom autograd and
handwritten kernels. Use `/kernel-numerics-verifier` for a new low-precision
kernel and `/numerics_debugging` to find where unexplained differences start,
when those skills are available.

## Check MoE routing

Small score differences near a tie can change top-k expert choices. Check score
math, expert indexing, and selection logic before accepting different choices:

1. Compare scores, biases, selected expert IDs, and mixing weights aligned by
   expert ID. Include normalization, route scaling, group selection, shared
   experts, padding, and token filtering where used.
2. Measure routing agreement per layer and check the score gap between selected
   and excluded experts. Use higher precision or controlled inputs to check
   whether score differences explain the changed choices.
3. In a separate diagnostic run, reuse the reference expert IDs to isolate
   expert computation and the rest of the network. Report the normal run too,
   and validate the model's own router independently.

When imposing routing, record whether indices, weights, or both are replaced.
Overriding weights bypasses router scores and mixing-weight computation; check
those outputs and gradients separately.

## Check training and distributed execution

Every model port needs a golden from a remote debug run. First prove that
independent runs have exactly matching full-precision loss and grad_norm at every
step. Then create the golden. Follow [test_coverage.md](test_coverage.md) for the
repeatability check, export, replay, files for each GPU architecture, and CI
registration.
Reference parity checks correctness; repeated training checks determinism;
the golden detects later changes to the validated behavior.

After component and forward comparisons pass, validate training. Computation
changes require loss convergence on representative data. For refactors or
parallelism changes that preserve computation, compare matching configs with
seed 42 and deterministic execution. Require bitwise-identical loss and grad_norm
when parallelisms and GPU settings match. Use full-precision TensorBoard values.

For an existing model, adapt this command to the before and after refs
and a recipe present in both checkouts:

```bash
python3 scripts/loss_compare.py upstream/main HEAD \
  --baseline-module=torchtitan_recipes.tests.models.llama3 \
  --baseline-config=llama3_debugmodel --steps=10 \
  --metrics=loss,grad_norm --assert-equal
```

Set `--baseline-ngpus` and `--test-ngpus` to each run's GPU count. The script
enables deterministic execution and defaults to comparing loss; request both
metrics. For recipes with transforms, check whether seed creation needs a
matching `--seed-module` and `--seed-config`. See `scripts/loss_compare.py` and
the integration suite's `loss_compare_seed_config` examples.

## Checkpoint and layout checks

- **Native DCP compatibility.** If native checkpoint names, shapes, or parameter
  layouts change, test loading and regenerate affected converted fixtures.
  Check the actual keys after changing the module tree. Unexplained missing or
  unexpected keys, or uninitialized weights, fail the comparison. Preserve
  loading of previously supported checkpoints.
- **Complex RoPE buffers.** Casting to a real dtype can discard the imaginary
  part of a complex cache. Set buffer dtypes through the model lifecycle and
  rebuild buffers before loading weights. Preserve their intended dtype and
  keep loaded parameters unchanged while fixing a buffer.
- **Tied embeddings.** Match `Decoder.Config.enable_weight_tying` to the
  reference's `tie_word_embeddings`. Check that weights stay shared through
  construction, materialization, initialization, and loading. Equal values in
  separate parameters do not share gradients. Reuse adapter support for omitted
  tied output weights.
- **Norm variants.** Match eps and the weight convention. Account for offsets
  such as `(1 + weight) * norm(x)` in the math or adapter.
- **RoPE and projections.** Check partial rotation, adjacent-pair or split-half
  rotation, multimodal position axes, GQA head order, fused QKV splits, and gated
  projections. Reuse fused-linear adapter helpers.
- **Packed and distributed inputs.** Match document boundaries, position resets,
  masks and offsets, padding, CP input and label sharding, and recurrence resets.
  Test the selected backend's metadata handling.
- **Config and sharding.** Build configs for the intended parallelism. Rebuild
  them when sequence or expert parallelism changes. Preserve component sharding
  when applying transforms.
