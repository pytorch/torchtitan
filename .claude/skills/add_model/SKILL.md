---
name: add_model
description: Add or port a model to TorchTitan using shared components, model flavors, Python recipes, reference parity, CPU feature tests, combined parallelism training, and numerical golden baselines. Use when the user requests a new model or invokes /add_model.
---

# Adding a Model to TorchTitan

Establish a reference and an unparallelized numerical baseline, then validate
parallelism and training. Batch related edits before expensive GPU checks;
rerun affected checks after fixing a failure or changing computation.

Follow applicable `AGENTS.md`, `.claude/CLAUDE.md`, and `.claude/rules/`
instructions. Read the current [model guide](../../../torchtitan/models/README.md)
and [config guide](../../../torchtitan/config/README.md) before choosing an API.
Repository paths below are relative to the repository root. Verify helpers and
example harnesses in the checkout before using them.

- Read [numerical_parity.md](references/numerical_parity.md) when building or
  running reference comparisons.
- Read [test_coverage.md](references/test_coverage.md) when planning feature
  tests, combined parallelism training, and remote numerical golden baselines.
- Read [code_style_checklist.md](references/code_style_checklist.md) for the
  cleanup and review pass.
- When available, use `/numerics_debugging` to localize unexplained drift and
  `/pyrefly_ci_check` when local type checking disagrees with CI. Model-specific
  numerical skills can supply an existing harness for that family.

## Phase 0: Scope and reference

1. Identify the model variant, supported tasks, checkpoint and reference-code
   revisions, and requested parallelisms. Reference candidates include HuggingFace
   `transformers`, the model author's implementation, SGLang, and vLLM. Choose one
   that supports the exact variant and feature paths, and pin its revision.
2. Inventory attention and sequence boundaries, positional encoding, norms,
   dense FFN or MoE routing, shared experts, auxiliary objectives, weight tying,
   and any vision encoder or projector. Compare each with existing components.
3. Establish the available checkpoints, dependencies, and hardware. Prefer the
   model's official open weights on Hugging Face Hub when available; record the
   repository ID and pin the checkpoint revision. Plan three test artifacts:
   CPU coverage of important/new features, a real-GPU training test combining
   supported parallelisms, and a debug-recipe numerical golden baseline generated
   on remote GPUs. Normally reserve 8 H100s for combined FSDP/TP/EP/PP coverage.
   Also plan reference parity, convergence, and requested scale validation;
   record checks that cannot run.

## Phase 1: Model definition and public config API

Reuse the closest model and the shared components before adding code:

| Need | Current location |
| --- | --- |
| Model lifecycle, module and adapter contracts | `torchtitan/models/base/` |
| Decoder blocks and stack | `torchtitan/models/common/decoder.py` |
| Attention, metadata and CP backends | `torchtitan/models/common/attention/` |
| RoPE | `torchtitan/models/common/rope.py` (`ComplexRoPE`, `CosSinRoPE`) |
| Norms and standard module wrappers | `torchtitan/models/common/nn_modules.py`, `torchtitan/models/common/norm.py` |
| Dense FFN and MoE | `torchtitan/models/common/feed_forward.py`, `torchtitan/models/common/moe/`, `torchtitan/models/common/dist_moe/` |
| Embedding and linear projections | `torchtitan/models/common/embedding.py`, `torchtitan/models/common/linear/` |
| Vision and multimodal composition | `torchtitan/models/common/vision_encoder.py`, `torchtitan/models/common/multimodal.py` |

Use `torchtitan/models/common/config_utils.py` factories such as `make_gqa_config`,
`make_ffn_config`, `make_moe_config`, and `make_router_config` when they express the
architecture. Keep new concrete shared components in `torchtitan/models/common/`,
lifecycle contracts in `torchtitan/models/base/`, distributed helpers in
`torchtitan/distributed/`, and model-specific computation in its model package.

The model package exposes the following API:

- `flavors.py`: `MODEL_FLAVORS` and
  `build_model_config(flavor, *, parallelism=...)`, returning the concrete
  model's nested `Config`. Follow sibling builders for supported context lengths,
  attention backends, and flavor names; include a reduced `debugmodel` fixture.
- `model.py`: the model and its nested config. Inherit `Decoder` or another
  matching shared architecture where possible. Models used by the standard
  engine need `preprocess_inputs`, `Config.get_nparams_and_flops`, and the
  `BaseModel` lifecycle. Reuse inherited implementations when applicable.
- `sharding.py`: model-specific sharding constructors, only when shared
  constructors do not cover the components. Attach sharding to component configs
  where the flavor builder creates them.
- `state_dict_adapter.py`: HF/native key and layout mapping. Reuse shared adapter
  helpers and bind the adapter with the model class's `state_dict_adapter_cls`.
- `__init__.py`: re-export the model class, `MODEL_FLAVORS`, and
  `build_model_config` for recipe and checkpoint-conversion consumers.

Keep parameter initialization in configs or `init_states` overrides needed for
ordering, such as tying weights. Follow the nearest transformer block's remat
annotations when adding computation; see
[REMAT.md](../../../torchtitan/distributed/REMAT.md).

Wire stateful training features through `_register_optimizer_hooks` when needed:
balancing updates, accumulated auxiliary losses, and resets must run at the
intended optimizer/microbatch boundaries. Follow Kimi K3's quantile-balancing
hook and Qwen3.5/DeepSeek V4's balancing and auxiliary-loss hooks. Verify training,
evaluation, and recomputation behavior through the actual trainer path.

Training recipes belong in `torchtitan_recipes/`. Reusable CPU/CI model fixtures
belong in `torchtitan_recipes/tests/models/`; composed integration configurations
live with their definitions in `tests/integration_tests/`. Public training
recipes should describe configurations actually validated for training.

## Phase 2: Establish numerical parity

Use [numerical_parity.md](references/numerical_parity.md):

1. Build the matching model config with `ParallelismConfig()` for the baseline.
   Match the reference's weights, inputs, masks, positions, dtype, and evaluation
   mode. Reuse the generic checkpoint converter when a native DCP is needed.
2. Compare leaves, one block, and any vision/projector path before full logits.
   Preserve enough reference provenance to reproduce each comparison.
3. Check actual tensors with justified absolute and relative tolerances. Report
   logit/probability metrics as additional evidence. For MoE, validate the router
   separately and use imposed routing to diagnose the remaining computation.
4. Verify backward computation for new differentiable components, then establish
   a training baseline. Revalidate loading when native checkpoint keys, shapes,
   or layouts change.

## Phase 3: Recipes and distributed execution

1. Construct `ParallelismConfig` first and pass it to `build_model_config`.
   Sequence and expert parallelism affect the built sharding. Rebuild the model
   config when those settings change; copying a config or changing degrees later
   does not reconstruct its layouts.
2. Reuse `BaseModel.parallelize`: it binds `local_compile_regions`, applies
   declarative SPMD parallelism, applies `activation_remat`, then applies FSDP.
   `Decoder` supplies its FSDP implementation. Override lifecycle methods only
   for a demonstrated model-specific requirement.
3. Apply optional config transforms in the recipe for CP, async TP, or other
   supported implementations. Follow
   [transform/README.md](../../../torchtitan/config/transform/README.md) for ordering
   and compatibility. Enable registered local compile regions through the model
   config; FlexAttention manages its own compilation.
4. Use the shared sharding constructors and canonical mesh-axis order. Match
   sharding input keys to forward parameter names after shape-suffix renames.
   Validate mesh axes and placements and document gradient placements for
   `to_local`, `local_map`, or `full_tensor` boundaries.
5. Treat preprocessing outputs as explicit contracts: mask and RoPE positions,
   packed-document boundaries, padding, labels/loss denominators, and modality
   metadata must survive TP/SP, EP, CP, PP, and data-parallel execution. Test
   packed inputs against isolated documents and mixed modality presence across
   ranks where supported.
6. Reuse the model's pipeline hooks for PP. Extra cross-layer state needs an
   explicit stage payload, gradient route, and cleanup contract. Choose stage
   splits that exercise its lifetime and reuse; Kimi K3's attention-residual
   cache and exact block-gradient tests illustrate this requirement.

Select a recipe with `--module` (its importable module) and `--config` (its factory
name). Set model, training, parallelism, debug, and remat fields in Python. For
example, inspect a shipped recipe's model and settings before a launch:

```bash
python3 - <<'PY'
from torchtitan.config import ConfigLoader

config = ConfigLoader().load([
    "--module", "torchtitan_recipes.tests.models.llama3",
    "--config", "llama3_debugmodel",
])
print("model:", type(config.model).__qualname__)
print("parallelism:", config.parallelism)
print("debug:", config.debug)
PY
```

For a training launch, `NGPU` must match the recipe's physical world size. The
supported operational CLI options are documented in `torchtitan/config/README.md`.

## Phase 4: Test coverage, cleanup, and lint

Implement all three layers in [test_coverage.md](references/test_coverage.md):

1. **Important feature tests, mostly CPU.** Cover each new computation or
   stateful feature with independent expected results, edge cases, gradients
   where applicable, and lifecycle behavior. For quantile balancing, test the
   histogram/quantile math, padding, bias updates, and accumulation/reset timing.
   Extend shared flavor/config and adapter contracts as well.
2. **Combined parallelism training, on real GPUs.** Add a single model training
   test with FSDP, TP, EP, and PP active together for an MoE model, normally on
   8 H100s. Dense models cover FSDP/TP/PP and record EP as inapplicable. Verify the
   built model, mesh, pipeline microbatches, optimizer steps, and feature hooks;
   mark the definition `use_real_pg=True`. Check that CI selects the entry and
   hardware. Add focused cases for uncovered model-specific paths.
3. **Numerical regression golden, from remote GPUs.** Select a reduced debug
   recipe retaining the important architecture and feature paths. First prove
   repeatability across at least two independent remote launches, each with
   at least 10 optimizer steps: seed 42, strict deterministic execution, and
   bitwise-identical full-precision loss and grad_norm at every step. Then export
   and commit the small, architecture/mode-specific golden and its provenance,
   register `golden_numerics_path`, and rerun without export. A deterministic
   setting alone does not prove repeatability. Reuse the combined training test
   when repeatable; otherwise establish a deterministic numerical fixture and
   document the combined test's limitation. Keep golden creation gated on the
   repeatability proof.

- For non-computation changes, require bitwise-identical loss and grad_norm with
  `config.debug.seed = 42` and `config.debug.deterministic = True`, using matching
  parallelism and GPU settings. Never enable `deterministic_warn_only`. Use
  `scripts/loss_compare.py` with `--metrics loss,grad_norm --assert-equal` to compare
  full-precision TensorBoard values. Computation changes require convergence on
  representative data, such as C4.
- Run the relevant parity and integration matrix after the implementation and
  cleanup pass. For performance measurements, use at least 10 training steps.
- Apply [code_style_checklist.md](references/code_style_checklist.md), then run
  `pre-commit run --all-files` before review. Resolve local type-check environment
  differences before allowing a hook to remove unused ignores.

## Phase 5: Scale validation when requested

Run multi-node training when it is part of the requested validation and the
needed resources are available. Use the target cluster's supported launcher,
compiler, communication stack, and dependency versions; configure persistent
output storage through `--output-dir`. Investigate failures in both the model's
distributed behavior and its launch environment. Record the recipe, topology,
data, steps, and any unvalidated combinations in the handoff.

## Definition of done

- [ ] Public flavor/build API, model lifecycle, and adapter work for the new model.
- [ ] Reuse and placement follow current model/config guides; shared callers audited.
- [ ] Reference parity has reproducible inputs and justified tensor tolerances;
      MoE router behavior and new backward paths are validated.
- [ ] Checkpoint loading and adapter mappings remain compatible for affected models.
- [ ] Important/new features have meaningful CPU tests, including stateful behavior.
- [ ] Training hooks, input contracts, and any extra PP state/gradients are validated.
- [ ] A real-GPU model training test combines FSDP/TP/EP/PP where applicable;
      its actual config and CI selection are verified.
- [ ] The chosen debug recipe is demonstrably deterministic across independent
      remote launches before golden generation; every loss/grad_norm step matches.
- [ ] A remote debug-recipe golden records loss and grad_norm, provenance, and
      hardware; a registered regression run passes without export.
- [ ] Applicable exact-loss/grad_norm or convergence evidence is recorded.
- [ ] Style and pre-commit checks pass; measured performance uses sufficient steps.
- [ ] Requested scale validation is complete; remaining coverage is stated clearly.
- [ ] The PR explains the architectural choices and includes numerical evidence.
