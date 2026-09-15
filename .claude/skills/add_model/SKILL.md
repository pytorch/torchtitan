---
name: add_model
description: Add or port a model to TorchTitan using shared components, model flavors, Python recipes, forward/backward parity, CPU feature tests, parallelism validation, numerical goldens, and locally reviewed PR stacks. Use when the user requests a new model or invokes /add_model.
---

# Adding a Model to TorchTitan

Build the model in a PR stack. The initial model PR establishes reference
correctness and an FSDP+EP training baseline for MoE models (FSDP for dense
models). Add other parallelisms, any required new kernels, and requested
performance work in later PRs. Batch related edits before expensive GPU checks;
rerun affected checks after fixes or changes to computation. Review every PR
locally before submitting the stack.

Follow applicable `AGENTS.md`, `.claude/CLAUDE.md`, and `.claude/rules/`
instructions. Read the current [model guide](../../../torchtitan/models/README.md)
and [config guide](../../../torchtitan/config/README.md) before choosing an API.
Repository paths below are relative to the repository root. Verify helpers and
example harnesses in the checkout before using them.

- Read [enablement_checkpoints.md](references/enablement_checkpoints.md) before
  implementation, expensive validation, and qualification of each PR. Record
  the required design, memory, simplification, and test evidence before proceeding.
- Read [numerical_parity.md](references/numerical_parity.md) when building or
  running reference comparisons.
- Read [test_coverage.md](references/test_coverage.md) when planning feature
  tests, combined parallelism training, and remote numerical golden baselines.
- Read [code_style_checklist.md](references/code_style_checklist.md) for cleanup
  and the required `pr_review` -> revise -> `pr_review` loop for each PR.
- When available, use `/numerics_debugging` to localize unexplained drift and
  `/pyrefly_ci_check` when local type checking disagrees with CI. Model-specific
  numerical skills can supply an existing harness for that family.

## Phase 0: Scope and reference

1. Identify the model variant, supported tasks, checkpoint and reference-code
   revisions, and requested parallelisms. Reference candidates include Hugging Face
   `transformers`, Megatron-LM/Megatron Core, the model author's implementation,
   SGLang, and vLLM. Choose references that support the exact variant, features,
   and inference or training checks; pin their revisions.
2. Inventory attention and sequence boundaries, positional encoding, norms,
   dense FFN or MoE routing, shared experts, auxiliary objectives, weight tying,
   and any vision encoder or projector. Compare each with existing components.
3. Establish the available checkpoints, dependencies, and hardware. Prefer the
   model's official open weights on Hugging Face Hub when available; record the
   repository ID and pin the checkpoint revision. Plan three test artifacts:
   CPU coverage of important/new features, a real-GPU training test combining
   supported parallelisms, and a debug-recipe numerical golden baseline generated
   on remote GPUs. Normally reserve 8 H100s for combined FSDP/TP/EP/PP coverage.
   Also plan forward/backward reference parity, an offline parallelism matrix,
   convergence, and requested scale validation. Record checks that cannot run.
4. Plan the stack around these stages:
   - Prerequisite fixes with their own tests, when needed.
   - Model definition, FSDP+EP for MoE models (FSDP for dense models),
     forward/backward parity, and feature tests.
   - Additional parallelisms and any required new kernels, with validation.
     A suggested progression is TP+SP, then PP, then CP. Choose the order and
     PR boundaries based on the model's requirements and dependencies.
   - Performance optimization when requested.
   Split larger stages further, for example decoder and image encoder support.
   Each PR must be a working increment with its own scope, prerequisites,
   supported parallelisms, tests, and completion criteria.
   Reuse existing attention backends, including FlexAttention and varlen paths,
   and standard remat integration during initial enablement.
5. Review the component/parameter inventory, storage and optimizer policy,
   ownership, initial training memory estimate, and test plan before implementation.
   Follow [enablement_checkpoints.md](references/enablement_checkpoints.md) and
   record the decisions and evidence. Resolve required design choices first.

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

Before large reference/GPU checks, record the simplification pass, inspection of
tensors saved for backward, and test justifications from
[enablement_checkpoints.md](references/enablement_checkpoints.md).
Then use [numerical_parity.md](references/numerical_parity.md):

1. Build the matching model config with `ParallelismConfig()` for the unsharded
   reference comparison. Match the reference's weights, inputs, masks, positions,
   dtype, and mode. Reuse the generic checkpoint converter when a native DCP is
   needed.
2. Compare each component, one block, any vision/projector path, and full-model
   outputs in evaluation and training modes. Reuse shared component tests where
   they cover unchanged math, and verify the port's integration. Record enough
   reference details to reproduce each comparison.
3. Check actual tensors with justified absolute and relative tolerances. Report
   logit/probability metrics as additional evidence. For MoE, validate the router
   separately and use imposed routing to diagnose the remaining computation.
4. Compare input gradients where defined and parameter gradients for differentiable
   components and the full training model. Check normalized training loss,
   auxiliary objectives, and state updates. Use references that support backward
   or independent differentiable oracles; record any missing checks as gaps.
   Revalidate loading when native checkpoint keys, shapes, or layouts change.

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
7. Run an offline numerical matrix covering each claimed parallelism and important
   combinations. Match logical inputs, checkpoints, global batch, and loss
   normalization. Compare outputs, loss, gradients, and global gradient norms
   with justified tolerances. Follow
   [numerical_parity.md](references/numerical_parity.md) for reconstruction and
   scaling checks. Summarize results in the PR; keep matrix scripts and logs in
   generated outputs.
8. Check gradient participation on ranks with empty work, trainability when
   objectives are disabled, auxiliary-loss resets, and checkpoint state. Follow
   the distributed lifecycle checks in
   [enablement_checkpoints.md](references/enablement_checkpoints.md), including
   optimizer moments for changed parameter storage.

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

Use [test_coverage.md](references/test_coverage.md) for the completed stack's
three test layers. Implement each PR's planned coverage before qualifying it:

1. **Important feature tests, mostly CPU.** Cover each new computation or
   stateful feature with independent expected results, edge cases, gradients
   where applicable, and lifecycle behavior. For quantile balancing, test the
   histogram/quantile math, padding, bias updates, and accumulation/reset timing.
   Extend shared flavor/config and adapter contracts as well.
2. **Combined parallelism training, on real GPUs.** Start the test with FSDP+EP
   for the first MoE PR and extend it as parallelisms are added. The completed
   stack needs a test with FSDP, TP, EP, and PP active together, normally on
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

- Maintain the compact coverage table for every component and the full model,
  including mode, forward/backward checks, reference, recipe/parallelism, and
  evidence. Commit focused feature tests, the combined integration test, and the
  golden; report the wider offline matrix in the PR description.
- For each proposed test, record the uncovered invariant, existing coverage,
  independent expected result, owning PR, and hardware before writing it.
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

## Phase 5: Performance work when requested

Prepare performance changes as later PRs after correctness and parallelism
validation. Choose a feasible model, batch, sequence length, and parallelism
config for the available hardware, then submit benchmark jobs with the cluster's
supported launcher. Compare matching workloads: training with HF/Megatron and
supported inference paths with HF/vLLM/SGLang, matching prefill/decode and cache
settings. Record dtype, backend, kernels, compilation, and hardware. Use at least
10 measured optimizer steps after warmup for training; report throughput, step
time, and peak memory. Use `local-compile-perf` when available for compile tuning,
and revalidate numerics after changes.

## Phase 6: Scale validation when requested

Run multi-node training when it is part of the requested validation and the
needed resources are available. Use the target cluster's supported launcher,
compiler, communication stack, and dependency versions; configure persistent
output storage through `--output-dir`. Investigate failures in both the model's
distributed behavior and its launch environment. Record the recipe, topology,
data, steps, and any unvalidated combinations in the handoff.

## Phase 7: Local review and stack submission

Follow [code_style_checklist.md](references/code_style_checklist.md) for each PR.
Resolve and read the `pr_review` skill (TorchTitan also names it `pr-review`).
Use its local branch mode with the PR's actual base and head refs. Review,
revise, and review again until the current version has local approval, no open
must-fix findings, and its required checks pass. Re-review affected dependent
PRs after earlier changes. Submit the stack targeting `upstream/main` only when
every PR passes; include dependencies, numerical evidence, and review results.
If a required skill, resource, or decision is missing, report the blocker and
keep stack submission pending.

## Definition of done

Qualify each PR against its own recorded criteria. The complete model port must
satisfy this checklist:

- [ ] A PR stack has working increments and validation criteria for each PR.
- [ ] Reviewed design and optimizer policy are recorded before implementation.
- [ ] Training memory estimates and baseline measurements establish feasibility.
- [ ] Simplification decisions and test justifications precede expensive validation.
- [ ] The initial model PR validates an FSDP+EP training baseline for MoE models
      (FSDP for dense models), including numerical evidence.
- [ ] Public flavor/build API, model lifecycle, and adapter work for the new model.
- [ ] Reuse and placement follow current model/config guides; shared callers audited.
- [ ] Component and full-model forward/backward parity has reproducible inputs,
      justified tolerances, and appropriate references; MoE routing is validated.
- [ ] HF checkpoint compatibility and native save/load round trips are verified,
      including affected optimizer state and logical parameter storage.
- [ ] Important/new features have meaningful CPU tests, including stateful behavior.
- [ ] Training hooks, input contracts, and any extra PP state/gradients are validated.
- [ ] A real-GPU model training test combines FSDP/TP/EP/PP where applicable;
      its actual config and CI selection are verified.
- [ ] Offline comparisons cover claimed parallelisms and important combinations,
      including gradient scaling; the PR records configs, metrics, and evidence.
- [ ] The chosen debug recipe is demonstrably deterministic across independent
      remote launches before golden generation; every loss/grad_norm step matches.
- [ ] A remote debug-recipe golden records loss and grad_norm, provenance, and
      hardware; a registered regression run passes without export.
- [ ] Applicable exact-loss/grad_norm or convergence evidence is recorded.
- [ ] Style and pre-commit checks pass; measured performance uses sufficient steps.
- [ ] Requested performance comparisons use matched workloads and record results.
- [ ] Requested scale validation is complete; remaining coverage is stated clearly.
- [ ] Every PR's current version passes `pr_review` after all required revisions;
      affected dependent PRs have been re-reviewed before stack submission.
- [ ] The PR explains the architectural choices and includes numerical evidence.
