# Model Port Cleanup and Review Checklist

Use this checklist for new model code and shared code changed by the port. Follow
`AGENTS.md`, `.claude/CLAUDE.md`, and applicable `.claude/rules/` instructions.
Confirm the early records in
[enablement_checkpoints.md](enablement_checkpoints.md), including design review,
training memory, simplification, and test justifications before expensive
validation.

## Reuse, placement, and public API

- Reuse existing attention, RoPE, FFN/MoE, embedding, linear, decoder, and adapter
  helpers. Put shared code in the appropriate base, common, or distributed package.
  Keep core training and parallelism code PyTorch-native.
- Put lifecycle contracts in `torchtitan/models/base/`, concrete shared model
  components in `torchtitan/models/common/`, and distributed helpers in
  `torchtitan/distributed/`. Keep experiment code in its experiment package.
- Follow the current package API: `flavors.py` provides `MODEL_FLAVORS` and
  `build_model_config`; `__init__.py` re-exports these and the model class.
  Set `state_dict_adapter_cls` on the model class. Add `sharding.py` when shared
  constructors cannot express the model's sharding.
- Keep runnable training recipes in `torchtitan_recipes/` and reusable test
  recipes in `torchtitan_recipes/tests/models/`. Keep integration configs beside
  their test definitions.
- Set `param_init` on component configs. Use `_init_self_buffers()` for buffer
  initialization. Reuse automatic `init_states()` recursion; override it only
  when ordering requires it. Check weight tying after materialization and loading.
  Keep routing and conditional computation visible in `forward`.
- Justify each new wrapper, override, file, and shared API against existing
  alternatives. Preserve inherited compile decorators, remat annotations, and
  lifecycle contracts when overriding methods.

## Naming, comments, and errors

- Give tensors shape suffixes in new or rewritten model code (`x_BLD`, `q_BLHK`,
  `out_THV`). Define the letters in each module. They describe logical tensor
  dimensions and keep their meaning under sharding.
- Name components for what they compute. Use plain descriptive names for
  internal components; the package identifies the model. Prefix private helpers
  with `_` and counts with `num_`, unless matching an upstream API. Follow sibling
  flavor names.
- Use `axis`/`axes` for specific DeviceMesh axes and `dim` for tensor dimensions
  or mesh shape. Match upstream spelling when calling its APIs.
- Use ASCII in comments and docstrings you add or rewrite. Explain non-obvious
  math, gradient placements, and workarounds. Remove obvious narration and unused
  code; keep TODOs for known limitations.
- Give unfamiliar APIs concise docstrings and a small usage or shape example.
  Keep function names clear and comments focused on the reason for the behavior.
- Remove unused settings and casts, fallbacks, or wrappers with no demonstrated
  need. Preserve supported backends and config paths; audit callers of any
  intentional removal.
- Use `ValueError` for bad config or input and `assert` for internal programmer
  errors. Required config fields must not default to `None`. Put important
  arguments first and prefer keyword-only options.
- `dataclasses.replace()` shares nested objects. Copy nested configs when they
  must change independently.

## Audit changed contracts

- Search all tracked model variants, recipes, and tests when changing a
  shared component, field, or distributed helper. Update every affected caller.
- Match sharding input keys to forward parameter names after tensor renames.
  Validate mesh axes, tensor placements, and gradient placements at DTensor
  boundaries. Build configs for the recipe's parallelism.
- Check that compile region names and remat patterns match the intended
  computation and are applied. Raise `ValueError` when requested configuration
  is unsupported or would be ignored.
- Check optimizer hooks and update/reset timing for stateful features. Check
  positions, masks, document and modality metadata, and any extra PP state.
  Trace its gradients and cache cleanup.
- Check expected gradients on ranks with empty work and parameter trainability
  when objectives are disabled. Verify auxiliary-loss resets and packed boundaries.
- Test HF import/export and native save/load round trips, including tied weights
  and optimizer moments for changed storage. Check affected sibling HF mappings
  and current native round trips too.
- Follow [test_coverage.md](test_coverage.md) for CPU feature tests, combined
  FSDP/TP/EP/PP training on real GPUs where applicable, and a remote debug golden.
  Verify the actual recipe, model, mesh, feature hooks, and CI suite and hardware.
- Check component and full-model forward coverage in evaluation and training
  modes, plus backward coverage for differentiable components and the full
  training model. Review the offline matrix, reconstructed gradients, global
  norm ratios, and the reported tolerances and evidence.
- Before creating a golden, require independent remote debug runs to match
  exactly for every full-precision loss and grad_norm step. Keep initialization,
  data order, optimizer, backend, and required features the same.
- Check that the golden covers all expected loss and grad_norm steps, records
  run details, and passes an independent replay. Include output and gradient
  comparisons for new computation and exact loss/grad_norm comparisons for
  refactors. Add convergence evidence for computation changes.

## Local review and stack submission

Run `pre-commit run --all-files` before review. If local Pyrefly results differ
from CI, use `/pyrefly_ci_check` when available. Inspect `pyrefly check` output
and fix environment differences before a hook removes unused ignores. Fix issues
caused by the port and report any existing or environment failures.

For every PR in the planned stack:

1. Prepare a working increment with its required tests and numerical evidence.
   The initial model PR validates FSDP+EP training for MoE models (FSDP for dense
   models).
   Verify prerequisites and this PR's completion criteria on the tested version.
   Include its design, memory, simplification, and test justification records.
   Record its base and head refs. The first PR starts from `upstream/main`;
   later PRs are reviewed against their immediate parent in the stack.
2. Resolve and read the `pr_review` skill (`pr-review` in TorchTitan's naming).
   Use local branch mode and explicitly supply the PR's base/head as review
   context, overriding its single-branch `main...HEAD` default for stacked PRs.
   Apply current user/repository rules and keep findings local.
3. Fix each valid must-fix finding in the PR that owns it. Re-run affected tests
   and numerical comparisons, then amend the appropriate commit(s).
4. Run `pr_review` again on the revised PR. Repeat review, revise, and re-review
   until the current version has an **Approve** verdict, no open must-fix items,
   and all required checks pass. An approval with remaining must-fix edits is
   still pending; **Request changes** or **Needs discussion** is not a pass.
5. Update dependent PRs after changes to an earlier PR. Re-review any PR whose
   code or base changed. Record the reviewed base/head, resolved findings, and
   final verdict so approval of an old version is not reused.
6. When every PR passes and the stack is stable, submit it using the repository's
   supported stack tooling, targeting `upstream/main`. Verify dependencies and
   that each published PR shows the reviewed changes. Include the scope,
   numerical evidence, and local review results in each description.

If the review skill, required validation resources, or a necessary decision is
missing, report the blocker and keep submission pending. Continue independent
implementation and validation work; never mark an incomplete review as passed.

Record reference and checkpoint revisions, tolerances and measured errors,
tests run, supported parallelism, and remaining limitations in the PR.
