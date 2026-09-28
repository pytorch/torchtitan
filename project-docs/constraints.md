# Project Constraints

## Scope boundaries

- Limit the project to MTP auxiliary-loss normalization.
- Do not redesign MTP model blocks, target construction, or parallelism abstractions.
- Do not create a generalized loss-weighting framework.
- Do not modify unrelated model architectures or training behavior.
- Keep the eventual implementation small enough for a focused upstream PR.

## Required semantics

- Preserve existing non-MTP behavior exactly.
- Preserve the ordinary main-loss normalization by the existing global valid-token count.
- Preserve `mtp_scale / num_mtp_layers` weighting.
- Use globally reduced per-depth denominators, never rank-local denominators.
- Preserve gradient-accumulation semantics: counts must cover every microbatch contributing to one optimizer step.
- Preserve `IGNORE_INDEX` behavior for padding, sequence tails, and packed-document boundaries.
- Preserve compatibility with relevant DeepSeek V3 and V4 paths.
- Preserve compatibility with chunked loss, pipeline execution, and the supported distributed configurations where MTP is enabled.

## API and implementation discipline

- Do not change the CLI unless it is absolutely required. A normalization correction should preferably use existing configuration and loss-call plumbing.
- Do not introduce speculative defensive checks or unrelated abstractions.
- Do not optimize prematurely. First establish correct counts, reductions, and numerics.
- Do not change the shared `AuxLoss` infrastructure unless the MTP path genuinely requires it. The MTP TODO concerns `MTPLoss`, while `AuxLoss` is primarily used for MoE auxiliary losses.
- Do not make speculative changes to context-parallel or pipeline behavior. Verify the required denominator semantics from existing execution paths first.

## Experiment boundaries

- Do not run the controlled training experiment during the documentation or initial implementation-planning phase.
- Do not benchmark performance yet.
- Do not change training schedules, optimizers, model sizes, datasets, or parallelism as part of the implementation itself.
- The later experiment must compare the current shared-denominator objective with the per-depth objective under fixed initialization, data order, token budget, optimizer, schedule, precision, model, and parallelism.

## Open questions for Phase 1

### Intended objective

Confirm whether TorchTitan intends each MTP depth to contribute its own mean token loss, or whether the shared denominator is deliberate and should remain the objective. The TODO and MTP documentation suggest per-depth normalization, but this should be confirmed before changing behavior.

### Source of counts

Decide whether per-depth counts should be derived from:

- shifted MTP labels;
- the existing shifted validity masks;
- a new metadata structure produced during preprocessing.

The chosen source must match the labels actually consumed by cross entropy.

### Reduction dimensions

Determine exactly which distributed meshes must participate in each count reduction under data parallelism, context parallelism, tensor parallelism, expert parallelism, and pipeline parallelism. The result must have the same global-token semantics as the current main-loss denominator.

### Transport through execution paths

Determine how a tuple of per-depth denominators should pass through:

- ordinary forward/backward execution;
- pipeline schedules;
- `ChunkedLossWrapper`;
- compiled loss paths, if enabled.

The solution should remain MTP-specific and avoid changing the general loss API more than necessary.

### Zero-valid-token depths

Define behavior when an MTP depth has zero valid targets, which can occur with very short documents, heavy packing, or aggressive padding. The implementation must avoid division by zero while preserving clear objective semantics.

### DeepSeek V4 coverage

Confirm whether the V4 implementation uses exactly the same `MTPLoss` call path in every supported configuration and identify the minimum test coverage needed for both V3 and V4.

### Numerical expectations

Define whether the proposed change is expected to preserve loss numerics in equal-count cases only, or whether a convergence comparison is required for the intentionally changed objective. The shared-denominator and per-depth objectives should not be expected to produce identical training curves when counts differ.

## Explicit non-goals

- No generalized per-token weighting system.
- No new MTP architecture.
- No changes to ordinary next-token prediction.
- No changes to MoE load-balancing loss normalization.
- No changes to data packing policy.
- No performance tuning before correctness is established.
- No controlled training experiment in this phase.
