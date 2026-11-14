# Model Enablement Checkpoints

Record decisions and evidence in the port plan or PR description as each
checkpoint is completed. Include the source revision and recipe they cover.
Resolve findings and update the records after changes to the design or code.

| Before | Required record |
| --- | --- |
| Implementation | Reviewed component/parameter inventory, optimizer policy, initial memory estimate, test plan, and criteria for each PR |
| Large reference/GPU checks | Simplification decisions, saved-tensor inspection, and justification for each proposed test |
| Long training / claiming feasibility | Measured baseline memory compared with the estimate and hardware budget |
| Qualifying each PR | Actual base/head, prerequisites, supported parallelisms, required results, and current local review |

## 1. Review the design before implementation

Build a small inventory from the pinned reference and closest TorchTitan models:

| Component / parameter group | Existing component or new code | Logical shape, storage layout, dtype | Optimizer / trainability | State owner / code location |
| --- | --- | --- | --- | --- |
| Fill per distinct group | Reuse choice and reason for new code | Include sharing and checkpoint mapping | Assignment and policy source | Module that registers state; model/common/base/distributed package |

Settle parameter storage, sharing, initialization, and ownership before coding.
Follow the model report's optimizer policy when available; justify assignments
such as Muon versus AdamW rather than choosing from tensor rank alone. Identify
the canonical parameter behind aliases and the owner of updates and resets.

Review the inventory locally against the reference and repository patterns.
Record findings, alternatives, and chosen design. Resolve decisions affecting
shapes, storage, optimizer behavior, or shared APIs before implementation.
List prerequisite fixes and assign each planned PR its own completion criteria.

## 2. Establish training memory feasibility

Estimate peak memory per rank for the intended baseline model, hardware, batch,
sequence length, dtype, FSDP/EP layout, optimizer, and compile/remat settings.
Include:

- Stored parameters and buffers, gradients, optimizer moments, and master weights
  where used.
- Transient unsharded weights during FSDP gathers and any additional copies or
  concatenations retained through backward.
- Saved backward tensors, including attention intermediates, gathered inputs,
  and transformed weights.
- Attention/expert workspaces, communication buffers, and other live state.

Account for shapes, storage aliasing, and overlapping lifetimes. Count shared
storage once and allow headroom for allocator behavior and uncertain workspaces.
Explain how the estimate scales with sequence length, microbatch size, and mesh.
Inference weight size or forward parity alone does not establish feasibility.

Before large reference/GPU checks, use reduced components to inspect tensors
retained for backward and check scaling assumptions. Before long training or
claiming the baseline fits, run a short forward/backward/optimizer check of the
intended recipe on its target hardware. Include lazy optimizer-state allocation
and compile/graph warmup when enabled. Record peak allocated and reserved memory
across ranks and check the worst rank against the budget. Reuse the baseline
smoke run for memory and lifecycle evidence when it covers the intended settings.

Compare measurements with the estimate and investigate unexpected saved tensors
or copies. Keep memory feasibility pending until required GPU measurements run;
continue independent implementation and CPU checks. Update estimates and
evidence after changes to storage, sharding, dtype, or remat.

## 3. Simplify before expensive validation

For each new wrapper, override, file, and shared API, record the concrete need,
existing alternative, and ownership. Remove additions without a demonstrated
need before long parity or training jobs.

Consider ordinary module composition before custom autograd. Check whether
small dtype/layout operations can stay in the owning module and whether existing
interfaces cover the callers. Inspect the backward storage cost of compositions.
Overrides must preserve relevant inherited compile decorators, remat annotations,
sharding contracts, and lifecycle hooks. Re-run affected checks after simplifying.

## 4. Check distributed lifecycle and checkpoint state

- List expected gradient participation for each parameter group. Exercise ranks
  with no local blocks, tokens, or routes where applicable. Preserve a
  zero gradient connected to the computation graph when collective participation
  requires it.
- Set trainability consistently for each logical parameter group across ranks.
  When an objective is disabled, freeze and exclude parameters that have no
  remaining training objective. Verify shared parameters still receive their
  expected gradients through active objectives.
- Check auxiliary-loss/statistics accumulation, updates at optimizer boundaries,
  and resets after steps and evaluation. Remat must not count statistics twice.
- Compare packed inputs with separate documents. Cover position and recurrence
  resets, padding, empty work, and cross-stage state where supported.
- Verify HF import/export and native save/load round trips, including optimizer
  moments, persistent buffers, and tied parameters. For indexed or unusual
  storage, compare logical rows/slices and their optimizer state after reload.
  Save at supported boundaries and check the next optimizer step against an
  uninterrupted run under matching deterministic settings. Reset temporary state
  according to its lifecycle.

## 5. Justify tests by uncovered invariant

Before writing each test, record the uncovered invariant or silent regression,
existing coverage and its gap, independent expected result, owning PR, and
hardware. Exercise the actual feature path. Avoid expected results that copy the
implementation, assertions of config constants, and repeated config-error checks
already covered elsewhere.

Prefer extending an existing integration recipe with missing save/load or
lifecycle coverage. Add a separate GPU test when an uncovered invariant needs
a different recipe or topology. Follow [test_coverage.md](test_coverage.md) for
the three permanent test layers and offline numerical evidence.

## 6. Qualify each PR on its own base and scope

Record the PR's actual base/head, prerequisites, implemented features and
parallelisms, required tests, and completion evidence. Required fixes must be
present in the tested base or PR before qualification; re-run affected checks
when they change.

The first MoE PR qualifies FSDP+EP training (FSDP for dense models). Later PRs
qualify the TP/SP, PP, or CP paths they add. State deferred work in the plan.
Include the design review, memory evidence, simplification decisions, and test
justifications in each PR's record. Link earlier records when they still apply
and update records affected by this PR. Only mark its criteria complete after the
required checks run on the intended recipe and current version. Then follow the
local review loop in [code_style_checklist.md](code_style_checklist.md).
