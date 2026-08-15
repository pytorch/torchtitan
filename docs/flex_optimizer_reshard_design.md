# Restoring `flex_optimizer_reshard` and adopting `torch.optim.Muon`

## Status

The accompanying TorchTitan change implements the target integration. Users
construct a normal `torch.optim.Muon`, pass that same object to
`flex_optimizer_reshard`, and retain the complete redistribution, batching,
overlap, and checkpoint behavior of the former custom optimizer. The returned
object remains the original exact `torch.optim.Muon` instance.

This implementation consumes the supported PyTorch step-executor API and
public phased Muon operations described below. It fails during configuration,
before process-group or planning side effects, when the installed PyTorch does
not provide those APIs. The historical rollout sections retain the design-only
name `DistMuon` for the implementation removed by this change; it is not an
exported or selectable optimizer.

## Motivation

`build_flex_shard_muon` is convenient for TorchTitan, but it combines two
operations:

1. Construct a particular optimizer.
2. Bind that optimizer to FlexShard compute layouts and physical buckets.

`flex_optimizer_reshard` separates those operations and provides a stable
entry point:

```python
optimizer = torch.optim.Muon(named_param_groups, **optimizer_kwargs)
optimizer = flex_optimizer_reshard(
    optimizer,
    compute_sharding_by_fqn=compute_sharding_by_fqn,
    bucket_configs=bucket_configs,
)

assert type(optimizer) is torch.optim.Muon
```

The function configures the optimizer to execute its work with temporary
compute shardings that may differ from persistent DTensor storage shardings.
It performs no runtime tensor redistribution until `optimizer.step()`.
Configuration may create process groups and run plan-validation collectives,
so all participating ranks must call it in a consistent order.

## Target end state

The final ownership boundary is:

- PyTorch owns the `Muon` optimizer, its hyperparameters, state schema,
  closures, update mathematics, and reusable Muon tensor operations.
- FlexShard owns FQN binding, storage-to-compute transition planning, physical
  communication buckets, packed collectives, stream ordering, overlap, and
  checkpoint-time replanning.
- A private `_MuonReshardIntegration` connects the two without subclassing
  `torch.optim.Muon`, replacing its `step` method, or changing its exact type.
- Persistent parameters and momentum remain keyed and sharded exactly as they
  are in the input optimizer. Only temporary optimizer inputs and directions
  use FlexShard compute layouts.
- The former optimizer class and locally maintained Muon kernels are deleted;
  the existing acceptance suite validates the stock-Muon path.

Physical buckets and optimizer parameter groups remain independent concepts.
Parameter groups describe optimizer hyperparameters and checkpoint state;
physical buckets describe communication-compatible work and private execution
ordering. FlexShard must not create one `Muon` instance or one parameter group
per physical bucket.

## Historical API and why its mechanism should not return

The removed API had this public signature:

```python
def flex_optimizer_reshard(
    optimizer: DistMuon,
    *,
    compute_sharding_by_fqn: Mapping[str, ComputeLayout],
    bucket_configs: Sequence[BucketConfig],
) -> DistMuon:
    ...
```

Its implementation dynamically replaced `optimizer.__class__` with a class
whose bases were `FlexOptimizer` and the original optimizer class. It then
called PyTorch's private `Optimizer._patch_step_function()` so the new mixin's
`step()` received optimizer profiling and hooks.

That mechanism had several problems:

- The generic-looking API rejected every optimizer except `DistMuon`.
- The exact Python class changed after construction.
- It depended on a private PyTorch method.
- Whole-optimizer pickling was unsupported.
- It had to run before an LR scheduler or another utility wrapped `step()`.
- The public FlexShard package imported a supposedly generic API from the Muon
  consumer module.

The useful parts were the call shape, same-object return, validation, frozen
topology, and generic planning/runtime seams. The dynamic mixin and class
mutation should not be restored.

## Goals

- Restore `flex_optimizer_reshard` as the public configuration front door.
- Return the same optimizer object without changing its exact type.
- Support a normally constructed `torch.optim.Muon` as the final Muon API.
- Preserve all behavior covered by the former optimizer acceptance suite, including
  logical matrix views, batched Muon compute, arbitrary legal physical
  buckets, redistribution overlap, pipeline parallelism, and checkpoint
  replanning.
- Reuse PyTorch-owned Muon mathematics instead of maintaining a forked
  optimizer implementation and kernels in FlexShard.
- Keep `build_flex_shard_muon` as a TorchTitan construction helper returning a
  configured `torch.optim.Muon`.
- Preserve one public `BucketConfig` as one physical bucket.
- Keep scheduling, communication, and overlap implementation private.
- Define a clean optimizer-integration boundary without claiming arbitrary
  `torch.optim.Optimizer` support.
- Keep FlexShard self-contained and PyTorch-only so the folder remains
  extractable into a standalone package.

## Non-goals

- Dynamically replacing or wrapping an optimizer's `step()` method.
- Depending on PyTorch private methods such as `Muon._init_group`,
  `Optimizer._patch_step_function`, or private functional Muon kernels.
- Making an arbitrary existing optimizer reshardable without algorithm-specific
  integration.
- Encoding physical communication buckets as optimizer parameter groups or as
  multiple optimizer instances.
- Moving persistent parameters or momentum into compute sharding between
  steps. Compute sharding is temporary execution state.
- Publishing a third-party optimizer adapter protocol before a second optimizer
  validates its shape.
- Serializing FlexShard plans, streams, buffers, or process groups in the
  optimizer state dict.
- Supporting whole-object pickling of a configured optimizer. Users reconstruct
  the optimizer and FlexShard binding, then load `state_dict()`.
- Inferring or silently splitting physical communication buckets.

## Public API

The historical Phase 1 call was explicit about the self-hosted optimizer:

```python
def flex_optimizer_reshard(
    optimizer: DistMuon,
    *,
    compute_sharding_by_fqn: Mapping[str, ComputeLayout],
    bucket_configs: Sequence[BucketConfig],
) -> DistMuon:
    """Configure a supported optimizer for FlexShard redistribution.

    Return the same optimizer object. Configuration freezes parameter
    membership, FQNs, compute shardings, and physical bucket topology.
    """
```

The Phase 1 implementation was generically typed so it preserved the concrete
optimizer type, then dispatched through a private optimizer hook. The call
shape did not need to change for the stock-Muon integration.

During the historical Phase 1, the package root exported:

- `flex_optimizer_reshard`
- `DistMuon`
- `build_dist_muon`
- The existing compute-layout and physical-bucket configuration types

`build_dist_muon` was a convenience composition:

```python
def build_dist_muon(...):
    normalized_param_groups = _prepare_named_param_groups(...)
    return flex_optimizer_reshard(
        DistMuon(normalized_param_groups, **optimizer_kwargs),
        compute_sharding_by_fqn=compute_sharding_by_fqn,
        bucket_configs=bucket_configs,
    )
```

Documentation must describe the function as configuring a *supported*
optimizer. A generic redistribution runtime cannot update arbitrary optimizer
state or run arbitrary optimizer compute on temporary tensors without an
algorithm-specific integration.

The current supported signature uses stock Muon:

```python
def flex_optimizer_reshard(
    optimizer: torch.optim.Muon,
    *,
    compute_sharding_by_fqn: Mapping[str, ComputeLayout],
    bucket_configs: Sequence[BucketConfig],
) -> torch.optim.Muon:
    """Configure this Muon instance to execute through FlexShard."""
```

The final FlexShard package exports `flex_optimizer_reshard` and the existing
`BlockShard`, compute-sharding, and bucket configuration types. A flat 2D
parameter uses `BlockShard(dim=0, block_size=R)` to declare complete Muon
matrix boundaries. The private Muon integration derives the local
matrix-batch view from that placement; there is no public compute-view wrapper
or separate matrix-count configuration. The package no longer exports a
`DistMuon` optimizer class. TorchTitan retains
`build_flex_shard_muon` as its construction factory:

```python
def build_flex_shard_muon(...):
    normalized_param_groups = _normalize_param_groups(...)
    return flex_optimizer_reshard(
        torch.optim.Muon(normalized_param_groups, **optimizer_kwargs),
        compute_sharding_by_fqn=compute_sharding_by_fqn,
        bucket_configs=bucket_configs,
    )
```

The factory name does not imply a distinct optimizer implementation. New
integration code may construct `torch.optim.Muon` directly.

## Object identity and step ownership

`flex_optimizer_reshard` must satisfy:

```python
original_type = type(optimizer)
configured_optimizer = flex_optimizer_reshard(optimizer, ...)

assert configured_optimizer is optimizer
assert type(configured_optimizer) is original_type
```

Historical Phase 1 satisfied this contract because `DistMuon` declared
`step()` normally. Its shape was conceptually:

```python
class DistMuon(Optimizer):
    _optimizer_reshard_binding: _MuonReshardBinding | None

    @torch.no_grad()
    def step(self, closure=None):
        binding = self._require_optimizer_reshard_binding()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._preflight_step()
        binding.runtime.run(
            binding.plans,
            local_tensor_spec=self._local_tensor_spec,
            prepare=self._prepare_local,
            compute=self._compute_update,
            finalize=self._apply_update,
            local_bucket_executor=binding,
        )
        return loss
```

This preserves normal PyTorch optimizer hooks, profiling, schedulers, class
identity, and state-dict behavior. FlexShard does not call
`Optimizer._patch_step_function()`.

The final stock-Muon path must additionally satisfy:

```python
optimizer = torch.optim.Muon(named_param_groups, **optimizer_kwargs)
declared_step = optimizer.step.__func__

configured = flex_optimizer_reshard(optimizer, ...)

assert configured is optimizer
assert type(configured) is torch.optim.Muon
assert configured.step.__func__ is declared_step
```

Current PyTorch optimizer pre- and post-step hooks are insufficient: they can
observe or modify step arguments, but they cannot report that the original
step body was fully handled. Calling stock `Muon.step()` after FlexShard has
updated a bucket would apply the update twice. A supported upstream execution
seam is therefore a prerequisite for removing `DistMuon`.

### Required PyTorch step-execution seam

The exact upstream API should be decided in PyTorch, but it must provide these
semantics:

- Register one exclusive executor on an existing `torch.optim.Muon` instance.
- Invoke that executor from the normal profiled and hooked `Muon.step()`
  envelope instead of the default body.
- Give the executor ownership of closure evaluation and the returned loss so
  the closure runs exactly once.
- Run optimizer work under Muon's normal no-grad context and enable gradients
  only while evaluating the closure.
- Preserve the optimizer object's identity, exact type, declared `step`
  function, parameter groups, state, and scheduler references.
- Keep the registered executor out of `state_dict()`. Whole-object pickling of
  a configured optimizer must fail clearly rather than silently drop the
  executor and resume stock execution after unpickling.
- Reject or detect parameter-group membership changes before state mutation or
  communication. The final path must retain the currently frozen topology.
- Make executor installation atomic at the optimizer object. A failed install
  rolls back its registration, while an active FlexShard executor cannot be
  removed into an unsafe stock-step fallback. Already-created process groups
  may remain after a failed distributed configuration, as they do today.

One possible Muon-side shape is illustrative only:

```python
@torch.no_grad()
def step(self, closure=None):
    executor = self._registered_step_executor
    if executor is not None:
        return executor.step(self, closure)
    return self._default_step(closure)
```

PyTorch may instead provide an optimizer-wide handled-step hook. In either
case, the supported API must perform this dispatch; FlexShard must not assign
an instance `step`, mutate `__class__`, install a dynamic subclass, or override
private `_init_group` behavior.

### Required reusable Muon operations

The current PyTorch Muon implementation is monolithic and private. Its stock
step initializes state, then its private functional updates supplied momentum,
orthogonalizes, and updates parameters while assuming parameter, gradient,
momentum, and update share one 2D layout. FlexShard needs supported phased
operations so communication can occur between algorithm phases.

The upstream interface should provide operations equivalent to:

```python
muon_prepare(
    gradient,
    momentum_buffer,
    *,
    momentum,
    nesterov,
    out,
)
muon_orthogonalize(
    prepared,
    *,
    ns_coefficients,
    ns_steps,
    eps,
    out,
)
muon_apply(
    parameter,
    direction,
    *,
    lr,
    weight_decay,
    adjust_lr_fn,
    logical_matrix_shape,
)
```

The names are not part of this design. Their required semantics are:

- `muon_prepare` updates persistent momentum in storage layout and writes the
  temporary tensor that FlexShard may redistribute.
- `muon_orthogonalize` supports 2D matrices and independent batch-first
  matrices, permits runtime-owned output buffers, and does not own optimizer
  state.
- `muon_apply` applies weight decay and the returned direction exactly once to
  persistent storage. Learning-rate adjustment uses the logical matrix's last
  two tensor dimensions rather than an incidental flattened storage shape.
- Stock `torch.optim.Muon.step()` uses the same operations so default and
  FlexShard execution cannot silently drift.
- `torch.optim.Muon` construction supports the batch-first 3D expert
  parameters needed by the current integration. Today it rejects non-2D
  parameters before FlexShard can be attached.

These operations may remain Muon-specific. A generic optimizer decomposition
is not required for this migration.

## Internal integration and binding

In Phase 1, successful configuration installs one private, per-instance
binding owned by `DistMuon`:

```python
@dataclass(slots=True)
class _MuonReshardBinding(
    _LocalBucketExecutor[_ParameterComputeLayout]
):
    optimizer: DistMuon
    bucket_specs: tuple[_BucketSpec, ...]
    plans: tuple[_BucketExecutionPlan[_ParameterComputeLayout], ...]
    plan_items: tuple[_ParameterComputeLayout, ...]
    runtime: _BucketedRedistributionRuntime[_ParameterComputeLayout]
    parameter_group_membership_signature: tuple[
        tuple[tuple[int, str], ...], ...
    ]
    _local_execution_plans: dict[
        tuple[str, ...], tuple[_ParameterComputeLayout | _LocalMatrixBatch, ...]
    ]
```

The deliberately Muon-specific name reflects that this binding also owns
Muon's local matrix-batching cache. A second optimizer may prove a smaller
optimizer-neutral binding later. The exact fields remain private.
Conceptually, the binding contains all resolved FlexShard state that belongs
to this optimizer instance:

- Selected physical buckets and redistribution groups.
- Resolved optimizer work items.
- Communication and overlap plans.
- Runtime streams and reserved buffers.

It must not be stored in `Optimizer.state`, included in `state_dict()`, or
exposed as a public handle.

Configuration should compile a complete binding before installing it on the
optimizer. A validation or planning error therefore leaves the optimizer
object unconfigured instead of partially initialized. Cartesian process-group
creation is a distributed side effect and cannot be rolled back if a later
validation fails; the atomicity guarantee applies to the optimizer object.

The stock-Muon implementation moves optimizer behavior out of the optimizer
class and into a private integration:

```python
@dataclass(slots=True)
class _MuonReshardIntegration(
    _OptimizerStepExecutor,
    _LocalBucketExecutor[_ParameterComputeLayout],
):
    optimizer: torch.optim.Muon
    binding: _MuonReshardBinding
    first_step_validated: bool

    def step(self, optimizer, closure=None):
        assert optimizer is self.optimizer
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._preflight_step()
        self.binding.runtime.run(
            self.binding.plans,
            local_tensor_spec=self._local_tensor_spec,
            prepare=self._prepare_local,
            compute=self._compute_update,
            finalize=self._apply_update,
            local_bucket_executor=self,
        )
        return loss
```

`_MuonReshardIntegration` is illustrative and remains private. It contains the
Muon-specific interpretation of optimizer state and compute configuration. A
separate `_MuonReshardBinding` contains the compiled, runtime-only state:

- Selected physical buckets and redistribution groups.
- Resolved per-parameter compute layouts and original parameter-group indexes.
- Ordered local, redistribution, and overlap plans.
- Packed forward and reverse collective schedules.
- Runtime streams, events, and reserved double-buffer slots.
- The frozen parameter/FQN membership signature.
- Muon-compatible local matrix batching plans.

The integration reads hyperparameters from the original
`torch.optim.Muon.param_groups` and reads or creates momentum under
`torch.optim.Muon.state[parameter]`. Temporary compute tensors never become
optimizer parameters or state keys. The binding and integration are not
included in the optimizer state dict.

The phased operations run on local tensor views. After an in-place momentum or
parameter mutation, the integration increments the version counter of the
outer persistent DTensor, matching the current behavior and preserving
autograd's mutation tracking.

Configuration compiles and validates the complete binding before registering
the executor. If executor registration fails, the optimizer remains on its
normal stock step path. As in Phase 1, process-group creation is a distributed
side effect that cannot be rolled back.

## Optimizer integration boundary

FlexShard core owns optimizer-independent mechanics:

- Pattern matching and FQN-to-bucket binding.
- Validation of explicit physical redistribution axes.
- Process-group and participant-order resolution.
- Physical bucket compatibility validation.
- Communication-route execution and buffer management.
- Ordered local/redistributed execution and overlap.

The optimizer integration owns algorithm-specific semantics:

- Interpreting its per-FQN compute configuration.
- Resolving logical compute views.
- Resolving storage-to-compute transitions.
- Initializing and validating optimizer state.
- Estimating compute cost and assigning dynamic owners.
- Planning and executing local batches.
- Preparing optimizer inputs, computing updates, and applying updates.

PyTorch owns the reusable Muon phase operations that implement the last item.
The FlexShard integration owns when and on which temporary layout each phase
runs.

For the first restoration, this boundary remains a private optimizer hook on
`DistMuon`. The final implementation instead uses a private FlexShard
registry keyed by explicitly supported optimizer types. Conceptually:

```python
_OPTIMIZER_INTEGRATIONS = {
    torch.optim.Muon: _configure_muon_reshard,
}
```

`flex_optimizer_reshard` resolves the stock Muon integration, compiles its
binding, and registers `_MuonReshardIntegration` through the supported
PyTorch execution seam. `torch.optim.Muon` must not import or otherwise depend
on FlexShard. Initially, exact `torch.optim.Muon` is supported; subclasses are
rejected until their step and state semantics are explicitly evaluated.

After a second optimizer integration exists, the common boundary can be
evaluated and, if useful, promoted to a public
`OptimizerReshardIntegration` protocol. Publishing it before then would freeze
Muon-specific assumptions into an allegedly generic interface.

## Physical bucket contract

Restoring the function does not change the explicit physical-bucket design in
[`flex_shard_physical_bucket_design.md`](flex_shard_physical_bucket_design.md).

- One selected `BucketConfig` produces at most one physical `_BucketSpec`.
- `NoRedistribution()` means compute-ready local work, not unsharded storage.
- A nonlocal config declares the exact ordered redistribution mesh axes.
- Repeated patterns may describe explicit runtime-topology alternatives.
- Binding selects exactly one alternative from the resolved transition.
- Local and communicating buckets remain distinct and may be overlapped only
  by private scheduling plans.
- Incompatible parameters require separate public bucket configs; the runtime
  never silently splits one.

The optimizer-specific integration resolves compute sharding before the common
binder selects the matching physical bucket alternative.

### Arbitrary bucket execution and synchronization

No layer boundary is part of the physical-bucket contract. A legal bucket may
contain parameters from unrelated layers. Before the first collective,
preflight validation requires every configured gradient to exist, so execution
does not depend on backward layer readiness.

Arbitrary configuration does not mean arbitrary collective order. Binding
turns the user-provided config sequence into one deterministic ordered plan,
sorts FQNs within each bucket, and validates that all participants agree on
membership, tensor regions, rank order, dtype, device, and routes. The runtime
uses a bounded rolling pipeline rather than a general dependency scheduler.
With two buffer slots, it may prefetch the next redistribution bucket while the
current bucket computes.

For each redistribution bucket, one runtime work object binds the physical
plan, packed buffer spans, buffer slot, and stream events. The GPU dependency
chain is:

```text
caller stream produces gradients
    -> transfer stream prepares storage-local Muon input
    -> storage-to-compute all-to-all
    -> record compute_input_ready
    -> compute stream waits for compute_input_ready
    -> unpack this bucket and run Muon orthogonalization
    -> record compute_done
    -> transfer stream waits for compute_done
    -> compute-to-storage all-to-all
    -> apply this bucket's direction to persistent parameters
    -> record done before slot reuse or step return
```

For Muon, the forward collective redistributes the prepared gradient/momentum
input, not the persistent parameter. Orthogonalization runs only after that
bucket's `compute_input_ready` event. The returned direction is restored to
storage layout before weight decay and the parameter update. Local-only
buckets use the same phase callbacks without collectives and may run while a
communicating bucket is in flight.

This schedule remains owned by one registered executor on one
`torch.optim.Muon`. Multiple optimizer instances would fragment closure,
hook, scheduler, and checkpoint semantics without providing any additional
communication dependency mechanism.

Parameter groups cannot encode the same plan. Repeated bucket patterns are
explicit topology alternatives, and binding selects only one for a parameter;
an optimizer parameter cannot appear in multiple parameter groups. Even after
selection, parameter groups encode persistent hyperparameter and checkpoint
semantics, while buckets may be reordered or replanned for communication.
Using multiple Muon instances would still require an outer controller to run
the closure once and impose this event and collective order, recreating the
executor with a fragmented public optimizer API.

## Lifecycle contract

The supported sequence is:

1. Construct the concrete optimizer with aligned `params` and `param_names`.
2. Call `flex_optimizer_reshard` exactly once and in a consistent order on all
   ranks participating in its declared redistribution groups.
3. Construct schedulers or other training components as usual.
4. Load an optimizer state dict if resuming.
5. Call `step()`.

The new implementation does not technically need to precede scheduler
construction because it does not replace `step()`. Keeping configuration near
construction nevertheless makes the frozen topology explicit.

Applying FlexShard freezes:

- Parameter-group membership.
- Parameter identity and FQNs.
- DTensor storage topology expected by the plan.
- Compute-sharding configuration.
- Bucket configuration and physical topology.

Scalar optimizer settings such as learning rate remain mutable so schedulers
continue to work. Loaded settings that affect planning, such as Muon's number
of Newton-Schulz steps, trigger plan and buffer rebuilding in the existing
post-load hook. A pre-load hook rejects checkpoint `param_names` that differ
from the configured FQNs because PyTorch would otherwise overwrite them.

The state-dict workflow reconstructs the optimizer and FlexShard binding from
configuration before loading optimizer state. The binding itself is not
checkpointed.

## Validation and errors

Use user-facing exceptions for public contract violations:

- `TypeError` when the optimizer's exact type has no supported FlexShard
  integration.
- `RuntimeError` when the installed PyTorch version lacks the required Muon
  execution or phased-functional API.
- `ValueError` when `flex_optimizer_reshard` is applied more than once.
- `ValueError` for missing, duplicate, or misaligned FQNs.
- `ValueError` when checkpoint parameter names differ from configured FQNs.
- `ValueError` for unsupported compute configurations or storage transitions.
- `ValueError` when bucket declarations do not match resolved redistribution
  axes or cannot form one packed collective.
- An unconfigured stock `torch.optim.Muon` continues to execute its normal
  PyTorch step; only an explicitly configured instance uses FlexShard.
- `RuntimeError` when parameter-group membership changes after configuration.
- `RuntimeError` before any collective when a configured gradient is missing
  or a parameter, gradient, or momentum storage layout has changed.

An unsupported optimizer error should explain that the optimizer must provide
a registered FlexShard integration; it should not suggest that every PyTorch
optimizer is accepted.

## Feature-parity and removal gate

The migration treated deleting `DistMuon` as its last step, not as the
mechanism for forcing completion. The stock-Muon path first had to preserve
these contracts.

### Optimizer API and state

- The configured object is the original exact `torch.optim.Muon` instance and
  retains its declared `step` function.
- Closure execution, returned loss, optimizer pre/post hooks, profiling,
  `zero_grad`, mutable scalar hyperparameters, and LR schedulers retain normal
  PyTorch semantics.
- Momentum remains keyed by each original parameter, uses its persistent
  DTensor storage layout, and is updated exactly once before redistribution.
- Weight decay and the final direction are applied exactly once to each
  persistent parameter.
- The initial integration retains the current one-parameter-group contract.
  Supporting more groups is a separate extension; physical buckets must not be
  represented as groups.
- Parameter/FQN membership remains frozen after configuration. Mutation is
  rejected immediately if the upstream seam supports that contract, or is
  detected before checkpoint mutation, optimizer state mutation, or the first
  collective.

### Compute and layout behavior

- Compute-ready 2D matrices execute locally.
- Flattened attention storage can use independent per-head logical matrices.
- Native batch-first 3D expert parameters execute as independent matrices.
- `Owned`, `Shard`, and `Replicate` compute distributions retain their current
  validation and execution semantics.
- Dynamic `Owned` participant assignment retains its compute-cost balancing
  and deterministic cross-rank plan validation.
- Oversharded heads, explicit replicated compute, orthogonal preserved shards,
  and Cartesian multi-axis redistribution remain supported.
- Learning-rate adjustment uses logical matrix rows and columns for every
  storage representation.
- Compatible local matrices retain the current batching optimization.

### Buckets and distributed execution

- One selected `BucketConfig` remains one physical bucket; incompatible work
  raises instead of being split silently.
- Repeated-pattern topology alternatives still select exactly one physical
  config for each parameter.
- Packed forward and reverse collective counts, participant order, tensor
  regions, bounded prefetch, local/communication overlap, and buffer reuse
  ordering remain unchanged.
- Pipeline-parallel stages, stage-local parameter sets, and multiple storage
  meshes retain their current behavior.
- Local, `dp_shard`, `efsdp`, `ep`, and Cartesian-axis routes retain their
  current physical collective semantics.
- Missing gradients or changed storage layouts fail before any collective and
  before partial parameter or optimizer-state mutation.
- An exception after communication or mutation has been queued remains fatal
  for that configured optimizer; it never falls back to stock Muon execution.

### Checkpoint behavior

- `state_dict()` contains only standard Muon parameter groups and state. It
  never serializes integrations, process groups, streams, events, plans, or
  temporary buffers.
- `load_state_dict()` validates parameter names and storage state, then rebuilds
  planning-dependent data and buffers.
- A failed load or rebuild never falls through to stock Muon execution on a
  partially configured optimizer. The registered executor remains installed
  and raises a clear error until the optimizer is reconstructed.

### Numerical behavior

- A 2D compute-ready path matches stock `torch.optim.Muon` exactly when both
  use the same upstream kernels and schedule.
- Redistributed 2D paths match the current `DistMuon` update after
  gathering persistent parameters and momentum.
- Batched expert and per-head views remain mathematically equivalent to one
  stock Muon update per logical matrix. Any accepted tolerance must be limited
  to documented BF16 batched-kernel reduction-order differences.
- Deterministic TorchTitan before/after runs use the same parallelism,
  `--debug.seed=42`, and `--debug.deterministic`, and must satisfy the project's
  loss and gradient-norm comparison requirements.

The class and self-hosted kernels were removed only after the supported
PyTorch version provided the execution seam and phased Muon operations and the
stock path passed this parity gate. TorchTitan no longer keeps a private Muon
algorithm implementation as a fallback.

## Testing

Phase 1 retains focused tests for the restored front door:

- The returned object is the input object and its exact type is unchanged.
- No dynamic class is created and no private step-patching API is called.
- `build_flex_shard_muon` and explicit construction produce equivalent plans
  and numerical results.
- Repeated configuration raises `ValueError`.
- An unsupported optimizer raises `TypeError`.
- An unconfigured stock `torch.optim.Muon` keeps its normal step behavior.
- Failed configuration does not install a binding or executor.
- Parameter groups cannot be added after configuration.
- Optimizer pre/post hooks and LR schedulers still work.
- `state_dict()` and `load_state_dict()` rebuild planning-dependent buffers
  without serializing the binding.
- A checkpoint cannot replace the configured parameter FQNs.

The PyTorch prerequisite changes need upstream tests for:

- Default Muon remains unchanged when no executor is registered.
- The executor receives the same optimizer and closure, fully handles the
  step, and returns its loss without the default update also running.
- Profiling, optimizer pre/post hooks, and closure evaluation each occur once.
- Executor registration is exclusive, installation rollback is atomic, and an
  active configured executor cannot silently fall back to the default step.
- Executor registration does not alter `state_dict()` or scheduler behavior.
- Whole-object pickling of a configured optimizer fails clearly instead of
  producing an unconfigured stock optimizer.
- Phased operations match the default Muon update for tall, wide, and square
  matrices across Nesterov, momentum, weight decay, and every learning-rate
  adjustment mode.
- Batch-first Muon matches independent 2D Muon references for each logical
  matrix, subject only to the documented BF16 schedule tolerance.
- Output-buffer and aliasing contracts used by FlexShard are explicitly
  tested.

The stock path retains the former distributed acceptance suite and compares
its updates with plain stock Muon where applicable. That suite covers:

- Exact input identity, exact `torch.optim.Muon` type, and unchanged declared
  `step` function after configuration.
- Compute-ready 2D, flattened per-head, and native batch-first expert inputs.
- Local, `Owned`, `Shard`, and `Replicate` compute distributions.
- Oversharding, orthogonal shards, Cartesian routes, pipeline parallelism,
  local batching, overlap, and deterministic collective counts.
- Persistent parameter and momentum values and placements after each step.
- Checkpoint save, load, binding rebuild, and a second post-load step.
- Missing gradients, invalid gradient or momentum placements, and changed
  topology failing before collectives and without partial state creation or
  parameter updates.
- Closure return values, optimizer hooks, schedulers, and scalar
  hyperparameter changes.
- Rejection of a second parameter group until that behavior is deliberately
  generalized.

The final source audit rejects imports from `torch.optim._muon`, calls to
private `_init_group` or `_patch_step_function`, dynamic class changes, and
remaining self-hosted Muon update kernels.

## Rollout

The following phases describe the logical migration sequence reflected in the
final implementation. Phases 1 and 2 describe the former custom optimizer;
the completed TorchTitan outcome is Phase 6, consuming the PyTorch foundations
from Phase 3. The intermediate phases need not land as separate commits.

### Phase 1: restore the stable front door

- Add and export `flex_optimizer_reshard`.
- Export `DistMuon` for explicit construction.
- Add the private binding and use-once state.
- Keep `DistMuon.step()` as the declared optimizer method.
- Implement `build_dist_muon` through the restored function.
- Add lifecycle, identity, hook, and state-dict tests.

### Phase 2: externalize the Muon integration

- Move preflight validation, compute-layout resolution, planning callbacks,
  state access, local batching, and checkpoint rebuilding from
  `DistMuon` methods into `_MuonReshardIntegration` and
  `_MuonReshardBinding`.
- Keep `DistMuon.step()` temporarily as a thin host for the extracted
  integration so this refactor can be tested without changing the public
  optimizer type at the same time.
- Keep the existing Muon mathematics temporarily so this phase is a structural
  refactor with no computation change.
- Preserve the existing public function signature and distributed tests.

### Phase 3: add the supported PyTorch foundations

- Add supported phased Muon operations and make stock `Muon.step()` use them.
- Add batch-first independent-matrix support, including construction with the
  3D expert parameters required by the current TorchTitan integration.
- Add a supported exclusive step-execution seam that preserves normal
  profiling, hooks, closure, scheduler, type, and state-dict semantics.
- Define topology-mutation handling sufficient to reject membership changes
  before FlexShard state mutation or communication.
- Land upstream tests before TorchTitan consumes the new APIs.

This work may use multiple PyTorch pull requests, but all parts are hard gates
for full feature parity. The private `_init_group` interception prototype is
not an acceptable intermediate dependency.

### Phase 4: support stock Muon side by side

- Add private FlexShard type dispatch for exact `torch.optim.Muon`.
- Replace self-hosted Muon mathematics in `_MuonReshardIntegration` with the
  supported upstream phased operations.
- Register `_MuonReshardIntegration` through the upstream execution seam only
  after binding compilation and validation succeed.
- Reuse the former `DistMuon` acceptance coverage for configured stock
  Muon and compare parameters, momentum, placements, collectives, and
  checkpoint continuation with plain stock Muon where applicable.
- Add deterministic TorchTitan before/after numerical evidence for the Kimi
  configurations that exercise dense, attention, routed-expert, EP, EFSDP,
  and PP paths.

### Phase 5: switch TorchTitan to stock Muon

- Change `build_dist_muon` to construct and return
  `torch.optim.Muon`, then configure that object with
  `flex_optimizer_reshard`.
- Switch the TorchTitan optimizer registry and Kimi integration to the stock
  object. Retain an old configuration-name alias only for an explicit
  deprecation window if compatibility requires it.
- Retain `DistMuon` temporarily as a differential oracle rather than a
  selectable production fallback.
- Update the minimum PyTorch version and fail clearly on older versions; do
  not silently select the self-hosted optimizer.

### Phase 6: remove the self-hosted optimizer

- Remove the `DistMuon` class, its export, its constructor-only tests,
  and the self-hosted `_adjust_muon_learning_rate`, `_prepare_muon_input`,
  `_compute_muon_direction`, `_apply_muon_update`, and
  `_zeropower_via_newtonschulz` implementations.
- Rename the remaining Muon file and private symbols to describe an
  integration rather than an optimizer implementation.
- Rename the construction helper to `build_flex_shard_muon`; the name does not
  imply a distinct optimizer implementation.

### Phase 7: validate broader extensibility

- Implement a second optimizer integration only when there is a concrete use
  case.
- Compare both integrations and promote a public adapter protocol only if its
  boundary is stable and optimizer-neutral.
- Carry the same public API into a standalone FlexShard package without a
  TorchTitan dependency.
