# Restoring the `flex_optimizer_reshard` API

## Status

The accompanying change implements Phase 1 of this design: restoring
`flex_optimizer_reshard` as a public configuration entry point without
restoring the former dynamic optimizer wrapper. Phases 2 and 3 remain future
work.

Before this change, the public API exposed only `build_dist_muon`.
`DistMuon` already owned its normal `step()` method, and the FlexShard
schedule and runtime were private implementation details.

## Motivation

`build_dist_muon` is convenient for TorchTitan, but it combines two
operations:

1. Construct a particular optimizer.
2. Bind that optimizer to FlexShard compute layouts and physical buckets.

Restoring `flex_optimizer_reshard` separates those operations and provides a
stable entry point for explicit optimizer construction:

```python
optimizer = flex_optimizer_reshard(
    DistMuon(named_param_groups, **optimizer_kwargs),
    compute_sharding_by_fqn={
        fqn: ComputeLayout(
            shardings_by_mesh_axis={
                "dp_shard": BlockShard(dim=0, block_size=rows_per_matrix),
            }
        )
    },
    bucket_configs=bucket_configs,
)
```

The function configures the optimizer to execute its work with temporary
compute shardings that may differ from persistent DTensor storage shardings.
It does not redistribute parameter data until `optimizer.step()`.
Configuration may create process groups and run plan-validation collectives,
so all participating ranks must call it in a consistent order.

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
- Keep `build_dist_muon` as the convenient TorchTitan factory.
- Preserve one public `BucketConfig` as one physical bucket.
- Keep scheduling, communication, and overlap implementation private.
- Define a clean optimizer-integration boundary without claiming arbitrary
  `torch.optim.Optimizer` support.
- Keep FlexShard self-contained and PyTorch-only so the folder remains
  extractable into a standalone package.

## Non-goals

- Dynamically replacing or wrapping an optimizer's `step()` method.
- Making an arbitrary existing optimizer reshardable without algorithm-specific
  integration.
- Publishing a third-party optimizer adapter protocol before a second optimizer
  validates its shape.
- Serializing FlexShard plans, streams, buffers, or process groups in the
  optimizer state dict.
- Inferring or silently splitting physical communication buckets.

## Public API

The initial supported call is explicit about Muon:

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

The implementation is generically typed so it preserves the concrete optimizer
type, then dispatches through a private optimizer hook. Future supported
optimizers may add type overloads for their compute-sharding configuration.
The call shape does not need to change.

The package root should export:

- `flex_optimizer_reshard`
- `DistMuon`
- `build_dist_muon`
- The existing compute-layout and physical-bucket configuration types

`build_dist_muon` becomes a convenience composition:

```python
def build_dist_muon(...):
    normalized_param_groups = _normalize_param_groups(...)
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

## Object identity and step ownership

`flex_optimizer_reshard` must satisfy:

```python
original_type = type(optimizer)
configured_optimizer = flex_optimizer_reshard(optimizer, ...)

assert configured_optimizer is optimizer
assert type(configured_optimizer) is original_type
```

The concrete optimizer continues to declare `step()` normally. For Muon, the
shape is conceptually:

```python
class DistMuon(Optimizer):
    _optimizer_reshard_binding: _DistMuonReshardBinding | None

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

## Internal binding

Successful configuration installs one private, per-instance binding:

```python
@dataclass(slots=True)
class _DistMuonReshardBinding(
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
- Deriving optimizer-local matrix views from compute sharding.
- Resolving storage-to-compute transitions.
- Initializing and validating optimizer state.
- Estimating compute cost and assigning dynamic owners.
- Planning and executing local batches.
- Preparing optimizer inputs, computing updates, and applying updates.

For the first restoration, this boundary should remain a private protocol or
private optimizer hook. `flex_optimizer_reshard` asks the optimizer's supported
integration to compile a binding and then installs it. Core must not import
`DistMuon` merely to dispatch the public API.

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

- `TypeError` when the optimizer has no supported FlexShard integration.
- `ValueError` when `flex_optimizer_reshard` is applied more than once.
- `ValueError` for missing, duplicate, or misaligned FQNs.
- `ValueError` when checkpoint parameter names differ from configured FQNs.
- `ValueError` for unsupported compute configurations or storage transitions.
- `ValueError` when bucket declarations do not match resolved redistribution
  axes or cannot form one packed collective.
- `RuntimeError` when `step()` is called before configuration.
- `RuntimeError` when parameter-group membership changes after configuration.

An unsupported optimizer error should explain that the optimizer must provide
a FlexShard integration; it should not suggest that every PyTorch optimizer is
accepted.

## Testing

The initial restoration needs focused tests for:

- The returned object is the input object and its exact type is unchanged.
- No dynamic class is created and no private step-patching API is called.
- `build_dist_muon` and explicit construction produce equivalent plans
  and numerical results.
- Repeated configuration raises `ValueError`.
- An unsupported optimizer raises `TypeError`.
- Calling an unconfigured `DistMuon.step()` raises a clear error.
- Failed configuration does not leave a binding installed.
- Parameter groups cannot be added after configuration.
- Optimizer pre/post hooks and LR schedulers still work.
- `state_dict()` and `load_state_dict()` rebuild planning-dependent buffers
  without serializing the binding.
- A checkpoint cannot replace the configured parameter FQNs.
- Existing physical-bucket, collective-count, overlap, pipeline-parallel, and
  Cartesian-redistribution tests remain unchanged.
- DistMuon loss and updates match the current builder path.

## Rollout

### Phase 1: restore the stable front door

- Add and export `flex_optimizer_reshard`.
- Export `DistMuon` for explicit construction.
- Add the private binding and use-once state.
- Keep `DistMuon.step()` as the declared optimizer method.
- Implement `build_dist_muon` through the restored function.
- Add lifecycle, identity, hook, and state-dict tests.

### Phase 2: consolidate proven optimizer-neutral code

- Move only genuinely shared binding and lifecycle operations into core.
- Keep Muon matrix views, transition resolution, state, batching, and update kernels
  in the Muon integration.
- Preserve the current public function signature.

### Phase 3: validate extensibility

- Implement a second optimizer integration.
- Compare both integrations and promote a public adapter protocol only if its
  boundary is stable and optimizer-neutral.
- Carry the same public API into a standalone FlexShard package without a
  TorchTitan dependency.
