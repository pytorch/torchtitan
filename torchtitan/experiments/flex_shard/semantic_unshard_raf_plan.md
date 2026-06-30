# Semantic Unshard RAF Plan

## Goal

Replace the current selective-checkpointing op classification with a semantic
FlexShard unshard boundary.

The intended behavior is:

- FlexShard reshard-after-forward (RAF) recomputes bucket unshards.
- FlexShard RAF does not classify normal module compute ops such as matmul,
  attention, RMSNorm, or all-to-all.
- If the user already applied activation checkpointing (AC), the user's AC
  policy remains responsible for normal forward compute and communication.
- FlexShard only overrides the semantic unshard operation so full parameters can
  be freed after forward and rematerialized in backward.

This removes the need for heuristic lists such as
`_FLEX_SHARD_MUST_SAVE_OPS` and `_FLEX_SHARD_PREFER_SAVE_OPS`.

## Current State

FlexShard currently has two RAF paths:

- RAF-only modules use saved-tensor hooks. Saved full parameters are replaced by
  handles, and unpacking a handle recomputes the corresponding bucket unshard.
- Modules already wrapped by AC are unwrapped and rewrapped with a composed
  checkpoint policy.

The composed AC policy currently matches low-level c10d ops:

- FlexShard unshard collectives are forced to `MUST_RECOMPUTE`.
- Selected non-FlexShard collectives are forced to `MUST_SAVE`.
- Selected compute-heavy ops are `PREFER_SAVE`.
- Everything else defaults to `PREFER_RECOMPUTE`.

This works, but it makes FlexShard own policy for normal module forward logic.
That is the wrong abstraction boundary.

## Target Design

### 1. Introduce a Semantic Unshard Boundary

Define one replay-visible FlexShard operation for bucket materialization, for
example:

```text
flex_shard::unshard_bucket(local_shards, bucket_runtime_key) -> full_params
```

The exact implementation can be a custom op, a higher-order op, or another
trace-visible boundary. The important requirement is that checkpoint policies
can identify "this is FlexShard bucket unshard" without inspecting raw c10d ops.

The semantic op should cover:

- placement-specific unshard preparation;
- placement-specific collective launch;
- view/copy-out of full parameters;
- eager stream/event lifetime through the existing `UnshardHandle`;
- compile tracing without exposing placement internals as policy decisions.

Short-term implementation should keep the existing placement lifecycle:

```text
prepare_unshard_bucket
run_prepared_unshard
finish_prepared_unshard
```

The semantic op is a boundary around that lifecycle, not a replacement for
placement ownership.

### 2. Keep RAF-Only on Saved-Tensor Hooks

For modules without user AC, do not wrap module compute in SAC.

Use the existing saved-tensor hook strategy:

1. Forward unshards bucket params normally.
2. Parameter getters register saved-tensor handles for RAF full params.
3. Autograd saves handles instead of full parameter tensors.
4. Backward unpacking a handle runs only the bucket unshard.
5. Module compute is not replayed just because FlexShard is doing RAF.

This path should not need any `CheckpointPolicy` op list. It should be the
default RAF implementation for non-AC modules.

### 3. Preserve User AC Policy

If the user already wrapped a module with AC, FlexShard should compose only the
minimum additional behavior.

For user selective AC:

- keep the user's original policy for all normal ops;
- override only `flex_shard::unshard_bucket` to `MUST_RECOMPUTE`;
- mark the recompute context so bucket runtime can use RAF recompute state;
- do not special-case matmul, attention, RMSNorm, all-to-all, or other forward
  ops.

For user full AC:

- preserve full AC semantics;
- do not convert full AC into a FlexShard-defined selective policy;
- add only a recompute-context marker so bucket runtime knows that the module is
  being replayed for backward;
- rely on full AC to replay normal module compute because that is what the user
  requested.

This makes user AC own compute-memory tradeoffs and makes FlexShard own only
parameter materialization.

## Proposed Implementation Steps

### Phase 1: Add Characterization Tests

Add tests before changing the mechanism:

- RAF-only module with no AC:
  - full params are not retained after forward;
  - backward recomputes bucket unshards;
  - normal module compute is not replayed solely due to FlexShard RAF.
- User SAC module:
  - original user policy is invoked for non-FlexShard ops;
  - FlexShard unshard is forced to recompute;
  - no FlexShard policy decision is made for `mm`, attention, RMSNorm, or
    all-to-all.
- User full AC module:
  - full AC still recomputes normal module forward;
  - FlexShard recompute state is active during the replay.

These tests should count operation launches or use small instrumented modules
instead of relying only on final gradients.

### Phase 2: Define the Semantic Unshard Op

Prototype the smallest viable boundary.

Open implementation choices:

- `torch.library` custom op:
  - pros: clear operator identity for checkpoint policy;
  - cons: cannot directly carry arbitrary Python runtime state, so it likely
    needs a runtime registry key.
- Higher-order op:
  - pros: can model "run this unshard function" more explicitly;
  - cons: more integration complexity.
- Custom autograd boundary plus policy-visible marker op:
  - pros: least disruption to current `_BucketUnshard`;
  - cons: must verify SAC sees the marker at the right level.

The first milestone is policy visibility, not compile completeness. It is
acceptable to keep the existing eager implementation underneath the semantic
boundary while the compile path is validated separately.

### Phase 3: Route Bucket Runtime Through the Boundary

Update bucket pre-forward unshard so RAF full params originate from the semantic
operation.

Keep these existing responsibilities unchanged:

- `BucketRuntime` still owns bucket scheduling, prefetch, and RAF recompute
  state.
- Placements still own packing, collectives, and view-out.
- `UnshardedParamSlot` still owns forward-scoped parameter exposure.
- Saved-tensor handles still own RAF backward rematerialization.

The changed responsibility is policy identity: checkpointing should see
`flex_shard::unshard_bucket`, not raw c10d collectives, as the thing FlexShard
wants to recompute.

### Phase 4: Simplify AC Composition

Replace the current policy composition with semantic-op-only composition.

For user SAC:

```python
def merged_policy(ctx, func, *args, **kwargs):
    if func is flex_shard_unshard_op:
        return CheckpointPolicy.MUST_RECOMPUTE
    return original_policy(ctx, func, *args, **kwargs)
```

For user full AC, do not synthesize a selective checkpoint policy. Rewrap with a
context function only if needed to install the recompute-state marker:

```text
forward context: no-op
recompute context: _MarkRecomputeContext(no-op, recompute_state, bucket_ids)
```

This preserves full AC behavior while letting FlexShard know it is in backward
recompute.

### Phase 4a: Remove Nested Python Dispatch from RAF Recompute

The current recompute marker is implemented as a `TorchDispatchMode`:

```python
class _MarkRecomputeTorchDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **kwargs)
```

It does not classify operators. It only enters
`_ReshardAfterForwardRecomputeState` and delegates every intercepted op. When
FlexShard composes with user selective AC, each recompute op is therefore seen
by both the user's SAC dispatch mode and FlexShard's marker mode. A rank0
S=4096 trace showed:

```text
FSDP2 PythonDispatchMode:      9377 total,    0 nested
FlexShard PythonDispatchMode: 15459 total, 5490 nested
```

The fix is to replace the marker dispatch mode with a plain context manager:

```python
class _MarkRecomputeContext:
    def __enter__(self):
        self.ctx.__enter__()
        self._token = self.recompute_state.enter_recompute(self.bucket_ids)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.recompute_state.exit_recompute(self._token)
        return self.ctx.__exit__(exc_type, exc_val, exc_tb)
```

The composed AC policy still uses the user's SAC `PythonDispatchMode` as the
top-level operator classifier. FlexShard only contributes context-var state for
`BucketRuntime.in_reshard_after_forward_recompute()`. Acceptance for this phase:

- nested `PythonDispatchMode` events in the FlexShard trace drop to zero;
- top-level `PythonDispatchMode` remains, matching the SAC baseline behavior;
- FlexShard still marks `flex_shard::unshard_bucket` as `MUST_RECOMPUTE`;
- full AC still has RAF recompute state active during replay;
- existing FlexShard tests continue to pass.

Execution result:

```text
S=4096/B=4/EP=4 rank0 active step after replacing the marker:
PythonDispatchMode total:     9969
PythonDispatchMode top-level: 9969
PythonDispatchMode nested:       0
PythonDispatchMode max depth:    0
```

The remaining top-level events are the user's SAC dispatch mode, matching the
FSDP2 shape. FlexShard no longer adds a nested operator-intercepting dispatch
mode during RAF recompute.

### Phase 5: Remove Heuristic Op Lists

After the semantic boundary and tests are in place, remove:

- `_FLEX_SHARD_MUST_SAVE_OPS`
- `_FLEX_SHARD_PREFER_SAVE_OPS`

Then shrink the FlexShard RAF policy to only:

- semantic unshard op -> `MUST_RECOMPUTE`
- everything else -> delegated user AC policy, or no FlexShard policy at all

If `_FLEX_SHARD_COLLECTIVE_OPS` remains temporarily for fallback, mark it as a
compatibility bridge and remove it after all unshard paths emit the semantic op.

## Acceptance Criteria

- No FlexShard policy table for normal module compute ops.
- No FlexShard policy table for non-FlexShard communication ops.
- RAF-only modules do not use SAC over module compute.
- User SAC policy controls non-FlexShard ops.
- User full AC remains full AC.
- FlexShard unshard is still recomputed in backward when RAF is enabled.
- Existing FlexShard runtime, bucket, Muon, GroupedOwned, and compile tests pass.

## Risks

- A custom op may require a runtime registry key, which introduces lifetime and
  graph-capture concerns.
- A semantic op must preserve stream/event ordering that is currently handled by
  `UnshardHandle`.
- Saved-tensor hooks must continue to handle views and dtype casts of full
  parameters.
- User AC composition must preserve existing `CheckpointWrapper` kwargs.
- Compile may need a separate fake/meta implementation or tracing rule for the
  semantic op.

## Suggested Test Additions

- `test_raf_only_saved_tensor_hooks_do_not_replay_compute`
- `test_user_sac_policy_delegates_non_flex_shard_ops`
- `test_user_full_ac_semantics_preserved_with_flex_shard_raf`
- `test_semantic_unshard_policy_recomputes_only_flex_shard_unshard`
- `test_moe_all_to_all_policy_owned_by_user_ac`

## Migration Notes

This should be staged as a mechanical design cleanup, not a behavior rewrite for
placements.

Placements should not need to know whether RAF is implemented through SAC or
saved-tensor hooks. They should continue implementing only the placement
contract. The new semantic boundary belongs between `BucketRuntime` and the
checkpointing policy layer.
