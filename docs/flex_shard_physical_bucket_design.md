# Explicit Physical FlexShard Buckets

## Status

The accompanying FlexShard changes implement this design. At the time of
writing, the submitted #4060, #4122, and #4124 stack does not yet implement this
contract.

### Submitted behavior before this change

- `BucketConfig` contains only `patterns` and `name`; it represents an ordered
  logical scheduling bucket.
- `ComputeLayout.shardings_by_mesh_axis` declares compute sharding by named
  storage-mesh axis. The runtime compares it with each parameter's DTensor
  placements and infers the communication axes.
- #4122 can split one `BucketConfig` into multiple private `_BucketSpec`
  instances. Its physical grouping key contains inferred transport-axis names,
  communication rank order, dtype, device, and a transport-compatibility key.
- Compute-ready items are attached to the first inferred communication group so
  their work can overlap that group's collective. An entirely local config
  produces a `_BucketSpec` without a communication mesh.
- #4124 sorts inferred storage-mesh axes and flattens multiple axes into a 1D
  communication mesh. Axis topology and order are not declared by the user.

Consequently, the submitted public configuration does not expose the number or
topology of physical collectives. A single `BucketConfig` can produce multiple
physical communication buckets.

### Implemented behavior

The new contract makes the public configuration the physical bucket boundary.
One active `BucketConfig` produces one private `_BucketSpec` and is never split
automatically. An alternative that receives no parameters produces no spec.

Physical does not mean that every parameter has identical storage and compute
shardings. It means that all parameters use one exact redistribution group, or
need no redistribution for local work, and one bucket-wide collective
strategy. A local parameter may still be stored on an EP- or EFSDP-sharded
`DeviceMesh`; local means its storage shard is already compute-ready.
Heterogeneous per-parameter routes may share the bucket when an explicit typed
strategy can safely pack them into the same collective. Incompatible routes
require separate public configs.

## Public contract

One selected `BucketConfig` with at least one matched parameter corresponds to
exactly one physical optimizer-work bucket:

```python
@dataclass(frozen=True, slots=True)
class NoRedistribution:
    """Declare a bucket whose storage shards are already compute-ready.

    Parameters in this bucket may still use sharded DTensor storage. The marker
    means only that optimizer compute needs no storage-to-compute redistribution.
    """

    pass


@dataclass(frozen=True, slots=True)
class BucketConfig:
    """One physical optimizer-redistribution bucket.

    ``redistribution_mesh_axis_names`` declares the ordered storage-mesh axes
    whose Cartesian product forms this bucket's exact redistribution group.
    ``NoRedistribution`` declares local-only work. A config is never split
    automatically.
    """

    patterns: tuple[str, ...]
    redistribution_mesh_axis_names: tuple[str, ...] | NoRedistribution
```

There is no public bucket name. Diagnostics are derived on demand from the
declared redistribution choice and bound FQNs; neither a config sequence index
nor a persisted diagnostic label is part of physical bucket identity.

`redistribution_mesh_axis_names` is required. Its value is:

- `NoRedistribution()` for compute-ready local work with no redistribution
  group. This marker does not mean unsharded storage: an expert may remain
  sharded across EP or EFSDP when its local storage shard is already the
  required compute shard.
- `("dp_shard",)` for DP communication.
- `("efsdp",)` for routed-expert communication after EP has already been
  applied.
- A nonempty ordered tuple for a transition that redistributes across multiple
  axes' Cartesian product.
- Validated to reject an empty tuple, empty or non-string axis names, and
  duplicate axes.

Runtime parallelism overrides can change whether the same FQN is local or
communicates over DP or EFSDP. Configs may therefore repeat the same patterns
with different redistribution choices. These are explicit physical-bucket
alternatives, not logical buckets: after storage and compute layouts resolve,
each parameter must match exactly one config by both pattern and resolved axis
set. Two matching alternatives with the same axis set are ambiguous and
invalid. Matching compares the axis set. The declared tuple order controls the
`participant_by_shard_index` mapping used to interpret sharded compute. The
process group uses canonical participant order; the private redistribution
group records the declared shard order separately in that mapping.

A tuple with multiple axes is allowed only when the resolved transition moves
data across their Cartesian product. The bound process group covers exactly
that product, and the declared order determines its
`participant_by_shard_index` mapping. A preserved storage axis is not a
redistribution axis. In particular, an expert parameter that remains `Shard(0)`
across EP while redistributing across EFSDP declares only `("efsdp",)`.

Cartesian communicators are created directly from their canonical participant
ranks with group-local synchronization. Only subgroup members participate, so
one pipeline stage can bind a multi-axis bucket without requiring unrelated
stages to call `DeviceMesh._flatten()`.

The bound private types separate public bucket identity from runtime
communication state:

```python
@dataclass(frozen=True, slots=True)
class _RedistributionGroup:
    process_group: dist.ProcessGroup
    participants: tuple[int, ...]
    participant_by_shard_index: tuple[int, ...]
    local_participant: int


@dataclass(frozen=True, slots=True)
class _BucketSpec:
    fqns: tuple[str, ...]
    redistribution_mesh_axis_names: tuple[str, ...] | NoRedistribution
    redistribution_group: _RedistributionGroup | None
```

A nonlocal spec holds the exact runtime group resolved from its matched
parameters. A local spec has `redistribution_group=None` while retaining its
declared `NoRedistribution()` marker. Parameter storage remains on its original
DTensor mesh in either case, so `NoRedistribution()` does not imply replicated,
unsharded, or mesh-free storage.

## Bucket identity and collective compatibility

Bucket identity does not require a private aggregate layout signature. Such a
signature would conflate bucket identity with per-parameter redistribution
planning and reject valid packed collectives.

Bound bucket identity is represented by the exact FQNs, declared redistribution
choice, and optional redistribution group in `_BucketSpec`. DTensor storage
placements, `ComputeLayout`, tensor shapes, and compute views remain inputs to
each parameter's route planner; they are not copied into the bucket config or
spec and compared wholesale for equality.

Every resolved parameter transition must instead report a typed bucket
compatibility key. The key means that the same bucket-wide implementation can
pack its route with other routes into one outbound collective and one return
collective. It should capture only constraints that affect that promise,
including:

- Collective and packing strategy.
- Any in-place safety constraint, such as a replicated fanout whose writeback
  could overwrite another route's unread input.

The exact process-group participants, `participant_by_shard_index` mapping,
dtype, and device are validated separately while binding `_BucketSpec`. They do
not need to be repeated in the compatibility key.

Different storage placements, compute shardings, views, and tensor shapes may
share a bucket when the explicit composite planner supports them. For example,
`Owned()` and `Shard(0)` routes on the same DP group are not inherently
incompatible. A composite packed A2A planner may give both the same
compatibility key. If it cannot guarantee one safe collective, it must give
them different keys and binding must require separate `BucketConfig` objects.

This compatibility hook is the extensibility mechanism. Supporting a new
mixture means implementing and testing a typed composite collective strategy,
not adding another raw-layout field to bucket identity.

## Binding rules

For each `BucketConfig`:

1. Find the configs whose patterns match each local FQN.
2. Resolve each parameter's storage-to-compute transition and redistribution
   axis set.
3. Require exactly one matching config to declare that resolved axis set.
4. If an alternative config receives no parameters, emit no `_BucketSpec`.
5. Resolve the selected config's declared axes to one exact
   `_RedistributionGroup`, record `participant_by_shard_index`, and require the
   same participants and shard-index mapping for every matched parameter. A
   `NoRedistribution()` config uses no group.
6. Require one dtype, device, and typed bucket compatibility key.
7. Bind the exact FQNs once into one `_BucketSpec`.
8. Never split the config or attach local work merely to overlap a collective.

For a local config, every matched parameter must be compute-ready, but its
storage may remain sharded on axes such as EP or EFSDP. For a nonlocal config,
each nontrivial transition must participate in that bucket's packed exchange.
Local-only parameters belong in a separate config; the scheduler may overlap
their computation without changing bucket membership.

Examples of invalid configurations:

- A parameter requiring DP redistribution whose matching configs declare only
  `NoRedistribution()`.
- A compute-ready expert whose matching configs declare only `("efsdp",)`.
- Two matching configs declaring the same redistribution axis set.
- `("dp_shard",)` containing a route that resolves to the EFSDP process group.
- Different process-group participants, `participant_by_shard_index` mappings,
  dtype, or device.
- Routes with incompatible typed packing strategies.
- A replicated fanout mixed with another route when in-place writeback cannot
  be proven safe.

These are user configuration errors and raise `ValueError`:

```text
BucketConfig(redistribution_mesh_axis_names=('dp_shard',),
fqns=('layers.1.attention.wq_a.weight', 'layers.2.attention.wq_a.weight'))
cannot form one packed collective:
'layers.1.attention.wq_a.weight' resolves to participant-by-shard-index mapping (0, 2),
but 'layers.2.attention.wq_a.weight' resolves to (1, 3);
use separate BucketConfigs
```

## Communication behavior

A nonlocal physical bucket produces one packed round trip per optimizer step:

1. Storage-to-compute `all_to_all_single`.
2. Optimizer computation.
3. Compute-to-storage `all_to_all_single`.

A local bucket produces no collective. Its parameters may still have sharded
DTensor storage; their existing local shards are used directly.

Therefore, collective count is visible from the selected nonlocal configs and
cannot change because the runtime silently subdivided a bucket. An unselected
alternative or selected local config naturally produces no remote
communication.

## Overlap scheduling

Physical identity and overlap remain separate.

The internal plan types are:

```python
_SingleBucketPlan = _LocalBucketPlan | _RedistributionBucketPlan


@dataclass(slots=True)
class _BucketOverlapPlan:
    redistribution_bucket: _RedistributionBucketPlan
    local_buckets: tuple[_LocalBucketPlan, ...]


_BucketExecutionPlan = _SingleBucketPlan | _BucketOverlapPlan
```

Execution is:

1. Enqueue the communication bucket's outbound A2A.
2. Execute distinct local buckets on the compute stream.
3. Wait for the communication result.
4. Compute the communication bucket.
5. Enqueue its return A2A.

These types are not required for bucket identity.
`_compose_bucket_overlap_plans` composes ordered `_SingleBucketPlan` objects
without changing their FQNs or redistribution groups, and
`_flatten_bucket_plan_items` derives the corresponding optimizer `plan_items`.
When no redistribution bucket exists, local work remains in distinct
`_LocalBucketPlan` objects; an overlap plan always has one redistribution
bucket. If explicit user-controlled overlap is later needed, add a typed
`BucketOverlapConfig`; do not use free-form overlap-group strings.

## Kimi configuration

Dense and routed experts are separate semantic bucket families, but runtime
topology can require multiple explicit physical alternatives for each layer
group:

- Dense FQNs declare `NoRedistribution()` and `("dp_shard",)` alternatives.
  Aligned per-head matrices can be local while `Owned()` or oversharded matrices
  communicate over DP in the same run.
- Routed-expert FQNs declare `NoRedistribution()`, `("dp_shard",)`, and
  `("efsdp",)` alternatives. The resolved storage transition selects DP, EFSDP,
  or local; with EP disabled, expert work may be DP or local depending on
  storage. EP remains pre-applied and preserved when present, so it is not part
  of the expert redistribution group.

Only alternatives selected by local runtime layouts emit `_BucketSpec`
objects. Every possible physical bucket remains explicit in configuration, and
no active config is split.

Additional public buckets are required only when parameters resolve to a
different redistribution participant set or `participant_by_shard_index`
mapping, dtype/device, or collective compatibility key. Raw storage or compute
sharding differences alone do not force a split.

## Migration from the submitted stack

### #4060

Submitted behavior:

- `BucketConfig` has only `patterns` and `name`.
- The binder infers a single communication mesh from parameters that require
  redistribution.
- Compute-ready parameters can share a logical config with communicating
  parameters.

Implemented change:

- Add required `redistribution_mesh_axis_names`, using `NoRedistribution()` for
  local-only work.
- Remove the public `name`; derive diagnostics on demand from declared axes and
  matched FQNs without persisting a config index or label.
- Bind one config to at most one `_BucketSpec`.
- Validate the declared axes against every matched parameter's resolved
  transition.
- Add explicit local/DP alternatives for Kimi dense FQNs and
  local/DP/EFSDP alternatives for routed-expert FQNs.
- Allow compatible heterogeneous routes through the packed A2A planner.
- Preserve overlap through separate private bucket-overlap plans.

### #4122

Submitted behavior:

- Automatically groups one logical config by inferred transport axes, rank
  order, dtype, device, and transport compatibility.
- Emits multiple `_BucketSpec` instances when those keys differ.
- Attaches local items to the first inferred communication group.

Implemented change:

- Replace automatic grouping with validation against one declared
  redistribution group, its `participant_by_shard_index` mapping, and one typed
  compatibility key.
- Keep orthogonal-shard planning.
- Preserve compatible heterogeneous route packing.
- Raise `ValueError`, rather than split, when participant sets,
  participant-by-shard-index mappings, or compatibility keys differ.
- Keep local-only work in its own public bucket and overlap it through the
  bucket-overlap plan.

### #4124

Submitted behavior:

- Infers and sorts multiple storage-mesh axes.
- Flattens the inferred axes into a 1D communication mesh.

Implemented change:

- Bind exactly the declared axes. Construct one canonical subgroup communicator
  locally on its members and preserve `participant_by_shard_index` separately.
- Do not infer or split public bucket topology; resolved axes only select an
  explicit alternative.
- Use only `("efsdp",)` for the routed-expert path when EP is already applied
  and preserved.
- Support a multi-axis tuple only for a transition that actually communicates
  across the declared Cartesian product.

## Validation requirements

- Compatible heterogeneous routes pack into one outbound and one return A2A.
- Runtime-resolved axes select exactly one explicit pattern-and-axis
  alternative.
- Duplicate alternatives with the same axis set raise as ambiguous.
- Incompatible compatibility keys raise instead of causing an automatic split.
- Different redistribution participant sets or `participant_by_shard_index`
  mappings raise.
- Wrong redistribution axes and mixed dtype/device raise.
- Local work is not silently attached to a communication bucket.
- Preserved EP is excluded from an EFSDP-only expert bucket.
- A permuted storage `DeviceMesh` produces the correct
  `participant_by_shard_index` mapping.
- One selected public config with matched parameters produces exactly one
  physical spec.
- Separate local and communication buckets still overlap numerically.
- Cartesian axes succeed only when explicitly declared and required.
