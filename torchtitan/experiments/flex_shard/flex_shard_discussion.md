# FlexShard API and Runtime Discussion

PR: https://github.com/pytorch/torchtitan/pull/3239

This note is meant to make the current FlexShard shape reviewable. It separates
the current minimal implementation from open API questions that we should settle
before growing the experiment.

## 1. API Spec and Transformer Examples

Current public entry point:

```python
flex_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    buckets: list[BucketSpec],
) -> FlexShardModule
```

Current `BucketSpec`:

```python
BucketSpec(
    patterns: list[str],
    shard_placement_fn: PlacementFn,
    mp_policy: MixedPrecisionPolicy | None = None,
    offload_policy: OffloadPolicy | None = None,
    reshard_after_forward: bool = True,
)
```

Current `PlacementFn`:

```python
PlacementFn = Callable[
    [list[tuple[str, nn.Parameter]], DeviceMesh],
    dict[str, tuple[Placement, ...]],
]
```

Current behavior:

- One `flex_shard(root, mesh, buckets=[...])` call owns the sharded parameter
  set.
- Bucket assignment is root-level and FQN-based.
- Each parameter must match exactly one `BucketSpec`.
- Each bucket gets one `DStorage`, one all-gather, and one reduce-scatter.
- The minimal eager path currently accepts one placement tuple per parameter,
  but validates that all parameters in a bucket have the same placement tuple.
- The minimal eager path requires one original parameter dtype per bucket.
- Nested FlexShard wrapping is not part of the intended API.

### Transformer: Bucket Each Execution Unit

This is the main shape for `reshard_after_forward=True`, because each bucket hook
must replay in activation checkpoint recompute before the module that reads the
full params.

```python
buckets = [
    BucketSpec(
        ["tok_embeddings.*"],
        shard_placement_fn=per_param_placements,
        reshard_after_forward=True,
    ),
    BucketSpec(
        ["pos_embeddings.*"],
        shard_placement_fn=per_param_placements,
        reshard_after_forward=True,
    ),
    *[
        BucketSpec(
            [f"layers.{idx}.*"],
            shard_placement_fn=per_param_placements,
            reshard_after_forward=True,
        )
        for idx in range(args.n_layers)
    ],
    BucketSpec(
        ["norm.*"],
        shard_placement_fn=per_param_placements,
        reshard_after_forward=True,
    ),
    BucketSpec(
        ["output.*"],
        shard_placement_fn=per_param_placements,
        reshard_after_forward=True,
    ),
]

flex_shard(model, mesh, buckets=buckets)
```

### Transformer: Whole Model Bucket

This is useful for simple eager experiments where we do not need per-layer
recompute-safe hooks.

```python
buckets = [
    BucketSpec(
        ["*"],
        shard_placement_fn=per_param_placements,
        reshard_after_forward=False,
    )
]

flex_shard(model, mesh, buckets=buckets)
```

### Open API Questions

**Multiple placements per parameter / bucket**

The `PlacementFn` returns `tuple[Placement, ...]`, matching the idea that a
parameter can be described by a placement stack. The current runtime calls into a
single placement and rejects mixed placement tuples within one bucket.

Questions:

- Should the first landable API intentionally support only one placement per
  parameter?
- If we keep the tuple in the API, should the error message say this is reserved
  for future multi-axis or hybrid sharding?
- Can one bucket ever contain parameters with different placements while still
  preserving the "one bucket, one all-gather, one reduce-scatter" contract?
- If different placements need different reduce-scatter packing, should that
  necessarily imply different buckets?

**Mixed precision as hook points**

Current `MixedPrecisionPolicy` is bucket-level and carries `param_dtype` and
`reduce_dtype`. A clearer runtime model may be to express this as hook points:

- `pre_gather`: prepare local shards before all-gather.
- `post_gather`: transform full params before module compute sees them.
- `pre_grad_reduce`: transform full-param grads before reduce-scatter packing.
- `post_grad_reduce`: transform reduced local shards before accumulation.

Questions:

- Should bucket-level mixed precision be implemented through these hook points
  instead of special cases inside collectives?
- Does `param_dtype` belong in `post_gather`?
- Does `reduce_dtype` belong in `pre_grad_reduce`?
- Do we need a module-boundary input/output dtype policy, or should FlexShard
  only own parameter and gradient dtypes?

**CPU offload as hook points**

`OffloadPolicy` exists as a placeholder and is currently rejected by
`flex_shard()`.

A hook model would make CPU offload explicit:

- `pre_gather`: move local shards from CPU storage to the CUDA communication
  device.
- `post_gather`: release temporary GPU gather inputs when full params are ready.
- `pre_grad_reduce`: ensure full-param grads are on the reduce-scatter device.
- `post_grad_reduce`: optionally move reduced sharded grads or sharded params
  back to CPU-owned storage.

Questions:

- Should CPU offload share the same hook surface as mixed precision?
- Is offload a property of bucket storage, collective inputs, optimizer state, or
  all three?
- What is the minimum offload behavior worth supporting in the experiment?

## 2. Placement Contract

`Placement` is the user-pluggable policy that tells FlexShard how local shards
are shaped, stored, assembled, and reduced. The core runtime owns bucket order,
byte storage, hooks, and collective scheduling; the placement owns tensor layout
semantics.

### Function Roles

`compute_local_shape(global_shape, rank, world_size)`

- Returns the tensor shape this rank owns for one full parameter.
- Used when building `ParamInfo` and when assembling per-rank all-gather results.

`compute_local_numel(global_shape, rank, world_size)`

- Returns the number of local elements this rank owns.
- Defaults to `math.prod(compute_local_shape(...))`.
- Used to size all-gather buffers and rank-specific offsets.

`local_storage_layout(global_shape, dtype, rank, world_size)`

- Returns `LocalStorageLayout(local_shape, local_numel, storage_nbytes)`.
- Lets a placement reserve more storage than the exposed local tensor view if it
  needs padding or a custom layout.

`extract_local_shard(param, rank, world_size)`

- Extracts this rank's local shard from an original full parameter.
- Used at initialization to copy full parameter data into bucket storage.

`copy_param_to_storage(byte_storage, info, param, rank, world_size)`

- Default helper that calls `extract_local_shard()` and copies the shard bytes
  into the bucket byte storage at `info.byte_offset`.
- A placement can override it for a custom packed storage layout.

`make_local_storage_view(byte_storage, info)`

- Returns the local parameter tensor view exposed as `nn.Parameter`.
- The default view uses `info.local_numel`, `info.dtype`, and
  `info.local_shape`.

`assemble_from_shards(per_rank_shards, global_shape, dtype)`

- Reconstructs a full parameter from the per-rank shards produced by all-gather.
- Used in all-gather copy-out.

`pack_reduce_grad(tensors, infos, world_size)`

- Packs full-parameter grads into the flat input expected by reduce-scatter.
- Returns `(send_buf, layout)`, where `layout` is placement-defined metadata
  needed for unpacking.

`unpack_reduced_grad(recv_buf, infos, layout, rank, world_size)`

- Converts this rank's reduce-scatter output back into local sharded gradients.
- Used before accumulating into the original sharded parameters.

### Flow

```text
Initialization
--------------
full params
  -> BucketSpec FQN assignment
  -> shard_placement_fn(bucket_named_params, mesh)
  -> Placement.local_storage_layout(...)
  -> ParamInfo byte offsets
  -> Placement.copy_param_to_storage(...)
  -> Placement.make_local_storage_view(...)
  -> module params become local sharded views

Forward all-gather
------------------
local sharded params from bucket storage
  -> concatenate local shards
  -> all-gather per rank buffers
  -> Placement.compute_local_numel/shape(...) for offsets and views
  -> Placement.assemble_from_shards(...)
  -> full params exposed to parameter accessors
  -> module forward reads full params

Backward reduce-scatter
-----------------------
full-param grads from autograd
  -> Placement.pack_reduce_grad(...)
  -> reduce-scatter
  -> Placement.unpack_reduced_grad(...)
  -> local sharded grads accumulated into original sharded params
```

## 3. Activation Checkpoint Region and Backward Communication Order

`reshard_after_forward=True` is implemented with activation checkpointing rather
than `storage.resize_(0)`. The bucket all-gather outputs full parameter tensors
inside the checkpointed region. The selective activation checkpoint policy marks
FlexShard collective outputs for recompute, so full params can be released after
forward and regenerated in backward recompute.

### Forward Region

For layer buckets:

```text
AC region for layers.0:
  layers.0 pre-forward hook
    consume pending all-gather for layers.0, or launch layers.0 all-gather
    finish all-gather handle and assemble full params
    _BucketAllGather outputs full params
    prefetch next forward bucket all-gather: layers.1
  layers.0.forward()
    parameter accessors read full params
  layers.0 post-forward hook
    clear full-param access state

AC region for layers.1:
  layers.1 pre-forward hook
    consume pending all-gather for layers.1
    finish all-gather handle and assemble full params
  layers.1.forward()
    parameter accessors read full params
  layers.1 post-forward hook
    clear full-param access state
```

Important boundary property:

```text
_BucketAllGather full-param outputs -> all full-param uses by the layer
```

Both sides live inside the activation checkpointed region for the layer. Forward
prefetch can still exist because prefetch produces a pending all-gather handle;
the current layer still finishes the handle and exposes full params inside its
own checkpointed region.

### Backward Recompute and Backward All-Gather Prefetch

Backward recompute visits checkpointed layers in reverse execution order. The
same pre-forward hooks run during recompute, so FlexShard reverses the prefetch
order.

```text
AC recompute region for layers.1:
  layers.1 pre-forward hook
    launch layers.1 all-gather on demand
    finish all-gather handle and expose full params
    prefetch next backward bucket all-gather: layers.0
  layers.1.forward() recompute
    parameter accessors read full params

AC recompute region for layers.0:
  layers.0 pre-forward hook
    consume pending all-gather for layers.0
    finish all-gather handle and expose full params
  layers.0.forward() recompute
    parameter accessors read full params
```

This avoids the naive activation-checkpoint behavior where every backward
all-gather runs only when its layer recompute starts.

### Deferred Reduce-Scatter for Next All-Gather Priority

`_BucketAllGather.backward()` receives full-param grads at the bucket boundary.
It packs those grads and either launches reduce-scatter or defers the launch.

The deferral condition is about backward all-gather priority, not about this
bucket's memory policy:

```text
after bucket i backward produces grads:
  if bucket i - 1 has a backward all-gather to prioritize:
    queue bucket i reduce-scatter
  else:
    launch bucket i reduce-scatter

next pre-forward/recompute hook:
  launch or consume bucket i - 1 all-gather first
  then flush one queued reduce-scatter
```

This keeps the next backward all-gather on the critical path ahead of the
previous bucket's reduce-scatter while still allowing reduce-scatter to overlap
with later backward compute where possible.

## 4. Dynamo Tracing Boundary

The compile goal is not to trace eager side-stream scheduling perfectly. The
goal is to make Dynamo see the same FlexShard bucket semantics that eager uses:

```text
module pre-forward hook
  -> collect local sharded params for this bucket
  -> _BucketAllGather.apply(...)
  -> expose full params through parameter accessors
module forward
  -> normal parameter reads
_BucketAllGather.backward(...)
  -> reduce full-param grads back to local sharded grads
```

Compile mode should trace into:

- bucket hook dispatch;
- local shard collection;
- `_BucketAllGather` forward and backward boundary;
- parameter accessor logic that returns hook-provided full params;
- synchronous all-gather / reduce-scatter operations needed for the bucket.

Compile mode should not trace eager-only runtime machinery:

- side-stream event handoff;
- forward or backward prefetch queues;
- autograd engine callbacks used to wait for async reduce-scatter;
- profiler-only `record_function` / `record_comm` ranges;
- stale pending work cleanup.

Current direction:

- eager uses async handles, side streams, events, and prefetch;
- compile uses synchronous current-stream collective handles;
- both paths share the same bucket-level public runtime contract;
- `torch.compiler.is_compiling()` should only gate runtime scheduling details,
  not switch to a different parameter bucketing algorithm.

Open tracing questions:

- Should FlexShard collectives be wrapped in a custom op namespace so the
  activation-checkpoint policy can distinguish FlexShard all-gathers from TP or
  other user collectives?
- Should Dynamo trace through placement pack/unpack Python, or should placement
  implementations eventually provide trace-friendly/custom-op packers?
- What is the exact graph boundary we want for `_BucketAllGather` if we later
  replace c10d collectives with functional collectives?

## 5. PR #3239 Review Comment Triage

Source: `gh pr view https://github.com/pytorch/torchtitan/pull/3239` plus
GraphQL `reviewThreads(first:100)`, checked on 2026-05-14. GitHub reports 70
review threads. The priority below is for landing the first FlexShard PR, not
for long-term feature importance.

Priority definitions:

- `P0`: correctness/API blocker or likely reviewer blocker before landing.
- `P1`: should address or give a strong reply before landing.
- `P2`: cleanup, naming, documentation, or follow-up design question.
- `P3`: already addressed, outdated, informational, or low-risk nit.

### Suggested Fix Order

1. `P0` activation-checkpoint policy / FlexShard collective provenance:
   non-FlexShard collectives must not be forced to recompute.
2. `P0` placement and communication extensibility: custom collectives,
   bucket-level pack/unpack ownership, fused/direct collective paths.
3. `P0` hook order and recompute safety: make the current wrapping/hook order
   defensible and explain unsupported bucket shapes clearly.
4. `P0` backward communication priority: reduce-scatter deferral should depend
   on the next backward all-gather, not this bucket's
   `reshard_after_forward`. Code has been updated; reply/resolve the thread.
5. `P1` placement API shape: naming, inheritance vs protocol, equality/hash,
   padding semantics, multiple placements per bucket, and example placements.
6. `P1` parameter access / storage API clarity: public vs internal structs,
   state machine docs, `DStorage` naming, and initialization copy semantics.
7. `P2/P3` smaller nits and already-addressed threads.

### P0 - Highest Priority

1. [reshard_after_forward.py:64 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3223616696) - AC policy can interfere with non-FlexShard collectives such as TP all-gather; current op-set matching is too broad. Status: replied, still needs design/code plan. Next: propose FlexShard-specific collective provenance or custom op namespace and clarify no override of existing AC policy.
2. [reshard_after_forward.py:78 @wconstab](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3236523500) - Placement-specific collectives need an extensible mechanism. Status: replied, unresolved. Next: tie to the same custom collective/provenance design as the previous item.
3. [bucket_storage.py:47 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231125350) - Need open registration for custom communication kernels, stochastic rounding, even/uneven algorithms, and placement/mixed-precision composition. Status: open. Next: answer in the API discussion and likely add TODO/design hook rather than hard-code current collectives as final.
4. [placement_contract.py:120 @aditvenk](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3222235967) - Is it intentional that `Placement.pack_reduce_grad()` / `unpack_reduced_grad()` operate on the entire bucket rather than per parameter? Status: open. Next: decide if placement owns bucket packing or if bucket runtime owns multi-param packing.
5. [placement_contract.py:63 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231168651) - Current contract may force all-gather then copy/assemble instead of expressing a direct/fused collective. Status: open. Next: document current default and leave extension point for optimized collective implementations.
6. [flex_shard.py:251 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3223753643) - Hook install order vs `reshard_after_forward` wrapping looks logically reversed. Status: replied, still current. Next: give precise explanation that current implementation wraps first to validate checkpoint wrappers, then installs hooks on the unwrapped target; point to tests.
7. [bucket_runtime.py:593 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3237771956) - Reduce-scatter queueing should depend on next backward all-gather, not this bucket's `reshard_after_forward`. Status: addressed in code and outdated after submit. Next: reply with the new scheduler helper and ordering.
8. [bucket_runtime.py:477 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231486766) - Reduce-scatter could delay the next critical-path all-gather. Status: addressed/replied and outdated. Next: resolve after pointing to deferred reduce-scatter ordering.
9. [flex_shard.py:245 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3222109746) - Should `reshard_after_forward=False` use `MUST_SAVE` to unify treatment? Status: open. Next: explain why correctness does not require AC for `False`, while unified `_BucketAllGather` now covers autograd reduce-scatter.
10. [flex_shard.py:251 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231186693) - Make automatic all-gather/reduce-scatter hooks optional so users can manually schedule collectives. Status: replied lightly, unresolved. Next: mark as follow-up API mode or explain why first PR keeps hooks mandatory.

### P1 - Address or Strongly Reply Before Landing

1. [flex_shard.py:180 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3215649148) - When can bucket hooks fail to run in both original forward and AC recompute? Status: open. Next: reply with the layer-bucket vs root/rest-bucket example.
2. [utils.py:196 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3215667322) - Require params to be either all meta or all non-meta. Status: open. Next: likely add validation.
3. [placement_contract.py:78 @aditvenk](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3222235946) - `local_storage_layout()` default assumes no padding; is that implementation-dependent? Status: open. Next: clarify default and override behavior.
4. [placement_contract.py:35 @aditvenk](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3222235955) - Is `compute_local_numel()` needed if it can be computed from local shape? Status: current, effectively addressed by default implementation. Next: reply.
5. [placement_contract.py:28 @aditvenk](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3222235960) - Need `__eq__` / `__hash__` contract for placement subclasses. Status: open. Next: decide ABC/protocol/dataclass direction.
6. [bucket_storage.py:222 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3223026032) - Is only one placement enough when all placements in the same bucket should be identical? Status: open. Next: answer with current single-placement bucket constraint and future multi-placement question.
7. [utils.py:284 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3223419359) - What does equality mean for `FlatShard`? Status: open. Next: cover in placement equality contract.
8. [placement_contract.py:28 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231058014) - `Placement` name conflicts with DTensor's `Placement`. Status: open. Next: consider rename or justify with namespace.
9. [placement_contract.py:45 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231107051) - Prefer lightweight `Protocol` / composition over inheritance. Status: open. Next: decide before API hardens.
10. [example/ragged_shard.py:7 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231096852) - Implement exact veScale semantics to validate abstraction. Status: open. Next: likely follow-up placement PR, but should acknowledge.
11. [example/flat_shard.py:7 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231098587) - Land alternate placements sooner to flush out contract issues; parameter order control question was discussed. Status: replied, unresolved. Next: track as follow-up but keep contract discussion in this doc.
12. [bucket_storage.py:64 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231176295) - Remove unimplemented `OffloadPolicy` from first PR. Status: open. Next: either remove or strongly justify as reserved API.
13. [bucket_collectives.py:393 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231174054) - `ReduceOp.AVG` is dangerous; Torchtitan should expect better scaling semantics. Status: TODO added and replied. Next: decide whether TODO is enough for first PR.
14. [placement_contract.py:88 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231086198) - Is `copy_param_to_storage()` mandatory? What about checkpoints already in local shard form? Status: replied with init-time behavior, unresolved. Next: document local-shard checkpoint path as future work.
15. [bucket_runtime.py:444 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3223729405) - Root-level bucket with checkpointed children is hard to understand. Status: replied and code comment added, still current. Next: resolve if reviewer is satisfied.
16. [param_access.py:249 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3222086914) - Explain parameter accessor mechanism vs SimpleFSDP tricks. Status: open. Next: add reply or docstring.
17. [param_access.py:245 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3223591098) - Add docstring for parameter access state machine. Status: open. Next: add concise state transition docstring.
18. [bucket_runtime.py:699 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3223638234) - `_pre_gathered` name is confusing because it represents after-all-gather state. Status: open. Next: rename to `full_param`/`gathered_param` or explain.
19. [bucket_runtime.py:660 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3222069975) - Needs more detailed docstring. Status: open. Next: add docstring to hook installation/runtime helper.
20. [bucket_storage.py:26 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231188664) - Public API vs internal bookkeeping unclear. Status: replied and `__init__.py` exports narrowed. Next: resolve or add explicit docs.
21. [bucket_runtime.py:5 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231179880) - Is runtime modeled after FSDP2? Status: replied, but answer drifted to async reduce-scatter. Next: give a direct architecture comparison.

### P2 - Cleanup, Naming, or Follow-Up Design

1. [utils.py:156 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3215663763) - Add TODO or `NotImplementedError` for nD mesh support. Status: open. Next: easy code/comment fix.
2. [utils.py:183 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3215666073) - Why optional dependency? Status: open. Next: reply or simplify.
3. [flex_shard.py:136 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3215683530) - Should bucket placement validation live with assignment validation? Status: open. Next: explain split between FQN assignment and placement/dtype validation.
4. [flex_shard.py:56 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3215687044) - Does `param_placements` already cover `named_params`? Status: open. Next: likely answered by `_validate_placements()`.
5. [flex_shard.py:229 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3221816322) - Why not pass `inputs` directly? Status: open. Next: cleanup call signature if still relevant.
6. [bucket_storage.py:411 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3221853207) - Generate `fqn_to_bucket_spec` in prepare stage? Status: open. Next: consider moving derived map.
7. [bucket_storage.py:174 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3221883285) - Rename `DStorage` to something like `ShardedBucketStorage`. Status: open. Next: decide naming.
8. [bucket_storage.py:396 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3221892346) - Group return values into a larger dataclass. Status: open. Next: consider internal dataclass.
9. [bucket_storage.py:316 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3221938686) - Function name/docstring should say it packs local sharded param into byte storage. Status: open. Next: rename or docstring.
10. [bucket_storage.py:394 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3222017909) - Use clearer name `fqn_to_bucket_spec`. Status: open, likely already addressed in newer code. Next: reply/resolve.
11. [param_access.py:154 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3222066670) - Reviewer note: did not understand this, skipped for now. Status: open. Next: low-priority doc if area changes.
12. [example/shard.py:52 @aditvenk](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3222235937) - DTensor uneven chunk semantics. Status: replied and code updated. Next: resolve if reviewer accepts.
13. [placement_contract.py:127 @aditvenk](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3222235969) - Naming nit: `reduced` vs `reduce` in pack/unpack API. Status: open. Next: rename before API hardens.
14. [utils.py:179 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3223428253) - What makes validation eager-only? Status: open. Next: reply or rename message.
15. [param_access.py:49 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3223496682) - Does a parameter always need mesh? Status: open. Next: explain sharded metadata requirements.
16. [param_access.py:230 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3223503438) - What makes bucket 0 special? Status: open. Next: rename or clarify primary storage alias.
17. [placement_contract.py:5 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231061230) - Make placement contract docs more elaborate. Status: open. Next: discussion doc covers it, but file docstrings may still need updates.
18. [bucket_storage.py:24 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231090681) - Circular import/type hierarchy is annoying. Status: open. Next: refactor module boundaries if easy.
19. [placement_contract.py:135 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231127788) - Explicitly state whether dtype conversion happens in unpack. Status: open. Next: docstring fix.
20. [placement_contract.py:53 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231133730) - Does output tensor need to be contiguous? Status: open. Next: document/validate placement contract.
21. [bucket_storage.py:93 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231182167) - Normalize this internal structure eventually. Status: open. Next: follow-up cleanup.
22. [bucket_storage.py:103 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231191103) - Use type aliases to clarify list positions/buckets. Status: open. Next: easy readability fix.
23. [flex_shard.py:146 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231314303) - Bucket-centric API and internal dataclasses. Status: outdated, likely addressed by moving placement function into `BucketSpec`; still mention in API discussion.

### P3 - Already Addressed, Outdated, or Informational

1. [flex_shard.py:157 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3215647518) - Nested wrapping should be banned in favor of root-level FQN buckets. Status: replied and outdated; code cleaned up.
2. [flex_shard.py:104 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3215673189) - Mixed dense/sparse mesh TODO. Status: outdated; API moved.
3. [utils.py:293 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3215682592) - "oh, makes sense". Status: informational.
4. [flex_shard.py:81 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3215684643) - `FlatShard` implementation question. Status: outdated; broader placement discussion remains P1/P2.
5. [bucket_runtime.py:566 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3222033555) - Non-CUDA bucket reason. Status: replied and outdated; runtime now expects CUDA bucket storage.
6. [bucket_runtime.py:552 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3222038539) - Whether reshard-after-forward can avoid autograd path. Status: replied and outdated; both paths now use `_BucketAllGather`.
7. [bucket_runtime.py:554 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3222040358) - When no bucket params exist. Status: replied and outdated; now asserts.
8. [example/shard.py:64 @aditvenk](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3222235943) - Replace manual numel with `math.prod`. Status: outdated; default moved to base placement.
9. [placement_contract.py:38 @aditvenk](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3222235950) - Use `abc.ABC` / `abstractmethod`. Status: outdated relative to protocol/inheritance discussion; keep under API follow-up if needed.
10. [example/shard.py:89 @aditvenk](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3222235973) - Empty shard assembly case. Status: replied and outdated; assertion added.
11. [bucket_runtime.py:39 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3223654889) - Clarify tuple fields. Status: replied and outdated; changed to dataclass.
12. [bucket_runtime.py:87 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3223667662) - Whether every bucket creates streams. Status: replied and outdated; split `get()` / `create()`.
13. [bucket_runtime.py:270 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3223741297) - Function hard to understand. Status: replied and outdated; renamed to `resolve_bucket_forward_hook_module`.
14. [example/shard.py:65 @ezyang](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231102384) - `compute_local_numel` should always derive from local shape. Status: replied and outdated; base default added.
15. [bucket_runtime.py:480 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231363077) - Noted conditional prefetch order for forward vs backward. Status: informational.
16. [bucket_runtime.py:524 @tianyu-l](https://github.com/pytorch/torchtitan/pull/3239#discussion_r3231489014) - Difference between `register_multi_grad_hook` and `_BucketAllGather.backward`. Status: replied and outdated; hook removed.
