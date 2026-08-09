# Distributed Muon Plan for Oversharded QKV Heads

## Baseline and goal

This plan is based on commit
`a70e37fcf5084d4c2b821f99056feb8bb08ed87f`.

Support FSDP2 `Shard(0)` storage when a shard boundary cuts through a logical
QKV head. Muon must reconstruct complete head matrices before Newton-Schulz
(NS) computation and return the resulting direction to the original FSDP2
storage layout.

The target flow is:

```text
FSDP2 storage [H * R, C], sharded by rows
    -> packed all-to-all on dp_shard
complete local heads [local_H, R, C]
    -> batched Muon NS
    -> reverse packed all-to-all
original FSDP2 row shards
```

Here `H` is the number of logical heads, `R` is the number of rows per head,
and `C` is the matrix column count.

## Original limitation

`BatchedMatrixComputeView` describes the correct logical transformation from
`[H * R, C]` to `[H, R, C]`, and `Shard(0)` means sharding the logical head
axis. The baseline implementation only supported this placement when
every FSDP2 shard already contained complete heads.

The split-head case was rejected because:

- `muon/prep_parameters.py` required every local row count and global row offset
  to be a multiple of `R`.
- Preparation eagerly viewed each local storage shard as a batch of complete
  matrices.
- The original distributed Muon implementation treated `Shard(0)` as a
  local-compute-only placement.
- `flex_optimizer_reshard.py` only supported local computation or redistribution
  of a complete tensor to one `Owned` rank.
- The reshard runtime assumed one input and output span per redistributed
  parameter and that every compute destination received the full tensor.

Removing the alignment validation alone would be incorrect. NS normalization
and matrix products must cover the complete `[R, C]` head matrix.

## Public contract

Keep the existing configuration:

```python
MuonComputeSharding(
    view_before_placement=BatchedMatrixComputeView(num_matrices=num_heads),
    placement=Shard(0),
)
```

Its contract becomes:

- The view is applied logically before compute placement.
- `Shard(0)` partitions complete matrices along the logical head axis.
- Storage alignment is an implementation detail and fast-path decision.
- Head-aligned storage computes locally without communication.
- Split-head storage is redistributed to complete-head compute partitions.
- The destination head ranges follow
  `Shard.local_shard_size_and_offset(num_heads, world_size, rank)` so the
  behavior matches PyTorch `Shard(0)` semantics, including empty partitions.

Buckets describe execution grouping. A bucket binds a communication mesh only
when at least one resolved parameter transition requires redistribution; an
entirely compute-ready bucket remains mesh-free. Muon balances single-rank
`Owned` compute from the resolved parameter sizes, then immediately converts
those choices to endpoint routes. Neither `Shard(0)` nor `Replicate` compute
requires a single-rank assignment. The Muon-local
`balance_loads_across_partitions` helper accepts `(load, stable key)` pairs;
the adapter uses parameter sizes as loads and FQNs only as deterministic keys.

`muon/distributed_muon.py` owns the optimizer runtime and its private Tensor-level
Muon operations. `muon/prep_parameters.py` is the trainer-facing adapter for
parameter views and groups, while `muon/storage_to_compute.py` owns transition
policy and route construction.

## Implementation plan

### 1. Separate storage metadata from the compute view

Update `torchtitan/components/distributed_optimizers/muon/prep_parameters.py`:

- Continue validating the global 2D shape and resolving `H`, `R`, and `C`.
- Continue requiring supported, contiguous storage layouts.
- Stop requiring every FSDP2 shard to begin and end on a head boundary.
- Preserve global compute-view metadata without eagerly viewing a split local
  shard as `[local_H, R, C]`.
- Represent the local storage view as optional. Construct a direct alias only
  for the aligned local fast path.
- Derive validation from globally identical DTensor metadata and inspect every
  mesh coordinate before optimizer construction. Do not introduce rank-local
  validation that could strand peers in a later collective.

### 2. Resolve an explicit storage-to-compute transition

Update `muon/storage_to_compute.py` for placement planning and
`muon/distributed_muon.py` for the optimizer runtime:

- Represent storage-to-compute behavior with explicit transitions: no
  redistribution, whole-tensor single-participant or replicated compute, or
  dimension-0 sharded compute with view-aware repartitioning.
- Fingerprint the mathematical matrix view: FQN, global storage shape, compute
  view, and global compute shape. Do not fingerprint `Owned`, `Replicate`, or
  `Shard` execution distribution.
- Do not include generated routes, participant choices, world size, buckets,
  or storage alignment in the checkpoint fingerprint.
- Keep gradients and momentum in their original FSDP2 DTensor layout.
- Update momentum and prepare the Nesterov direction before redistribution,
  because these operations are elementwise and commute with repartitioning.
- Apply the returned direction and weight decay once to the reconstructed local
  FSDP2 direction.

### 3. Add head-aware reshard planning

Update
`torchtitan/components/distributed_optimizers/flex_optimizer_reshard.py`:

- Generalize the bucket planner beyond local and whole-tensor `Owned` work.
- Keep optimizer-specific compute placement and balancing outside the generic
  bucket and transport layer.
- Build the storage partition from the exact FSDP2 row ranges.
- Build the compute partition by sharding `H`, then map each head range back to
  `[H * R, C]` storage rows.
- Intersect storage ranges with compute head ranges to generate routes.
- Keep separate route endpoint descriptions:
  - a source or destination region in flat 2D storage coordinates;
  - the corresponding region in logical 3D head coordinates.
- Require equal element counts between route endpoints.
- Validate that storage routes cover the original tensor exactly and compute
  routes collectively cover the logical tensor exactly.
- Validate each compute participant against only its assigned head partition,
  rather than requiring every destination to receive the full tensor.
- Build the reverse plan as the exact inverse of the forward plan.
- Continue validating rank-stable plan digests before runtime communication.

The existing packed variable-size `all_to_all_single` transport remains. All
parameters in a bucket should share at most one forward and one
reverse collective.

When every transfer in one direction selects a local replica, execute that
packed direction as a local copy and skip the collective launch.

### 4. Support multiple fragments in the runtime

Update the redistribution runtime to:

- Prepare each parameter once into local-storage-shaped scratch.
- Pack any number of source fragments from that scratch into the bucket
  exchange buffer.
- Assemble only the rank-local complete-head tensor `[local_H, R, C]`.
- Skip NS when `local_H == 0` while still entering both collectives.
- Run batched NS for all heads assigned to the rank.
- Pack the computed fragments for the reverse collective.
- Reassemble all returned fragments into one local-storage-shaped direction.
- Invoke finalization once per parameter.

The aligned fast path must continue to alias local storage and avoid scratch,
packing, and collectives.

### 5. Preserve overlap and memory bounds

Keep the existing schedule:

```text
gather current bucket
return previous bucket
compute current bucket
```

The new path should require two collectives per communicating bucket, not per
parameter or per head. It should not all-gather a full projection on every
rank or gather the full projection to one compute participant.

Long-lived parameter and optimizer state remain FSDP2-sharded. Additional peak
memory should be bounded by exchange buffers, one reusable local storage
scratch tensor, and one rank-local complete-head compute tensor. No rank should
allocate the full QKV projection solely for this transition.

## Mesh scope

For split-head storage, the first implementation should support:

- an exact one-dimensional FSDP2 communication mesh with `(Shard(0),)`
  storage placement;
- the Kimi recipe's `dp_shard` axis;
- contiguous 2D storage;
- independent execution within each data-parallel replica.

The following remain out of scope:

- `_StridedShard` produced by TP and FSDP sharding the same tensor dimension;
- BlockShard;
- composite or multi-axis FSDP communication meshes;
- communication across `dp_replicate`;
- arbitrary fused-QKV layouts that cannot be represented by the existing
  uniform `BatchedMatrixComputeView`;
- distributed NS using reductions inside each NS iteration.

These layouts should continue to fail with explicit validation errors.

## Test plan

### CPU planning and validation tests

Update `tests/unit_tests/test_muon_prep_parameters.py` and
`tests/unit_tests/test_flex_optimizer_reshard.py`:

- Replace the split-head construction rejection with successful route
  construction.
- Test 3 heads with 4 rows per head over 2 ranks. The middle storage boundary
  cuts a head.
- Test 5 heads with 3 rows per head over 4 ranks. Canonical head counts are
  `[2, 2, 1, 0]`.
- Test 2 heads over 4 ranks so ranks with no compute heads still have valid
  send and return routes.
- Validate exact, non-overlapping storage and compute coverage.
- Reject missing, overlapping, out-of-bounds, or non-invertible routes.
- Keep invalid global shape, dimension-1 storage sharding, `_StridedShard`, and
  inconsistent mesh tests.
- Test a bucket containing aligned local work, `Owned` work, and split-head
  `Shard(0)` work.

### Distributed numerical tests

Update `tests/unit_tests/test_distributed_muon.py`:

- Compare split-head computation with plain `torch.optim.Muon` operating on
  one parameter per complete logical head.
- Compare update directions with an explicit BF16 tolerance because batched
  `bmm`/`baddbmm` and independent `mm`/`addmm` calls are not bitwise
  equivalent.
- Compare momentum exactly after multiple steps, including Nesterov, weight
  decay, and nonzero momentum.
- Assert that parameters, gradients, and momentum retain the original FSDP2
  `Shard(0)` placement.
- Assert exactly two `all_to_all_single` calls per communicating bucket.
- Assert no communication for a fully aligned bucket.
- Cover more ranks than heads and verify zero-head ranks participate correctly.
- Save and reload the flat optimizer state, then compare another step.
- Retain rejection when `num_matrices` changes across checkpoint restore.

### FSDP2 and DP+EP integration tests

- Add a small module wrapped by actual FSDP2, rather than relying only on
  manually constructed DTensors, and choose dimensions that split a head.
- Extend the mixed DP+EP test with dense QKV redistributed on `dp_shard` and
  routed experts computing locally on `(efsdp, ep)`.
- Verify that EP does not participate in dense QKV redistribution.

## Validation and performance acceptance

- Run focused CPU and distributed Muon tests.
- Run `pre-commit run --all-files`.
- Run an FSDP2 integration workload with at least 10 training steps.
- Compare the aligned path before and after the change with deterministic
  settings to prove that the existing path is unchanged.
- Compare split-head results against the complete-head reference within the
  expected numerical tolerance.
- Capture a profiler trace and verify:
  - one forward and one reverse packed all-to-all per communicating bucket;
  - no per-head collective launches;
  - no full-QKV all-gather allocation;
  - the existing one-bucket-ahead overlap remains present.

## Suggested implementation sequence

1. Generalize route representation and validation in
   `flex_optimizer_reshard.py`, with CPU tests.
2. Extend packed preparation, assembly, and finalization to multiple spans.
3. Wire the matrix-batch transition into Muon while preserving the local and
   `Owned` paths.
4. Replace split-head rejection tests with distributed numerical parity tests.
5. Add the actual FSDP2 and mixed DP+EP integration coverage.
6. Run lint, numerical validation, and profiling before declaring the change
   review-ready.

## Completion status

Status: complete as of 2026-08-07.

The implementation now supports exact one-dimensional FSDP2 `Shard(0)`
storage whose row boundaries split logical QKV heads. It keeps aligned heads
on the local fast path and redistributes split heads through the existing
packed all-to-all transport. The optimizer state fingerprint describes the
mathematical matrix view and remains independent of execution distribution,
generated routes, buckets, and world size.

Completed coverage includes:

- CPU validation of head-aware routes, exact forward/reverse inversion,
  multi-fragment packing, invalid storage placements, and aligned aliases.
- Distributed parity with one plain `torch.optim.Muon` parameter per logical
  head, including momentum, Nesterov, weight decay, state-dict continuation,
  uneven head counts, and ranks with empty compute partitions.
- An actual FSDP2-wrapped projection whose storage shard cuts head boundaries.
- Mixed DP+EP coverage in which dense QKV redistribution uses only
  `dp_shard`, while routed experts compute locally on `(efsdp, ep)`.

Validation results:

- Split-head update directions match the per-matrix `torch.optim.Muon`
  reference within an explicit BF16 tolerance. Momentum, checkpoint state,
  layouts, and communication remain exact.

- Focused CPU suites: 25 passed and 9 subtests passed.
- Distributed Muon suite: 26 passed and 7 subtests passed.
- Formatting, lint, documentation, spelling, and link hooks passed. The local
  Pyrefly hook reports five baseline `torch.version.hip` stub errors in
  unchanged files.
- The aligned fast path produced bitwise-identical parameter and momentum
  hashes on both ranks before and after this change, using the same two-step
  deterministic input from the baseline commit.
- A four-GPU FSDP2 smoke workload completed 10 steps with two independent
  `[3 * 1024, 1024]` projections. Each 768-row storage shard split a 1024-row
  logical head. The slowest-rank mean for the six profiled steps was 7.717 ms,
  peak allocated memory was 155.1 MiB, and the final loss was 0.18964693.

The [rank-0 profiler trace](https://fburl.com/zh9aygtz) contains four optimizer
all-to-alls for two communicating buckets: gather bucket 0, gather bucket 1,
return bucket 0, and return bucket 1. This confirms two collectives per bucket,
no per-head launches, and the intended one-bucket-ahead schedule. Within the
DistributedMuon step, the largest positive allocation was 3 MiB; the optimizer
did not allocate a 12 MiB full projection for the transition.

The out-of-scope layouts listed above remain explicitly unsupported.
