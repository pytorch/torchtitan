# FlexShard RaggedShard Example Plan

PR context: FlexShard example placements live under
`torchtitan/experiments/flex_shard/example/`.

External references:

- veScale-FSDP paper: https://arxiv.org/abs/2602.22437
- veScale repo: https://github.com/volcengine/veScale
- veScale RaggedShard docs:
  https://github.com/volcengine/veScale/blob/main/docs/texts/raggedshard.md
- veScale implementation reference:
  `vescale/dtensor/placement_types.py` and
  `vescale/dtensor/vescale_utils/ragged_shard_utils.py`

## Goal

Implement `torchtitan/experiments/flex_shard/example/ragged_shard.py` as a
FlexShard example placement that demonstrates veScale-style ragged sharding
within the current FlexShard placement contract.

The first version should prove the core semantics:

- arbitrary uneven distribution across ranks through `local_units`;
- sharding over contiguous flattened prefix dimensions through `dims`;
- bucket all-gather unshard to full parameters;
- bucket gradient reduction back to ragged local shards;
- compatibility with FlexShard bucket storage and unsharded parameter getters.

## Background From veScale

The paper identifies two separate ideas:

1. `RaggedShard` as a flexible sharding format.
   It supports arbitrary sharding granularity over contiguous memory and
   arbitrary distribution across ranks. This is the abstraction that preserves
   block boundaries for structure-aware training, such as block-wise quantized
   optimizers and Muon/Shampoo-style matrix optimizers.

2. Grouped RaggedShard planning plus DBuffer for performance.
   The planner permutes tensors and inserts padding between tensors so grouped
   communication respects non-shardable block boundaries, balances per-rank
   communication sizes, and enables zero-copy access through a distributed
   buffer.

For FlexShard, `RaggedShard` should land first as an example `Placement`.
The grouped planner and DBuffer-like zero-copy layout should be treated as a
follow-up because FlexShard's current placement contract already owns bucket
collective packing, while bucket storage owns local byte storage.

## Current FlexShard Constraints

The current contract in `placement_contract.py` has two API layers:

- per-parameter local storage APIs:
  `compute_local_shape()`, `compute_local_numel()`,
  `extract_local_shard()`, `local_storage_layout()`,
  `make_local_storage_view()`;
- bucket collective APIs:
  `prepare_unshard_bucket()`, `run_prepared_unshard()`,
  `finish_prepared_unshard()`, `prepare_reduce_grad()`,
  `reduce_prepared_grad()`.

The current bucket validation requires all parameters in a bucket to share one
dtype and one placement tuple. Therefore the first `RaggedShard` example should
assume all params in one bucket use the same `RaggedShard(dims, local_units)`.
If different params need different block granularities, users should split them
into different buckets until FlexShard has a composite/mixed placement story.

`example/ragged_shard.py` currently only contains a TODO stub.

## Proposed API

```python
class RaggedShard(Placement):
    """Uneven contiguous-memory sharding over flattened prefix dimensions.

    Args:
        dims: Prefix dimensions to flatten and shard. Only
            `tuple(range(len(dims)))` is supported in the first implementation.
        local_units: Relative non-negative unit counts for each rank. Length
            must equal the mesh size. A unit count may be zero.
    """
```

Examples:

```python
RaggedShard(dims=(0,), local_units=(1, 2, 1, 0))
RaggedShard(dims=(0, 1), local_units=(1, 1, 2, 0))
```

For a tensor with shape `(n, m, k)`:

- `dims=(0,)` shards rows. The local shape is
  `(local_rows, m, k)`.
- `dims=(0, 1)` flattens the `(n, m)` prefix and shards contiguous prefix
  units. The local shape is `(local_prefix_units, k)`.

This follows the veScale model: `dims` describes a flattened prefix, and the
local payload is reconstructed as `(-1, *global_shape[len(dims):])`.

## Shape And Range Semantics

Validation:

- `dims` must be non-empty.
- `dims` must equal `tuple(range(len(dims)))` for the first implementation.
- `len(local_units) == world_size`.
- all `local_units` must be non-negative.
- `sum(local_units) > 0`.
- `prod(global_shape[:len(dims)]) % sum(local_units) == 0`.

Derived values:

```text
prefix_numel = prod(global_shape[:len(dims)])
suffix_shape = global_shape[len(dims):]
suffix_numel = prod(suffix_shape)
unit_prefix = prefix_numel // sum(local_units)
local_prefix_units[rank] = unit_prefix * local_units[rank]
local_numel[rank] = local_prefix_units[rank] * suffix_numel
local_shape[rank] = (local_prefix_units[rank], *suffix_shape)
```

The flattened element range for rank `r` is:

```text
start_prefix = unit_prefix * sum(local_units[:r])
end_prefix = start_prefix + local_prefix_units[r]
start_numel = start_prefix * suffix_numel
end_numel = end_prefix * suffix_numel
```

`extract_local_shard()` should return:

```python
param.contiguous().view(-1)[start_numel:end_numel].view(local_shape)
```

## Bucket Unshard Design

`prepare_unshard_bucket()`:

- concatenate local ragged shards from all params in bucket order;
- compute `per_rank_sizes` and `per_rank_param_offsets`;
- allocate one gathered buffer per source rank with that rank's exact ragged
  bucket size;
- store private state:
  `infos`, `world_size`, `pg`, `debug_fqn`, `per_rank_param_offsets`.

`run_prepared_unshard()`:

- call `dist.all_gather(gathered, send_buf, group=pg)`.

`finish_prepared_unshard()`:

- for each parameter, slice each gathered rank buffer using
  `per_rank_param_offsets`;
- concatenate rank slices in rank order;
- reshape to the original global shape;
- return `PlacementUnshardResult(full_params, buffers)`.

This is analogous to `Shard.prepare_unshard_bucket()` but replaces per-rank
row chunks with contiguous flattened-prefix ragged chunks.

## Gradient Reduction Design

The reduce path should keep one bucket-level collective contract.

Recommended first implementation:

1. Compute each target rank's true ragged bucket size:

   ```text
   per_rank_size[r] = sum(local_numel(param, r) for param in bucket)
   padded_segment_numel = max(per_rank_size)
   ```

2. Allocate `send_buf` with shape `(world_size, padded_segment_numel)`.

3. For each full gradient tensor and each target rank:

   - slice that target rank's ragged shard from the full grad;
   - write it into row `target_rank` at that target rank's bucket offset;
   - leave remaining padding zeros.

4. Run `dist.reduce_scatter_tensor()` with `AVG` initially, matching the
   existing `Shard` example's current policy.

5. Unpack this rank's valid prefix `recv_buf[:per_rank_size[rank]]` back into
   local ragged gradient tensors using the same local shapes and offsets.

Why not `all_reduce` first:

- `all_reduce` on full flattened gradients is easier, but it does not exercise
  the bucket reduce contract that FlexShard needs.
- Padded `reduce_scatter_tensor()` preserves the one bucket reduce collective
  shape while accepting ragged per-rank output sizes.

Follow-up:

- add a grouped RaggedShard planner to reduce padding by permuting params and
  padding between params, following the paper's Grouped RaggedShard section;
- add a scaling/reduction policy instead of hardcoding `AVG`.

## Placement Helper Functions

Add internal helpers in `ragged_shard.py`:

- `_validate_world_size(world_size: int)`;
- `_validate_dims(global_shape: torch.Size)`;
- `_prefix_numel(global_shape: torch.Size)`;
- `_suffix_shape(global_shape: torch.Size)`;
- `_rank_prefix_range(global_shape, rank, world_size)`;
- `_rank_flat_range(global_shape, rank, world_size)`;
- `_local_shape(global_shape, rank, world_size)`.

Keep these private to the example placement. Do not add them to
`placement_contract.py`.

## Placement Function Examples

Add at least one helper placement function:

```python
def per_param_ragged_placements(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
) -> dict[str, tuple[Placement, ...]]:
    return {
        fqn: (RaggedShard(dims=(0,), local_units=(1,) * mesh.size()),)
        for fqn, _ in named_params
    }
```

This is mostly a smoke-test default. A more useful follow-up helper can accept
`local_units` and `dims` as arguments:

```python
def make_ragged_placement_fn(
    *,
    dims: tuple[int, ...],
    local_units: tuple[int, ...],
) -> PlacementFn:
    ...
```

## Tests

Add tests to `test_flex_shard_buckets.py` or a dedicated
`test_flex_shard_ragged_shard.py`.

Unit tests:

- constructor equality/hash/repr;
- reject non-prefix dims, negative units, all-zero units, and length mismatch;
- `compute_local_shape()` and `compute_local_numel()` for:
  - 1D tensor, `local_units=(1, 2, 1, 0)`;
  - 2D tensor with `dims=(0,)`;
  - 3D tensor with `dims=(0, 1)`;
- `extract_local_shard()` returns contiguous flattened-prefix slices;
- local storage copy/view round-trips through `ShardedBucketStorage`.

Distributed tests:

- bucket all-gather reconstructs full params for uneven `local_units`;
- zero-unit rank gets empty local tensors and still participates correctly;
- padded reduce-scatter returns the expected ragged local gradients;
- `flex_shard()` on a small module with `RaggedShard` passes local parameter,
  state_dict, forward, and backward checks.

Regression tests:

- invalid shapes with `prefix_numel % sum(local_units) != 0` raise a clear
  `ValueError`;
- mixed `RaggedShard` placements in one bucket still raise through
  `_validate_bucket_uniform_dtype_and_placement()`.

## Documentation Updates

- Export `RaggedShard` and helper placement function from
  `torchtitan/experiments/flex_shard/example/__init__.py`.
- Mention it in `flex_shard_discussion.md` or a short example doc once the
  implementation lands.
- Keep the example scoped: it is not a DTensor `RaggedShard`, DCP integration,
  or grouped planner yet.

## Open Questions

1. Should the first implementation require
   `prefix_numel % sum(local_units) == 0`, or should it reserve padded local
   storage for invalid shapes?

   Recommendation: require divisibility first. Add grouped planning and
   inter-param padding later.

2. Should `RaggedShard(dims=(0, 1))` expose local tensors as
   `(local_prefix_units, *suffix)` or preserve the original rank with an
   irregular first two dimensions?

   Recommendation: expose `(local_prefix_units, *suffix)`, matching veScale's
   flattened-prefix semantics and keeping local views regular.

3. Should bucket validation allow different `RaggedShard` placements in one
   user bucket?

   Recommendation: no for the first version. Current FlexShard uses one
   placement instance to own one physical bucket collective layout.

4. Should the reduce path use `reduce_scatter_tensor()` with padded segments or
   `all_reduce()` plus local slicing?

   Recommendation: use padded `reduce_scatter_tensor()` so the example preserves
   the bucket reduce collective model.

5. Should we implement `StridedRaggedShard` for composition with TP/EP now?

   Recommendation: no. Keep that as a follow-up after single-placement
   `RaggedShard` is validated in FlexShard.

## Implementation Order

1. Replace the TODO stub in `example/ragged_shard.py` with the `RaggedShard`
   class, private state dataclasses, validation helpers, and local shard
   helpers.
2. Implement local storage APIs and per-param extraction.
3. Implement bucket all-gather unshard.
4. Implement padded reduce-scatter gradient reduction.
5. Add example placement function and export from `example/__init__.py`.
6. Add focused unit tests for shape/range/storage behavior.
7. Add distributed tests for unshard and reduce-grad.
8. Run:

   ```bash
   python -m py_compile \
     torchtitan/experiments/flex_shard/example/ragged_shard.py \
     torchtitan/experiments/flex_shard/tests/test_flex_shard_buckets.py
   python -m pytest -q torchtitan/experiments/flex_shard/tests
   ```

9. If the tests expose meaningful padding overhead, add a small planner note
   and defer performance planning to a follow-up PR.

## Implementation Status

Implemented:

- `RaggedShard` in `example/ragged_shard.py`.
- Prefix-dim validation, `local_units` validation, equality/hash/repr, local
  shape/numel, and local shard extraction.
- Bucket unshard through placement-owned all-gather:
  `prepare_unshard_bucket()`, `run_prepared_unshard()`,
  `finish_prepared_unshard()`.
- Bucket gradient reduction through padded `reduce_scatter_tensor()`:
  `prepare_reduce_grad()` and `reduce_prepared_grad()`.
- `make_ragged_placement_fn()` and `per_param_ragged_placements()`.
- Exports from `torchtitan/experiments/flex_shard/example/__init__.py`.
- Focused tests in `test_flex_shard_ragged_shard.py`.

Confirmed behavior:

- `dims=(0,)` and `dims=(0, 1)` both use flattened-prefix local payloads.
- Zero-unit ranks produce empty local tensor views.
- `ShardedBucketStorage.from_bucket()` materializes ragged local shards into
  bucket byte storage.
- Uneven CPU distributed all-gather reconstructs full parameters.
- Padded CPU distributed reduce-scatter returns the expected ragged local
  gradients and only unpacks the valid segment for this rank.

Verification run:

```bash
/usr/local/bin/black --check \
  torchtitan/experiments/flex_shard/example/ragged_shard.py \
  torchtitan/experiments/flex_shard/example/__init__.py \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_ragged_shard.py

python -m ruff check \
  torchtitan/experiments/flex_shard/example/ragged_shard.py \
  torchtitan/experiments/flex_shard/example/__init__.py \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_ragged_shard.py

python -m py_compile \
  torchtitan/experiments/flex_shard/example/ragged_shard.py \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_ragged_shard.py

python -m pytest -q \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_ragged_shard.py

python -m pytest -q \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_buckets.py \
  -k 'BucketPlacementValidation or BucketStorageLayout'
```

Results:

- `test_flex_shard_ragged_shard.py`: 9 passed.
- Selected existing bucket tests: 18 passed, 10 deselected.
- `ruff`: passed.
- `black --check`: passed.
- `py_compile`: passed.

Not yet covered:

- End-to-end `flex_shard()` runtime with `RaggedShard` on CUDA. The current
  CPU tests exercise the placement contract and storage materialization, while
  FlexShard runtime validation still requires a CUDA mesh.
- Grouped RaggedShard planning / DBuffer-style layout optimization from the
  veScale paper. The current implementation intentionally keeps one placement
  instance owning one bucket collective layout.
