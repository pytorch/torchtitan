# Owned Placement Support Plan

PR reference: https://github.com/pytorch/torchtitan/pull/3291

Goal: support `Owned` as a placement-side collective contract in the current
FlexShard experiment. `Shard` should own the all-gather / reduce-scatter pattern,
while `Owned` should own broadcast / reduce-to-owner.

## Current Direction

Move the collective choice out of `bucket_collectives.py` and into the
`Placement` contract:

```text
BucketRuntime
  -> asks placement to prepare/run/finish bucket unshard
  -> asks placement to prepare/reduce grads

Shard placement
  forward unshard: all-gather local shards
  backward reduce: reduce-scatter full-param grads

Owned placement
  forward unshard: broadcast full param from owner rank
  backward reduce: reduce full-param grad to owner rank
```

The bucket runtime still owns stream/event scheduling, pending unshard prefetch
handles, deferred reduce-grad launch ordering, and buffer lifetime. The
placement owns the communication algorithm and tensor packing semantics.

## API Shape

Add placement-side methods:

```python
class Placement:
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard: ...

    def run_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> None: ...

    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult: ...

    def prepare_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad: ...

    def reduce_prepared_grad(
        self,
        prepared: PlacementPreparedReduceGrad,
    ) -> PlacementReduceGradResult: ...
```

Default `Placement` behavior is the `Shard`-style path:

```text
prepare_unshard_bucket:
  concatenate local shards
  allocate per-rank all-gather buffers

run_prepared_unshard:
  all-gather per-rank buffers

finish_prepared_unshard:
  assemble full params

prepare_reduce_grad:
  pack full-param grads into reduce-scatter input

reduce_prepared_grad:
  reduce-scatter
  unpack reduced local shards
```

`Owned` overrides the collective methods:

```text
prepare_unshard_bucket:
  owner rank uses its full local tensor
  non-owner ranks allocate full tensor

run_prepared_unshard:
  broadcast owner tensor to all ranks

finish_prepared_unshard:
  return broadcast full params

prepare_reduce_grad:
  make contiguous full-grad tensors

reduce_prepared_grad:
  reduce each full-grad tensor to owner rank and average on owner
  owner rank returns full local grad
  non-owner ranks return empty local grad
```

## Owned Placement Semantics

`Owned(owner_rank)` means:

- `owner_rank` stores the full parameter locally.
- Other ranks store empty local tensors.
- Forward needs broadcast from `owner_rank`.
- Backward needs reduce to `owner_rank`.
- Optimizer state for that parameter lives only on the owner rank.

Example:

```text
global param shape: (100, 256)
world size: 4

Owned(2)
  rank 0 local shape: (0, 0)
  rank 1 local shape: (0, 0)
  rank 2 local shape: (100, 256)
  rank 3 local shape: (0, 0)
```

## Bucket Constraint

The current first-PR bucket runtime validates one placement tuple per bucket.
For `Owned`, that means all parameters in a bucket must share the same
`owner_rank`.

This preserves the current bucket invariant:

```text
one physical bucket runtime
one placement contract
one communication schedule
```

Open question:

- Should we later allow one `Owned` bucket to contain multiple owner ranks?

That would require either multiple broadcasts/reduces inside one logical bucket
or a coalesced owner-grouped implementation. It weakens the current "one bucket,
one placement" invariant, so the first implementation should keep same-owner
validation and let bucket specs group `Owned` params by owner.

## Example API

Manual same-owner bucket:

```python
from torchtitan.experiments.flex_shard import BucketSpec, flex_shard
from torchtitan.experiments.flex_shard.example.owned import Owned


def owned_rank0(named_params, mesh):
    return {fqn: (Owned(0),) for fqn, _ in named_params}


flex_shard(
    model,
    mesh,
    buckets=[
        BucketSpec(
            ["layers.0.*"],
            shard_placement_fn=owned_rank0,
            reshard_after_forward=False,
        ),
    ],
)
```

Greedy parameter-boundary assignment can be exposed as an example helper, but it
must be paired with bucket specs that keep equal `owner_rank` values together.

## Verification Plan

1. Unit-test `Owned` local layout:
   - owner rank has full shape and numel;
   - non-owner ranks have empty shape and zero numel.

2. Unit-test bucket storage materialization:
   - owner rank copies the full param into bucket storage;
   - non-owner rank reserves zero bytes and exposes an empty local view.

3. Multi-rank correctness test:
   - initialize a small module with rank-shared full params;
   - apply `Owned(0)`;
   - forward output matches the unsharded reference after broadcast;
   - backward grad on owner matches the averaged reference grad;
   - non-owner grad is empty.

4. Runtime scheduling regression:
   - ensure existing `Shard` tests still pass;
   - ensure deferred reduce-grad queueing works through the generic placement
     reduce contract, not reduce-scatter-specific fields.

5. Compile follow-up:
   - decide whether `Owned` compile uses eager c10d broadcast/reduce or
     functional collectives/custom ops;
   - make AC policy match FlexShard-owned broadcast without catching unrelated
     TP/user broadcasts.
