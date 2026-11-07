# Row-sharded embedding lookup

`row_sharded_embedding.py` routes embedding requests to persistent row owners
using PyTorch functional collectives and `F.embedding`. Qwen3.8-Flash-Next uses
it for PLE. There is no TorchRec or FBGEMM dependency, pooling, cache, planner or
optimizer update in backward.

## Routing and shapes

The caller supplies a named one-dimensional `DeviceMesh`. Rows use contiguous
`torch.chunk` boundaries, including empty owners. Each row group has the same
global row count, weight dtype and local feature width. Separate row groups can
own different feature columns. Input IDs can be int32 or int64.

```python
from torchtitan.distributed.row_sharded_embedding import (
    prepare_row_sharded_embedding,
    row_sharded_embedding,
)

# Once per microbatch, in preprocessing:
routing = prepare_row_sharded_embedding(
    hash_ids_TN, num_embeddings=num_global_rows, row_mesh=row_mesh
)

# Forward: FP32 local master rows, BF16 returned vectors.
embedding_TND = row_sharded_embedding(weight_RD, routing, dtype=torch.bfloat16)
```

For an eight-row table on two owners, owner 0 stores rows 0-3 and owner 1 stores
rows 4-7. Requests `[6, 1, 6]` are deduplicated to `[1, 6]`. Owner 0 looks up row
1 and owner 1 looks up its local row 2. The response restores `[E[6], E[1], E[6]]`.
No rank gathers the table.

| Phase | Tensors and communication |
| --- | --- |
| Prepare | Deduplicate local IDs, sort by owner, exchange counts and unique IDs. Owners receive `[M]` local row indices. |
| Forward | Read `[M,D]` vectors from local `[R,D]` weights, cast selected vectors, return with all-to-all, expand duplicates in original order. |
| Backward | Accumulate duplicates on requesters, reverse the vector exchange, sum requester gradients into local master-row gradients. |
| Optimizer | Accumulate microbatches, clip globally, then update local rows with ordinary AdamW. |

Qwen uses `[T,16]` global hash IDs and receives `[T,16,160]` vectors at released
geometry. Hash head offsets are already included in the IDs; the helper knows
nothing about token history or model layers.

Preparation performs one count-dependent CPU synchronization outside forward.
Reuse the metadata during recomputation. Every owner participates even with
empty inputs or empty table shards. All owners must replay response collectives
together; nonreentrant PyTorch checkpointing must disable early stopping.
Qwen's full `torch_remat` policy is tested with the actual model.

The functional collectives currently have a private import path; their public
release is tracked by [PyTorch #113669](https://github.com/pytorch/pytorch/issues/113669).

## Precision and optimizer semantics

Only selected vectors are cast to the requested compute dtype. With FP32 masters
and BF16 compute, duplicate gradients accumulate in BF16 on each requester, then
sum in FP32 on the owner. This matches an ordinary BF16 FSDP embedding's local
accumulation. For example, local contributions `[256,1]` round to 256 before a
second requester's -256 is added; the result is zero, rather than one.

There is no mesh-size division: TorchTitan already normalizes its loss by the
global valid-token count. Weights stay fixed through accumulated microbatches.
Gradients and AdamW moments are dense local shards; unvisited rows still receive
ordinary momentum and weight-decay updates.

Qwen's recipes apply existing regex optimizer configurations to the lookup and
the remaining parameters, with identical hyperparameters. The lookup weight,
its gradient and ordinary AdamW moments are DTensors.

## Parameter lifecycle, clipping and checkpoints

The model allocates local rows from its declarative SPMD sharding, then wraps
them in a DTensor parameter on the backbone's dense storage mesh. DP storage
axes use `Shard(0)`. An ignored-only `fully_shard` child excludes the table from
ancestor FSDP gathers. The lookup uses
`weight.to_local(grad_placements=weight.placements)` so backward returns a
DTensor gradient with the same placements. `to_empty()`, optimizer moments,
dtype conversion and native DCP retain the DTensor's global layout.

The existing clipper combines the table gradient with the other DTensor
gradients. Using the backbone's storage mesh preserves the norm grouping and
reduction order. Only scalar norms communicate; the trainer uses L2 clipping.
Native DCP saves weights and AdamW moments by global coordinates and supports
resume with a changed owner count.

Qwen's HF adapter uses the public DTensor chunk metadata to intersect local
rows with global numbered HF parts. Each local fragment carries the public
`CheckpointableTensor` coordinate fields for the HF storage format. Loading
concatenates only the local fragments and reconstructs the destination DTensor.
Splitting the global DTensor directly would gather its row shards.

Fresh initialization writes through the local view using owner-seeded normal
streams; it depends on owner count. Common-checkpoint comparisons qualify
training numerics. Qwen's SPMD meta allocation requires table rows divisible by
the DP degree, checked once the owner mesh is known. The generic lookup accepts
uneven and empty shards when the supplied weight has the correct local shape.
Qwen rejects CPU offload, deferred FSDP gradient reduction, and a reduction dtype
different from the master dtype.

## Capacity

The released table is `320,001,536 x 160`: 51,200,245,760 parameters.
Ownership removes the 95.37 GiB BF16 full-table gather. It retains dense AdamW
state and updates every local row at each optimizer step.

| Table storage | Global GiB |
| --- | ---: |
| FP32 parameters | 190.74 |
| FP32 gradients | 190.74 |
| Two FP32 AdamW moments | 381.47 |
| Total | 762.94 |

At eight owners the table alone costs 95.37 GiB per owner; at four it costs
190.74 GiB. Other model state and activations are additional. The full 177.4B
meta inventory's default FP32 AdamW state exceeds eight GB300s even with ideal
balance. Released-model capacity and throughput are unqualified.

Current Qwen support uses DP owners. A PP stage without PLE should own no table
and join no lookup collectives; PP ranks on other stages do not add capacity to
the table's owner mesh. Separate feature-column owners are supported by the
helper, while their model integration belongs to the TP/PP stack increments.

## Validation

Four-rank CPU Gloo and two-GB300 NCCL tests compare lookup values, accumulated
gradients, weights and AdamW moments exactly with a dense reference. Cases
include duplicate IDs and BF16 cancellation, uneven shards, empty requesters,
empty owners, all-empty inputs, mixed integer dtypes, non-contiguous/scalar
inputs, and PyTorch/`torch_remat` recomputation. Invalid IDs fail on every owner
before request routing. Active L2 clipping and synchronous/asynchronous
native checkpoints cover resume with fewer owners.

CPU adapter tests exercise real DTensors, HF parts crossing row and feature
owners, empty owners, dtype conversion, and distributed safetensors save,
consolidation and load. Communication instrumentation verifies that HF format
conversion does not gather the table.

Whole-model two-GB300 comparisons use seed 42, strict determinism, a common
checkpoint, two accumulated microbatches and fused AdamW. Native DTensor storage
and the plain owned-row baseline match first-step logits and 190 gradient shards,
ten-step loss/norm, and all 763 final checkpoint tensors bitwise. Full remat and
resume after five updates, including loading a plain-row checkpoint, preserve
the same result. Both sources use the same committed deterministic GDN policy.
The owned lookup also has reduced-model parity with a full-table FSDP embedding.
Changing ownership or collective reduction trees can change floating-point
summation order; these checks qualify the tested layouts.
