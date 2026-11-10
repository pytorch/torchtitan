# Row-sharded embedding lookup

`row_sharded_embedding.py` routes individual embedding requests to persistent
row owners using PyTorch functional collectives and `F.embedding`.

## Routing and Shapes

The caller supplies a named one-dimensional `DeviceMesh`; each rank owns an
equal contiguous block of rows, so the global row count must be divisible by
the mesh size. Every rank uses the same weight dtype and feature width.
Input IDs can be int32 or int64 and have any shape.

```python
from torchtitan.distributed.row_sharded_embedding import (
    prepare_row_sharded_embedding,
    row_sharded_embedding,
)

# Once per microbatch, before model forward:
routing = prepare_row_sharded_embedding(
    input_ids, num_embeddings=num_global_rows, row_mesh=row_mesh
)

# Forward: local master rows, selected vectors returned in the compute dtype.
embeddings = row_sharded_embedding(weight_RD, routing, dtype=torch.bfloat16)
```

For an eight-row table on two owners, owner 0 stores rows 0-3 and owner 1
stores rows 4-7. Requests `[6,1,6]` become sorted unique IDs `[1,6]`, which
are already grouped by owner. Owner 0 reads local row 1; owner 1 reads local
row 2. Returned vectors expand to `[E6,E1,E6]` in the original request order.

| Phase | Tensors and communication |
| --- | --- |
| Prepare | Deduplicate IDs, exchange counts and unique IDs. Owners receive local row indices `[M]`. |
| Forward | Read `[M,D]` vectors from local weights, cast selected vectors, return with all-to-all, expand duplicates. |
| Backward | Sum duplicate gradients on requesters, reverse the vector exchange, accumulate into owner-local row gradients. |

The result has shape `[*input_ids.shape, D]`, where `D` is the local feature
width. Only requested vectors and their gradients cross the row mesh.

## Precision and Optimizer Semantics

Only selected vectors are cast to the requested compute dtype. With FP32
masters and BF16 compute, duplicate gradients accumulate in BF16 on each
requester, then sum in FP32 on the owner. This matches a mixed-precision
embedding's local accumulation: `[256,1]` rounds to 256 before another
requester's -256 is added, producing zero.

Collectives sum gradients without mesh-size division. The caller normalizes
the loss and keeps weights fixed through accumulated microbatches. The helper
uses ordinary autograd; optimizer updates happen after backward. Gradients
and optimizer moments remain dense local shards.

## Parameter and Recompute Contracts

The caller allocates local rows, owns the storage mesh, and excludes the
parameter from FSDP gathers. The weight is a `Shard(0)` DTensor over the row
mesh. Its gradient retains that placement, so existing clipping, optimizers and
native DCP can consume it.

Preparation performs one count-dependent device-to-host synchronization
before forward. The prepared metadata can be reused during recomputation.
Every owner participates, including ranks with no input IDs.
Invalid IDs fail on every owner before request/response exchanges begin.

All owners must replay response collectives together. Nonreentrant PyTorch
checkpointing must disable early stopping so one rank cannot skip a
collective that another rank replays.

## Validation

Four-rank CPU/Gloo tests compare values, accumulated gradients and optimizer
updates with a dense reference. Cases include duplicate IDs, BF16
cancellation, ranks with no input IDs, integer dtypes and recomputation.

```bash
pytest tests/unit_tests/cpu/distributed/test_row_sharded_embedding.py -q
```
