# Row-sharded embedding lookup

`row_sharded_embedding.py` routes individual embedding requests to persistent
row owners using PyTorch functional collectives and `F.embedding`.

## Routing and Shapes

The caller supplies a named one-dimensional `DeviceMesh`; each rank owns an
equal contiguous block of rows, so the global row count must be divisible by
the mesh size. Every rank uses the same weight dtype and feature width.
Input IDs can be int32 or int64 and have any shape.

```python
from torchtitan.models.qwen3_8_flash_next.row_sharded_embedding import (
    prepare_row_sharded_embedding,
    row_sharded_embedding,
)

# Once per microbatch, before model forward. max_ids_per_owner bounds the
# distinct IDs any requester sends to any one owner, and must be equal on
# every rank of row_mesh.
routing = prepare_row_sharded_embedding(
    input_ids,
    num_embeddings=num_global_rows,
    row_mesh=row_mesh,
    max_ids_per_owner=max_ids_per_owner,
)

# Forward: local master rows, selected vectors returned in the compute dtype.
embeddings = row_sharded_embedding(weight_RD, routing, dtype=torch.bfloat16)
```

Every requester sends each owner a fixed slot of `C = max_ids_per_owner`
distinct IDs, so shapes and all-to-all splits do not depend on the IDs. This
keeps the lookup capturable in a CUDA graph and keeps counts off the host.

For an eight-row table on two owners with `C=2`, owner 0 stores rows 0-3 and
owner 1 stores rows 4-7. Requests `[6,1,6]` become sorted distinct IDs `[1,6]`,
which are already grouped by owner, and fill slots `[[1,-],[6,-]]`. Owner 0
reads local row 1; owner 1 reads local row 2; unused slots read local row 0,
but the requester never selects them. Returned vectors expand to `[E6,E1,E6]`
in the original request order.

| Phase | Tensors and communication |
| --- | --- |
| Prepare | Sort and deduplicate IDs, exchange `[P,S]` request slots (`S = C + 1`: `C` IDs and the requester's validity flag). Owners receive local row indices `[P*C]`. |
| Forward | Read `[P*C,D]` vectors from local weights, cast them, return with an equal-split all-to-all, expand duplicates. |
| Backward | Sum duplicate gradients on requesters, reverse the vector exchange, accumulate into owner-local row gradients. Unused slots carry zero gradient. |

The result has shape `[*input_ids.shape, D]`, where `D` is the local feature
width. Each exchange moves `P*C` IDs or vectors per rank, so a tight `C` keeps
traffic close to the requested vectors. The equal splits are sized from each
rank's local `C`, so every rank of `row_mesh` must pass the same value; a
mismatch mis-sizes the exchange instead of raising. Derive `C` from
rank-uniform quantities (the table layout and the configured tokens per
microbatch), never from the local IDs. For example, Flash-Next's hash heads
each own a contiguous row range, so an owner receives at most one ID per token
from each head meeting its rows.

## Precision and Optimizer Semantics

Only selected vectors are cast to the requested compute dtype. With FP32
masters and BF16 compute, duplicate gradients accumulate in BF16 on each
requester, then sum in FP32 on the owner. This matches a mixed-precision
embedding's local accumulation: `[256,1]` rounds to 256 before another
requester's -256 is added, producing zero.

Collectives sum gradients without mesh-size division. The caller normalizes
the loss and keeps weights fixed through accumulated microbatches. The lookup
is an `autograd.Function` whose backward runs only the reverse all-to-all and
builds the DTensor gradient locally; optimizer updates happen after backward.
Gradients and optimizer moments remain dense local shards.

## Parameter and Recompute Contracts

The caller allocates local rows, builds the weight's DTensor, and excludes the
parameter from FSDP gathers. The weight is a DTensor sharded `Shard(0)` on the
row axis; its mesh may carry further axes (Flash-Next uses the dense SPMD mesh).
Its gradient retains the weight's placements, so existing clipping, optimizers
and native DCP can consume it.

Preparation performs no device-to-host synchronization. The prepared
metadata can be reused during recomputation, and is a registered pytree so
CUDA graph replay refreshes its tensors. Every owner participates, including
ranks with no input IDs. Each requester's slots carry a validity flag; IDs
outside the table or more than `C` distinct IDs for an owner fail an
asynchronous assertion on every owner after the request exchange.

All owners must replay response collectives together. Nonreentrant PyTorch
checkpointing must disable early stopping so one rank cannot skip a
collective that another rank replays.

## Validation

Four-rank CPU/Gloo tests compare values, accumulated gradients and optimizer
updates with a dense reference. Cases include duplicate IDs, BF16
cancellation, ranks with no input IDs, integer dtypes, recomputation, and
both full and unused request slots. Invalid IDs and capacity overflow fail on
every owner.

```bash
pytest tests/unit_tests/cpu/models/qwen3_8_flash_next/test_row_sharded_embedding.py -q
```
