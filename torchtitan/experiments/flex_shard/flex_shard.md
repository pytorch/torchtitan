# FlexShard

Implementation: https://github.com/pytorch/pytorch/pull/174267

## Motivation

FSDP1, FSDP2, and veScale-FSDP each exist because of a different sharding strategy: FSDP1 introduced `FlatShard` (flat-concat sharding), FSDP2 introduced per-param `Shard(0)`, and veScale-FSDP introduced `RaggedShard` for asymmetric chunk sizes. Yet the scaffolding around these strategies is the same every time — unshard/reshard lifecycle hooks, all-gather/reduce-scatter scheduling, buffer management, CPU offloading, gradient reduction. Each new sharding strategy required reinventing this scaffolding from scratch.

FlexShard exposes the key abstraction — `Placement` with `pack_unshard`/`unpack_unshard`/`comm_reduce` — so users can define arbitrary sharding strategies without rewriting the FSDP scaffolding. The scaffolding (byte buffer, unshard/reshard hooks, AG/RS scheduling, CPU offloading, mixed-dtype support) is written once and shared across all placements. This is analogous to [FlexAttention](https://pytorch.org/blog/flexattention/), where `score_mod`/`block_mask` are the abstraction layer and FlashAttention becomes a pluggable backend. Similarly, `FlatShard`, `Shard`, and `RaggedShard` become pluggable placements within FlexShard.

Concrete use cases enabled by pluggable placements:

- **Co-design sharding with quantization**: Block-wise quantized training (e.g., 32x32 blocks) requires shard boundaries to align with quantization block boundaries. Standard `Shard(0)` breaks blocks arbitrarily. `RaggedShard` with `local_units` respects block structure.
- **Co-design sharding with structure-aware optimizers**: Muon and Shampoo need full parameter matrices on a single device for matrix-sign or eigenvalue preconditioners. `Owned` gathers full params to one rank with efficient batched collectives.
- **Mixed placements in a single module**: `shard_placement_fn(fqn, param) → Placement` gives per-parameter control. No need to fit into a one-size-fits-all sharding scheme.

## API

### `flex_shard(module, mesh, **options) → FlexShardModule`

```python
from torchtitan.experiments.flex_shard import flex_shard

# Basic usage
mesh = init_device_mesh("cuda", (world_size,))
flex_shard(model, mesh)

# model is now a FlexShardModule
model.unshard()   # all-gather (for prefetching / manual control)
model.shard()     # release unsharded parameter, restore sharded state

# Nested wrapping (inner-first)
for layer in model.layers:
    flex_shard(layer, mesh)
flex_shard(model, mesh)  # Only wraps root params
```

## Why not DTensor

DStorage manages the full parameter lifecycle: hooks handle unshard (all-gather) before compute and reshard (reduce-scatter) after backward. The actual forward/backward runs on plain unsharded tensors. DTensor's dispatch machinery is never used, making it pure overhead:

| | DTensor | Tensor + hidden attrs |
|---|---|---|
| `__torch_dispatch__` | Active — intercepts every op | None — zero dispatch overhead |
| `torch.compile` | Graph breaks from dispatch + `.data` + `resize_` | Clean tracing — plain tensor, no subclass |
| Placement info | `param.placements`, `param._spec` | `get_placements(param)`, `get_global_shape(param)` |
| DCP integration | Native `DTensor.__create_chunk_list__` | Custom handler using hidden attributes |
| Optimizer compatibility | DTensor-aware optimizers | Standard optimizers work directly on local tensor |
| Memory | DTensorSpec + TensorMeta objects per param | A few attributes per param |
| `type(param)` | `DTensor` | `nn.Parameter` (standard) |

### Parameter swap during forward/backward

```
SHARDED state:
  model.weight = nn.Parameter(local_view)     # view into byte buffer
    ._placements = (Shard(0),)
    ._global_shape = (100, 256)
    .shape = (25, 256)                         # local shard

unshard():
  model.weight = nn.Parameter(unsharded_tensor)  # plain tensor, full shape
    .shape = (100, 256)

reshard():
  model.weight = nn.Parameter(local_view)     # back to sharded view
    ._placements = (Shard(0),)
```

During forward/backward compute, parameters are plain unsharded tensors — no subclass overhead on the compute path.

## Memory Layout

Buffer is organized by region for mixed Shard/Owned/RaggedShard placements:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                               Byte Storage                                          │
├─────────────────────────┬──────────────────────────┬───────────────────────────────┤
│     Shard Region        │   RaggedShard Region     │      Owned Region             │
│  (symmetric sharding)   │  (asymmetric sharding)   │  (full param on one rank)     │
│  padded to uniform size │  variable size per rank   │  only owner rank allocates    │
└─────────────────────────┴──────────────────────────┴───────────────────────────────┘
```

The Shard region is padded to uniform size across ranks (ceiling-based chunk sizes). The RaggedShard region has **different sizes per rank** determined by `local_units`. The Owned region only allocates on the owner rank. Total buffer size varies per rank.

### Unshard: One All-Gather for the Whole Buffer

DStorage issues a **single `dist.all_gather`** for the entire byte buffer. Per-rank buffer sizes are computed locally by replaying the buffer layout algorithm for each rank (`_compute_per_rank_byte_offsets`) — no size exchange needed.

```
unshard:
  comm:    dist.all_gather(gathered_buffers, byte_storage)   # one call, variable-size
  unpack:  per param: placement.unpack_unshard(gathered_buffers, per_rank_offsets, ...)

reshard (gradient reduction):
  Shard:       batched reduce_scatter_tensor on the Shard region
  RaggedShard: per-param variable-size dist.reduce_scatter
  Owned:       per-param dist.reduce to owner rank
```

The `_compute_per_rank_byte_offsets()` method replays the buffer allocation for all ranks, returning a dict mapping `fqn → list[int]` (byte offset of that param in each rank's buffer) plus a `list[int]` of per-rank total buffer sizes. This is computed once and cached.

### Non-Contiguous Collectives ([pytorch/pytorch#177427](https://github.com/pytorch/pytorch/issues/177427))

The RFC proposes pushing non-contiguity into the collective layer itself:

```
Current:  byte_storage → all_gather → per-param unpack_unshard → full params
With RFC: param views → all_gather_column_shard (NCCL Device API) → full params
                        ↑ zero-copy, in-place strided access
```

With RFC primitives, the reduce path simplifies to:
1. **RaggedShard**: `all_gather_column_shard` on strided views (no packing)
2. **Owned**: `grouped_reduce` for fused per-param reduces (replaces per-param `dist.reduce` loop)

## Placement Types

Per-parameter control via `shard_placement_fn(fqn, param) → Placement`. Default is `Shard(0)`.

All placements inherit from `Placement`, which defines the shard/unshard/reduce contract for DStorage. Users can register custom placements by subclassing `Placement` and implementing 6 methods.

```python
from torchtitan.experiments.flex_shard import Placement

class Placement:
    """Base class for FlexShard placement strategies (separate from DTensor's Placement)."""

    def compute_local_numel(self, global_shape, rank, world_size) -> int:
        """How many elements this rank holds for a param with global_shape."""
        ...

    def compute_local_shape(self, global_shape, rank, world_size) -> torch.Size:
        """Local shape this rank holds for a param with global_shape."""
        ...

    # Shard: extract local shard from full param → byte buffer
    def pack_unshard(self, param, buffer, byte_offset, rank, world_size) -> None:
        """Extract this rank's local shard from full param and copy into buffer."""
        ...

    # Unshard: gathered per-rank buffers → full param
    def unpack_unshard(self, gathered_buffers, per_rank_byte_offsets, global_shape, dtype) -> Tensor:
        """Reconstruct full param from all_gather'd per-rank byte buffers."""
        ...

    # Reduce: full grad → local reduced grad
    def comm_reduce(self, send_buf, recv_buf, mesh) -> None: ...
    def unpack_reduce(self, buffer, offset, local_shape, dtype) -> Tensor: ...
```

- **`pack_unshard`** is called during `flex_shard()` init and `_sync_unsharded_to_storage()` — it extracts this rank's shard from a full param and packs it into the byte buffer.
- **`unpack_unshard`** is called after the single `dist.all_gather` — it reconstructs the full param from gathered per-rank buffers using `per_rank_byte_offsets` (one offset per rank, computed by replaying buffer layout).
- **`comm_reduce`** and **`unpack_reduce`** handle gradient reduction. DStorage groups params by placement type: Shard uses batched `reduce_scatter_tensor`, RaggedShard uses per-param variable-size `reduce_scatter`, Owned uses per-param `dist.reduce`.

Built-in placements: `Shard`, `FlatShard`, `Owned`, `RaggedShard`.

### `Shard(dim)` — Symmetric Sharding

Parameter is split evenly along `dim` across all ranks. Supports uneven sharding where `dim % world_size != 0` using ceiling-based chunk sizes.

```
World size = 4, Parameter shape = (100, 256)
Shard(0) → Each rank gets (25, 256) local shard
```

```python
flex_shard(model, mesh)  # default: Shard(0) for all params
```

### `FlatShard(flat_offset, numel)` — Flat-Concat Sharding

A contiguous 1D slice of the flattened parameter. This arises when parameters are flattened and concatenated into a single 1D tensor (FSDP1-style), then evenly split across ranks — a rank's chunk may span a parameter boundary, producing a shard that has no valid ND shape.

```
Original params: p0 shape [3, 4] (12 elem), p1 shape [3, 2, 4] (24 elem)
world_size = 2, flat = [p0(12) | p1(24)] = 36 elements
rank 0: elements 0-17, rank 1: elements 18-35

rank 0's portion of p1: FlatShard(flat_offset=0, numel=6)
  Unflattened to [3, 2, 4], this is NOT a rectangular sub-tensor:
  [0][0][0..3] = 4 elements  ← complete row
  [0][1][0..1] = 2 elements  ← partial row

  ┌───────────────────┐
  │  ✓   ✓   ✓   ✓   │  row [0,0]: complete
  │  ✓   ✓   ✗   ✗   │  row [0,1]: partial
  ├───────────────────┤
  │  ✗   ✗   ✗   ✗   │
  │  ✗   ✗   ✗   ✗   │  rank 1 owns the rest
  ├───────────────────┤
  │  ✗   ✗   ✗   ✗   │
  │  ✗   ✗   ✗   ✗   │
  └───────────────────┘
```

The local tensor is always 1D with shape `(numel,)`. The ND rectangular regions can be derived from `flat_offset + numel + global_shape` when needed (e.g., for checkpoint resharding).

```python
@dataclass
class FlatShard:
    flat_offset: int   # start position in the C-contiguous flattened param
    numel: int         # number of contiguous elements
```

**When FlatShard is needed**: FSDP1-style flat-concat sharding where shard boundaries don't align with parameter boundaries. This is the only placement where the local shard has no valid ND shape — it's inherently 1D.

### `Owned(owner_rank)` — Parameter-Boundary Sharding

Entire parameter lives on one rank, other ranks have empty tensor. Memory-balanced via greedy bin-packing.

```
World size = 4
Owned(2) → Rank 2 has full (100, 256), other ranks have empty tensor
```

`Owned(r)` is a special case of `RaggedShard(dims=(0,), local_units=(..., 1, ..., 0, ...))` where only rank `r` has a non-zero unit. It exists as a simpler API for the common case, and DStorage can optimize it with `dist.reduce` to the owner instead of variable-size reduce-scatter.

```python
flex_shard(model, mesh, shard_strategy="param_boundary")
```

### `RaggedShard(dims, local_units)` — Asymmetric Sharding

Parameter is distributed across all ranks with variable chunk sizes. `local_units` specifies relative allocation ratios.

```
World size = 4, Parameter shape = (100, 256)
RaggedShard(dims=(0,), local_units=(1, 2, 1, 1))
→ Rank 0 gets (20, 256)   # 1/5 of rows
→ Rank 1 gets (40, 256)   # 2/5 of rows
→ Rank 2 gets (20, 256)   # 1/5 of rows
→ Rank 3 gets (20, 256)   # 1/5 of rows
```

For block-wise quantized training with 32x32 quantization blocks:
```python
def placement_fn(fqn: str, param: nn.Parameter):
    if param.dim() >= 2 and param.shape[-1] >= 32:
        return RaggedShard(dims=(0,), local_units=(1, 1, 1, 1))
    else:
        return Shard(0)

flex_shard(model, mesh, shard_placement_fn=placement_fn)
```

For structure-aware optimizers (Muon/Shampoo) — gather full parameter to one device:
```python
def placement_fn(fqn: str, param: nn.Parameter):
    owner = hash(fqn) % world_size
    return RaggedShard(
        dims=(0,),
        local_units=tuple(1 if r == owner else 0 for r in range(world_size)),
    )

flex_shard(model, mesh, shard_placement_fn=placement_fn)
```

### Mixed Placements

Combine Shard, Owned, RaggedShard, and FlatShard per parameter:

```python
def placement_fn(fqn: str, param: nn.Parameter):
    if "embed" in fqn:
        return Owned(0)
    elif param.numel() < 1024:
        return Owned(1)
    elif "quantized" in fqn:
        return RaggedShard(dims=(0,), local_units=(1, 1, 1, 1))
    else:
        return Shard(0)

flex_shard(model, mesh, shard_placement_fn=placement_fn)
```


## References

- [PyTorch FSDP2 Design](https://github.com/pytorch/pytorch/tree/main/torch/distributed/fsdp)
- [DTensor Specification](https://github.com/pytorch/pytorch/tree/main/torch/distributed/tensor)
- [RaggedShard Placement (veScale)](https://github.com/volcengine/veScale/blob/main/docs/texts/raggedshard.md)
- [veScale-FSDP Paper](https://arxiv.org/abs/2602.22437)
- [RFC: Non-Contiguous Collective Primitives](https://github.com/pytorch/pytorch/issues/177427)
- [SymmMem: Multi-Root Tile Reduction](https://github.com/pytorch/pytorch/pull/164757)
