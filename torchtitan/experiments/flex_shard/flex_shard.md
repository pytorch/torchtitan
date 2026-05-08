# FlexShard

Implementation: https://github.com/pytorch/pytorch/pull/174267

## Motivation

FSDP1, FSDP2, and veScale-FSDP each exist because of a different sharding strategy: FSDP1 introduced `FlatShard` (flat-concat sharding), FSDP2 introduced per-param `Shard(0)`, and veScale-FSDP introduced `RaggedShard` for asymmetric chunk sizes. Yet the scaffolding around these strategies is the same every time — unshard/reshard lifecycle hooks, all-gather/reduce-scatter scheduling, buffer management, CPU offloading, gradient reduction. Each new sharding strategy required reinventing this scaffolding from scratch.

FlexShard exposes the key abstraction — `Placement` — a small interface for how parameters are split, reassembled, and communicated — so users can define arbitrary sharding strategies without rewriting the FSDP scaffolding. The scaffolding (byte buffer, unshard/reshard hooks, AG/RS scheduling, CPU offloading, mixed-dtype support) is written once and shared across all placements. This is analogous to [FlexAttention](https://pytorch.org/blog/flexattention/), where `score_mod`/`block_mask` are the abstraction layer and FlashAttention becomes a pluggable backend. Similarly, `FlatShard`, `Shard`, and `RaggedShard` become pluggable placements within FlexShard.

Concrete use cases enabled by pluggable placements:

- **Co-design sharding with quantization**: Block-wise quantized training (e.g., 32x32 blocks) requires shard boundaries to align with quantization block boundaries. Standard `Shard(0)` breaks blocks arbitrarily. `RaggedShard` with `local_units` respects block structure.
- **Co-design sharding with structure-aware optimizers**: Muon and Shampoo need full parameter matrices on a single device for matrix-sign or eigenvalue preconditioners. `Owned` gathers full params to one rank with efficient batched collectives.
- **Unified front-end across eager and compiler**: FlexShard's placement abstraction can serve as a single sharding specification that works in both eager execution and compiler-based (torch.compile) paths, avoiding the need for separate sharding APIs.
- **Parameter server as a backend**: A custom `Placement` subclass can implement parameter-server-style communication via `unshard()`/`reduce_grad()`, where parameters are fetched from and reduced to dedicated server ranks instead of using all-gather/reduce-scatter collectives.

## API

> **Note:** FlexShard is a prototype. The API below reflects the current design but is not finalized and may change as we iterate.

### `flex_shard(module, mesh, shard_placement_fn=None) → FlexShardModule`

```python
from torchtitan.experiments.flex_shard import (
    flex_shard,
    per_param_placements,      # Shard(0) per param (FSDP2-style, default)
    flat_shard_placements,     # flat-concat sharding (FSDP1-style)
    param_boundary_placements, # one param per rank (veScale-FSDP-style)
)

mesh = init_device_mesh("cuda", (world_size,))

flex_shard(model, mesh)                                              # default: per_param
flex_shard(model, mesh, shard_placement_fn=flat_shard_placements)    # FSDP1-style
flex_shard(model, mesh, shard_placement_fn=param_boundary_placements)# veScale-FSDP-style
```

Same scaffolding, different `shard_placement_fn` — no need to rewrite unshard/reduce_grad hooks, buffer management, or gradient reduction for each strategy. Custom placement functions just return a `dict[str, tuple[Placement, ...]]` mapping each parameter FQN to its placement.

## Placement — the abstraction layer

`Placement` defines how a parameter is split across ranks and which collective to use. To add a new sharding strategy, implement a `Placement` subclass with these methods:

- `extract_local_shard(param, rank, world_size)` — extract this rank's shard from the full parameter
- `assemble_from_shards(per_rank_shards, global_shape, dtype)` — reassemble the full parameter from all ranks' shards
- `unshard(tensors, infos, mesh)` — batched gather communication across all parameters in a storage unit
- `reduce_grad(tensors, infos, mesh)` — batched reduce communication for all param gradients in a storage unit
- `compute_local_shape(global_shape, rank, world_size)` — local shape on this rank
- `compute_local_numel(global_shape, rank, world_size)` — local element count on this rank

`flex_shard` and `DStorage` work with any `Placement` subclass unchanged.

```python
class Placement:
    def extract_local_shard(self, param, rank, world_size) -> Tensor: ...
    def assemble_from_shards(self, per_rank_shards, global_shape, dtype) -> Tensor: ...
    def compute_local_shape(self, global_shape, rank, world_size) -> Size: ...
    def compute_local_numel(self, global_shape, rank, world_size) -> int: ...

    @classmethod
    def unshard(cls, tensors, infos, mesh) -> list[Tensor]: ...
    @classmethod
    def reduce_grad(cls, tensors, infos, mesh) -> list[Tensor]: ...
```

## Implementation details

`flex_shard` is the entry point. It calls `shard_placement_fn` to get per-param `Placement` objects, creates a `DStorage`, and registers forward/backward hooks that schedule `unshard()` and `reduce_grad()` at the right time.

`DStorage` is the storage unshard/reduce_grad manager. It holds a unified byte buffer backing all sharded parameters, along with each parameter's global shape, dtype, placement, and byte offset within the buffer. On `unshard()`, it extracts local shards from the buffer and calls `Placement.unshard()` to gather full parameters. On `reduce_grad()`, it collects gradients and calls `Placement.reduce_grad()` to reduce them back to sharded form.

```python
class DStorage:
    def unshard(self) -> None: ...
    def reduce_grad(self) -> None: ...
    def get_local_view(self, fqn) -> Tensor: ...
    def get_unsharded_view(self, fqn) -> Tensor: ...
```

### How the components interact

```
┌──────────────────────────────────────────────────────────────────┐
│                     flex_shard (entry point)                       │
│                                                                    │
│  1. calls shard_placement_fn to get per-param Placements           │
│  2. creates DStorage with byte buffer + param metadata             │
│  3. registers forward/backward hooks:                              │
│       pre-forward  ──▶ DStorage.unshard()                          │
│       post-backward ──▶ DStorage.reduce_grad()                         │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│         DStorage (storage unshard/reduce_grad manager)             │
│                                                                    │
│  holds: byte_buffer, per-param shape/dtype/placement/offset        │
│                                                                    │
│  unshard():                                                        │
│    calls ──▶ Placement.unshard(shards, ...)                        │
│                  ↑ gather direction                                │
│                                                                    │
│  reduce_grad():                                                    │
│    calls ──▶ Placement.reduce_grad(grads, ...)                     │
│                  ↑ reduce direction                                │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                         Placement                                  │
│                                                                    │
│  Shard, FlatShard, Owned, RaggedShard                              │
│                                                                    │
│  Per-param:                   Batched across storage unit:          │
│    extract_local_shard()        unshard()      → gather collective  │
│    assemble_from_shards()       reduce_grad()  → reduce collective  │
│    compute_local_shape/numel()                                     │
└──────────────────────────────────────────────────────────────────┘
```

### Parameter lifecycle

```
                    unshard()                        reduce_grad()
                  (pre-forward)                     (post-backward)
                       │                                  │
  ┌────────────┐       │       ┌────────────────┐         │       ┌────────────┐
  │  SHARDED   │───────┼──────▶│   UNSHARDED    │─────────┼──────▶│  SHARDED   │
  │            │       │       │                │         │       │            │
  │ local shard│    gather     │  full tensor   │     reduce      │ local shard│
  │ (25, 256)  │       │       │  (100, 256)    │         │       │ (25, 256)  │
  │            │       │       │                │         │       │ + gradient │
  └────────────┘       │       └────────────────┘         │       └────────────┘
                       │         forward/backward         │
                       │         computes here            │
```

During forward/backward, parameters are plain unsharded tensors — no subclass overhead on the compute path.

### `Shard(dim)` — Symmetric Sharding

Parameter is split evenly along `dim` across all ranks. Supports uneven sharding where `dim % world_size != 0` using ceiling-based chunk sizes.

```
World size = 4, Parameter shape = (100, 256)
Shard(0) → Each rank gets (25, 256) local shard
```

```python
flex_shard(model, mesh)  # default: Shard(0) for all params
```

### `FlatShard(flat_offset, numel, total_flat_numel)` — Flat-Concat Sharding

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
class FlatShard(Placement):
    flat_offset: int       # start position in the global flat buffer
    numel: int             # number of elements in this param
    total_flat_numel: int  # total elements across all params in the flat buffer
```

```python
flex_shard(model, mesh, shard_placement_fn=flat_shard_placements)
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
flex_shard(model, mesh, shard_placement_fn=param_boundary_placements)
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

## DTensor and SPMD Types Interface

FlexShard adopts DTensor's placement and metadata vocabulary — `Shard`, `Replicate`, placements, global shape/stride — so that migrating between DTensor and plain-tensor (SPMD-type) interfaces requires minimal effort. The core concepts (`Placement`, `DeviceMesh`, per-param sharding semantics) are shared; what differs is the runtime representation:

- **Placement and metadata**: FlexShard stores the same information DTensor does (`placements`, `global_shape`, `global_stride`, `mesh`) as hidden attributes on plain tensors via `set_sharding_info()` / `get_placements()` / `get_global_shape()`. Code that queries placement metadata can migrate with minimal changes.
- **DStorage manages the lifecycle**: Unshard/reduce_grad hooks and collective scheduling are handled by `DStorage`, not by DTensor's dispatch machinery. The forward/backward compute path sees plain unsharded tensors.

By focusing on placement and metadata as the shared abstraction, FlexShard minimizes the migration effort from DTensor's interface to an SPMD-type interface.

## Status and Next Steps

### Completed

- Core `Placement` abstraction with four implementations: `Shard`, `FlatShard`, `Owned`, `RaggedShard`
- `DStorage` unified byte buffer with automatic unshard/reduce_grad lifecycle hooks
- Nested module wrapping (wrap inner modules first, then outer)
- Mixed-dtype support within a single buffer (proper alignment per dtype)
- Meta device initialization
- DTensor-compatible metadata interface (`set_sharding_info`, `get_placements`, `get_global_shape`)
- Multi-rank test coverage with `torchrun`

### Not Yet Started

- Checkpointing / state dict integration (save/load sharded checkpoints)
- CPU offloading (move sharded params to host memory between forward/backward)
- Backward prefetching (overlap next module's all-gather with current backward)
- Mixed-precision training (compute in low precision, communicate in full precision)
- `torch.compile` compatibility
- Multi-dimensional mesh support (currently 1D only)
- Multi-dim `RaggedShard` (single-dim only, raises `NotImplementedError` for multi-dim)
- Numerical parity validation against DDP (blocked on [ThreadBasedRNGTracker](https://github.com/pytorch/pytorch/pull/174446))

### Path to Production

1. **Harden the API**: Gather feedback on the `Placement` interface and `shard_placement_fn` signature. The current API is a prototype and may change.
2. **Feature parity with FSDP2**: Add checkpointing, CPU offloading, backward prefetching, and mixed-precision support to match FSDP2's feature set.
3. **Compiler integration**: Validate `torch.compile` compatibility and explore FlexShard as a unified front-end across eager and compiled execution.
4. **Upstream to PyTorch**: Move core components (`Placement`, `DStorage`) from `torchtitan/experiments/` into `torch.distributed` once the API stabilizes.
5. **Convergence testing**: Validate loss convergence on representative workloads (e.g., Llama on C4) across all placement types and parallelism configurations.

## References

- [PyTorch FSDP2 Design](https://github.com/pytorch/pytorch/tree/main/torch/distributed/fsdp)
- [DTensor Specification](https://github.com/pytorch/pytorch/tree/main/torch/distributed/tensor)
- [RaggedShard Placement (veScale)](https://github.com/volcengine/veScale/blob/main/docs/texts/raggedshard.md)
- [veScale-FSDP Paper](https://arxiv.org/abs/2602.22437)
- [RFC: Non-Contiguous Collective Primitives](https://github.com/pytorch/pytorch/issues/177427)
- [SymmMem: Multi-Root Tile Reduction](https://github.com/pytorch/pytorch/pull/164757)
