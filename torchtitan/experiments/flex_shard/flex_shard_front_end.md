# FlexShard as Graph Trainer Frontend

> **Canonical status:** This file is the current implementation plan and status log
> for FlexShard. `flex_shard.md` is a high-level overview; when status differs, this
> document is authoritative.

## Motivation

Graph Trainer's value comes from making communication ops visible as FX graph nodes so compiler passes can reorder, bucket, and overlap them. SimpleFSDP achieves this via DTensor's `redistribute()` which traces to `_c10d_functional` ops.

FlexShard offers richer sharding flexibility (`Shard`, `FlatShard`, `Owned`, `RaggedShard`) and batched communication via unified byte storage (DStorage), but currently hides communication in hooks and uses `dist.*` calls that are opaque to `make_fx` tracing.

Goal: combine FlexShard's flexible sharding API with Graph Trainer's compilation pipeline.

## DStorage vs SimpleFSDP: Storage and Communication

SimpleFSDP has no unified storage — each parameter is independently converted to a DTensor, and communication is per-parameter via `redistribute()`.

DStorage uses a single contiguous `uint8` byte buffer backing all parameters in a bucket. Communication is batched — one all-gather for the entire buffer, then views are carved out per-parameter.

```
SimpleFSDP (per-parameter):
  param_A (DTensor) ---> all_gather_A ---> compute
  param_B (DTensor) ---> all_gather_B ---> compute
  param_C (DTensor) ---> all_gather_C ---> compute

DStorage (batched):
  [param_A | param_B | param_C] (one uint8 buffer)
       |
       └---> one all_gather ---> carve views ---> compute
```

In the proposed design, DStorage **replaces** SimpleFSDP's DTensor-based sharding rather than layering on top of it. The key design decision is how DStorage emits communication during tracing (see "Communication Strategy" below).

## Current Architecture Comparison

| Aspect | SimpleFSDP | FlexShard |
|--------|-----------|-----------|
| Param storage | DTensor (per-parameter) | Plain tensors in unified uint8 byte buffer (DStorage) |
| Comm mechanism | DTensor `redistribute()` -> `_c10d_functional` ops | `dist.all_gather()` / `dist.reduce_scatter()` via hooks |
| FX traceability | Yes -- `_c10d_functional` ops appear as FX nodes | No -- hooks + `dist.*` calls are opaque to make_fx |
| Lifecycle | Parametrization (property-based access triggers comm) | Module hooks (pre-forward unshard, post-backward reduce) |
| Sharding flexibility | `Shard(dim)` only | `Shard`, `FlatShard`, `Owned`, `RaggedShard` |
| Graph pass compat | Passes find `all_gather_into_tensor` / `wait_tensor` nodes | Passes can't see the comm ops |

## Gaps to Close

### 1. Switch `dist.*` to `_c10d_functional` ops

FlexShard's `Placement.unshard()` and `Placement.reduce_grad()` use `torch.distributed` APIs (`dist.all_gather`, `dist.reduce_scatter_tensor`, `dist.broadcast`, `dist.reduce`). These are not FX-traceable.

**Change**: Replace with `torch.ops._c10d_functional` equivalents:
- `dist.all_gather_into_tensor` -> `torch.ops._c10d_functional.all_gather_into_tensor`
- `dist.reduce_scatter_tensor` -> `torch.ops._c10d_functional.reduce_scatter_tensor`
- Add `wait_tensor` calls after async collectives

**Uneven sharding complication**: `Shard.unshard()` currently uses `dist.all_gather()`
(list-of-tensors variant) with per-rank variable sizes to handle cases where
`dim_size % world_size != 0`. `_c10d_functional.all_gather_into_tensor` assumes uniform
shard sizes across ranks. This is the same class of problem as `RaggedShard` (Gap #6)
but for vanilla `Shard`.

**Phased approach**:

**Phases 1-3: Require even divisibility.** Assert at init that `shape[dim] % world_size
== 0` for all `Shard(dim)` placements. This keeps the `_c10d_functional` migration
straightforward — uniform `all_gather_into_tensor` only. In practice this rarely
triggers; transformer dimensions are multiples of 64/128.

```python
for fqn, placement in resolved_placements.items():
    if isinstance(placement, Shard):
        dim_size = params[fqn].shape[placement.dim]
        if dim_size % world_size != 0:
            raise ValueError(
                f"Shard({placement.dim}) on '{fqn}' requires dim {placement.dim} "
                f"(size {dim_size}) to be divisible by world_size ({world_size}). "
                f"Use FlatShard instead, or pad the parameter."
            )
```

**Phase 5: Handle uneven via pad-to-uniform.** Pad each rank's shard to
`ceil(dim_size / world_size)`, use uniform `all_gather_into_tensor`, then slice out
padding. Same pattern as the `RaggedShard` fallback (Gap #6, option B).

Affected code in `flex_shard.py`:
- `Shard.unshard()` / `Shard.reduce_grad()`
- `FlatShard.unshard()` / `FlatShard.reduce_grad()`
- `Owned.unshard()` / `Owned.reduce_grad()`
- `RaggedShard.unshard()` / `RaggedShard.reduce_grad()`

### 2. Replace hooks with parametrization or inline tracing

Module hooks (`register_forward_pre_hook`, `register_forward_hook`, gradient hooks via `register_hook`) are invisible to `make_fx`. The unshard/reduce_grad lifecycle needs to be part of the traced computation graph.

**Options**:

A. **Parametrization pattern** (like SimpleFSDP): Parameter access triggers unshard via property. Each parameter read calls `_c10d_functional.all_gather_into_tensor`. Backward autograd generates the reduce-scatter.

B. **Inline in FwdBwdStepModule**: Make `DStorage.unshard()` and `DStorage.reduce_grad()` explicit calls inside the traced forward/backward step, using traceable ops.

Option A is more aligned with SimpleFSDP and Graph Trainer's existing tracing infrastructure.

**Parametrization by placement type**: each placement type has a different
parametrization forward. `Shard` uses `all_gather_into_tensor` which always concatenates
along dim 0. For `Shard(0)` the output is already correct. For `Shard(dim)` where
`dim > 0`, the parametrization chunks the gathered tensor along dim 0 and re-cats along
the shard dimension — the same pattern DTensor's `redistribute()` uses internally.
The backward works through standard autograd: `cat`'s backward splits along `shard_dim`,
`chunk`'s backward cats along dim 0, and the all-gather backward is `reduce_scatter`
along dim 0, producing the correct local gradient shard with no custom autograd function.
`FlatShard` is structurally different — at init time, DStorage stores the param as a 1D
flat shard, so the parametrization all-gathers the flat buffer and reshapes to the
original shape:

```python
class ShardParametrization(nn.Module):
    def __init__(self, shard_dim: int, group_name: str, world_size: int):
        super().__init__()
        self.shard_dim = shard_dim
        self.group_name = group_name
        self.world_size = world_size

    def forward(self, local_shard: Tensor) -> Tensor:
        # all_gather_into_tensor always concatenates along dim 0.
        # For Shard(0) the output shape is already correct.
        # For Shard(dim) where dim > 0, chunk along dim 0 then cat along
        # shard_dim to move the gathered data to the right dimension.
        # Backward is handled by autograd: cat's backward splits along
        # shard_dim, chunk's backward cats along dim 0, and the all-gather
        # backward is reduce_scatter along dim 0 — producing the correct
        # local gradient shard. Same pattern as DTensor's redistribute().
        full = torch.ops._c10d_functional.all_gather_into_tensor(
            local_shard, self.world_size, self.group_name
        )
        full = torch.ops._c10d_functional.wait_tensor(full)
        if self.shard_dim != 0:
            chunks = full.chunk(self.world_size, dim=0)
            full = torch.cat(chunks, dim=self.shard_dim)
        return full


class FlatShardParametrization(nn.Module):
    def __init__(self, group_name: str, world_size: int, original_shape: torch.Size):
        super().__init__()
        self.group_name = group_name
        self.world_size = world_size
        self.original_shape = original_shape

    def forward(self, flat_shard: Tensor) -> Tensor:
        # flat_shard: 1D, size = numel / world_size (DStorage stores it this way)
        full_flat = torch.ops._c10d_functional.all_gather_into_tensor(
            flat_shard, self.world_size, self.group_name
        )
        full_flat = torch.ops._c10d_functional.wait_tensor(full_flat)
        return full_flat.view(self.original_shape)
```

Both are per-parameter under the hybrid approach (Gap #3). The compiler can re-fuse
adjacent `FlatShard` all-gathers since params within a bucket are contiguous in the
byte buffer by construction. `Owned` parametrization is deferred — see Gap #4.

**Registration mechanism**: FlexShard must use SimpleFSDP's custom property-based
registration pattern (`simple_fsdp.py:133-160`), **not** `nn.utils.parametrize`.
`nn.utils.parametrize` mangles `state_dict()` keys (e.g., `weight` becomes
`parametrizations.weight.original`), which breaks DCP checkpoint compatibility.
SimpleFSDP's pattern dynamically creates a module subclass with property getters that
call the parametrization's `forward()` — no key mangling, DCP works unchanged. The
parametrization classes above (`ShardParametrization`, `FlatShardParametrization`) are
the `forward()` callables; they are registered via property, not via
`nn.utils.parametrize.register_parametrization()`.

**Init guard**: parametrization must be disabled during initialization and inspection.
Without a guard, accessing a parameter during `flex_shard()` setup — before process
groups are ready or the byte buffer is populated — would trigger an all-gather and
crash or produce garbage. SimpleFSDP uses a module-level `_active_parametrization` flag:
the property getter checks the flag and returns the raw parameter when disabled. The
flag is off during `flex_shard()` init and re-enabled after setup completes. It is also
temporarily disabled during `state_dict()` / `load_state_dict()` calls so DCP sees
raw (sharded) parameters, not unsharded ones.

**Validation in parametrization**: by using raw `_c10d_functional` ops instead of
DTensor's `redistribute()`, FlexShard loses DTensor's built-in shape and placement
validation. The parametrization `forward()` is the natural boundary for cheap runtime
checks that catch misconfiguration before emitting ops. For example,
`ShardParametrization` should assert the local shard size matches
`full_size // world_size`, and `FlatShardParametrization` should assert 1D input.
These checks run once during tracing (not per-iteration in compiled mode) and
provide clear errors instead of silent numerical corruption.

### 3. Communication Strategy: Batched vs Per-Parameter

Graph passes in `passes.py` use helpers like `is_all_gather_into_tensor()`, `is_wait_tensor()` to identify communication nodes. FlexShard's batched byte-buffer communication produces a different pattern than SimpleFSDP's per-parameter redistribution.

**Three options**:

A. **Batched collectives** (DStorage's natural pattern): One all-gather per bucket. More efficient at runtime (fewer NCCL launches), but graph passes need new patterns to recognize a single large all-gather that maps to multiple parameters.

B. **Per-parameter collectives** (like SimpleFSDP): Less efficient but drops into existing pass infrastructure with no changes.

C. **Hybrid (recommended)**: DStorage manages the buffer and metadata, but during tracing it emits per-parameter `_c10d_functional` ops. The compiler then re-buckets via `auto_bucketing` pass. This way the compiler decides optimal bucket boundaries rather than hardcoding them in DStorage.

The hybrid approach is preferred because:
- It reuses existing graph passes unchanged
- The compiler has global visibility to make better bucketing decisions (e.g., overlap comm with compute across layers)
- DStorage still provides the runtime storage efficiency (unified buffer, aligned views)
- Bucketing is a compiler concern, not a storage concern

**Eager-mode trade-off**: the hybrid approach means eager execution uses per-parameter
all-gathers instead of DStorage's current batched-per-bucket all-gathers. This is a
regression in NCCL launch count for eager mode. For a 32-layer Llama3 with ~8 params
per layer, this is ~256 per-parameter all-gathers per fwd+bwd step vs ~32 batched ones
(one per bucket). Each extra NCCL launch adds ~5-10 us of kernel launch overhead,
totaling ~1-2 ms of additional latency — small relative to the all-gather transfer
time itself (~10-100 ms per layer depending on model size and interconnect), but
non-negligible at scale. This trade-off is deliberate — see Gap #9 ("Design principle:
single wrapping path") for why a unified parametrization path is worth the eager-mode
cost, and how a future prefetch optimization inside the parametrization can recover
batched behavior without reintroducing a mode split.

### 4. Handle `Owned` placement under tracing

`Owned` is structurally different from `Shard`/`FlatShard`: forward uses broadcast
(single source), backward uses all-reduce + zero on non-owner ranks. This asymmetry
doesn't fit the all-gather/reduce-scatter pattern that graph passes expect.

**Phased approach**:

**Phases 1-3: Reject at init.** `Owned` is the least common placement type
(parameter-server-style). The core value is FSDP-style training (`Shard`,
`FlatShard`). Reject `Owned` with a clear error at init:

```python
for fqn, placement in resolved_placements.items():
    if isinstance(placement, Owned):
        raise ValueError(
            f"Owned placement on '{fqn}' is not yet supported. "
            f"Use Shard or FlatShard instead."
        )
```

**Phase 5: Implement via `_c10d_functional.broadcast` / `all_reduce`.**

Both ops are FX-traceable. The asymmetry is not a fundamental blocker — it just
needs pass infrastructure that doesn't exist yet.

Forward parametrization (runs on all ranks identically):

```python
class OwnedParametrization(nn.Module):
    def __init__(self, owner_rank: int, group_name: str):
        super().__init__()
        self.owner_rank = owner_rank
        self.group_name = group_name

    def forward(self, param: Tensor) -> Tensor:
        full = torch.ops._c10d_functional.broadcast(
            param, self.owner_rank, self.group_name
        )
        return torch.ops._c10d_functional.wait_tensor(full)
```

Backward: the autograd derivative of broadcast is `all_reduce(grad, op=SUM)` back
to the source, then zero on non-owner ranks. If `_c10d_functional.broadcast` doesn't
have a backward registered (needs pre-investigation), use a custom autograd function:

```python
class _OwnedUnshard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param, owner_rank, group_name):
        ctx.owner_rank = owner_rank
        ctx.group_name = group_name
        result = torch.ops._c10d_functional.broadcast(param, owner_rank, group_name)
        return torch.ops._c10d_functional.wait_tensor(result)

    @staticmethod
    def backward(ctx, grad):
        reduced = torch.ops._c10d_functional.all_reduce(
            grad, "sum", ctx.group_name
        )
        reduced = torch.ops._c10d_functional.wait_tensor(reduced)
        rank = dist.get_rank()
        if rank != ctx.owner_rank:
            reduced = reduced.zero_()
        return reduced, None, None
```

Graph pass extension — add recognition helpers in `passes.py`:

```python
def is_broadcast(node: fx.Node) -> bool:
    return node.target == torch.ops._c10d_functional.broadcast.default

def is_all_reduce(node: fx.Node) -> bool:
    return node.target == torch.ops._c10d_functional.all_reduce.default
```

**Pre-investigation items**: verify `_c10d_functional.broadcast` backward registration;
confirm broadcast traces correctly under `FakeTensorMode`; verify `_OwnedUnshard`
custom autograd function survives the `inductor_decomposition_pass` retrace — this
pass retraces the joint graph with `make_fx()` using Inductor decomposition tables
(`passes.py:282-379`), and custom autograd functions must be decomposable or
pass-through-safe to survive.

### 5. Handle byte buffer under FakeTensorMode

`make_fx` tracing uses `FakeTensorMode`. The unified `uint8` byte buffer with typed
views (`buffer.view(dtype).view(shape)`) needs to work correctly under symbolic
tracing. This is a potential blocker for all three compilation modes.

**Concrete test matrix** — verify these op sequences under `FakeTensorMode` in Phase 1:

```python
with FakeTensorMode():
    # 1. Basic byte buffer allocation
    buf = torch.empty(N, dtype=torch.uint8)

    # 2. dtype reinterpretation (the critical path)
    typed = buf.view(torch.bfloat16)
    param = typed[offset:offset+numel].view(shape)

    # 3. Symbolic shape variant
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    shape_env = ShapeEnv()
    sym_N = shape_env.create_symint(...)
    buf_sym = torch.empty(sym_N, dtype=torch.uint8)
    typed_sym = buf_sym.view(torch.bfloat16)
```

**Specific failure modes to check**:
- Does `view(dtype)` correctly adjust the symbolic size (`N bytes / dtype.itemsize`)?
- Does aliasing tracking survive the dtype cast (mutations to `typed` must reflect
  in `buf`)?
- Does byte offset arithmetic (`offset * itemsize`) work with symbolic values?
- Do mixed-dtype buckets (e.g., bf16 weights + fp32 biases in the same uint8 buffer)
  produce correct symbolic shapes for each dtype region?

**Fallback: typed-per-bucket storage in graph mode.** If byte buffer reinterpretation
doesn't trace correctly, DStorage can use per-dtype contiguous buffers instead of a
single `uint8` buffer in graph mode. Each bucket already enforces one placement type —
extend to also group by dtype. The buffer becomes `torch.empty(numel, dtype=param_dtype)`
instead of `torch.empty(nbytes, dtype=uint8)`, eliminating `view(dtype)` entirely.
This costs slightly more memory (one buffer per dtype per bucket instead of one unified
buffer) but avoids the FakeTensor pain point completely.

```
uint8 byte buffer (current):
  [param_A (bf16) | param_B (fp32) | param_C (bf16)] as uint8
       |                |                |
       v                v                v
  .view(bf16)      .view(fp32)      .view(bf16)   <-- problematic under FakeTensor

Typed-per-bucket fallback (graph mode only):
  bucket_bf16: torch.empty(numel_A + numel_C, dtype=bf16)
  bucket_fp32: torch.empty(numel_B, dtype=fp32)
       |                |
       v                v
  .view(shape_A)   .view(shape_B)                 <-- no dtype cast needed
```

**Decision gate**: ~~The FakeTensorMode investigation is a **blocking item in Phase 1**
before proceeding to Phase 2. If `view(dtype)` reinterpretation fails under symbolic
tracing, adopt the typed-storage fallback before building parametrization on top of it.~~
**Resolved (Phase 1)**: all three test patterns pass under `FakeTensorMode` —
basic `view(dtype)`, byte offset slice + `view(dtype)` + `view(shape)`, and
mixed-dtype regions (bf16 + fp32 in one uint8 buffer). The typed-per-bucket
fallback is **not needed**. See `test_flex_shard_tracing.py::TestFakeTensorModeByteBuffer`.

### 6. Handle `RaggedShard` under symbolic tracing

`RaggedShard` uses `local_units` to give each rank a different slice size. A
variable-size all-gather requires each rank to know its own send size and every other
rank's receive size. Under `FakeTensorMode`, there is only one process — it cannot
represent N different rank-local shapes simultaneously.

For `Shard(dim)`, every rank has the same local shape (`dim // world_size`), so one
FakeTensor suffices. For `RaggedShard`, rank 0 might have shape `(20, 4096)` while
rank 1 has `(40, 4096)`. The tracer can only represent one rank's view.

**The key question**: does `_c10d_functional.all_gather_into_tensor` (or a
variable-size variant like `all_gather_into_tensor_v`) accept per-rank size metadata
as explicit arguments, or does it infer sizes from the tensor shape? If sizes are
explicit arguments, tracing works — the metadata flows through the graph as constants.
If inferred from tensor shape, it breaks because the tracer only sees one rank's shape.

**Phased approach**:

**Phases 1-3: Reject at init.** `RaggedShard` is the most exotic placement —
asymmetric expert assignment is a niche use case. Reject at init with a clear error
(same pattern as `Owned`):

```python
for fqn, placement in resolved_placements.items():
    if isinstance(placement, RaggedShard):
        raise ValueError(
            f"RaggedShard placement on '{fqn}' is not yet supported. "
            f"Use Shard or FlatShard instead."
        )
```

**Phase 5: Investigate variable-size collective tracing.** Two paths depending on
`_c10d_functional` support:

A. **Explicit split sizes** (preferred): If `_c10d_functional` has a variable-size
   all-gather that takes split-size lists as arguments, these lists are constants per
   rank and can be baked into the graph as literal arguments. The parametrization
   passes `local_units`-derived sizes explicitly:

   ```python
   # split_sizes = [20, 40, 20, 20] derived from local_units at init time
   result = torch.ops._c10d_functional.all_gather_into_tensor_v(
       local_shard, split_sizes, group_name
   )
   ```

B. **Pad-to-uniform fallback**: If no variable-size collective exists in
   `_c10d_functional`, pad each rank's shard to the maximum rank size, use a uniform
   all-gather, then slice out padding. This trades bandwidth for traceability:

   ```
   RaggedShard (rank 0: 20 rows, rank 1: 40 rows, max = 40):
     rank 0: pad 20 → 40, all_gather uniform, slice [0:20, 40:80, 80:100, 100:120]
     rank 1: no pad,       all_gather uniform, slice accordingly
   ```

   The padding overhead is bounded by `(max_units - min_units) / max_units` per rank.

**Pre-investigation item**: check whether `torch.ops._c10d_functional` exposes a
variable-size all-gather variant, and whether FakeTensorMode can handle rank-dependent
output shapes when split sizes are passed as constant arguments.

### Clarification: backward pass is decoupled from the byte buffer

The hybrid approach emits per-parameter all-gathers in forward. The backward generates
per-parameter reduce-scatters. The gradient flow does NOT pass back through the byte
buffer — this is intentional, not a gap, but stated explicitly so readers don't wonder.

**`view(dtype)` is init-time, not trace-time.** DStorage creates typed parameter views
at initialization (`buf[offset:offset+nbytes].view(bf16).view(shape)`). The
parametrization's `forward()` receives an already-typed tensor as its input. Under
`make_fx`, this input is a leaf tensor (graph input) — the `view(dtype)` from init
does not appear in the FX graph. Autograd never sees the dtype reinterpretation, so
there is no gradient issue.

**Reduce-scatter outputs are independent tensors.** The backward produces local
gradient shards as standalone tensors assigned to `param.grad`. The byte buffer is
forward-only parameter storage. Gradient storage does not need to be unified.

**Optimizer writes through the view.** Since parameters are views into the byte buffer,
`optimizer.step()` writes through to the underlying storage in-place. The byte buffer
stays coherent without explicit sync.

**In graph mode, the byte buffer is transparent to the compiler.** The compiler sees
per-parameter tensor nodes. The view relationship into the byte buffer may not survive
compilation — Inductor may materialize independent buffers. The byte buffer's value in
graph mode is primarily at init time (parameter management, metadata tracking via
ParamInfo), not runtime memory efficiency. Runtime memory layout is the compiler's
domain.

```
Init time (not traced):
  byte buffer (uint8) → .view(bf16) → .view(shape) → param (leaf tensor)

Traced forward (FX graph):
  param (leaf) → all_gather → full_param → compute → loss

Traced backward (FX graph):
  loss → grad → reduce_scatter → local_grad_shard (independent tensor)
                                        |
                                        v
                                   param.grad

Optimizer (not traced):
  param.grad + param → optimizer.step() → writes through view into byte buffer
```

**Checkpointing**: parametrized modules change `state_dict()` key structure (e.g.,
`parametrizations.weight.original` instead of `weight`). This is a non-issue if
FlexShard uses DCP's `get_model_state_dict()` / `set_model_state_dict()` — these APIs
abstract over parametrization internals and expose logical parameter keys, same as
SimpleFSDP. The byte buffer is invisible to checkpointing: parameters are views into
the buffer, saving/loading operates on these views, and the buffer is reconstructed at
init time. No FlexShard-specific checkpoint logic is needed. Cross-format loading
(SimpleFSDP checkpoint → FlexShard model) should work if both use DCP's logical keys,
but needs explicit testing (see Phase 4).

### 8. Gradient accumulation

Gradient accumulation (running N micro-batches before optimizer step) uses the same
semantics in eager and graph modes: the reduce-scatter is generated by autograd through
the parametrization and runs for each backward. The caller accumulates `param.grad`
across micro-batches.

FlexShard cannot conditionally skip reduce-scatter within a single traced graph — the
graph is static and does not support runtime conditionals. This is a training loop /
Graph Trainer infrastructure problem, not a FlexShard problem.

Two options for the training infrastructure:

A. **Two compiled graphs**: trace one variant with reduce-scatter (sync step) and one
   without (accumulation step). `FwdBwdStepModule` can be parameterized to include or
   exclude the gradient sync, producing two compiled artifacts. The training loop
   alternates between them.

B. **Always sync**: every micro-step does reduce-scatter. Mathematically equivalent if
   gradients are scaled by `1/N` and the optimizer accumulates. Wastes bandwidth on
   N-1 unnecessary reduce-scatters, but requires no graph duplication.

Whether Graph Trainer currently supports gradient accumulation with SimpleFSDP is an
open question. If it doesn't, this is a shared infrastructure gap — FlexShard should
follow whatever solution Graph Trainer adopts rather than inventing its own.

**FlexShard's responsibility is narrow**: keep parametrization tracing clean so the
training loop can compose either always-sync accumulation or future infrastructure that
uses separate compiled graph variants.

### 9. Design principle: single wrapping path (no eager/graph mode split)

FlexShard uses one wrapping strategy — parametrization — for all execution contexts.
There is no separate `mode="eager"|"graph"` parameter, and one must not be introduced.

**Why this matters**: if wrapping were split into a hooks path (eager) and a
parametrization path (graph), a user who wraps with hooks then passes the model to
`torch.compile` would get silent incorrect results — hooks are invisible to the tracer,
so communication ops would be missing from the compiled graph. A single parametrization
path eliminates this class of bugs entirely. SimpleFSDP validates this approach: the
same parametrization-wrapped model works in eager, `torch.compile`, and `make_fx`.

Parametrization covers all execution paths:

| Execution | How it works |
|---|---|
| Plain eager | Parametrization fires on param access, per-param all-gathers |
| `torch.compile` | Dynamo traces through parametrization, captures `_c10d_functional` ops |
| `make_fx` / AOT | Same — parametrization emits traceable ops |
| Eager + memory opt | `reshard_after_forward=True` + checkpoint policy frees unsharded params |

How each concern is addressed without a mode split:

- **Eager memory staging** (unshard one bucket at a time): achieved through
  `reshard_after_forward` + checkpoint policy annotations on the parametrization,
  same as SimpleFSDP. No hooks needed.
- **Eager batched communication** (one all-gather per bucket instead of per-param):
  the one trade-off of a parametrization-only design. This is an optimization, not a
  correctness concern. Accept per-param all-gathers in eager initially. If batched
  eager communication is needed later, add a transparent prefetch optimization inside
  the parametrization: when the first param in a bucket is accessed, prefetch the
  entire bucket. This is a runtime detail, not a structural mode switch.
- **Graph compiler bucketing**: `buckets` are passed to `get_buckets()` (renamed from
  `get_transformer_block_buckets()` in Phase 3) for the `transformer_block_bucketing`
  pass, exactly as before.

### 10. Annotate FX nodes with placement metadata for bucketing passes

The hybrid approach emits per-parameter `_c10d_functional.all_gather_into_tensor` ops
during tracing. The `auto_bucketing` compiler pass fuses multiple small all-gathers
into fewer large ones to reduce NCCL launch overhead. In SimpleFSDP, all params use
the same placement, so any fusion is safe. With FlexShard, `Shard(0)` and `Shard(1)`
params cannot be fused — they concatenate along different dimensions. But the
all-gather nodes in the FX graph look identical; the pass has no way to distinguish
them without metadata.

**Fix**: the parametrization tags the **final output node** of the unshard sequence
with placement metadata. This is the node whose output is the full parameter — the
`wait_tensor` for `Shard(0)`, the `cat` for `Shard(dim != 0)`, the `view` for
`FlatShard`. Tagging the final output (rather than the all-gather itself) ensures
that `reshard_after_forward` annotates the right node for memory freeing, and that
all intermediate ops (chunk, cat, view) are included in the recomputation scope.

```python
# Inside the parametrization forward, tag the final output node:
node.meta["flex_shard_placement"] = placement  # e.g., Shard(0), Shard(1), FlatShard()
```

**Phase 3 finding**: `auto_bucketing` delegates to PyTorch's
`schedule_overlap_bucketing`, which does communication **scheduling** (reordering
collectives for compute/comm overlap on the same CUDA stream), not all-gather
**fusion**. Each all-gather remains an independent node with its own output. Different
placement types don't conflict because their post-processing (chunk+cat for
Shard(dim!=0), view for FlatShard) operates on independent tensors. The placement
metadata is available for future passes that need to distinguish placement types, but
the existing bucketing passes work without it.

**Potential future fusion constraint**: if a fusion pass is added, two all-gather nodes
can be fused only if they have identical placement type and shard dimension.
`Shard(0)` + `Shard(0)` is safe. `Shard(0)` + `Shard(1)` is not. `Shard` +
`FlatShard` is not.

**Implemented in `flex_shard/reshard_after_forward.py`.** The generalized pass
`annotate_flex_shard_all_gather` reuses `is_wait_tensor_from_fsdp()` for detection
(it already handles FlexShard — the backward walk through single-input ops to
`placeholder` works for all patterns including `.to()` for offloading). It then
walks FORWARD from `wait_tensor` to annotate the full unshard sequence:
`chunk` + `getitem` + `cat` for `Shard(dim != 0)`, `view` for `FlatShard`,
`slice` for SimpleFSDP, and `convert_element_type` for mixed precision. It also
walks BACKWARD from `all_gather` to annotate pre-processing ops (`.to()` for
offloading, `convert_element_type` for SimpleFSDP mixed precision). The terminal
node is tagged with `node.meta["flex_shard_placement"] = True`.

The pass is registered as `flex_shard_reshard_after_fwd` in `graph_trainer/passes.py`.
It is a superset of `fsdp_reshard_after_fwd` — handles both SimpleFSDP and FlexShard
patterns. Users working with FlexShard should use this pass instead of the original.
Selective activation checkpointing (SAC) composes naturally: when SAC recomputes
forward ops during backward, it re-triggers the parametrization's all-gathers — this
is the intended FSDP memory model, not a bug. The two passes are independent
(`reshard_after_forward` handles parameter memory, SAC handles activation memory) and
require no FlexShard-specific coordination.

### 11. Mixed precision training

Mixed precision training (e.g., bf16 compute, fp32 master weights) introduces a
storage-dtype vs compute-dtype split. FSDP2 supports per-module `MixedPrecisionPolicy`;
FlexShard supports per-bucket policies via `BucketSpec`.

**Storage vs compute dtype**:

- **Storage dtype**: what the byte buffer holds (e.g., fp32 master weights or bf16
  shards)
- **Compute dtype**: what forward/backward uses (e.g., bf16)

~~Two options for where master weights live:~~

~~A. **Store fp32, cast during unshard**~~ **Implemented (Phase 2c)**: the byte buffer
holds fp32 master weights. The parametrization all-gathers the fp32 shard, then casts
to bf16 for compute via `_MixedPrecisionCast.apply()`. The backward casts the grad
to `reduce_dtype` before reduce-scatter. This is the "gather-then-cast" pattern,
matching SimpleFSDP/FSDP2 semantics.

B. **Store bf16, optimizer holds fp32**: the byte buffer holds bf16 shards. The
   optimizer maintains separate fp32 state. Simpler for DStorage (no dtype mismatch),
   but fp32 optimizer state lives outside the byte buffer. (Not implemented — option A
   is the standard pattern.)

**Implementation**: `_MixedPrecisionCast` is a custom `torch.autograd.Function` that
decouples forward and backward dtype control. Forward casts to `param_dtype`, backward
casts grad to `reduce_dtype`. Both `ShardParametrization` and `FlatShardParametrization`
apply the cast after all-gather + reshape:

```python
def forward(self, local_shard: Tensor) -> Tensor:
    full = all_gather(local_shard)      # storage dtype (e.g., fp32)
    full = wait_tensor(full)
    # ... dim-fix (chunk+cat for Shard(dim!=0), view for FlatShard) ...
    if self.param_dtype is not None or self.reduce_dtype is not None:
        full = _MixedPrecisionCast.apply(full, self.param_dtype, self.reduce_dtype)
    return full                          # compute dtype (e.g., bf16)
```

**Cast location** is a bandwidth vs memory trade-off: cast-then-gather sends less
data but requires bf16 storage; gather-then-cast sends more data but preserves fp32
master weights in the buffer. The current implementation uses gather-then-cast.
This is controlled by `MixedPrecisionPolicy.param_dtype`
and `reduce_dtype`, specified per bucket via `BucketSpec`:

```python
flex_shard(
    model, mesh, dp_mesh_dims,
    shard_placement_fn={"*": Shard(0)},
    buckets=[
        BucketSpec([f"layers.{i}.*"],
                   mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))
        for i in range(32)
    ] + [
        # norm/output head in fp32 for stability
        BucketSpec(["norm.*", "lm_head.*", "output.*"],
                   mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32)),
    ],
)
```

### 12. CPU offloading

FSDP2 supports per-module CPU offloading; FlexShard supports per-bucket offloading
via `BucketSpec`. The bucket is the natural offload unit — it maps to one DStorage
buffer and one CPU→GPU transfer.

~~`BucketSpec.offload_policy` is typed as `Any` — a placeholder with no behavior
until Phase 2d implements it.~~ **Implemented (Phase 2d)**: `OffloadPolicy` frozen
dataclass with `pin_memory: bool = True`, matching FSDP2's `CPUOffloadPolicy`
semantics. `BucketSpec.offload_policy` typed as `OffloadPolicy | None`.

**Bucket-level offloading**: the entire bucket buffer lives on CPU (optionally
pinned). The parametrization handles H2D transfer before all-gather via
`.to(compute_device, non_blocking=True)`. Backward autograd handles D2H
automatically (`.to()` is differentiable — its backward generates `.to(cpu)`).
Gradients end up on CPU; optimizer runs on CPU.

```
Forward:  CPU param → .to(cuda) → GPU shard → all_gather → full param (GPU)
Backward: full grad → reduce_scatter → GPU grad → .to(cpu) → CPU grad
```

**Implementation**: `ShardParametrization` and `FlatShardParametrization` accept a
`compute_device` kwarg. When set (non-None), the forward does H2D before all-gather:

```python
def forward(self, local_shard: Tensor) -> Tensor:
    if not _active_parametrization:
        return local_shard
    if self.compute_device is not None and local_shard.device != self.compute_device:
        local_shard = local_shard.to(self.compute_device, non_blocking=True)
    full = all_gather(local_shard)
    ...
```

Per-bucket offloading via `BucketSpec`:

```python
offload = OffloadPolicy(pin_memory=True)

flex_shard(
    model, mesh, dp_mesh_dims,
    shard_placement_fn={"*": Shard(0)},
    buckets=[
        BucketSpec([f"layers.{i}.*"], offload_policy=offload)
        for i in range(32)
    ] + [
        ["norm.*", "lm_head.*", "output.*"],  # plain list — no offload
    ],
)
```

Both mixed precision and CPU offloading are handled inside the parametrization.
Neither requires changes to the compilation story — compiled graphs see the same
`_c10d_functional` ops regardless of storage dtype or device. `BucketSpec` policies
compose naturally: offloaded fp32 params on CPU → transfer to GPU → all-gather →
cast to bf16 → compute.

**Not yet implemented**: prefetch optimization (overlap H2D of next bucket with
compute of current bucket), optimizer state offloading.

**Implemented**: `reshard_after_forward` graph pass integration — `flex_shard_reshard_after_fwd_pass`
in `flex_shard/reshard_after_forward.py` annotates all FlexShard unshard sequence nodes
(including chunk+cat, view, .to(), convert_element_type) with `CheckpointPolicy.MUST_RECOMPUTE`
or `MUST_SAVE`, and tags terminal nodes with `node.meta["flex_shard_placement"]`.
Registered as `flex_shard_reshard_after_fwd` in `graph_trainer/passes.py`.

### 13. Tensor parallelism composition

FlexShard handles data parallelism (FSDP). In practice, models combine FSDP with
tensor parallelism (TP) and/or expert parallelism (EP) on a multi-dimensional mesh.
When a parameter is both TP-sharded (inner DTensor) and FSDP-sharded (FlexShard),
the parametrization must handle the nesting correctly.

SimpleFSDP's `ReplicateComputation` (`simple_fsdp.py:184-240`) handles this: it checks
whether the input parameter is a nested DTensor with inner-mesh placements (TP/EP),
extracts the inner DTensor metadata, calls `redistribute()` on the outer (DP) mesh,
then calls `to_local()` with correct `grad_placements` to produce a plain tensor for
compute. The forward/backward correctly handles the multi-mesh gradient flow.

FlexShard's parametrization must handle the same cases:

- **FSDP-only** (1D mesh): parameter is a plain tensor. Parametrization all-gathers
  it directly. This is the simple case covered by `ShardParametrization` above.
- **FSDP + TP** (2D mesh): parameter is a DTensor on the TP mesh. The FSDP
  parametrization receives this DTensor and must all-gather it on the DP mesh
  without disturbing the TP sharding. The all-gather input and output are both
  DTensors — `_c10d_functional.all_gather_into_tensor` must work on DTensor inputs,
  or the parametrization must call `redistribute()` on the DP mesh dimension.
- **FSDP + EP** (2D mesh): similar to TP but the inner mesh is the expert mesh.

**Decision**: this ties directly to Open Question #3 (reuse `ReplicateComputation`
vs. implement own). If FlexShard reuses `ReplicateComputation`, TP/EP composition
is handled by DTensor's `redistribute()` — the same code path SimpleFSDP uses. If
FlexShard implements its own parametrization with raw `_c10d_functional` ops, it must
handle nested DTensors explicitly, which is significant additional complexity.

**Phased approach**:

**Phases 1-3: FSDP-only (1D mesh).** Assert at init that the mesh is 1D. This covers
the core FSDP use case and avoids the nested DTensor complexity:

```python
if mesh.ndim != 1:
    raise ValueError(
        f"FlexShard currently requires a 1D mesh (got {mesh.ndim}D). "
        f"Use SimpleFSDP for multi-dimensional mesh (FSDP + TP/EP)."
    )
```

**Phase 5+: Multi-mesh composition.** Extend the parametrization to handle nested
DTensors, either by reusing `ReplicateComputation` or by adding explicit DTensor
handling to `ShardParametrization`.

### 14. `reassign_to_pg_pass` interaction

`passes.py` includes `reassign_to_pg_pass` which rewrites process group name arguments
on all-gather FX nodes (e.g., reassigning FSDP all-gathers from one process group to
another for hierarchical sharding). This pass must run **before** bucketing passes,
because bucketed all-gathers must inherit the reassigned process group.

FlexShard's `_c10d_functional` ops use the same argument format as SimpleFSDP's
(process group name as a string argument), so `reassign_to_pg_pass` works
without modification.

**Verified (Phase 3)**: `test_flex_shard_passes.py::TestReassignToPgComposition`
confirms (a) all-gather nodes are correctly reassigned across all patterns
(Shard(0), Shard(dim!=0), FlatShard), (b) post-gather `chunk`/`cat`/`view` nodes
are not affected, (c) mixed `Shard(0)` + `Shard(dim!=0)` all-gathers both get
reassigned, and (d) reassign + reshard passes compose correctly.

## What FlexShard Brings to Graph Trainer

1. **Flexible per-parameter sharding**: Different placement types per parameter (e.g., `Shard(1)` for EP modules, `Shard(0)` for dense layers)
2. **Batched communication**: One collective per DStorage bucket instead of per-parameter
3. **Multiple sharding strategies**: `FlatShard` (FSDP1-style), `Owned` (parameter-server-style), `RaggedShard` (asymmetric)
4. **Unified storage**: Memory-efficient byte buffer backing all sharded parameters

## Compilation Modes and FlexShard

Graph Trainer supports three compilation modes. All three rely on the same mechanism:
parameter access triggering traceable communication. The difference is *who* does the
tracing (Dynamo vs `make_fx`) and *what scope* is traced (forward only vs joint fwd+bwd).

**If FlexShard adopts the parametrization pattern with `_c10d_functional` ops, it works
across all three modes with no mode-specific code.**

### JIT mode (`--compile.mode jit`)

`model.compile(backend=custom_backend, fullgraph=True)` — Dynamo traces forward only;
autograd handles backward. Passes are applied via the custom backend callback.

- Dynamo triggers property access on parameters → parametrization fires →
  `_c10d_functional.all_gather_into_tensor` is captured in the graph.
- Hooks would NOT work: Dynamo skips hooks in `fullgraph=True` mode.
- Simplest mode — Dynamo only traces forward.

### AOT mode (`--compile.mode aot`)

`export_joint()` uses `dynamo_graph_capture_for_export()` to trace forward, then
`aot_export_joint_with_descriptors()` to get the joint fwd+bwd graph. Joint passes
run on the combined graph, then it's partitioned into fwd/bwd, and compiler passes
run on each.

- Same parametrization mechanism: property access → `_c10d_functional` ops during
  Dynamo capture.
- AOTAutograd generates the backward graph with matching `reduce_scatter` ops.
- The joint graph contains both all-gather (forward) and reduce-scatter (backward)
  as visible nodes.
- Joint passes like `fsdp_reshard_after_fwd` can annotate the all-gather nodes with
  `CheckpointPolicy.MUST_RECOMPUTE`.
- Compiler passes (`auto_bucketing`) can reorder/bucket them after partitioning.
- Supports precompilation: compile on one rank, serialize, load on all ranks.
- **Precompilation compatibility**: FlexShard's parametrization state (placement
  metadata in `node.meta`, process group names in `_c10d_functional` op args) must
  survive serialization/deserialization. Process group names are strings (serializable).
  Placement metadata in `node.meta` must be picklable — `Shard(0)`, `FlatShard()` are
  simple dataclasses and should serialize cleanly, but needs explicit testing in Phase 4.
  The `_ops_filter_with_distributed` override in `regional_inductor_pass` already allows
  `_c10d_functional` ops through the serialization filter.

### aot_fx_trace mode (`--compile.mode aot_fx_trace`)

`FwdBwdStepModule` wraps model + loss_fn + `autograd.grad()` into one module. On the
first training iteration, `trace_module()` calls `make_fx()` to capture the entire
fwd+loss+bwd as a single FX graph. No Dynamo involved.

- `make_fx` traces through Python code more directly than Dynamo.
- `FwdBwdStepModule.forward()` calls `model(inputs)` which triggers parameter access →
  parametrization fires → `_c10d_functional` ops appear in the graph.
- The `autograd.grad()` call generates reduce-scatter ops.
- `make_fx` uses `FakeTensorMode` — byte buffer views (`buffer.view(dtype).view(shape)`)
  must work under symbolic tracing (see Gap #5).
- After tracing, `apply_default_graph_passes()` runs on the captured graph.
- Most flexible for debugging (errors are deferred until training begins).

### Summary

| Aspect | JIT | AOT | aot_fx_trace |
|--------|-----|-----|--------------|
| Tracer | Dynamo | Dynamo + AOTAutograd | `make_fx` |
| Scope traced | Forward only | Joint fwd+bwd | Fwd+loss+bwd together |
| Graph partitioning | Autograd handles bwd | AOTAutograd partitions | Already combined |
| Joint passes | N/A | Yes (before partition) | Via `apply_default_graph_passes` |
| Compiler passes | Via custom backend | On partitioned fwd/bwd | On combined graph |
| Precompile support | No | Yes | No |
| FlexShard requirement | Parametrization | Parametrization | Parametrization |

All three modes require the same thing from FlexShard: **parametrization-based parameter
access that emits `_c10d_functional` ops**. No mode-specific code is needed in FlexShard.

## Unified API: Wrap Once with Bucket Spec

FlexShard currently supports nested per-module wrapping to control communication
granularity. However, nested wrapping exists to solve an **eager-mode problem**:
controlling *when* unshard/reshard happens for memory efficiency. In graph trainer
mode, the compiler decides bucketing via graph analysis — nesting is unnecessary.

The solution: **wrap the whole model once with an explicit bucket spec**. One calling
convention serves both eager and graph trainer paths.

### API Design

```python
flex_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    dp_mesh_dims: DataParallelMeshDims,
    shard_placement_fn: PlacementFn | dict[str, Placement] | None = None,
    buckets: list[list[str] | BucketSpec] | None = None,
    apply_reshard_checkpoint: bool = True,
)

@dataclass
class BucketSpec:
    patterns: list[str]
    mp_policy: MixedPrecisionPolicy | None = None
    offload_policy: OffloadPolicy | None = None
    reshard_after_forward: bool = True
```

`buckets` accepts plain `list[str]` (no policies) or `BucketSpec` (with per-bucket
policies), mixed freely. Plain lists mean no mixed precision, no offloading, and
`reshard_after_forward=True`.
`BucketSpec` attaches policies directly to the bucket — no global defaults, no
precedence rules. For uniform policies, use a Python variable:

```python
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
offload = OffloadPolicy(offload_device="cpu")

flex_shard(
    model, mesh, dp_mesh_dims,
    shard_placement_fn={"*": Shard(0)},
    buckets=[
        BucketSpec([f"layers.{i}.*"], mp_policy=mp, offload_policy=offload)
        for i in range(32)
    ] + [
        BucketSpec(["norm.*", "lm_head.*", "output.*"],
                   mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32)),
    ],
)
```

**`shard_placement_fn`** accepts two forms:

1. **Dict with fnmatch glob patterns** (simple cases): Keys are glob patterns matched
   against parameter FQNs using `fnmatch.fnmatch()`. First matching pattern wins.
   A `"*"` key serves as default.

   ```python
   flex_shard(
       model, mesh, dp_mesh_dims,
       shard_placement_fn={
           "*.experts.*": Shard(1),     # EP params: shard along dim 1
           "*.bias":      Shard(0),     # all biases: shard along dim 0
           "*":           FlatShard(),  # everything else: flatten
       },
       buckets=[...],
   )
   ```

   Common fnmatch patterns:
   - `"*"` — match all FQNs
   - `"*.experts.*"` — any FQN containing `.experts.`
   - `"layers.0.*"` — all params in layers.0
   - `"layers.?.fc1.*"` — fc1 in single-digit layers (0-9)

2. **Callable** (complex cases): Full control when glob patterns aren't enough.

   ```python
   PlacementFn = Callable[
       [list[tuple[str, nn.Parameter]], DeviceMesh],
       dict[str, tuple[Placement, ...]],
   ]
   ```

   ```python
   def my_placements(named_params, mesh):
       return {
           fqn: (Shard(1),) if p.shape[0] > 4096 else (Shard(0),)
           for fqn, p in named_params
       }
   ```

If `shard_placement_fn=None`, defaults to `{"*": Shard(0)}` (FSDP2-style).

**`buckets`** groups params into communication units. Each bucket is a list of
fnmatch glob patterns matched against param FQNs — same matching semantics as
`shard_placement_fn`. Every param must match exactly one bucket (no overlap, no
orphans). `flex_shard` validates this at init and provides actionable diagnostics
on failure:

**Orphan params** (match zero buckets):
```
flex_shard: 3 parameters not covered by any bucket:
  layers.2.gate.weight
  layers.2.gate.bias
  output.weight
Add these to an existing bucket or add a catch-all bucket: ["*"]
```

**Overlapping params** (match multiple buckets):
```
flex_shard: 2 parameters matched multiple buckets:
  layers.0.fc1.weight → bucket 0 ["layers.*"], bucket 2 ["*.fc1.*"]
  layers.1.fc1.weight → bucket 0 ["layers.*"], bucket 2 ["*.fc1.*"]
Ensure each parameter matches exactly one bucket.
```

**Coverage summary** (on success, `logger.debug` level):
```
flex_shard bucket coverage:
  bucket 0 ["tok_embeddings.*"]: 1 param
  bucket 1 ["layers.0.*"]: 8 params
  bucket 2 ["layers.1.*"]: 8 params
  bucket 3 ["norm.*", "lm_head.*", "output.*"]: 3 params
  total: 20 params across 4 buckets
```

```python
flex_shard(
    model, mesh, dp_mesh_dims,
    shard_placement_fn={"*": Shard(0)},
    buckets=[
        ["tok_embeddings.*"],                       # one module
        ["layers.0.*"],                             # one module
        ["layers.1.*"],                             # one module
        ["layers.*.fc1.*"],                         # fc1 across all layers
        ["layers.*.fc2.*", "layers.*.fc3.*"],       # fc2+fc3 across all layers
        ["norm.*", "lm_head.*", "output.*"],        # output modules together
    ],
)
```

Both `shard_placement_fn` and `buckets` use fnmatch patterns on param FQNs — one
matching system, no "module level" vs "param level" distinction.

If `buckets=None`, all parameters go into a single bucket (whole-model unshard).

### How Buckets Work in Each Execution Context

Since `flex_shard` always uses parametrization (see Gap #9), the same wrapped model
works in both eager and compiled execution. Buckets serve different purposes in each
context:

**Eager execution**:
- Parametrization fires on parameter access, emitting per-parameter all-gathers.
- `reshard_after_forward=True` frees unsharded params after forward via checkpoint
  policy annotations (same mechanism as SimpleFSDP).
- DStorage provides unified byte buffer for memory-efficient storage.
- Buckets control memory staging granularity: only one bucket's parameters are
  unsharded at a time during forward.

```
Forward pass (eager, reshard_after_forward=True):
  bucket 0: param access → all-gather → embedding forward → reshard (free)
  bucket 1: param access → all-gather → layer 0 forward → reshard (free)
  bucket 2: param access → all-gather → layer 1 forward → reshard (free)
  ...
  bucket N: param access → all-gather → norm + output forward → reshard (free)

Backward pass (eager):
  bucket N: all-gather (recompute) → norm + output backward → reduce-scatter
  ...
  bucket 1: all-gather (recompute) → layer 0 backward → reduce-scatter
  bucket 0: all-gather (recompute) → embedding backward → reduce-scatter
```

**Compiled execution** (`torch.compile`, `make_fx`, AOT):
- Parametrization emits per-parameter `_c10d_functional` ops during tracing.
- All communication appears as individual FX graph nodes — buckets don't affect
  what gets traced.
- User-specified `buckets` are passed through to `get_buckets()` (renamed from
  `get_transformer_block_buckets()`), which feeds into the `transformer_block_bucketing`
  compiler pass for manual comm/compute overlap scheduling.
- If `auto_bucketing` is used instead, buckets are ignored — the compiler derives
  optimal boundaries from graph analysis.

```
Tracing (compiled):
  flex_shard wraps model once → parametrization on all params
  make_fx / Dynamo traces forward:
    param access → _c10d_functional.all_gather (per param, in graph)
  autograd traces backward:
    _c10d_functional.reduce_scatter (per param, in graph)

Compiler passes:
  auto_bucketing: ignores user buckets, derives from graph analysis
  OR transformer_block_bucketing: uses user buckets via get_buckets()
```

### Why Not Nested Wrapping

| | Nested wrapping | Bucket spec |
|--|-----------------|-------------|
| Calling convention | Multiple `flex_shard()` calls, inner-first order matters | Single `flex_shard()` call |
| Eager memory control | Per-module DStorage with hooks | Per-bucket DStorage with parametrization + checkpoint policy |
| Compilation compat | Hooks invisible to tracer, nesting complicates parametrization | Parametrization on all params, works in eager and compiled |
| User complexity | Must understand wrapping order, excluded params, module hierarchy | Flat list of FQN groups |
| Bucket boundaries | Implicit (one bucket per `flex_shard()` call) | Explicit (user specifies, or auto-derived) |

### Interaction Between `shard_placement_fn` and `buckets`

`shard_placement_fn` specifies per-param sharding strategy. `buckets` groups params
for communication. DStorage handles buffer layout and collective ops internally.

| Concern | Who decides |
|---|---|
| Sharding strategy per param | `shard_placement_fn` |
| Which params communicate together | `buckets` |
| Buffer layout, flat offsets, collective ops | DStorage (internal) |

**Constraint: one placement per bucket (type AND parameters).** All params in a bucket
must share the same placement type and the same placement parameters — `Shard(0)` +
`Shard(0)` is valid, but `Shard(0)` + `Shard(1)` is not, because they concatenate
along different dimensions and cannot share a batched all-gather. `flex_shard` validates
this at init.

| Placement in bucket | What DStorage does |
|---|---|
| `Shard(dim)` (same dim) | One all-gather on the buffer, local reshape per param |
| `FlatShard` | One all-gather over flat buffer |
| `Owned(rank)` (same rank) | One broadcast from owner rank |
| `RaggedShard` (same local_units) | One variable-size all-gather |
| Mixed types or mismatched params | Invalid — rejected at init |

**Additional constraint for `Owned` buckets**: all params in an `Owned` bucket must
share the same `owner_rank`. Broadcast is single-source — you cannot batch two
broadcasts from different source ranks into one op. `flex_shard` validates this at init.

### Placement-Bucket Coupling

Placements fall into two categories based on how they relate to buckets:

**Self-contained placements** (`Shard`, `Owned`, `RaggedShard`): Fully describe one
param's distribution across ranks. Independent of bucket — you can re-bucket freely
without changing the sharding. The bucket is just grouping for communication batching.

**Bucket-coupled placement** (`FlatShard`): Describes one param's position in a flat
buffer shared with other params. The bucket defines the flat buffer scope — which params
get flattened together. `flex_shard` computes `flat_offset`, `numel`, and
`total_flat_numel` from the bucket contents. The user's placement fn returns
`FlatShard()` as a marker (no offsets); `flex_shard` fills them in.

| | Self-contained | Bucket-coupled |
|--|---|---|
| Placements | `Shard`, `Owned`, `RaggedShard` | `FlatShard` |
| Depends on bucket | No | Yes — bucket defines flat buffer scope |
| Re-bucketing | Safe — sharding unchanged | Recomputes offsets |
| User specifies | Minimal intent (see below) | Marker only (`FlatShard()`), offsets computed by `flex_shard` |

### What the User Specifies vs What `flex_shard` Computes

`shard_placement_fn` captures user **intent** — the minimal information that defines
the sharding strategy. `flex_shard` derives all metadata (local shape, numel, byte
offsets) from intent + param shape + mesh + bucket.

| Placement | User specifies | `flex_shard` computes |
|---|---|---|
| `Shard(dim)` | `dim` — which dimension to shard | local shape, local numel, chunk sizes |
| `FlatShard()` | nothing (marker) | `flat_offset`, `numel`, `total_flat_numel` from bucket |
| `Owned()` | nothing, or `Owned(rank)` for explicit assignment | owner via bin-packing (if not explicit), local shape |
| `RaggedShard(dims, local_units)` | `dims` + `local_units` — distribution ratios | per-rank shapes, split sizes |

`local_units` is a tuple of relative weights, one per rank, that controls how a
dimension is split unevenly. Example with `RaggedShard(dims=(0,), local_units=(1, 2, 1, 1))`
on a param with dim 0 = 100, world_size = 4:

```
total_units = 1 + 2 + 1 + 1 = 5

rank 0: 1/5 → 20 rows
rank 1: 2/5 → 40 rows
rank 2: 1/5 → 20 rows
rank 3: 1/5 → 20 rows
```

Compare with `Shard(0)` which gives 25 rows to each rank uniformly. `local_units`
represents a deliberate distribution choice (e.g., asymmetric capacity, uneven expert
assignment) — `flex_shard` cannot derive it.

All derived metadata lives in `ParamInfo`, computed internally:

```python
# ParamInfo (computed by flex_shard, not by the user)
ParamInfo(
    fqn="layers.0.fc1.weight",
    global_shape=(4096, 4096), dtype=bf16,
    placements=(Shard(0),),
    local_shape=(2048, 4096),       # from shape + mesh
    local_numel=8388608,            # from local_shape
    byte_offset=0,                  # from bucket layout
    unsharded_byte_offset=0,        # from bucket layout
    ...
)
```

```python
flex_shard(
    model, mesh, dp_mesh_dims,
    shard_placement_fn={
        "*.experts.*": Shard(1),     # self-contained: shard along dim 1
        "*":           FlatShard(),  # bucket-coupled: marker, no offsets
    },
    buckets=[
        ["layers.0.*"],              # FlatShard — scope is layers.0 params
        ["layers.1.*"],              # FlatShard — scope is layers.1 params
        ["*.experts.*"],             # Shard(1) — independent of bucket grouping
        ["norm.*", "lm_head.*", "output.*"],  # FlatShard — output params
    ],
)
# flex_shard validates: one placement type per bucket
# flex_shard computes: FlatShard offsets within each bucket
```

### Buckets: One Knob for Both Execution Contexts

`buckets` specifies **"which params belong together for communication"** — an
execution-independent concept. How each execution context uses that grouping is an
implementation detail:

- **Eager execution**: per-bucket DStorage + checkpoint policy for staged memory management
- **Compiled execution**: bucket FQNs passed to `get_buckets()` → `transformer_block_bucketing`
  pass for comm/compute overlap scheduling

In practice, the optimal boundaries for memory staging (eager) and comm/compute overlap
(compiled) align — transformer blocks are the natural unit for both. One knob serves
both execution contexts without leaking abstractions.

When the compiler can do strictly better (e.g., non-obvious overlap patterns),
`auto_bucketing` ignores user buckets entirely — so the user is never constrained.

### `get_buckets()` (renamed from `get_transformer_block_buckets()` in Phase 3)

`get_buckets()` (formerly `get_transformer_block_buckets()`, renamed in Phase 3)
currently hardcodes transformer-specific model structure (`model.tok_embeddings`,
`model.layers`, etc.). Future work: make it a simple pass-through for user-specified
buckets:

```python
def get_buckets(model, user_buckets=None) -> list[list[str] | str]:
    """Get bucket FQNs for manual bucketing passes.

    If user_buckets is provided, use them directly.
    Otherwise, generate one bucket per direct child module.
    """
    if user_buckets is not None:
        return user_buckets
    return [[name] for name, _ in model.named_children()]
```

Model-specific bucket logic (e.g., grouping `norm` + `output` together for
transformers) belongs in the model's `parallelize.py`, not in a general-purpose
utility. Models that need custom bucket layouts pass them via `flex_shard(buckets=...)`.

**Migration**: ~~existing callers of `get_transformer_block_buckets()` in Graph Trainer
(e.g., `graph_trainer.py`, pass setup code) need to be updated to call `get_buckets()`
and pass user-specified buckets through from the model's `parallelize.py`.~~
**Done (Phase 3)**: renamed to `get_buckets()` in `common_utils.py`, updated callers
in `compile.py` and `graph_utils.py`. Future work: add `user_buckets` parameter to
accept user-specified bucket layouts from `flex_shard(buckets=...)`.

### Auto-Bucket Generation

`auto_buckets()` is a public helper that generates one bucket per direct child module:

```python
def auto_buckets(module: nn.Module) -> list[list[str]]:
    """Generate one bucket per direct child module."""
    children = list(module.named_children())
    if not children:
        return [["*"]]
    return [[f"{name}.*"] for name, _ in children]
```

When `buckets=None` (the default), all parameters go into a single bucket. Users
call `auto_buckets()` explicitly when they want per-child bucketing:

```python
flex_shard(model, mesh, dp_mesh_dims, buckets=auto_buckets(model))
```

This separation keeps the default simple (whole-model = one bucket) while making
per-child bucketing a one-liner when needed.

## Proposed Architecture

```
User API
  flex_shard(model, mesh, dp_mesh_dims, shard_placement_fn, buckets=[...])
       |
       v
FlexShard Core (always parametrization-based)
  DStorage: unified byte buffer, per-param metadata (ParamInfo)
  Bucket spec: groups params into communication units
  Parametrization: per-param _c10d_functional ops (FX-traceable)
       |
       ├── Eager execution
       |     model(inputs) — parametrization fires on param access
       |     Per-bucket memory staging via reshard_after_forward + checkpoint policy
       |
       └── Compiled execution (torch.compile / make_fx / AOT)
             Tracer captures _c10d_functional ops as FX graph nodes
             buckets passed to get_buckets() for transformer_block_bucketing
                  |
                  v
             Graph Trainer Pipeline
               FX graph with visible comm nodes
               Compiler passes:
                 auto_bucketing (ignores user buckets, graph-derived)
                 OR transformer_block_bucketing (uses user buckets via get_buckets())
               + SAC, cudagraph, inductor
                  |
                  v
             Optimized Execution
```

## Implementation Plan

### Phase 1: Traceable communication ✓

- [x] Add `_c10d_functional` versions of unshard/reduce_grad: `ShardParametrization` (with chunk+cat for dim != 0) and `FlatShardParametrization` in `flex_shard.py`
- [x] Assert even divisibility: `shape[dim] % world_size == 0` for all `Shard(dim)` placements, `numel % world_size == 0` for `FlatShard` (see Gap #1; **relaxed in Phase 5** — uneven splits now use pad-to-uniform)
- [x] Reject `Owned` and `RaggedShard` placements with clear errors at init via `_validate_placements_for_tracing()` called from `flex_shard()` (see Gaps #4 and #6). **Restored in Phase 5** — `RaggedShardParametrization` added, validation relaxed
- [x] Assert 1D mesh at init (**removed in Phase 5** — multi-mesh TP/EP composition via `DTensorAwareParametrization`, see Gap #13)
- [x] **Blocking gate**: validated under `FakeTensorMode` that byte buffer `view(dtype).view(shape)` operations trace correctly — all three patterns pass (basic view, byte offset slice, mixed-dtype regions)
- [x] Unit tests in `test_flex_shard_tracing.py`: FakeTensorMode byte buffer tests, FX graph structure tests (verify `all_gather_into_tensor`/`wait_tensor`/`chunk`/`cat`/`view` nodes), init validation tests, distributed correctness tests

### Phase 2a: Core parametrization ✓

- [x] Replace hook-based unshard/reduce_grad with parametrization pattern using `ShardParametrization` and `FlatShardParametrization` — property-based param access triggers `_c10d_functional` all-gather; backward autograd generates reduce-scatter
- [x] Use SimpleFSDP's custom property-based registration (`_register_parametrization()` in `flex_shard.py`) — dynamic subclass creation with property getters, not `nn.utils.parametrize`, to avoid `state_dict()` key mangling
- [x] Implement `_active_parametrization` guard (`disable_active_parametrization()` context manager) to disable parametrization during init and `state_dict()` calls
- [x] Parametrization is the single wrapping path for `flex_shard()`
- [x] `flex_shard()` creates per-param parametrization instances grouped by leaf module, wiring `Shard` → `ShardParametrization`, `FlatShard` → `FlatShardParametrization`
- [x] Unit tests in `test_flex_shard_parametrization.py`: guard behavior (disable/restore/exception safety), property registration (dynamic subclass, state_dict bypass, multi-param), distributed correctness (param access triggers all-gather, state_dict returns sharded, guard disables all-gather, forward correctness)

### Phase 2b: BucketSpec and bucket validation ✓

- [x] Implement `BucketSpec` dataclass with `patterns`, `mp_policy`, `offload_policy`, and `reshard_after_forward` fields (`mp_policy`/`offload_policy` are placeholders — behavior implemented in Phase 2c)
- [x] Implement bucket validation: `_assign_params_to_buckets()` rejects orphan params and overlapping params with actionable error messages; `_validate_bucket_placements()` rejects mismatched placement type or shard dimension within a bucket; coverage summary emitted at `logger.debug` level
- [x] Implement fnmatch-based `shard_placement_fn` dict form via `_resolve_placement_fn()` — accepts `None` (default `per_param_placements`), `dict[str, Placement]` (fnmatch patterns, first match wins), or callable (pass-through)
- [x] Create per-bucket DStorage instances: `flex_shard()` body refactored to loop over bucket assignments, creating independent `byte_storage`, `param_infos`, and `DStorage` per bucket; module stores `_dstorages` (list) and `_dstorage` (first element, backward compat)
- [x] `FlexShardModule` updated: `unshard()`/`reduce_grad()` iterate all storages; `.dstorage`/`.dstorages` properties added
- [x] `buckets=None` defaults to single bucket (`["*"]`); `auto_buckets()` is a public helper generating one bucket per direct child module (user calls explicitly, not the default)
- [x] `flex_shard()` signature updated: `shard_placement_fn` accepts `PlacementFn | dict[str, Placement] | None`; `reshard_after_forward` is a per-bucket `BucketSpec` policy.
- [x] Unit tests in `test_flex_shard_buckets.py`: fnmatch placement dict (5 tests), bucket assignment (6 tests), placement validation (5 tests), auto_buckets (3 tests), BucketSpec (2 tests), distributed per-bucket DStorage (4 tests)

### Phase 2c: Mixed precision ✓

- [x] Implement `MixedPrecisionPolicy` frozen dataclass (`param_dtype`, `reduce_dtype`) matching SimpleFSDP's definition
- [x] Implement `_MixedPrecisionCast` custom autograd function for decoupled forward/backward dtype control — forward casts to `param_dtype`, backward casts grad to `reduce_dtype`, traceable by `torch.compile` and `make_fx`
- [x] Update `ShardParametrization` and `FlatShardParametrization` to apply gather-then-cast pattern: all-gather in storage dtype, then cast to compute dtype via `_MixedPrecisionCast.apply()`
- [x] Wire per-bucket `mp_policy` from `BucketSpec` → parametrization instances via `fqn_to_mp_policy` dict built during per-bucket loop in `flex_shard()`
- [x] Type `BucketSpec.mp_policy` as `MixedPrecisionPolicy | None` (was `Any`); `offload_policy` typed in Phase 2d
- [x] Unit tests in `test_flex_shard_mixed_precision.py`: MixedPrecisionPolicy, _MixedPrecisionCast, BucketSpec mp_policy wiring, and distributed mixed precision

### Phase 2d: CPU offloading ✓

- [x] Implement `OffloadPolicy` frozen dataclass (`pin_memory: bool = True`) matching FSDP2's `CPUOffloadPolicy` semantics
- [x] Type `BucketSpec.offload_policy` as `OffloadPolicy | None` (was `Any`)
- [x] Update `ShardParametrization` and `FlatShardParametrization` with `compute_device` kwarg — H2D transfer via `.to(device, non_blocking=True)` before all-gather; backward autograd handles D2H automatically (`.to()` is differentiable)
- [x] Wire per-bucket `offload_policy` from `BucketSpec` → CPU byte_storage allocation + parametrization `compute_device` via `fqn_to_offload_policy` dict in `flex_shard()`
- [x] Unit tests in `test_flex_shard_offload.py`: OffloadPolicy (3 tests), BucketSpec offload_policy wiring (6 tests), H2D transfer (2 tests), distributed offloading (6 tests)
- [x] `reshard_after_forward` graph pass: `flex_shard_reshard_after_fwd_pass` in `reshard_after_forward.py` — generalizes `annotate_fsdp_all_gather` to handle all FlexShard unshard patterns (Shard(0), Shard(dim!=0) chunk+cat, FlatShard view) plus offload `.to()` and mixed precision `convert_element_type`; tags terminal nodes with `node.meta["flex_shard_placement"]`; registered in `graph_trainer/passes.py` as `flex_shard_reshard_after_fwd`
- [x] Unit tests in `test_flex_shard_reshard.py`: pattern detection (8 tests across Shard(0)/Shard(dim!=0)/FlatShard/SimpleFSDP with offload and mp variants), recompute policy (2 tests), metadata tagging (4 tests), SAC composition (2 tests), pass signature (2 tests)
- [x] Unit tests in `test_flex_shard_passes.py`: reassign_to_pg composition (3 tests), pass ordering (3 tests)
- [ ] Prefetch optimization (overlap H2D of next bucket with compute of current) — deferred

### Phase 3: Graph pass compatibility ✓

- [x] Attach placement metadata (`node.meta["flex_shard_placement"]`) to final output node of each parametrization's unshard sequence (see Gap #10) — implemented in `flex_shard/reshard_after_forward.py`
- [x] Generalize `annotate_fsdp_all_gather` for FlexShard — `annotate_flex_shard_all_gather` in `flex_shard/reshard_after_forward.py` handles all patterns; registered as `flex_shard_reshard_after_fwd` in `graph_trainer/passes.py`
- [x] Wire `flex_shard_reshard_after_fwd_pass` into pipeline — `get_joint_custom_passes_from_config()` in `graph_utils.py` skips hardcoded `fsdp_reshard_after_fwd_pass` when `flex_shard_reshard_after_fwd` is in `joint_pass_names`
- [x] Rename `get_transformer_block_buckets()` → `get_buckets()` — updated in `common_utils.py`, `compile.py`, `graph_utils.py`
- [x] Verify `apply_sac_pass` works with FlexShard — SAC's `PREFER_RECOMPUTE` on unshard nodes is correctly overridden by reshard pass's `MUST_RECOMPUTE` (reshard pass always runs last). Tested in `test_flex_shard_reshard.py::TestSACComposition`
- [x] Verify `reassign_to_pg_pass` works with FlexShard — only modifies `all_gather_into_tensor` node PG args; post-gather nodes (chunk, cat, view) are untouched. Tested in `test_flex_shard_passes.py::TestReassignToPgComposition`
- [x] `auto_bucketing` works without changes — delegates to PyTorch's `schedule_overlap_bucketing` which does communication scheduling (reordering for compute/comm overlap), not all-gather fusion. Each all-gather remains independent; different placement types don't conflict.
- [x] `transformer_block_bucketing` works without changes — uses `manual_overlap_bucketing` with module FQNs. FlexShard's parametrization fires during each module's forward, so nodes inherit correct module context. FQN-based bucketing is placement-agnostic.
- [x] `cudagraph` and `full_inductor_compilation` work without changes — no FSDP-specific logic; FlexShard uses the same `_c10d_functional` ops
- [x] Mixed-placement model tested — `Shard(0)` + `Shard(dim!=0)` in same graph; `reassign_to_pg_pass` correctly reassigns both all-gathers. Tested in `test_flex_shard_passes.py::TestReassignToPgComposition::test_reassign_mixed_placements`

### Phase 4: End-to-end integration ✓

**Status**: All tests pass. 6 unit tests (2 GPUs), 7 integration tests (4 GPUs),
2 full-model loss comparisons (8 GPUs, JIT + AOT). All Phase 4 items complete
including deferred items (NCCL benchmark, CPU offload integration test).

- [x] Create `graph_trainer/flex_shard_llama3/` experiment module — `__init__.py` (model_registry), `model.py` (FlexShardLlama3Model with `disable_active_parametrization`), `parallelize.py` (1D mesh FlexShard + BucketSpec per transformer block), `config_registry.py` (debugmodel/8b configs with `joint_passes=["flex_shard_reshard_after_fwd"]`). Registered in `experiments/__init__.py` as `"graph_trainer.flex_shard_llama3"`.
- [x] Integration test configs — `flex_shard/tests/integration_tests.py` with 7 configs: JIT 1D + auto_bucketing, JIT 1D + transformer_block_bucketing, AOT 1D, AOT 1D + auto_bucketing, aot_fx_trace 1D, JIT 1D + cpu_offload, JIT 1D + mixed_precision. Run: `python -m torchtitan.experiments.flex_shard.tests.integration_tests <output_dir> --test_suite flex_shard --ngpu 4`
- [x] Numerical equivalence tests — `flex_shard/tests/test_numerics.py`: 14 pass, 2 full-model tests deselected. Run: `torchrun --standalone --nproc_per_node=2 -m pytest torchtitan/experiments/flex_shard/tests/test_numerics.py -q -k "not FullModel"`
- [x] Checkpoint round-trip tests — `flex_shard/tests/test_checkpoint.py`: 3 pass. Uses per-rank `torch.save/load` (FlexShard params are regular tensors, not DTensors). Run: `torchrun --standalone --nproc_per_node=2 -m pytest torchtitan/experiments/flex_shard/tests/test_checkpoint.py -q`
- [x] Precompile round-trip tests — `flex_shard/tests/test_precompile_roundtrip.py`: 4 pass (CPU-only). Run: `pytest torchtitan/experiments/flex_shard/tests/test_precompile_roundtrip.py -x`
- [x] Full-model loss comparison test class — `flex_shard/tests/test_numerics.py` `TestFlexShardNumericsFullModel` with `_run_flex_shard_loss_compare()`. Run: `pytest torchtitan/experiments/flex_shard/tests/test_numerics.py -x -k FullModel`
- [x] Benchmark script — `flex_shard/benchmark_flex_shard.sh` runs FlexShard vs SimpleFSDP back-to-back with TensorBoard output
- [x] FlatShard + parametrization mode — `flex_shard()` decomposes bucket-level FlatShard into per-parameter flat sharding (each param independently flattened and evenly divided), enabling `FlatShardParametrization` in all execution contexts
- [x] Mixed precision unit test passes — `test_flex_shard_mixed_precision_vs_simple_fsdp` (bf16 param / fp32 reduce)
- [x] DCP checkpoint hooks — `_register_dcp_hooks()` in `model.py` stores sharding info (`_placements`, `_mesh`) on modules (not tensors) so it survives `to_empty()`. `_state_dict_post_hook` all-gathers shards to full tensors on save; `_load_state_dict_pre_hook` chunks full tensors to shards on load. Enables `loss_compare.py` seed checkpoint workflow (save on 1 GPU, load on 8 GPUs).
- [x] Integration tests — all 7 configs pass (JIT+auto_bucketing, JIT+transformer_block_bucketing, AOT, AOT+auto_bucketing, aot_fx_trace, JIT+cpu_offload, JIT+mixed_precision)
- [x] Full-model loss comparison — both JIT and AOT pass (20 steps, FlexShard vs FSDP2 eager baseline)

#### Numerical equivalence results

Test code in `flex_shard/tests/test_numerics.py`, run via `pytest` (FSDPTest spawns processes).

| Pair | What it validates | Status |
|---|---|---|
| FlexShard eager vs SimpleFSDP eager | Sharding correctness | PASS |
| FlexShard eager vs FlexShard + `torch.compile` | Compilation numerics | PASS |
| FlexShard `reshard_after_forward=True` vs `False` | Recomputation correctness | PASS |
| FlexShard mixed precision vs SimpleFSDP mixed precision | Mixed precision correctness | PASS |
| FlexShard `Shard(0)` vs `FlatShard` | FlatShard correctness | PASS (parametrization mode) |

#### Full-model loss comparison results

FlexShard compiled vs FSDP2 eager produces a small per-step numerical difference
(~2e-5) that compounds over training. This is because compiled FlexShard uses
`_c10d_functional` ops for reduce-scatter, which accumulate in a different order
than FSDP2's reduce-scatter. This is not a correctness issue — the unit test
`test_flex_shard_vs_simple_fsdp` verifies bitwise eager-mode equivalence. Both
JIT and AOT produce identical losses (7.749865531921387 at step 2), confirming
the difference is in the reduce-scatter accumulation order, not in FlexShard logic.

| Comparison | Steps | Status |
|---|---|---|
| FlexShard JIT vs FSDP2 eager (full Llama3 debugmodel, 8 GPUs) | 20 | PASS |
| FlexShard AOT vs FSDP2 eager (full Llama3 debugmodel, 8 GPUs) | 20 | PASS |

#### Deferred items (completed)

- [x] Benchmark eager-mode overhead — `benchmark_nccl_overhead.py` measures per-parameter vs batched all-gather NCCL launch count/latency via `torch.cuda.Event` and `torch.profiler`
- [x] CPU offloading integration test — wired `OffloadPolicy` in `parallelize.py` (transformer layer buckets only), added `flex_shard_jit_1d_cpu_offload` integration test config

### Phase 5: `Owned`, `RaggedShard`, uneven sharding, and multi-mesh composition

**Status**: Implementation and validation complete. All Phase 5 items done. ✓

#### 5a: Owned graph mode ✓

- [x] `OwnedParametrization` + `_OwnedBroadcast` already implemented — `_c10d_functional.broadcast` (forward) / `all_reduce` (backward). Pre-investigation confirmed: no backward registered for `broadcast` (custom autograd function needed), ops survive `inductor_decomposition_pass` retrace (not in decomposition table)
- [x] `_validate_bucket_placements()` already enforced same `owner_rank` within a bucket
- [x] Added `_is_broadcast()` / `_is_all_reduce()` helpers to `reshard_after_forward.py`
- [x] Added `_is_wait_tensor_from_owned()` detection + `_annotate_owned_sequence()` to recognize the `placeholder → [_to_copy] → broadcast → wait_tensor → [convert_element_type]` pattern
- [x] Updated `annotate_flex_shard_all_gather()` to handle both `is_wait_tensor_from_fsdp` and `_is_wait_tensor_from_owned`
- [x] Extended `reassign_to_pg_pass` in `graph_trainer/passes.py` to rewrite broadcast node PG args (Owned placement), matching existing all-gather rewriting logic
- [x] Tests: `TestOwnedPattern` (7 tests — basic, offload, MP, recompute/save policy, tagging, MP tagging), SAC composition extended to include "owned" pattern, `test_reassign_broadcast_nodes` + `test_reassign_mixed_all_gather_and_broadcast` in `test_flex_shard_passes.py`

#### 5b: Uneven Shard(dim) and FlatShard via pad-to-uniform ✓

- [x] Extended `ShardParametrization` with `padded_shard_size` / `global_dim_size` kwargs — pads local shard to `ceil(dim_size / world_size)`, uniform all-gather, `narrow()` to slice out padding. Backward is autograd-generated (no custom autograd function)
- [x] Extended `FlatShardParametrization` with `padded_shard_size` / `global_numel` kwargs — same pad-to-uniform pattern for flat 1D shards with `numel % world_size != 0`
- [x] Relaxed `_validate_placements_for_tracing()`: removed `dim_size % world_size != 0` rejection for `Shard`, removed `numel % world_size != 0` rejection for `FlatShard`. Added `dim >= ndim` check for `Shard`
- [x] Wired both into `flex_shard()`: detects uneven split and passes padding parameters
- [x] Added `_is_narrow()` helper to `reshard_after_forward.py`; extended `_annotate_unshard_sequence()` to handle `narrow` after `wait_tensor` (uneven Shard(0)) and after `cat` (uneven Shard(dim!=0))
- [x] Tests: `test_uneven_shard_dim0_traces` (tracing), `test_accepts_uneven_shard` / `test_accepts_uneven_flat_shard` / `test_rejects_shard_invalid_dim` (validation), `test_flex_shard_uneven_shard_convergence` / `test_flex_shard_uneven_shard_eager_vs_compiled` (numerics)

#### 5c: RaggedShard graph mode ✓

- [x] No `all_gather_into_tensor_v` (variable-size) available in `_c10d_functional` — using pad-to-uniform fallback
- [x] Created `RaggedShardParametrization` class: pre-computes per-rank `split_sizes` and `max_shard_size` at init; forward pads to `max_shard_size`, uniform all-gather, chunk + `narrow()` per rank to real size + `cat`. Exported via `__init__.py`
- [x] Relaxed `_validate_placements_for_tracing()`: replaced unconditional `RaggedShard` rejection with `len(local_units) != world_size` check
- [x] Added `RaggedShard` bucket validation in `_validate_bucket_placements()`: same `local_units` within a bucket
- [x] Wired into `flex_shard()`: `isinstance(placement, RaggedShard)` branch constructs `RaggedShardParametrization`
- [x] Tests: `test_ragged_shard_parametrization_traces` (tracing), `test_accepts_ragged_shard` / `test_rejects_ragged_shard_bad_local_units` (validation), `test_flex_shard_ragged_shard_convergence` / `test_flex_shard_ragged_shard_eager_vs_compiled` (numerics)

#### 5d: Multi-mesh composition (FSDP + TP/EP) ✓

- [x] Added global SPMD mesh support: `flex_shard()` can now take the full named mesh plus `dp_mesh_dims`, derive the 1D DP shard mesh internally, and validate that DTensor params live on the full mesh with DP dims replicated
- [x] Created `DTensorAwareParametrization` wrapper: `to_local(grad_placements=inner_placements)` to peel off non-DP DTensor layer → delegate to inner parametrization → `DTensor.from_local()` to re-wrap with original non-DP placements. Explicit `grad_placements` prevents double-reduction between TP and DP backward passes
- [x] Updated `_create_param_infos()` and `_write_params_to_dstorage()` to handle DTensor params (extract TP-local tensor via `to_local()`)
- [x] Wired into `flex_shard()`: wraps inner parametrization with `DTensorAwareParametrization` when param is a DTensor
- [x] Updated `flex_shard_llama3/parallelize.py`: call `model.parallelize()` on the full SPMD state mesh so params are full-mesh DTensors directly, use the TP submesh for activation wrappers, and call FlexShard with `DataParallelMeshDims(shard="fsdp")`
- [x] Updated DCP checkpoint hooks in `model.py` for multi-mesh DTensor params: `_state_dict_post_hook` uses `to_local()` + DP all-gather + `full_tensor()` to reconstruct globally-full tensors; `_load_state_dict_pre_hook` reverses via `redistribute()` + DP chunking; `_register_dcp_hooks` captures DTensor mesh and placements
- [x] Added FSDP+TP integration test configs: JIT + AOT 2D (dp=2, tp=2, 4 GPUs)
- [x] Tests: `test_accepts_multidim_mesh` (validation)

#### 5e: Integration test infrastructure and EP verification ✓

- [x] Added `shard_placement` config field to `GraphTrainerCompileConfig` in `graph_trainer/configs.py` — options: `per_param` (default), `flat_shard`, `param_boundary`, `ragged`. CLI: `--compile.shard_placement <policy>`
- [x] Wired config-driven placement lookup in `flex_shard_llama3/parallelize.py` — replaces hardcoded `Shard(0)` with lookup table; RaggedShard dynamically computes `local_units` from the derived FSDP mesh size
- [x] Integration test configs for Owned: JIT + AOT `param_boundary` (`flex_shard_jit_1d_owned`, `flex_shard_aot_1d_owned`)
- [x] Integration test config for uneven Shard: JIT with `ngpu=3` (`flex_shard_jit_1d_uneven`)
- [x] Integration test config for RaggedShard: JIT `--compile.shard_placement ragged` (`flex_shard_jit_1d_ragged`)
- [x] EP double-wrapping verification: `TestFlexShardDoubleWrapping` unit tests (`test_double_wrap_convergence`, `test_double_wrap_expert_params_excluded`) validate that `_get_managed_named_params` correctly excludes already-wrapped sub-module params. Full DeepSeek EP validation deferred to DeepSeek FlexShard experiment.

#### 5f: Reshard-after-forward across all execution modes ✓

Reshard-after-forward frees unsharded parameter memory after forward and recomputes (re-all-gathers) in backward. Previously only worked in AOT mode via the joint graph pass. Now works in all three modes:

| Mode | Mechanism | AllGathers (fwd+bwd) |
|------|-----------|---------------------|
| **Eager** | `saved_tensors_hooks` — pack replaces unsharded param with `_FlexShardSavedHandle`, unpack re-runs parametrization | 113 |
| **JIT (inductor)** | Joint graph pass `flex_shard_reshard_after_fwd_pass` marks all-gather nodes `MUST_RECOMPUTE`, min-cut partitioner recomputes | 113 |
| **AOT** | Same joint graph pass, same min-cut partitioner | 114 |

Without reshard: 57 AllGathers (forward only, unsharded params held in memory through backward).

**Eager implementation** (`saved_tensors_hooks`, inspired by FSDP2 design doc "FSDP2 as Saved Tensor Hooks"):

- [x] Each parametrization's `forward()` calls `_tag_for_reshard(full, orig_shard, self)` which registers the unsharded tensor's storage in `_active_unsharded_storages` (keyed by `id(untyped_storage)` with strong ref to prevent id reuse)
- [x] `_FlexShardSavedTensorHooks` installs `pack`/`unpack` hooks: `pack` checks the registry by `id(storage)` — catches views like `w.T` that share the same underlying storage — and replaces with `_FlexShardSavedHandle`; `unpack` re-runs the parametrization to reconstruct
- [x] `_install_reshard_hooks()` registers pre/post forward hooks on the root module: pre-forward activates `_FlexShardSavedTensorHooks`, post-forward deactivates and clears the registry
- [x] No `storage.resize_(0)` — memory freed by GC when the handle replaces the only reference to the unsharded tensor
- [x] Verified with profiler traces: eager AllGathers went from 57 → 113, matching compiled modes

**JIT fix**:

- [x] Replaced `fsdp_reshard_after_fwd_pass` with `flex_shard_reshard_after_fwd_pass` in `jit_backend.py` — the FlexShard pass is a superset that handles all SimpleFSDP patterns plus FlexShard-specific patterns (Owned broadcast, chunk+cat, view, narrow)
- [x] Note: only the `inductor` backend respects `MUST_RECOMPUTE` annotations; `aot_eager` does not (its partitioner saves everything)

#### Pad-to-Uniform Bandwidth Overhead

The pad-to-uniform approach used by `RaggedShardParametrization` and uneven `ShardParametrization` pads each rank's local shard to a uniform size before `all_gather_into_tensor`. This introduces bandwidth overhead proportional to the padding ratio:

```
overhead = (max_shard_size * world_size - global_dim_size) / global_dim_size
```

**Uneven Shard example**: For `Shard(0)` with `dim_size=7, world_size=4`: `padded_shard_size=2`, total gathered `= 2*4 = 8`, `overhead = (8-7)/7 = 14%`. In general, uneven Shard overhead is bounded by `(world_size - 1) / dim_size` — typically negligible for production-scale parameters.

**RaggedShard example**: For `RaggedShard(dims=(0,), local_units=(1, 3, 1, 1))` on a `[24, D]` param: `split_sizes = [4, 12, 4, 4]`, `max = 12`, total gathered `= 12*4 = 48`, `overhead = (48-24)/24 = 100%`. Skewed `local_units` can produce significant waste.

**Guidance**: For RaggedShard with extreme skew (>50% overhead), consider `Owned` placement instead — each param is stored fully on one rank (broadcast instead of all-gather), with zero padding waste.

## Open Questions

1. ~~Should DStorage emit one batched collective per bucket, or per-parameter collectives?~~ **Resolved**: Hybrid approach — DStorage emits per-parameter `_c10d_functional` ops during tracing; compiler re-buckets via `auto_bucketing` pass.
2. ~~How to handle `Owned` placement under tracing? Broadcast is structurally different from all-gather/reduce-scatter.~~ **Resolved (Phase 5)**: `OwnedParametrization` with `_OwnedBroadcast` custom autograd function. `_c10d_functional.broadcast` (forward) / `all_reduce` (backward). Reshard pass extended with `_annotate_owned_sequence()`.
3. ~~Should FlexShard reuse SimpleFSDP's `ReplicateComputation` parametrization class, or implement its own?~~ **Resolved (Phase 5)**: Own parametrization with `DTensorAwareParametrization` wrapper for multi-mesh. The wrapper peels off the non-DP DTensor layer via `to_local()`, delegates to the inner parametrization, then re-wraps via `DTensor.from_local()`. This avoids coupling to `ReplicateComputation` while supporting FSDP + TP/EP composition.
6. ~~Mixed precision: should the byte buffer store fp32 master weights (cast during unshard) or bf16 shards (optimizer holds fp32)? Cast before or after all-gather?~~ **Resolved (Phase 2c)**: gather-then-cast — all-gather in storage dtype (fp32), then cast to compute dtype (bf16) via `_MixedPrecisionCast`. Per-bucket policy via `BucketSpec.mp_policy`. See Gap #11.
7. ~~CPU offloading: bucket-level or per-param offloading?~~ **Resolved (Phase 2d)**: per-bucket offloading via `BucketSpec.offload_policy`. `OffloadPolicy(pin_memory=True)`. H2D via `.to()` in parametrization; D2H via autograd `.to()` backward. See Gap #12.
4. ~~How to handle mixed placement types within a single DStorage?~~ **Resolved**: Enforce one placement type per bucket. Different placements → different buckets. Validated at init.
5. ~~How does `FlatShard` interact with buckets?~~ **Resolved**: The bucket IS the FlatShard scope. `flex_shard` computes flat offsets internally from bucket contents. User's placement fn returns `FlatShard()` as a marker with no offsets.

## References

- SimpleFSDP paper: arXiv:2411.00284
- SimpleFSDP impl: `torchtitan/experiments/graph_trainer/simple_fsdp.py`
- FlexShard impl: `torchtitan/experiments/flex_shard/flex_shard.py`
- Graph passes: `torchtitan/experiments/graph_trainer/passes.py`
- Reshard-after-forward (SimpleFSDP): `torchtitan/experiments/graph_trainer/reshard_after_forward.py`
- Reshard-after-forward (FlexShard): `torchtitan/experiments/flex_shard/reshard_after_forward.py`
- FlexShard design doc: `torchtitan/experiments/flex_shard/flex_shard.md`
- FlexShard Llama3 experiment: `torchtitan/experiments/graph_trainer/flex_shard_llama3/`
- FlexShard integration tests: `torchtitan/experiments/flex_shard/tests/integration_tests.py`
- FlexShard benchmark: `torchtitan/experiments/flex_shard/benchmark_flex_shard.sh`
- FlexShard NCCL overhead benchmark: `torchtitan/experiments/flex_shard/benchmark_nccl_overhead.py`
- FlexShard numerical equivalence tests: `torchtitan/experiments/flex_shard/tests/test_numerics.py`
- FlexShard DCP checkpoint tests: `torchtitan/experiments/flex_shard/tests/test_checkpoint.py`
- FlexShard precompile round-trip tests: `torchtitan/experiments/flex_shard/tests/test_precompile_roundtrip.py`
