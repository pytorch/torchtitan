# FlexShard: One API Across Eager and Compile

**Authors**: Wei Feng, Tianyu Liu, Ailing Zhang

## Front-End API

### `flex_shard()` — the single entry point

```python
flex_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    dp_mesh_dims: DataParallelMeshDims,
    shard_placement_fn: PlacementFn | dict[str, Placement] | None = None,
    buckets: list[list[str] | BucketSpec] | None = None,
    apply_reshard_checkpoint: bool = True,
)
```

One call shards a module. The same function works for eager, JIT, and AOT — no mode-specific API.

### Multi-Mesh Example: Parallelize on the Global SPMD Mesh

For FSDP + TP composition, call `model.parallelize()` with the full SPMD
state mesh so parameters become DTensors on that mesh with DP dims replicated.
During the current migration, pass the TP submesh as the activation mesh so
existing local-map and input/output redistribution specs keep their TP-only
compute contract. Then pass the same global state mesh to `flex_shard()` with
`dp_mesh_dims`. `flex_shard()` derives the DP shard mesh internally and
re-wraps unsharded parameters with the original non-DP placement metadata.

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import DataParallelMeshDims

from torchtitan.experiments.flex_shard import (
    BucketSpec,
    Shard,
    flex_shard,
)

mesh = init_device_mesh(
    "cuda",
    (dp_size, tp_size),
    mesh_dim_names=("fsdp", "tp"),
)

model = Transformer(args)

# model.parallelize() creates full-mesh DTensor parameters directly.
model.parallelize(mesh, activation_mesh=mesh["tp"], distribute_buffers=False)

# FlexShard owns the data-parallel dim of the global mesh.
flex_shard(
    model,
    mesh,
    DataParallelMeshDims(shard="fsdp"),
    shard_placement_fn={"*": Shard(0)},
    buckets=[
        BucketSpec(["tok_embeddings.*"]),
        *[BucketSpec([f"layers.{i}.*"]) for i in range(args.n_layers)],
        BucketSpec(["norm.*", "lm_head.*", "output.*"]),
    ],
)
```

```python
mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
model = Transformer(args)
model.parallelize(mesh, wrap_forward=False, distribute_buffers=False)
dp_mesh_dims = DataParallelMeshDims(shard="fsdp")

mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
buckets = [
    BucketSpec(["tok_embeddings.*"], mp_policy=mp),
    *[BucketSpec([f"layers.{i}.*"], mp_policy=mp, offload_policy=OffloadPolicy()) for i in range(n_layers)],
    BucketSpec(["norm.*", "lm_head.*", "output.*"], mp_policy=mp),
]

# Shard(0): per-param all-gather, one bucket per transformer block (FSDP2-style)
# per_param_placements is a built-in factory:
#
#   def per_param_placements(named_params, mesh):
#       return {fqn: (Shard(0),) for fqn, _ in named_params}
#
# {"*": Shard(0)} is shorthand — flex_shard resolves it via fnmatch to the same result.
flex_shard(
    model,
    mesh,
    dp_mesh_dims,
    shard_placement_fn=per_param_placements,
    buckets=buckets,
)

# FlatShard: flatten params per bucket into one 1D buffer, single all-gather per bucket (FSDP1-style)
# flat_shard_placements is a built-in factory that computes contiguous flat offsets:
#
#   def flat_shard_placements(named_params, mesh):
#       total = sum(p.numel() for _, p in named_params)
#       result, offset = {}, 0
#       for fqn, p in named_params:
#           result[fqn] = (FlatShard(offset, p.numel(), total),)
#           offset += p.numel()
#       return result
#
# Example for a bucket with weight (256x768) + bias (768):
#   weight → FlatShard(flat_offset=0,      numel=196608, total=197376)
#   bias   → FlatShard(flat_offset=196608, numel=768,    total=197376)
# Each rank stores total//world_size elements of the concatenated flat buffer.
flex_shard(
    model,
    mesh,
    dp_mesh_dims,
    shard_placement_fn=flat_shard_placements,
    buckets=buckets,
)

# Owned: each param lives fully on one rank, broadcast to all (veScale-style)
# param_boundary_placements is a built-in factory that uses greedy bin-packing:
#
#   def param_boundary_placements(named_params, mesh):
#       assignments = _assign_params_to_ranks(named_params, mesh.size())
#       return {fqn: (Owned(assignments[fqn]),) for fqn, _ in named_params}
#
# _assign_params_to_ranks greedily assigns each param to the least-loaded rank.
# Forward: broadcast from owner. Backward: all-reduce to average gradients.
flex_shard(
    model,
    mesh,
    dp_mesh_dims,
    shard_placement_fn=param_boundary_placements,
    buckets=buckets,
)
```

## Global SPMD Mesh Migration Plan

PyTorch `fully_shard(mesh, dp_mesh_dims)` treats `mesh` as the full SPMD mesh and
uses `dp_mesh_dims` to derive the data-parallel shard/replicate submeshes
internally. FlexShard now follows the same global-mesh model: callers pass a
full named mesh plus `dp_mesh_dims`, and FlexShard derives the DP shard mesh
internally. The old API where `mesh` meant the DP/FSDP submesh is removed.

### Current API

```python
from torch.distributed.fsdp import DataParallelMeshDims

mesh = init_device_mesh(
    "cuda",
    (dp_size, tp_size),
    mesh_dim_names=("fsdp", "tp"),
)

flex_shard(
    model,
    mesh,
    DataParallelMeshDims(shard="fsdp"),
    shard_placement_fn={"*": Shard(0)},
    buckets=buckets,
)
```

The `mesh` argument is the global SPMD mesh. `dp_mesh_dims` names which mesh
dimension(s) FlexShard owns for data parallelism. Multiple shard dimensions are
flattened into one DP shard mesh, matching PyTorch FSDP's global SPMD behavior.

For FSDP-only tests or callers, use a one-dimensional named global mesh and let
`model.parallelize()` record the full-SPMD state layout before sharding:

```python
mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
model.parallelize(
    mesh,
    wrap_forward=False,
    distribute_buffers=False,
    materialize_state=False,
)
flex_shard(model, mesh, DataParallelMeshDims(shard="fsdp"))
```

`materialize_state=False` leaves CPU-built parameters on CPU and records the
resolved SPMD mesh/placements as metadata. FlexShard then copies only this
rank's non-DP-local, DP-sharded slice into DStorage on the mesh device.

### Done

1. **Global mesh is required.** `flex_shard()` now requires
   `DataParallelMeshDims` and no longer supports passing a DP/FSDP submesh as
   the public API.

2. **FlexShard mesh metadata is implemented.** `FlexShardMeshInfo` stores the
   full `spmd_mesh`, the derived `dp_shard_mesh`, DP dim names, and DP dim
   indices. Multiple shard dims are flattened into one DP shard mesh.

3. **Global SPMD validation is implemented.** FlexShard validates that the mesh
   is named, that `dp_mesh_dims.shard` names exist, and that managed parameters
   are either DTensors on an equivalent full mesh or plain tensors annotated by
   `model.parallelize(..., materialize_state=False)`, with DP dims as
   `Replicate()`.

4. **SPMD metadata is preserved in storage.** `ParamInfo` records the original
   SPMD mesh, placements, and DP dim indices while DStorage continues to store
   DP-local plain shards.

5. **Parameter lifting helper is available for migration/tests.**
   `lift_params_to_global_spmd_mesh()` converts plain params or TP-only DTensor
   params to DTensors on the full named mesh with DP dims replicated. It is no
   longer used by the steady-state Graph Trainer FlexShard Llama path, which
   now records SPMD metadata without materializing full CUDA DTensors.

6. **Graph Trainer FlexShard Llama path is migrated.** The FSDP-only path calls
   `model.parallelize(fsdp_mesh, wrap_forward=False, distribute_buffers=False,
   materialize_state=False)` so parameters stay on their build device until
   FlexShard copies local shards into DStorage. The FSDP+TP path calls
   `model.parallelize()` on the full `("fsdp", "tp")` state mesh, uses the TP
   submesh for activation wrappers, and passes the full mesh to FlexShard with
   `DataParallelMeshDims(shard="fsdp")`.

7. **DTensor-aware parametrization is updated.** Pure-DP outputs stay plain
   tensors so existing module forwards with local tensor inputs still work.
   Parameters with non-DP mesh dims are rewrapped as DTensors on the non-DP
   compute submesh, including replicated TP/EP placements, so FlexShard compute
   composes with TP activation DTensors while storage still records full SPMD
   metadata.

8. **Checkpoint metadata is partially migrated.** DCP hooks now detect
   `_spmd_mesh` and `_spmd_placements` on FlexShard params so the original SPMD
   metadata is available after raw params are replaced by DStorage views.

9. **Initial tests are migrated and passing.** The single-process FlexShard suite
   passes under the required-global-mesh API. Two-rank distributed
   `test_flex_shard_parametrization.py` and `test_flex_shard_buckets.py` also
   pass after migration.

10. **Distributed tests and examples have had the final API pass.** Stale
    `flex_shard(model, mesh)` docs were updated to the required
    `flex_shard(model, mesh, DataParallelMeshDims(...))` form, old
    hooks/no_sync benchmark wording was removed, and the distributed checkpoint
    and numerics commands were rerun under the required-global-mesh API:
    `test_checkpoint.py` passes with 3 tests, and `test_numerics.py -k "not FullModel"`
    passes with 14 tests and 2 full-model tests deselected. The DTensor-aware
    wrapper now forwards the inner mixed-precision dtype policy so the eager
    batched all-gather path preserves bf16 parameter casts.

11. **The lifting helper is removed from the steady-state path.**
    `flex_shard_llama3/parallelize.py` no longer imports or calls
    `lift_params_to_global_spmd_mesh()`. `Module.parallelize()` can either
    distribute parameters on the full SPMD mesh or record the full-SPMD state
    metadata without materializing state. The FlexShard Llama path uses the
    non-materializing mode, separates the state mesh from the activation mesh
    where needed, and `resolve_placements()` maps the runtime `"fsdp"` mesh
    axis to the `DP_SHARD` sharding role.

12. **Compile mode uses graph-pass resharding only.** The Graph Trainer
    FlexShard path disables eager reshard checkpoint wrappers when compilation
    is enabled, so Dynamo does not trace checkpoint-side tensor metadata
    mutation. The compile graph pass still receives the global
    `reshard_after_forward` policy.

### TODO

1. **HSDP/global replicate dims.** `dp_mesh_dims.replicate` is still rejected.
   Add replicate-mesh support and test `DataParallelMeshDims(shard="fsdp",
   replicate="dp_replicate")`.

2. **DDP-only mode.** `DataParallelMeshDims(replicate=...)` without a shard dim
   is not supported yet.

3. **Same-tensor-dim TP+FlexShard sharding.** If a non-DP mesh dim and FlexShard
   shard the same logical tensor dimension, add strided-shard metadata equivalent
   to PyTorch's `_StridedShard` or reject the case with a clear validation error.

4. **Full global activation mesh.** The steady-state path no longer uses the
   lifting helper for parameters, but FSDP+TP activation wrappers still run on
   the TP submesh. A future pass can expand local-map/input/output placement
   specs to the full SPMD mesh once the model configs describe DP/CP activation
   placements at every local-map boundary.

5. **Fix compiled reshard graph-pass partitioning.** The 2D FSDP+TP JIT/AOT
   integration now reaches compilation, but `flex_shard_reshard_after_fwd_pass`
   still trips AOTAutograd partitioning with `Node tangents_2 was invalid, but
   is output`. This is separate from parameter lifting; the global-mesh
   migration now gets past parameter validation, bucket coverage, and local-map
   placement resolution.

4. **Broaden placement coverage under non-DP SPMD.** Validate `FlatShard`,
   `RaggedShard`, and `Owned` with TP/EP placements, not only pure-DP meshes.

5. **Finish checkpoint integration.** Add full DCP save/load coverage for the
   global SPMD Graph Trainer path, including FSDP+TP and non-default placements.

## CPU Init to CUDA Sharded State Plan

PyTorch `fully_shard()` supports CPU-built modules by deriving the target device
from the mesh, moving only managed sharded state to that device, and asserting
that the sharding path sees the expected device. For FlexShard global SPMD mode,
the model still needs full-mesh placement metadata before `flex_shard()` runs,
but that metadata no longer has to come from CUDA DTensors.

A CPU-built model is supported through:

```python
model = Transformer(args)  # built on CPU
model.parallelize(
    cuda_mesh,
    activation_mesh=tp_mesh,
    distribute_buffers=False,
    materialize_state=False,
)
flex_shard(model, cuda_mesh, DataParallelMeshDims(shard="fsdp"), buckets=buckets)
```

Directly calling `flex_shard()` on plain CPU parameters with `dp_mesh_dims`
remains unsupported. In global SPMD mode, FlexShard needs the full-mesh
placements produced by `model.parallelize()` to preserve TP/EP metadata; it
cannot infer those placements from plain parameters.

### Target Behavior

1. **CPU model construction is allowed.** Model builders may create real CPU
   tensors. `model.parallelize(cuda_mesh, materialize_state=False, ...)` records
   each parameter's full-SPMD mesh and placements without moving the parameter
   to CUDA.

2. **FlexShard sees full-mesh state metadata.** By the time `flex_shard()` runs,
   every managed parameter is either a DTensor on the same full SPMD mesh passed
   to `flex_shard()` or a plain tensor annotated with that mesh and placements.
   DP dims are `Replicate()`.

3. **DStorage owns the final sharded state.** In normal mode, bucket byte storage
   and sharded parameter views live on the CUDA mesh device. In CPU-offload mode,
   bucket byte storage intentionally lives on CPU, optionally pinned, while
   unsharded compute tensors are materialized on CUDA.

4. **Whole-model `.to(cuda)` is not part of the steady state.** After FlexShard
   replaces parameters with DStorage-backed views, broad module movement is too
   coarse. Device movement should happen before sharding through
   `model.parallelize(materialize_state=False)` plus explicit DStorage/offload
   paths.

### Implementation Steps

1. **Mirror FSDP2 device validation.** After global SPMD parameter validation,
   check that each DTensor parameter's mesh device type matches the FlexShard
   mesh device type and that the local tensor already lives on the target mesh
   device. Error clearly for CPU DTensors on a CUDA mesh or any mismatched mesh
   device.

2. **Keep `model.parallelize()` as the placement source, not the CUDA move.**
   `materialize_state=False` records model-specific full-mesh placements without
   calling `distribute_tensor()`. FlexShard consumes that metadata and copies
   only local shards to CUDA DStorage.

3. **Tighten DStorage device invariants.** For non-offloaded buckets, allocate
   `byte_storage` on the mesh compute device and assert the sharded views created
   from that storage live on the same device. For offloaded buckets, allocate
   CPU pinned storage and keep the parametrization's `compute_device` set to the
   CUDA device.

4. **Make buffer handling explicit.** Parameters are owned by FlexShard, but
   unmanaged buffers must still be placed correctly. For Graph Trainer FlexShard
   paths, initialize buffers with `buffer_device=cuda` or explicitly move only
   unmanaged buffers to CUDA after initialization. Avoid using broad
   `model.to(cuda)` as a post-FlexShard fixup.

5. **Preserve meta/deferred-init behavior.** Existing meta initialization support
   should continue to skip copying unmaterialized params and use the DStorage
   sync path after materialization. CPU-init support should not regress meta
   model flows.

6. **Document CPU offload separately.** CPU init means "start with model tensors
   on CPU and end with CUDA sharded state." CPU offload means "keep sharded
   storage on CPU and transfer to CUDA for compute." Tests and docs should keep
   those two cases distinct.

### Test Plan

1. Add a CUDA distributed test for CPU-built model ->
   `model.parallelize(cuda_mesh, materialize_state=False)` -> `flex_shard()`,
   asserting no CUDA allocation during `parallelize()` and DStorage byte storage
   plus sharded parameter views on CUDA after `flex_shard()`.

2. Add a 2D global SPMD test with `("fsdp", "tp")` mesh, CPU-built model,
   full-mesh SPMD metadata from `model.parallelize()`, and
   `DataParallelMeshDims(shard="fsdp")`.

3. Add a CPU-offload variant asserting byte storage is CPU/pinned while
   parametrized compute tensors materialize on CUDA.

4. Add a negative test that `flex_shard(plain_cpu_model, cuda_mesh,
   DataParallelMeshDims(...))` still fails with the global SPMD metadata error.

5. Add a buffer placement test for the FlexShard Llama path, especially RoPE
   buffers, to verify CPU-built models do not rely on post-FlexShard
   `model.to(cuda)`.

### Done

1. **Mesh-device validation is implemented.** FlexShard resolves the target
   device from the DP shard mesh and validates that full-mesh DTensor
   parameters have local tensors on that device before constructing DStorage.

2. **CPU-built model to CUDA sharded state is covered.** Tests now build a
   module on CPU, call `model.parallelize(..., materialize_state=False)`, assert
   that no CUDA allocation occurs during `parallelize()`, call `flex_shard()`,
   and assert CUDA DStorage plus CUDA sharded parameter views.

3. **2D global SPMD CPU init is covered.** The test suite covers CPU-built
   parameters annotated for a `("fsdp", "tp")` CUDA mesh, followed by
   `flex_shard(..., DataParallelMeshDims(shard="fsdp"))`. The test checks that
   TP-local parameter storage is derived from the CPU parameter without first
   materializing a full CUDA DTensor.

4. **CPU init and CPU offload are tested separately.** CPU-init plus
   `OffloadPolicy(pin_memory=True)` keeps DStorage on pinned CPU while
   parametrized access materializes CUDA compute tensors.

5. **Plain CPU params remain unsupported under `dp_mesh_dims`.** A negative test
   verifies that `flex_shard(plain_cpu_model, cuda_mesh, dp_mesh_dims)` still
   fails with the global SPMD metadata error.

6. **DStorage device invariants are explicit.** Non-offloaded buckets assert
   that sharded parameter views live on the mesh device. Offloaded buckets keep
   CPU-backed storage and CUDA compute device behavior.

7. **Graph Trainer no longer uses broad post-FlexShard `model.to(cuda)`.**
   FlexShard models initialize buffers with a CUDA `buffer_device` and then move
   only unmanaged buffers if needed, leaving DStorage-owned parameters alone.

8. **Tests were updated for parametrized gradient access.** Offload tests now
   inspect raw sharded gradients under `disable_active_parametrization()` instead
   of reading gradients through the active property getter.

### Verified

1. `python -m pytest torchtitan/experiments/flex_shard/test_flex_shard_parametrization.py -q -k 'not Distributed'`
2. `torchrun --standalone --nproc_per_node=2 -m pytest torchtitan/experiments/flex_shard/test_flex_shard_parametrization.py -q`
3. `torchrun --standalone --nproc_per_node=2 -m pytest torchtitan/experiments/flex_shard/test_flex_shard_offload.py -q`
4. `python -m pytest torchtitan/experiments/graph_trainer/tests/test_trainer_init.py -q`
5. `pre-commit run --files torchtitan/experiments/flex_shard/flex_shard.py torchtitan/experiments/graph_trainer/trainer.py torchtitan/experiments/graph_trainer/tests/test_trainer_init.py torchtitan/experiments/flex_shard/test_flex_shard_parametrization.py torchtitan/experiments/flex_shard/test_flex_shard_offload.py torchtitan/experiments/flex_shard/flex_shard_design.md`

### TODO

1. Add a real FlexShard Llama/Graph Trainer integration test that exercises the
   RoPE buffer path end to end with the production model config.

2. Add a meta/deferred-init regression test for CPU-init-adjacent paths to ensure
   the DStorage sync path still works when local DTensor params start on meta.

3. Audit other Graph Trainer model families that might rely on post-init
   `model.to(device)` for unmanaged buffers and add targeted tests if needed.

## Eager All-Gather Stream Plan

PyTorch composable FSDP (`code-review/pytorch/torch/distributed/fsdp/_fully_shard`)
uses a split unshard path for eager mode:

1. `FSDPCommContext` owns separate high-priority streams:
   `all_gather_copy_in_stream` and `all_gather_stream`.
2. `foreach_all_gather()` runs copy-in on `all_gather_copy_in_stream`.
3. `all_gather_stream.wait_stream(all_gather_copy_in_stream)` orders the NCCL
   all-gather after copy-in.
4. `foreach_all_gather_copy_out()` waits on the all-gather event before exposing
   unsharded parameter views to compute.

FlexShard should adopt the same launch/wait split for eager batched all-gather,
but with one intentional difference: **all-gather copy-in stays on the
default/current stream**. This keeps the local shard packing/copy ordered with
the surrounding eager forward work and avoids introducing a second copy-in
stream while still moving the NCCL all-gather itself to a dedicated stream.

### Target Behavior

1. **Copy-in remains on the default stream.** The local sharded parameter reads,
   packing, dtype casts, and send-buffer writes happen on the stream active when
   the FlexShard pre-forward hook runs.

2. **NCCL all-gather launches on a shared FlexShard all-gather stream.** The
   FlexShard root owns one high-priority `all_gather_stream` for all eager
   batched buckets on the same device. The stream waits on a copy-in completion
   event before launching each collective.

3. **Compute waits only at parameter use.** Before the property getter exposes
   `_pre_gathered` parameters to the module forward, the default stream waits on
   the all-gather completion event and runs any copy-out/assembly needed to
   produce per-parameter full tensors.

4. **Compiled mode is unchanged.** `torch.compile` paths continue using
   `_c10d_functional` ops from parametrization modules. The stream changes are
   eager-only and live under `_install_batched_allgather_hooks()`.

### Implementation Steps

1. **Add a FlexShard eager communication context.** Create a small context owned
   by the FlexShard root and shared across its eager batched `DStorage`
   buckets:

   ```python
   @dataclass
   class EagerAllGatherContext:
       all_gather_stream: torch.Stream
   ```

   Lazily initialize it for CUDA/XPU-like devices. CPU-offload buckets can
   continue using the current synchronous path initially. If one FlexShard
   module ever spans multiple CUDA devices, use one stream per device; the
   normal single-device case has one all-gather stream for all buckets.

2. **Introduce an all-gather result type.** Store the tensors and sync objects
   needed between launch and wait within the same pre-forward hook:

   ```python
   @dataclass
   class EagerAllGatherResult:
       gathered: list[torch.Tensor]
       infos: list[ParamInfo]
       event: torch.Event
   ```

   If we later switch from `dist.all_gather()` to `all_gather_into_tensor()`,
   this can instead hold a flat output buffer plus split metadata.

3. **Split `Placement.unshard()` for `Shard`.** Keep the current synchronous
   `Shard.unshard()` as a compatibility wrapper, and add eager-specific helpers:

   ```python
   result = Shard.begin_unshard(local_shards, infos, mesh, all_gather_stream)
   full_params = Shard.finish_unshard(result)
   ```

   `begin_unshard()` runs copy-in on the current/default stream:

   - collect local shard tensors
   - allocate/pack the send buffer
   - allocate gather output tensors
   - record `copy_in_done_event`

   Then it enters `all_gather_stream`, waits on `copy_in_done_event`, launches
   `dist.all_gather(..., async_op=False)`, records `all_gather_done_event`, and
   returns `EagerAllGatherResult`.

4. **Wait and assemble in the pre-forward hook.** In
   `_install_batched_allgather_hooks()`:

   - Call `begin_unshard()` for the current bucket.
   - Immediately call `finish_unshard()` for that same bucket before returning
     from the pre-forward hook.
   - `finish_unshard()` waits on the all-gather event from the default stream,
     assembles full tensors, and writes `param_p._pre_gathered`.

   This changes stream placement without changing when the current module may
   start compute.

5. **Prefetch the next eager bucket.** After the current bucket is ready, the
   eager hook launches the next bucket's all-gather on the shared all-gather
   stream. Normal forward uses bucket order:

   ```text
   embed -> layers.0 -> ... -> layers.N -> output
   ```

   FlexShard checkpoint recompute marks recompute execution with an internal
   context flag, and eager prefetch uses reverse bucket order there:

   ```text
   output -> layers.N -> ... -> layers.0 -> embed
   ```

   The output/lm_head bucket should remain `reshard_after_forward=True` by
   default. Setting it to `False` avoids the output bucket's backward
   recompute all-gather, but simply moves the exposed first all-gather to the
   last transformer layer because there is then no preceding recompute hook to
   launch that prefetch.

6. **Document the first backward-prefetch gap.** The current eager prefetch
   mechanism is one-bucket-ahead and is triggered from bucket pre-forward hooks.
   That means the first bucket in checkpoint recompute order cannot be
   prefetched by this mechanism:

   ```text
   output.reshard_after_forward=True:  output AG is exposed first
   output.reshard_after_forward=False: layers.N AG is exposed first
   ```

   Hiding that first recompute all-gather requires a separate backward-entry
   trigger, for example a FlexShard-owned eager scheduler hook at the downstream
   consumer boundary. A tensor grad hook can provide such a trigger internally,
   but it should be wrapped as a bucket scheduler primitive instead of exposed
   as model-specific output-hook logic.

7. **Handle cleanup.** On post-forward, reshard, exception cleanup, and repeated
   forward/backward:

   - clear `_pre_gathered`
   - avoid leaving references to gathered buffers after the bucket is consumed
   - make checkpoint recomputation replay hooks safely

8. **Add profiler ranges.** Add `torch.profiler.record_function()` scopes:

   - `FlexShard::all_gather_copy_in`
   - `FlexShard::all_gather`
   - `FlexShard::all_gather_copy_out`

   The expected trace should show copy-in on the default stream and NCCL
   all-gather on the FlexShard all-gather stream.

### Test Plan

1. Add a 2-rank eager batched test that runs forward/backward for multiple
   iterations and compares outputs plus sharded gradients with the existing
   synchronous eager path.

2. Add a checkpoint recomputation test to verify backward replay re-launches
   all-gather and does not reuse stale gathered tensors.

3. Add a CPU-offload regression test to ensure offloaded buckets either keep the
   synchronous path or explicitly copy to the compute device before launching
   stream-based all-gather.

4. Add a profiler smoke command for manual validation. The important visual
   checks are:

   - `FlexShard::all_gather_copy_in` on the default stream
   - NCCL all-gather on the FlexShard all-gather stream
   - default stream waiting before parameter compute consumes `_pre_gathered`

## Core Insight

FlexShard intercepts parameter access so that `module.weight` triggers an all-gather behind the scenes — the model code just reads `self.weight` as usual, unaware of sharding. This works identically across eager, JIT, and AOT. Under the hood, the interceptor (a `@property` on the module class) has two branches:

- **Compiled modes (JIT/AOT)**: falls through to `parametrization.forward()`, which emits `_c10d_functional` ops (per-param, async, FX-traceable). `torch.compile` traces these into the FX graph for compiler passes to optimize.
- **Eager mode**: short-circuits via `_pre_gathered` — batched hooks run `dist.*` collectives (per-bucket, synchronous) before the property getter fires, so `parametrization.forward()` never runs.

The parametrization classes (`ShardParametrization`, `FlatShardParametrization`, etc.) define the communication pattern once. Compiled modes execute them directly; eager mode uses them as the reference implementation but replaces the execution with batched collectives for performance (one NCCL kernel per bucket instead of N per-param kernels).

## Eager vs Compiled: Where Collectives Live

The key architectural difference between eager and compiled modes is where collectives sit relative to the autograd graph:

### Eager: collectives are OUTSIDE autograd, checkpoint controls lifetime

```
  ╔══ checkpoint_wrapper (per layer) ═══════════════════════════════════╗
  ║                                                                     ║
  ║  ┌─ outside autograd (torch.no_grad) ─────────────────────────┐    ║
  ║  │                                                             │    ║
  ║  │  pre_forward_hook:                                          │    ║
  ║  │    Placement.unshard()  ──→  dist.all_gather()  ──→ full_W  │    ║
  ║  │                                                             │    ║
  ║  └─────────────────────────────────────────────────────────────┘    ║
  ║                                                         │           ║
  ║                                             detach().requires_grad_ ║
  ║                                                         │           ║
  ║                                                         ▼           ║
  ║  ┌─ inside autograd ──────────────────────────────────────────┐    ║
  ║  │                                                             │    ║
  ║  │  full_W (leaf)  ──→  F.linear(input, full_W)  ──→  output  │    ║
  ║  │                                                             │    ║
  ║  └─────────────────────────────────────────────────────────────┘    ║
  ║                                                                     ║
  ╚═════════════════════════════════════════════════════════════════════╝
          │                                                 │
          │ after layer forward:                            │
          │ checkpoint replaces full_W                      │
          │ with _Holder ──→ full_W freed by GC             │
          │                                                 │
          ▼                                                 ▼
  ┌─ backward ────────────────────────────────────────────────────────┐
  │                                                                    │
  │  checkpoint unpack_hook fires ──→ re-runs layer forward:           │
  │    pre_forward_hook fires again ──→ new all-gather ──→ new full_W  │
  │    autograd computes grad on new full_W                            │
  │    AccumulateGrad hook fires ──→ _reduce_fn:                       │
  │      Placement.reduce_grad() ──→ dist.reduce_scatter()             │
  │      ──→ write to sharded_param.grad                               │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘
```

Autograd sees `full_W` as an ordinary leaf tensor — no knowledge of all-gather or reduce-scatter. Checkpoint controls `full_W`'s lifetime: discards it after each layer's forward, triggers re-all-gather during backward recomputation. This achieves per-layer memory release without autograd being aware of FSDP.

### JIT / AOT: collectives are INSIDE autograd

```
  ┌─ inside autograd ───────────────────────────────────────────┐
  │                                                              │
  │  sharded_W (leaf Parameter)                                  │
  │       │                                                      │
  │       ▼                                                      │
  │  _c10d_functional.all_gather_into_tensor(sharded_W)          │
  │       │                                                      │
  │       ▼                                                      │
  │  _c10d_functional.wait_tensor(full_W)                        │
  │       │                                                      │
  │       ▼                                                      │
  │  F.linear(input, full_W)  ──→  output                       │
  │                                                              │
  │  backward auto-generates:                                    │
  │    reduce_scatter_tensor(grad)  ──→  sharded_W.grad          │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

Autograd sees the all-gather as a differentiable op. The graph pass marks it `MUST_RECOMPUTE` so the compiler frees `full_W` after forward and re-all-gathers in backward.

## Parametrization: The Shared Front End

### How it works

`_register_parametrization()` creates a dynamic subclass of each leaf module's class with property descriptors. When `module.weight` is accessed, the property getter checks for a batched all-gather result first, falling through to the per-param parametrization if none:

```python
# Each layer is wrapped: checkpoint_wrapper(layer, context_fn=reshard_policy)
# checkpoint discards full_W after forward, re-creates via recomputation in backward

class FlexShardLinear_1(nn.Linear):
    @property
    def weight(self):
        # Eager only: _pre_gathered set by pre_forward_hook
        # (hook skips under torch.compiler.is_compiling(), so pre is always None in JIT/AOT)
        pre = parametrization._pre_gathered
        if pre is not None:
            # detach(): disconnect from batched AG (outside autograd graph)
            # requires_grad_(True): make it a leaf so autograd computes grad,
            #   which AccumulateGrad hooks capture for batched reduce-scatter
            # checkpoint treats this leaf like any other tensor — discards it
            #   after layer forward, re-creates it via re-all-gather in backward
            return pre.detach().requires_grad_(True)
        # JIT/AOT: per-param all-gather via _c10d_functional (traced into FX graph)
        return parametrization(self._parameters['weight'])
```

`state_dict()` reads `self._parameters` directly (bypasses the property), so checkpoints always see sharded tensors.

### Placement-specific parametrizations

| Class | Forward collective | Backward | Post-processing |
|-------|-------------------|----------|-----------------|
| `ShardParametrization` | `_c10d_functional.all_gather_into_tensor` | autograd `reduce_scatter_tensor` | chunk+cat (dim!=0), narrow (uneven) |
| `FlatShardParametrization` | `_c10d_functional.all_gather_into_tensor` | autograd `reduce_scatter_tensor` | view to original shape |
| `OwnedParametrization` | `_c10d_functional.broadcast` via `_OwnedBroadcast` | custom `all_reduce` / world_size | none |
| `RaggedShardParametrization` | `_c10d_functional.all_gather_into_tensor` | autograd `reduce_scatter_tensor` | chunk+narrow+cat per rank |

These are functional collectives (`torch.ops._c10d_functional.*`) — async, FX-traceable, with autograd support. The parametrization `forward()` containing these ops is the code that `torch.compile` traces into the FX graph.

**However, eager mode bypasses these entirely.** The property getter short-circuits before calling `parametrization.forward()`:
- The pre_forward_hook calls `Placement.unshard()` → `dist.all_gather()` (synchronous, per-bucket batched)
- The property getter returns the pre-gathered result as a detached leaf — `parametrization.forward()` never runs
- `AccumulateGrad` hooks call `Placement.reduce_grad()` → `dist.reduce_scatter_tensor()` (synchronous, per-bucket batched)

So in practice: **JIT/AOT use `_c10d_functional` (per-param, async). Eager uses `dist.*` (per-bucket, synchronous).**

## Reshard-After-Forward

Goal: **free unsharded parameter memory after each layer's forward, recompute in backward**.

### Eager mode

In eager mode, collectives are **outside** the autograd graph. Three mechanisms work together:

**Forward path** (per layer):

1. **Pre-forward hook** (`_install_batched_allgather_hooks`): calls `Placement.unshard()` — one batched `dist.all_gather` per bucket. Stashes the result on `_pre_gathered`.
2. **Property getter**: sees `_pre_gathered` is set, returns `pre.detach().requires_grad_(True)` — a detached leaf. The per-param `parametrization.forward()` (which calls `_c10d_functional` ops) is **never called** in eager.
3. **Post-forward hook**: registers `AccumulateGrad` hooks on each detached leaf for gradient capture.

**Reshard** (between layers):

Each layer is wrapped in `checkpoint_wrapper` with `_flex_shard_reshard_policy` (`_apply_reshard_checkpoint`). The policy marks collective ops as `MUST_RECOMPUTE`, everything else as `PREFER_RECOMPUTE`. Checkpoint discards all intermediates after each layer's forward — including the detached-leaf unsharded params. No explicit reshard call is needed; checkpoint's discard-and-recompute handles it.

If activation checkpointing (AC) is also applied, the two policies are merged into a single `checkpoint_wrapper`: FlexShard collectives → `MUST_RECOMPUTE`, AC compute ops → `MUST_SAVE`, everything else → `PREFER_RECOMPUTE`.

**Backward path** (per layer):

1. Checkpoint re-runs the layer's forward: the pre-forward hook fires again, doing another batched all-gather.
2. Autograd computes grad on the recomputed detached leaf.
3. `AccumulateGrad` hooks fire → `_reduce_fn` calls `Placement.reduce_grad()` — one batched `dist.reduce_scatter_tensor` per bucket, writing to `sharded_param.grad`.

**Key checkpoint interaction**: `_unsharded_for_reduce` is only stored on the FIRST forward (not during checkpoint recomputation). The original leaf is what checkpoint saves and autograd accumulates grad on. The recomputed leaf is a different object used for the recomputed backward graph.

### JIT / AOT (compiled modes)

In compiled modes, collectives are **inside** the autograd graph. The mechanism is entirely different:

**Forward path** (per layer):

1. **Property getter**: `_pre_gathered` is always `None` (batched hooks are disabled under `torch.compiler.is_compiling()`).
2. **`parametrization.forward()` runs**: calls `_c10d_functional.all_gather_into_tensor` / `broadcast` / `wait_tensor` per param. These are functional collectives that `torch.compile` traces into FX `call_function` nodes.
3. No pre/post-forward hooks run — the compiler sees the full op graph.

**Reshard** (between layers):

A **graph pass** (`flex_shard_reshard_after_fwd_pass`) annotates unshard node sequences with `CheckpointPolicy.MUST_RECOMPUTE`. The Inductor min-cut partitioner uses these annotations to free unsharded params after forward and recompute in backward. No `checkpoint_wrapper` is used.

- JIT: pass registered via `functorch_config` joint graph passes.
- AOT: same pass, applied through `get_joint_custom_passes_from_config()`, registered as `"flex_shard_reshard_after_fwd"` in `AVAILABLE_JOINT_PASSES`.

**Backward path** (per layer):

Autograd sees the all-gather as a differentiable op. The backward of `_c10d_functional.all_gather_into_tensor` auto-generates `reduce_scatter_tensor` — per-param, not batched. The compiler rebatches these via bucketing passes.

### Summary

| | Eager | JIT / AOT |
|---|---|---|
| **All-gather** | `dist.all_gather` (batched per bucket, outside autograd) | `_c10d_functional.all_gather_into_tensor` (per param, inside autograd) |
| **Reduce-scatter** | `dist.reduce_scatter_tensor` (batched per bucket, AccumulateGrad hooks) | `_c10d_functional.reduce_scatter_tensor` (per param, autograd-generated) |
| **Reshard mechanism** | `checkpoint_wrapper` + selective policy | Graph pass + min-cut partitioner |
| **Parametrization.forward()** | Skipped (short-circuited by `_pre_gathered`) | Runs (traced into FX graph) |
| **AC composition** | Merged policy in single `checkpoint_wrapper` | Separate graph pass (naturally composable) |

## Graph Pass Pattern Recognition

The reshard pass recognizes these FX node sequences (used in compiled modes):

```
Shard(0):           placeholder → [_to_copy] → all_gather → wait_tensor
                    → [narrow] → [convert_element_type]

Shard(dim!=0):      placeholder → [_to_copy] → all_gather → wait_tensor
                    → chunk → getitem(0..N) → cat → [narrow]
                    → [convert_element_type]

FlatShard:          placeholder → [_to_copy] → all_gather → wait_tensor
                    → view → [convert_element_type]

Owned:              placeholder → [_to_copy] → broadcast → wait_tensor
                    → [convert_element_type]
```

## Design Decisions

### Why parametrization over hooks?

Parametrization emits `_c10d_functional` ops into the graph, making communication visible to the compiler for reshard annotation, communication scheduling, and overlap optimization. Hooks are opaque to FX tracing.

### Why batched collectives in eager?

Per-param `_c10d_functional` ops emit one NCCL kernel per param (57 AllGathers for a 6-layer model). Batching via `Placement.unshard()` / `reduce_grad()` emits one NCCL kernel per bucket (8 AllGathers + 8 ReduceScatters). In compiled modes, the compiler rebatches per-param ops automatically via bucketing passes.

### Why `AccumulateGrad` hooks for reduce-scatter?

`register_hook` on detached leaf params fires when autograd computes their grad — guaranteed correct timing. Previous approaches (`queue_callback`, `_RegisterPostBackward` on inputs) failed because:
- `queue_callback` fires after the ENTIRE backward, losing per-layer timing
- `_RegisterPostBackward` on inputs doesn't work for modules with no-grad inputs (e.g., Embedding receives Long indices)
- `register_hook` on outputs fires BEFORE param grads are computed

`AccumulateGrad` hooks on the detached leaves fire at exactly the right time. Grads are captured from the hook argument (not `.grad`, which is None at hook time).

## Verified Behavior

Profiler traces confirm batched collectives across all modes:

| Mode | AllGather | ReduceScatter | Mechanism | Profiler Trace |
|------|----------|---------------|-----------|----------------|
| Eager | 17 | 8 | Batched `Placement.unshard()` / `reduce_grad()` | https://fburl.com/dt2ljemb |
| JIT (inductor) | 113 | 57 | Per-param, compiler rebatches | https://fburl.com/1qmg8vmh |
| AOT | 114 | 57 | Per-param, compiler rebatches | https://fburl.com/ss9bqtbv |

Convergence matches across all modes (loss ≈ 5.6-5.7 at step 5).

Repro commands (`batch=8, seq_len=6144, 4 GPUs`):
```bash
# Eager
NGPU=4 MODULE=graph_trainer.flex_shard_llama3 CONFIG=graph_trainer_flex_shard_llama3_debugmodel \
  ./run_train.sh --training.local_batch_size 8 --training.seq_len 6144 \
  --compile.mode None --profiling.enable_profiling --profiling.save_traces_folder profile_eager

# JIT
NGPU=4 MODULE=graph_trainer.flex_shard_llama3 CONFIG=graph_trainer_flex_shard_llama3_debugmodel \
  ./run_train.sh --training.local_batch_size 8 --training.seq_len 6144 \
  --compile.mode jit --compile.backend inductor --profiling.enable_profiling --profiling.save_traces_folder profile_jit

# AOT
NGPU=4 MODULE=graph_trainer.flex_shard_llama3 CONFIG=graph_trainer_flex_shard_llama3_debugmodel \
  ./run_train.sh --training.local_batch_size 8 --training.seq_len 6144 \
  --compile.mode aot --profiling.enable_profiling --profiling.save_traces_folder profile_aot
```

## Appendix

### Why store `_unsharded_for_reduce` only on first forward?

Checkpoint recomputation creates NEW detached leaves, but autograd accumulates grad on the ORIGINAL leaf (the one checkpoint saved via `_Holder`). If `_unsharded_for_reduce` were overwritten during recomputation, `_reduce_fn` would read from the recomputed leaf (which has no grad).

### Composition with SAC

Reshard and SAC use the same `CheckpointPolicy` annotation mechanism. They compose:
- SAC marks activations `PREFER_RECOMPUTE` / `MUST_SAVE`
- Reshard marks unshard sequences `MUST_RECOMPUTE` (overrides SAC)
- `ac_graph_id = 100000` prevents partitioner from treating reshard nodes as part of a user AC region
