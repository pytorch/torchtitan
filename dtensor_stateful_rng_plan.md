# DTensor Stateful RNG Init Plan

## Goal

Meta-device initialization can materialize parameters as DTensors before their
random initialization runs. Stateful random ops on those DTensors should produce
the same logical full tensor and generator progression as initializing one tensor
on a single CUDA device under strict SPMD execution.

Assumptions for the first implementation:

- all ranks run the same parameter traversal;
- all ranks make the same init calls in the same order;
- tensor-parallel DTensor local tensors are contiguous and use one `Shard(i)`
  placement on a 1D mesh.

## Chosen Contract

Use PyTorch's stateful DTensor Philox replay path. Do not introduce a
parameter-FQN RNG stream for this path.

For each DTensor random init call, PyTorch should:

1. start from the shared CUDA RNG generator state;
2. treat the logical DTensor as one full tensor;
3. fill each local tensor from its logical flat indices;
4. advance the generator by the full tensor's RNG increment.

Callers keep using stock stateful `nn.init` callables so the single-device and
DTensor paths consume the same global RNG stream under strict SPMD.

## Current Solution

PyTorch routes stateful DTensor initialization through a generic ATen flat-slice
operation. The normal initialization path is:

```text
nn.init.normal_
  -> aten.normal_ on a DTensor
  -> _normal_dtensor_handler
  -> _run_dtensor_local_rng_op(
       flat_slice_op_call=aten._philox_normal_flat_slice_.default)
  -> _run_stateful_rng_op
  -> flat_slice_op_call(
       local_tensor, total_numel, start_indices, block_sizes,
       block_strides, num_blocks, ...)
  -> ATen CUDA dispatch
  -> _philox_normal_flat_slice_cuda_
  -> distribution_flat_slice
  -> distribution_flat_slice_kernel
```

The layers have distinct responsibilities:

1. `pytorch/torch/distributed/tensor/_ops/_random_ops.py` registers
   `_normal_dtensor_handler` for `aten.normal_`. The handler passes
   `aten._philox_normal_flat_slice_.default` into the shared
   `_run_dtensor_local_rng_op` helper.
2. `_run_dtensor_local_rng_op` synchronizes the default CUDA generator when
   needed and describes the local output with `RNGIndexBlock` values. The
   blocks map the contiguous local tensor into the flattened logical tensor
   without exposing placements to ATen.
3. `pytorch/torch/distributed/_stateful_rng.py` defines `RNGIndexBlock`, the
   tensor metadata protocol, the shared `_run_stateful_rng_op` helper, and the
   scoped mode used by plain tensor producers.
4. `flat_slice_op_call` is a local alias for one of two ATen overloads:
   `aten._philox_normal_flat_slice_.default` or
   `aten._philox_uniform_flat_slice_.default`.
5. `pytorch/aten/src/ATen/native/native_functions.yaml` defines those generic
   operators and dispatches CUDA calls to `_philox_normal_flat_slice_cuda_` or
   `_philox_uniform_flat_slice_cuda_`. Their schemas contain no DTensor, mesh,
   rank, or shard concepts.
6. `pytorch/aten/src/ATen/native/cuda/PhiloxDistribution.cu` selects the normal
   or uniform sampler, then calls the shared `distribution_flat_slice` helper.
   That helper computes the launch policy from `total_numel`, advances the
   generator by the full tensor's reservation, and launches
   `distribution_flat_slice_kernel`.
7. The CUDA helper reserves RNG once for `total_numel`, then launches each
   descriptor against that state. Within a descriptor, the kernel maps each
   `local_idx` to its full-tensor logical index:

   ```text
   block_number = local_idx / block_size
   index_in_block = local_idx % block_size
   logical_index = start_index + block_number * block_stride + index_in_block
   ```

   It then reproduces the Philox thread, iteration, and lane that the
   full-tensor CUDA launch would have used.

The uniform path is identical except that `_uniform_dtensor_handler` binds
`flat_slice_op_call` to `aten._philox_uniform_flat_slice_.default`. An explicit
generator under LocalTensor simulation briefly routes through `_run_flat_slice`,
which then invokes the same ATen operation for each simulated rank from a common
captured generator state.

The DTensor adapter emits one descriptor for `Shard(i)` or `Replicate()` on a
1D CUDA mesh. Other producers can emit multiple descriptors through the
placement-independent protocol below.

## Placement-Independent Protocol

```python
@dataclass(frozen=True)
class RNGIndexBlock:
    start_index: int      # First global flat index
    block_size: int       # Consecutive elements per block
    block_stride: int     # Distance between global block starts
    num_blocks: int


@runtime_checkable
class StatefulRNGTensor(Protocol):
    rng_global_numel: int
    rng_index_blocks: tuple[RNGIndexBlock, ...]
```

Mappings consume the contiguous local tensor sequentially, so no local offset
is needed. For `(2, 4)` sharded on dimension 1, the second rank reports:

```python
RNGIndexBlock(
    start_index=2,
    block_size=2,
    block_stride=4,
    num_blocks=2,
)
```

This supports:

- `Shard(i)`: one block descriptor.
- Ragged contiguous ranges: one descriptor.
- Owned/expert ranges: one or several descriptors.
- Arbitrary custom placements: several descriptors, with a one-element block
  as the universal fallback.
- Empty ranks: no descriptors, but `rng_global_numel` still reserves the full
  dense increment.

The CUDA op consumes all descriptors, reserves Philox exactly once using
`rng_global_numel`, and launches every block from that shared state. It remains
placement-agnostic.

Attributes cannot intercept operations on a plain `torch.Tensor`. DTensor uses
its existing custom handlers; other tensor producers use a scoped
`TorchDispatchMode` during initialization to detect this protocol. That supports
direct `normal_`, `uniform_`, and `trunc_normal_` without tensor subclasses or
placement concepts in ATen.
