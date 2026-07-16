# DTensor Stateful RNG Init Plan

## Goal

Meta-device initialization can materialize parameters as DTensors before their
random initialization runs. Stateful random ops on those DTensors should produce
the same logical full tensor and generator progression as initializing one tensor
on a single CUDA device under strict SPMD execution.

Assumptions for the first implementation:

- all ranks run the same parameter traversal;
- all ranks make the same init calls in the same order;
- tensor-parallel DTensor local shards are contiguous slices in the flattened
  logical tensor space supported by the PyTorch DTensor RNG handlers.

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
  -> flat_slice_op_call(local_tensor, total_numel, start_index, ...)
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
   needed and describes the local output as a flat interval of the logical full
   tensor: `total_numel` is the full element count and `start_index` is the first
   logical element written by the local tensor.
3. `flat_slice_op_call` is a local alias for one of two ATen overloads:
   `aten._philox_normal_flat_slice_.default` or
   `aten._philox_uniform_flat_slice_.default`.
4. `pytorch/aten/src/ATen/native/native_functions.yaml` defines those generic
   operators and dispatches CUDA calls to `_philox_normal_flat_slice_cuda_` or
   `_philox_uniform_flat_slice_cuda_`. Their schemas contain no DTensor, mesh,
   rank, or shard concepts.
5. `pytorch/aten/src/ATen/native/cuda/PhiloxDistribution.cu` selects the normal
   or uniform sampler, then calls the shared `distribution_flat_slice` helper.
   That helper computes the launch policy from `total_numel`, advances the
   generator by the full tensor's reservation, and launches
   `distribution_flat_slice_kernel`.
6. The kernel maps each `local_idx` to
   `logical_index = start_index + local_idx`, then reproduces the Philox thread,
   iteration, and lane that the full-tensor CUDA launch would have used.

The uniform path is identical except that `_uniform_dtensor_handler` binds
`flat_slice_op_call` to `aten._philox_uniform_flat_slice_.default`. An explicit
generator under LocalTensor simulation briefly routes through
`_run_flat_slice_with_generator`, which then invokes the same ATen operation for
each simulated rank from a common captured generator state.

Exact replay currently targets 1D CUDA meshes whose local tensors form
contiguous flat intervals, such as `Shard(0)`; other devices and layouts retain
existing DTensor RNG behavior.
