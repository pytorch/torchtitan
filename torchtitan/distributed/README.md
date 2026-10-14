# Distributed

This package holds the PyTorch-native parallelisms and the training techniques
that have to compose with them. Not every module issues collectives: activation
checkpointing, CUDA graphs and local compile run per device, but they live here
because each one interacts with sharding, pipeline schedules or SPMD regions and
is applied during model parallelization.

## Parallelisms and communication

| Module | Purpose |
| --- | --- |
| `parallelism_context.py` | Device mesh construction and named mesh axes for every parallelism. See [PARALLELISM_CONTEXT.md](PARALLELISM_CONTEXT.md). |
| `spmd_types.py` | SPMD type annotations for tensor-parallel and other sharded regions. |
| `fsdp.py` | FSDP2 / HSDP application. See [FSDP.md](FSDP.md). |
| `pipeline_parallel.py` | Pipeline stages and schedules. See [PIPELINE_PARALLEL.md](PIPELINE_PARALLEL.md). |
| `context_parallel.py` | Context-parallel sequence sharding. |
| `deepep/` | DeepEP / HybridEP expert-parallel token dispatch. |
| `flex_shard/` | Optimizer-compute layouts that differ from parameter storage (DistMuon). See [flex_shard/README.md](flex_shard/README.md). |
| `utils.py` | Process-group initialization, distributed reductions, determinism, gradient clipping. |

## Techniques that compose with parallelism

| Module | Purpose |
| --- | --- |
| `activation_remat.py` | Activation remat (activation checkpointing) configs on `torch_remat`. See [REMAT.md](REMAT.md). |
| `cuda_graph.py` | CUDA graph capture and replay of the training step. |
| `local_compile.py` | `torch.compile` of named model regions. |
| `batch_invariant.py` | Batch-invariant (bitwise-reproducible) kernels. |
| `offloading.py` | NUMA binding of each rank to the CPU node local to its GPU. |

## First optimizer update

`Optim.step()` releases unused CUDA cache on its first call, before gradient
clipping. Each new `Optim` instance tracks its own first call, so a restored
step number does not skip the release. It waits for GPU work to finish before
releasing cache. Both calls run outside optimizer CUDA graph capture.

NCCL allocates its internal communication buffers outside PyTorch's caching
allocator. Unused PyTorch cache can block those allocations even when the live
tensors and NCCL buffers would fit after releasing the cache. See
[issue #5019](https://github.com/pytorch/torchtitan/issues/5019).

This workaround requires:

- Forward and backward to complete successfully.
- Enough unused CUDA cache that can be released for the NCCL buffers.
- The affected NCCL allocation to happen during the first optimizer call.

The release keeps live tensors and active CUDA graph pools. It does not reduce
NCCL buffer sizes or prevent allocation failures during forward or backward,
or later in the run. It does not release NCCL host shared memory.

## Environment settings

**`TORCH_NCCL_AVOID_RECORD_STREAMS=1` with tensor parallelism.** TP issues async
collectives to overlap communication with compute. By default the process group
calls [`Tensor.record_stream`](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)
on their inputs and outputs, which ties the caching allocator's memory reuse to
GPU kernel timing: while a collective runs, later allocations cannot reuse its
tensors' memory even when the ops that need them are ordered after it, so memory
stacks up. With this variable set, the process group instead holds references to
the tensors until `wait()` is called. That only regresses memory if `wait()` is
never called, which is not an expected usage.

**`CUDA_DEVICE_MAX_CONNECTIONS` with FSDP2 and native TP.** Set it to the number of
CUDA streams (for example 16 or 32) so compute and NCCL kernels can overlap. This
is the opposite of Megatron-style TP, which typically requires
`CUDA_DEVICE_MAX_CONNECTIONS=1`; do not copy that setting.
