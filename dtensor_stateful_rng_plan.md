# DTensor Stateful RNG Init Plan

## Goal

Preserve single-device PyTorch initialization numerics when TorchTitan initializes
DTensor parameters directly under strict SPMD execution.

Assumptions for the first implementation:

- all ranks run the same parameter traversal;
- all ranks make the same init calls in the same order;
- tensor-parallel DTensor local shards are contiguous slices in the dense
  flattened tensor space supported by the PyTorch DTensor RNG handlers.

## Chosen Contract

Use PyTorch's stateful DTensor Philox replay path. Do not introduce a
TorchTitan-level parameter-FQN RNG stream for this path.

For each DTensor random init call, PyTorch should:

1. start from the shared CUDA RNG generator state;
2. treat the logical DTensor as the dense tensor;
3. fill local shards from dense global flat offsets;
4. advance the generator by the full dense tensor's RNG increment.

TorchTitan should keep using stock stateful `nn.init` callables so the dense and
DTensor paths consume the same global RNG stream under strict SPMD.

## Current Solution

The solution is split between PyTorch and TorchTitan.

PyTorch owns the runtime behavior:

- `torch/distributed/tensor/_ops/_random_ops.py` registers custom DTensor
  handlers for `aten.normal_.default` and `aten.uniform_.default`.
- On CUDA, those handlers read the shared Philox seed/offset, compute the local
  shard's dense flattened start offset, call internal CUDA ops for the local
  dense-equivalent slice, and then advance the generator by the full dense
  tensor's RNG increment.
- The dense-equivalent replay contract is CUDA-only. CPU and other non-CUDA
  DTensor random init keep existing local-op behavior and are outside the
  single-device numerics guarantee.
- `aten/src/ATen/native/native_functions.yaml` defines internal generic
  `_philox_normal_dense_slice_` and `_philox_uniform_dense_slice_` ops. Their
  schemas contain no DTensor concepts: they accept the logical dense element
  count and the flat start of the output slice. Torchgen creates their functional
  and out variants as required for functionalization.
- `aten/src/ATen/native/cuda/PhiloxDistribution.cu` implements those ops by
  replaying the dense CUDA distribution kernel's Philox mapping. Each local
  output element is generated from its logical dense global flat index, not from
  its local shard index.

TorchTitan owns the integration contract and regression test:

- TorchTitan keeps model initialization on stock stateful `nn.init` callables.
  It does not add parameter-scoped seeds or a TorchTitan-side FQN RNG stream for
  this path.
- `tests/unit_tests/test_dtensor_stateful_rng_init.py` compares direct DTensor
  initialization with dense single-device initialization followed by gathering.
  It covers `nn.init.normal_`, `nn.init.uniform_`, and `nn.init.trunc_normal_`.
  `trunc_normal_` is covered through its stateful uniform draw plus deterministic
  elementwise transforms. The test uses multi-block tensors and also compares
  the generator state and next random draw after initialization.

The important invariant is generator-state parity. Under strict SPMD, every rank
enters the same init calls in the same order. For each DTensor init, PyTorch
consumes RNG as if the dense logical tensor were initialized on one CUDA device,
while each rank writes only its local shard.

## Validation Contract

Single-rank dense initialization followed by sharding is the test oracle, not the
runtime implementation. The runtime initializes DTensor shards in place, and the
test compares the gathered DTensor result against the dense single-device result.

## Supported Shape And Mesh Scope

The current PyTorch handler supports contiguous local shards in dense flattened
tensor space. For the TorchTitan test and target tensor-parallel initialization
path, this means a 1D CUDA mesh with `Shard(0)` over tensors whose remaining
dimensions are fully local on each rank.

The handler explicitly rejects CUDA local shards that are not contiguous slices
of the dense flattened tensor. That keeps unsupported shard layouts from
silently producing different numerics. Replicated tensors and empty local shards
follow the same handler path when their local tensor constraints are satisfied.

## Non-Goals

- Preserve numerics when ranks do not execute the same init calls in the same
  order.
- Preserve legacy full-model RNG streams for pipeline-parallel chunks that skip
  parameters owned by other chunks.
- Support every possible DTensor placement. Additional placements need separate
  dense-index mapping and tests.
- Introduce TorchTitan FQN-derived seeds. That remains useful for non-SPMD or
  model-surgery resilience, but it is not needed for the strict-SPMD contract and
  would change the stateful single-device RNG stream.

## Iterations

- [x] Capture the design decision in this markdown plan.
- [x] Remove the current TorchTitan FQN/stateless-init prototype from the active
  branch.
- [x] Restore model configs and HF backend initialization to stock stateful
  `nn.init` callables.
- [x] Add targeted TorchTitan tests that compare dense init against direct
  DTensor init for representative stateful initializers.
- [x] Run CPU/static checks that do not require the rebuilt PyTorch checkout.
- [x] After the PyTorch checkout is aligned with a rebuilt PyTorch commit, run
  DTensor/GPU parity tests.

## Deferred Work

Pipeline-parallel chunks do not satisfy the strict-SPMD traversal assumption
because each chunk may see only a subset of parameters. If we need PP parity with
legacy full-model single-device RNG streams, add a separate replay plan that
collects the full model init order and consumes RNG for skipped parameters.

## Execution Log

- `git diff --check`: passed.
- `python -m py_compile tests/unit_tests/test_dtensor_stateful_rng_init.py`:
  passed.
- Initial `python tests/unit_tests/test_dtensor_stateful_rng_init.py`: blocked
  at `import torch` by the PyTorch source/build mismatch described below.
- After rebuilding PyTorch from the trimmed DTensor RNG commit,
  `python tests/unit_tests/test_dtensor_stateful_rng_init.py`: passed.
- `ninja -C build install -j 96` in the PyTorch checkout: passed.
- PyTorch `lintrunner` on the four touched source and test files: passed.
- `python test/distributed/tensor/test_random_ops.py -k stateful_init`: two
  tests passed.
- The expanded TorchTitan test: three tests passed.
- Targeted TorchTitan pre-commit hooks passed with `pyrefly-check` skipped.
  Pyrefly still fails on existing repository-wide missing dependencies and
  baseline type errors unrelated to this change.

## Resolved Blocker

The TorchTitan parity test initially could not import torch because the editable
PyTorch install pointed at `/data/users/weif/code-review/pytorch`, but that
source checkout no longer matched the compiled extension built from `14753d46`.
Rebuilding PyTorch from the trimmed DTensor RNG commit resolved the mismatch.
