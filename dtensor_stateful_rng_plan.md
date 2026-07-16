# DTensor Stateful RNG Init Plan

## Goal

Preserve single-device PyTorch initialization numerics when TorchTitan initializes
DTensor parameters directly under strict SPMD execution.

Assumptions for the first implementation:

- all ranks run the same parameter traversal;
- all ranks make the same init calls in the same order;
- tensor-parallel DTensor local shards are contiguous slices in the dense
  flattened tensor space supported by the PyTorch DTensor RNG handlers;
- single-rank dense initialization followed by sharding is the test oracle, not
  the runtime implementation.

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

## Resolved Blocker

The TorchTitan parity test initially could not import torch because the editable
PyTorch install pointed at `/data/users/weif/code-review/pytorch`, but that
source checkout no longer matched the compiled extension built from `14753d46`.
Rebuilding PyTorch from the trimmed DTensor RNG commit resolved the mismatch.
