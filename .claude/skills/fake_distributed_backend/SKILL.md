---
name: fake_distributed_backend
description: Run TorchTitan with fully fake communication or real pipeline communication plus fake SPMD axes, especially for logical-rank memory debugging. Use for fake process groups, real-PP/fake-SPMD runs, large-model dry runs, or comparing rank-local CUDA memory ownership without launching the full logical world.
---

# Fake Distributed Backend

TorchTitan provides two distinct test modes:

- `fake` makes every distributed axis fake. Use it for configuration,
  shape, ownership, and rank-local allocation analysis without real transport.
- `real_pp_fake_spmd` uses one physical process per pipeline rank. PP
  traffic is real; data, tensor, context, and expert parallel axes are fake.
  Use it when pipeline communication and buffer lifetimes must be exercised.

Neither mode replaces a real distributed numerical or performance test.
The authoritative user-facing launch contract, including complete command
examples and environment-variable semantics, is in
[`docs/debugging.md`](../../../docs/debugging.md#fake-backend-debugging).

## Logical Rank Coordinates

Set `NGPU` to the logical world size. For pure fake PP, also set:

```bash
export FAKE_PP_RANK=<logical PP coordinate>
```

For PP degree `P`, the non-PP logical world size is `NGPU / P` and the global
logical rank is:

```text
FAKE_PP_RANK * (NGPU / P)
```

Both modes represent SPMD coordinate zero. Pure fake defaults to PP coordinate
zero when PP is disabled. In hybrid mode, physical `RANK` is the PP coordinate
and `WORLD_SIZE` must equal the PP degree; setting `FAKE_PP_RANK` is invalid.

`COMM_BACKEND=fake` is a convenience understood by `run_train.sh` for a
single-process pure-fake run. Launch hybrid mode with `torchrun`, set `NGPU` to
the logical world size, and pass
`--comm.backend real_pp_fake_spmd`; `torchrun` supplies the physical
`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, and rendezvous variables.

## Memory Debugging Workflow

1. Use the production model, recipe, dtype, parallel degrees, microbatch count,
   FSDP policy, activation checkpointing policy, and CUDA-graph setting.
2. Choose the PP coordinate whose SPMD-zero ownership is being investigated.
3. Capture allocator state after initialization, after complete optimizer
   warmup, during steady-state forward/backward, and after optimizer completion.
4. Compare the same logical rank and capture point between configurations.
5. Escalate in order: pure fake -> real PP/fake SPMD -> real distributed run.

Fake execution represents PyTorch-managed model parameters, optimizer state,
prepared quantized weights, activations, gradients, pipeline buffers, and
DistMoE scratch or activation arenas. It does not faithfully represent NCCL
communicator allocations, network registration, transport scratch, collective
latency, or communication overlap.

Use `torch.cuda.memory_snapshot()` for allocation provenance and
`torch.cuda.max_memory_allocated()` / `torch.cuda.max_memory_reserved()` for
comparable totals. A fake run can establish deterministic ownership and
lifetime behavior; it cannot establish distributed numerical parity or
throughput.

## Failure Interpretation

- A divisibility or coordinate error is a launch-contract failure; fix the
  logical topology rather than changing model configuration.
- A hybrid world-size error means the physical allocation does not contain
  exactly one process per PP rank.
- A pure-fake success followed by a hybrid failure usually isolates PP
  transport, send/receive ownership, or communicator initialization.
- A hybrid success followed by a real-run failure usually isolates an SPMD
  collective, real communication memory, or scale-dependent scheduling issue.
- Do not compare different PP coordinates as evidence of a memory regression;
  their model and activation ownership can differ intentionally.
