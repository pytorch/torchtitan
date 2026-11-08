---
description: Rules for distributed training code
globs: torchtitan/distributed/**
---

# Distributed Code Rules

## Mesh Axes and Placements
- Never assume a 1D mesh or a specific axis order (e.g. "PP is always outermost").
  (``axis``/``axes`` names a specific ``DeviceMesh`` axis like TP or
  ``dp_shard``; ``dim``/``dimensional`` describes the mesh's shape, e.g.
  "1D mesh" or "multi-dimensional mesh". Bare ``dim`` is reserved for
  tensor dimensions.)
- SPMD semantics are expressed with spmd_types (`torchtitan/distributed/spmd_types.py`)
  and declarative `ShardingConfig`s. Rely on spmd_types type checking
  (`config.debug.spmd_typechecking`) instead of hand-written placement asserts
  where it covers the case. `spmd.local_map` regions declare their `in_types`.
- Don't add new DTensor-specific code paths (`isinstance(x, DTensor)`,
  `.to_local()`, `DTensor.from_local`) in model forward or new features; FSDP's
  internal DTensor use is the exception.
- Where DTensor conversions remain, specify `grad_placements` explicitly for
  `to_local` / `full_tensor` / `local_map` (`in_grad_placements`), especially when
  the original placement is `Replicate` or `Partial`.
- Be careful with local vs global semantics: a value named `global_*` must be
  global on every rank, and a collective over several mesh axes must not leave
  ranks with different values.

## Consider All Parallelism Combinations
When adding or modifying distributed code, think through how it interacts with
every parallelism axis (FSDP/HSDP, TP/SP, PP including interleaved schedules,
CP, EP) and with gradient accumulation, activation remat, CUDA graphs, and
checkpoint resume. A fix for one configuration may break another. Include the
parallelism configuration in bug reports and test descriptions.

## Performance Hazards
- No hidden device-to-host syncs in the training step (`.item()`, `.tolist()`,
  `.cpu()`, `repeat_interleave` without `output_size`, data-dependent shapes).
- Code in the training step should stay CUDA-graph capturable and
  compile-friendly; don't do per-layer work that can be done once.
- Don't regress memory or throughput of the main path (e.g. FSDP-only bf16).

## Hardware
- Use `torch.accelerator` / the device module, not `torch.cuda.*`, in generic
  code. No NCCL- or vendor-specific tuning knobs in core.

## Model-Agnostic Code Lives Here
Helper functions that apply to any model belong in `torchtitan/distributed/`,
not in model-specific infrastructure files.

## Be Conservative with Changes
Distributed training code is hard to test exhaustively. When modifying existing
behavior:
- Verify numerics match before and after across multiple parallelism configs.
- Watch for silent correctness issues (wrong gradient placements, identity
  operations that break DCP).
- If changing something that's been converged and validated, provide strong
  justification and thorough testing.
