---
description: Rules for distributed training code
globs: torchtitan/distributed/**
---

# Distributed Code Rules

## Assert Mesh and Placements Explicitly
- Never assume a 1D mesh. Always assert on mesh axes before using them.
  (``axis``/``axes`` names a specific ``DeviceMesh`` axis like TP or
  ``dp_shard``; ``dim``/``dimensional`` describes the mesh's shape, e.g.
  "1D mesh" or "multi-dimensional mesh". Bare ``dim`` is reserved for
  tensor dimensions.)
- Validate tensor placements (Replicate, Shard, Partial) explicitly.
- When enforcing a field is not None for plain tensor inputs, do so with a clear
  error message.

## Document Parallelism Semantics
For any usage of `DTensor.to_local`, specify explicitly the `grad_placements`, especially when the original DTensor placement has `Replicate` or `Partial`. This includes the indirect calls from `local_map` (`in_grad_placements`), and `full_tensor` (`grad_placements`).

## Consider All Parallelism Combinations
When adding or modifying distributed code, think through how it interacts with
every parallelism dimension. A fix for one configuration may break another.
Include the parallelism configuration in bug reports and test descriptions.

## Model-Agnostic Code Lives Here
Helper functions that apply to any model (e.g. `maybe_enable_async_tp`,
`NoParallel`, tensor parallel utilities) belong in `torchtitan/distributed/`,
not in model-specific infrastructure files.

## Be Conservative with Changes
Distributed training code is hard to test exhaustively. When modifying existing
behavior:
- Verify numerics match before and after across multiple parallelism configs.
- Watch for silent correctness issues (wrong gradient placements, identity
  operations that break DCP).
- If changing something that's been converged and validated, provide strong
  justification and thorough testing.
