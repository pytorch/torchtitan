---
name: auto-fuser
description: Optimize GraphTrainer regions through kernel fusion with parallel implementation agents, preserving bitwise and memory parity while targeting the local GPU roofline.
---

# Auto Fuser

You are responsible for reaching the local GPU kernel roofline through kernel
fusion while preserving bitwise parity and memory parity with the original
model. Roofline performance is a best-effort target; the original implementation
is the minimum acceptable performance.

Apply this skill to the captured training graph, including forward, backward,
and optimizer operations when present. The coordinator owns region discovery,
fusion plans, integration, and acceptance. Delegate kernel implementation for
every fusion plan to parallel subagents.

## Acceptance criteria

Every kernel, fusion plan, and integrated graph must satisfy the same criteria:

1. **Bitwise parity.** Match the original implementation's output and gradient
   bit patterns exactly. Preserve observable shapes, dtypes, layouts, aliasing,
   mutations, and RNG state transitions. Model loss, grad_norm, parameter
   updates, and optimizer state must remain bitwise identical where applicable.
   Tolerance-based comparisons and rounded stdout values are insufficient.
2. **Memory parity.** Do not increase peak GPU memory or persistent and
   saved-for-backward storage relative to the original implementation. Removing
   intermediates is allowed. Include scratch buffers, copies, padding, retained
   tensors, and CUDA graph pools; moving allocations outside the timed region
   does not remove their cost. Preserve the configured checkpointing and
   offloading policy.
3. **Performance.** Approach the local roofline as closely as practical. A
   replacement must be at least as fast as the original region under matched
   conditions, and integration must not regress training-step performance.
   Confirm differences within measurement noise with repeated measurements.

Bitwise and memory parity are hard gates. Speed never compensates for a failure
of either gate.

## Establish the original reference

- Pin the model, recipe, input data, hardware, software versions, parallelism,
  precision, compilation, and CUDA graph settings. Use
  `config.debug.seed = 42` and `config.debug.deterministic = True`; never set
  `config.debug.deterministic_warn_only = True`.
- Keep an immutable original implementation as the numerical, memory, and
  performance reference. Reset model, optimizer, input, and RNG state before
  each comparison. Do not replace the reference with a previously fused graph.
- Capture the graph and profile representative training steps. Record region
  latency, launch count, and memory traffic, plus full-step latency and peak
  allocated and reserved GPU memory on every participating rank.
- Use the target workload's real shapes, strides, dtypes, device placements,
  mesh axes, and execution mode. Warm up compilation and CUDA graph capture
  before measuring steady state. Performance runs must use at least 10 training
  steps, with warmup excluded from the reported measurements.

## Partition into candidate fusible regions

Assume GEMMs, attention, and communication kernels are already well optimized.
Treat them as opaque partition boundaries: preserve their implementations,
input/output contracts, and synchronization dependencies. Do not fuse through
them or absorb operations into their internals.

Partition the remaining computation by data dependencies, not textual node
adjacency. A candidate region must execute without moving or including a
boundary operation. Split regions where needed to respect aliasing, mutation,
RNG, or synchronization ordering. Account for branches and shared consumers;
every value used outside a region remains part of its output contract.

For each region, record its operators, external inputs and outputs, observed
shape/dtype/stride variants, saved tensors, original kernel sequence, latency,
and memory footprint. Prioritize regions by their contribution to step time and
avoidable launches or memory traffic.

## Propose fusion plans

For each candidate region, propose plausible fusion plans, including partial
fusion when full fusion would increase register pressure, spills, or workspace.
Keep the original implementation as the fallback candidate.

Each plan must specify the operator groups and resulting kernels, launch and
tiling strategy, reduction and accumulation order, intermediate rounding points,
external outputs, saved tensors, and scratch-memory requirements. Explain the
expected reduction in launches or memory traffic and the likely bottleneck.

Estimate a local roofline using necessary memory traffic and achievable
throughput for the actual operations and hardware:

```python
roofline_time_s = max(
    required_bytes / achievable_bandwidth_bytes_per_s,
    required_flops / achievable_compute_flops_per_s,
)
```

Use bandwidth for the relevant memory level and throughput for the actual
instruction mix; GEMM tensor-core throughput is not a scalar-kernel roofline.
Account separately for launch latency, synchronization, and reduction or
special-function limits. Record assumptions and measured distance from the
estimate instead of claiming roofline performance from a speedup alone.

## Delegate implementation in parallel

Dispatch independent fusion plans to parallel subagents, one plan per agent,
up to available concurrency. Give each agent:

- The region graph and original executable reference, including all external
  inputs, outputs, and supported shape/dtype/stride variants.
- Its fusion plan, target hardware, roofline estimate, and baseline timings
  and memory measurements.
- The three acceptance criteria, deterministic setup, numerical checks, and
  benchmark commands.
- An isolated worktree or output path and an explicit file scope. Keep
  experiment-specific changes within GraphTrainer; reuse applicable existing
  implementations and avoid new dependencies.

Agents must not edit shared integration files or logs concurrently. The
coordinator integrates their patches. Parallelize implementation; run GPU
benchmarks concurrently only on disjoint, exclusively assigned GPUs. Otherwise
serialize GPU measurements to avoid contention invalidating comparisons.

Each agent returns its implementation patch, supported cases, bitwise-check
results, memory measurements, original and fused timings, roofline gap, and
remaining bottlenecks. Failed plans must report which criterion failed and why.

## Validate, integrate, and iterate

1. **Check each kernel against the original region.** Cover the workload's
   forward and backward paths, shape variants, reduction sizes, and tail cases.
   Compare raw tensor bit patterns, including signed zero and NaN payloads when
   present. Higher-precision oracles can diagnose errors but cannot replace the
   original implementation as the bitwise reference.
2. **Investigate mismatches before tuning.** Fusion can eliminate storage but
   must preserve the original rounding points, casts, reduction tree,
   accumulation precision, math operations, and FMA contraction behavior.
   Deterministic execution alone does not establish bitwise parity. Fix the
   cause or reject the plan; never relax tolerances to accept it.
3. **Check memory and performance.** Measure isolated regions and complete
   training steps under equivalent allocator, warmup, and capture conditions.
   Include forward-to-backward tensor lifetimes and allocations outside the
   kernel. Report allocated and reserved peaks. Time GPU execution with CUDA
   events or synchronized measurements, using the deployment's CUDA graph mode.
4. **Integrate qualifying plans and revalidate the model.** Compare against the
   immutable original model, including full-precision loss and grad_norm, all
   relevant gradients, parameter updates, and optimizer state. Use
   `scripts/loss_compare.py` for full-precision training metrics where applicable.
   Recheck memory on every rank and end-to-end timing: locally faster kernels
   can still hurt overlap or extend tensor lifetimes.
5. **Tune from evidence.** Reprofile the integrated graph and pursue the measured
   bottleneck with revised plans delegated to subagents. Keep a validated best
   implementation and retain the original path for any region or supported case
   without a qualifying replacement. Finish when regions have validated choices
   or documented reasons to retain the original, or the run's budget is reached.

Record per-region plans, patches, parity evidence, memory deltas, timings,
roofline estimates, and keep/discard decisions in `experiment_log.md`. Distill
reusable findings in `learnings.md`; record end-to-end measurements in
`results.tsv` using its existing schema. The final report must identify selected
fusions, retained original regions, measured speedups, and unresolved roofline
gaps without presenting unverified candidates as accepted results.
