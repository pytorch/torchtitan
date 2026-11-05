---
name: autokernel
description: Coordinate an agent swarm to optimize GPU kernel problem sets in CuTe DSL through operator fusion, schedule tuning, and profiling, while preserving bitwise parity and the complete input/output contract. Use when given kernel problems and reference implementations, or a substitute-kernels `cache_dir`, to optimize on available GPUs.
---

# Autokernel

Given a kernel problem set and reference implementations, spawn a coordinated
swarm of named agents to hillclimb kernel performance in CuTe DSL on available
GPUs. Minimize memory traffic through fusion, tune schedules for the actual
hardware, and use profiler evidence to work toward the applicable roofline.
Bitwise parity and the original input/output contract are hard acceptance gates.

## Substitute-kernels problem sets

When the problems come from a GraphTrainer `cache_dir` (written by
`substitute_with_kernels_pass` in `graph_trainer/substitute_kernels.py`):

1. **Ask the user for `cache_dir` before doing anything else.** Do not guess or
   fall back to `$TORCHTITAN_SUBSTITUTE_KERNELS_DIR` or the Inductor cache directory
   without confirmation. Only the coordinator writes into `cache_dir`, and only
   `<kernel_name>` files (see 3).
2. **The problem set is every specialization in `cache_dir`.** Each
   `pat_<l1>/specialization_<l2>.json` is one problem with ID `pat_<l1>/<l2>`:
   - `pat_<l1>/reference.py` is the immutable reference; its `run(*args)`
     defines the semantics. Specializations under one `pat_<l1>/` share it.
   - The spec's `signature`, `inputs` and `outputs` are the contract: build
     inputs with exactly the listed shapes, strides, dtypes, devices and
     values, and return outputs with exactly the listed structure and
     metadata. The entrypoint is `run(*args)`, not `kernel(reference, *args)`:
     the kernel does not receive the reference.
   - The kernel file is `pat_<l1>/<kernel_name>` from the spec. If it already
     exists, it is the current best implementation to beat; otherwise beat
     `reference.py`. It is imported as a standalone module, so a candidate
     must be a single self-contained file with no relative imports.
   - Never modify `reference.py` or the spec files. Writers, reviewers and
     verifiers work only in the run directory.
3. **Every claimed win gets a newly spawned verifier.** Each time a writer or
   reviewer reports a win, spawn a fresh verifier agent for that submission, one
   per submission, never the agent that wrote or reviewed it. On a leased GPU,
   the verifier independently:
   - loads `reference.py` and the candidate in a fresh process and checks that
     the candidate's `run` matches the spec's `signature`;
   - runs the exact correctness gate below on inputs built from the spec, with
     multiple seeds, and checks every output's structure, shape, stride, dtype
     and device against the spec's `outputs`;
   - times the candidate against the current `<kernel_name>` (or `reference.py`
     if none exists) under the measurement rules below;
   - records a pass/fail verdict with evidence on the board.

   Replace `pat_<l1>/<kernel_name>` with the candidate only when the verifier
   passes it and it is measurably faster than the current implementation
   (beyond run-to-run noise). The coordinator alone does the replacement: copy
   the previous file, if any, into the run directory for rollback, write the candidate to a temporary file in the same
   directory, then `os.replace` it onto `<kernel_name>`. A failed or
   inconclusive verification leaves `cache_dir` untouched and goes back to the
   writer.

The rest of this skill applies unchanged; for these problems, `reference.py` is
the reference and `<kernel_name>` is the canonical implementation file.

## Establish the contract and baseline

- Read the problem manifest, references, callsites, supported shapes, dtypes,
  strides, aliasing, mutations, and forward/backward requirements. Reuse existing
  deduplicated problem IDs so repeated occurrences share one optimization task.
- Keep the reference immutable. Record its source hash, environment, GPU model,
  math settings, and chosen backend algorithms. Establish repeatable reference
  results and timings before optimizing. A nondeterministic reference blocks an
  exact-parity claim; investigate it without substituting an easier reference.
- Inspect available GPUs and the installed CuTe DSL, CUDA, and profiling tools.
  Use GPUs available within the task's resource scope. Match the actual hardware
  and preserve existing workload and resource limits.
- Preserve the callable signature, input set, output tree and ordering, tensor
  shapes and dtypes, device placement, and required layouts, aliases, and mutation
  behavior. Do not remove saved intermediates that are outputs of the region.
  When the entrypoint is `kernel(reference, *args)`, retain that convention.
- For compiled attention or other runtime closures, use the live reference
  harness. A shared function name or matching shapes do not establish semantic
  equivalence. Resolve missing runtime dependencies before assigning such work.

## Coordinate the swarm

You are the coordinator. Split and prioritize problems, assign named writers and
independent cross-reviewers, collect their results, and reassign work as evidence
changes. Every agent reports back to you, including failed trials and blockers.
Prioritize aggregate workload impact using occurrence counts and critical-path
timings when available, rather than isolated kernel latency alone.

Use these collaboration mechanisms together:

| Mechanism | Required behavior |
| --- | --- |
| Coordinator | Own assignments, GPU allocation, handoffs, and promotion of reviewed winners (verified winners for substitute-kernels problems). |
| Shared task board | Agents claim work and post findings in one shared file so they can see progress and avoid duplicate work. |
| Stage handoff | A writer's candidate and evidence become a different reviewer's inputs; review findings feed the next writing iteration. |
| Named agents and messages | Reuse agents' context with the available messaging tool, such as `SendMessage` or `collaboration.send_message`; use a follow-up task to resume an idle agent. |

Create a run directory all agents can access, such as `/tmp/swarm/<run-id>/` on a
shared host. Initialize `board.md` from [assets/board.md](assets/board.md). Include
the board path, reference path, problem ID, candidate directory, and GPU assignment
in every agent brief. Messages accelerate coordination; record durable decisions
and evidence on the board as well.

All board claims and updates must use a shared lock, such as `flock` on
`board.lock`. Hold the lock for the entire read-modify-write operation and replace
the board atomically. Claim problems and GPU leases together when needed. Do not
reclaim a lease until the coordinator confirms the previous owner has stopped.

Lease each benchmarking or profiling GPU exclusively. Writers without a GPU can
inspect code or prepare candidates while another agent measures. Size the swarm
to the available GPUs and useful review work; concurrent timings on the same GPU
are not comparable.

Keep trials in separate directories such as
`candidates/<problem-id>/<agent-name>/<trial-id>/`. Give competing hypotheses
separate board rows and candidate directories under the same problem ID. The
coordinator alone promotes reviewed winners into the problem's canonical
implementation file (such as `kernel.py`) and its supporting files; for
substitute-kernels problems, only verified winners replace `<kernel_name>`.

## Hillclimbing loop

1. **Profile and form a hypothesis.** Identify the dominant cost and record the
   proposed change and expected effect on the board. Use Nsight Compute (`ncu`)
   or available equivalent tools to inspect memory traffic, tensor-core/SM
   utilization, occupancy, register pressure, spills, and stall reasons. Select
   relevant kernels and sections rather than profiling the entire workload on
   every trial. If counters are unavailable, record that limitation and use the
   measurements that are available; do not invent utilization results.
2. **Write a CuTe DSL candidate.** Fuse compatible operations to eliminate
   intermediate loads/stores. Keep required outputs and reference rounding
   points, even when intermediates remain in registers. Tune tile shapes,
   layouts/swizzles, warps, pipeline stages, asynchronous transfers, tensor-core
   instructions, and register/shared-memory use for the target GPU. A region may
   require multiple launches to preserve dependencies or its output contract.
3. **Pass exact correctness.** Follow the gate below before treating a candidate
   as a performance contender. Diagnose any mismatch before further tuning.
4. **Measure against the best valid implementation.** Exclude JIT compilation and
   warmup, synchronize correctly, and use the same inputs, cache conditions,
   device, and execution mode. Measure repeated unprofiled runs with CUDA events
   or the workload's CUDA-graph harness; report the median and variability.
   Profiler replay timings are diagnostic, not the final speedup measurement.
5. **Hand off for cross-review.** Give a different agent the candidate source and
   hashes, exactness results, reproduction commands, timings, and profiler
   artifacts. The reviewer inspects the implementation and independently reruns
   correctness and timing on a leased GPU. Return mismatches, unsupported cases,
   and gains indistinguishable from noise to the writer.
6. **Promote and reassign.** Preserve the best reviewed, exact candidate. Integrate
   it through the existing entrypoint and validate all affected callsites and
   gradients. Reprofile the workload after meaningful changes: eliminating one
   bottleneck can move the critical path. Assign the next experiment accordingly.

## Exact correctness gate

Use identical inputs for reference and candidate, preserving input alias
relationships and resetting mutable state between executions. Cover production
cases, multiple seeds, relevant numerical edge cases, and supported shape/tile
boundaries. Check every output and, where required, every input/parameter gradient
using the same upstream gradients. For joint forward/backward regions, gradient
tensors are part of the output contract.

Check structure and metadata separately from values. For dense tensors, compare
the logical elements as bytes, which also distinguishes signed zero and NaN
payloads:

```python
reference_bits = reference.detach().reshape(-1).contiguous().view(torch.uint8)
candidate_bits = candidate.detach().reshape(-1).contiguous().view(torch.uint8)
assert torch.equal(reference_bits, candidate_bits)
```

Check non-tensor outputs exactly and verify required aliases and mutation effects.
Repeat identical executions to detect nondeterminism. Tolerance-based comparisons
or a more accurate result cannot replace bitwise parity with the given reference.
Do not relax the gate to accept a faster candidate.

Fusion and schedule changes can alter reduction order, accumulation precision,
FMA contraction, intermediate casts, or transcendental approximations. Match the
reference's behavior at these boundaries. If parity prevents an optimization,
record the failed approach and keep the best exact implementation.

## Roofline objective and completion

Estimate a problem-specific lower bound using required arithmetic and compulsory
memory traffic, including all outputs that the contract requires:

```python
compute_floor_s = required_flops / attainable_flops_per_second
memory_floor_s = compulsory_bytes / attainable_bytes_per_second
roofline_floor_s = max(compute_floor_s, memory_floor_s)
```

Use ceilings appropriate to the GPU, precision, and instruction path, distinguishing
measured attainable rates from advertised peaks. Account for launch overhead,
synchronization, instruction throughput, and other profiler-observed limits when
the simple compute/memory model does not explain performance.

Continue the measured, profile-guided search toward that applicable roofline.
Honor supplied budgets and resource limits. Claim roofline performance only with
supporting measurements; if a budget, unavailable resource, or demonstrated
exactness constraint prevents reaching it, return the best verified candidate
with the remaining gap and blocker. A plateau alone is not evidence of roofline
performance.

Finish with the canonical implementations, shared board, reproducible validation
and benchmark commands, profile artifacts, and a per-problem table of reference
time, best exact time, speedup, tested coverage, and remaining bottleneck. Distinguish
isolated kernel gains from measured whole-workload gains. For training validation,
run at least 10 steps when measuring performance. Never describe an unchanged
reference-delegating stub as an optimized CuTe DSL implementation.
