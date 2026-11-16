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
   without confirmation. A `cache_dir` given in the invoking prompt counts as
   confirmed. Only the coordinator writes into `cache_dir`, and only the
   `<l2>.py` files of the problem set (see 3).
2. **The problem set is every `pat_<l1>/<l2>.py` file in `cache_dir`**, or, when
   the invoking prompt names a subset (problem IDs `pat_<l1>/<l2>` or whole
   `pat_<l1>` directories), those. Read anything in `cache_dir`, but never write
   outside the problem set: other runs may own the rest, including other files in
   the same `pat_<l1>/` directory, and may replace them while you read. Each
   `pat_<l1>/<l2>.py` is one problem with ID `pat_<l1>/<l2>`:
   - Its `SPEC` (`signature`, `inputs`, `outputs`) is the contract: build inputs
     with exactly the listed shapes, strides, dtypes, devices and values, and
     return outputs with exactly the listed structure and metadata.
   - Its `reference` function is the immutable reference and defines the
     semantics. Problems under one `pat_<l1>/` share the computation's structure
     but may differ in shapes, dtypes and constants, so a kernel written for
     another problem is a starting point that must pass the gate on this one.
   - `run(*args)` is the entrypoint the training run calls; the kernel does not
     receive the reference. While the file says `run = reference`, it has no
     kernel and the target is `reference`; otherwise `run` is the current best
     implementation to beat. The kernel goes into `impl` (and any helpers), and
     `run = impl` makes it the entrypoint.
   - The file is executed as a standalone module, so it must be self-contained
     with no relative imports. A candidate is a full copy of the problem file:
     the generated `SPEC` block and `reference` function unchanged, `impl`
     written, and `run = impl`.
   - Never modify `SPEC` or `reference`. Writers, reviewers and verifiers work
     only in the run directory.
3. **Every claimed win gets a newly spawned verifier.** Each time a writer or
   reviewer reports a win, spawn a fresh verifier agent for that submission, one
   per submission, never the agent that wrote or reviewed it. On a leased GPU,
   the verifier independently:
   - loads the problem file and the candidate in a fresh process and checks that
     the candidate's `run` matches the spec's `signature` and that the candidate
     keeps the problem's `SPEC` and `reference` unchanged;
   - runs the exact correctness gate below on inputs built from the spec, with
     multiple seeds, and checks every output's structure, shape, stride, dtype
     and device against the spec's `outputs`;
   - times the candidate against the problem file's current `run` (the incumbent,
     `reference` if there is no kernel yet) under the measurement rules below;
   - records a pass/fail verdict with evidence on the board.

   Replace `pat_<l1>/<l2>.py` with the candidate only when the verifier passes it
   and it is faster under the performance acceptance criterion below. The
   coordinator alone does the replacement: copy the previous file into the run
   directory for rollback, write the candidate to a temporary file in the same
   directory as `<l2>.py`, then `os.replace` it onto `<l2>.py`. A failed or
   inconclusive verification leaves `cache_dir` untouched and goes back to the
   writer. Append each replacement to `promotions.md` in the run directory:
   problem ID, old -> new sha256, incumbent and candidate median us, verifier.

**Performance acceptance criterion.** The metric is the median of 50 replays of a
CUDA graph that captures one call of `run(*args)`, each replay timed separately with
CUDA events, after warmup. The verifier measures, in one process on one GPU:

- **Alternating rounds:** incumbent (A) and candidate (B) in 3 rounds, ordered
  A B A B A B, one metric value per side per round. Alternation cancels slow drift
  (clocks, temperature, neighbours).
- **A/A noise:** the same 3 rounds with the incumbent against a second copy of
  itself loaded as a separate module. The A/A spread is the largest absolute
  difference between the two copies' per-round medians.

The candidate is faster only if it beats the incumbent in all 3 rounds and its
improvement (median over rounds of A minus B) exceeds the A/A spread. Record the
per-round values and the A/A spread in the verdict and in `promotions.md`. Eager,
pipelined-eager, kernel-only, profiler and multi-call-per-graph numbers are
diagnostics: they can explain a result but never justify a promotion.

The criterion gates promotion only, not exploration:

- Writers may build on any exact candidate, promoted or not, and stack gains that
  are individually within noise until the combination passes.
- Exact candidates that miss the criterion stay on the board with their numbers,
  as starting points for others, instead of being discarded.
- A structural rewrite may start slower than the incumbent while it is tuned; judge
  progress by the best exact candidate, not only by the promoted one.

The rest of this skill applies unchanged; for these problems, `reference` in
`<l2>.py` is the reference and `<l2>.py` is the canonical implementation file.

## Invocation

The invoking prompt may fix any of these up front; whatever it fixes counts as
the user's answer, and the run proceeds without asking:

| Parameter | Meaning |
| --- | --- |
| `cache_dir` | the substitute-kernels cache (or problem manifest) |
| problem subset | the problem IDs (`pat_<l1>/<l2>`) or `pat_<l1>` directories to tune, if not all of `cache_dir` |
| run directory | where the board, candidates, backups and `promotions.md` live |
| GPUs | the GPUs this run may lease; with one GPU, one agent holds the lease at a time and the others write code meanwhile |
| stop condition | when to wrap up, in addition to the completion criteria below |

If nobody is available to answer questions, make the reasonable choice, record
it on the board, and continue. Environment limits stated in the prompt (missing
profilers, unavailable real inputs) go into the run contract and into evidence.
If the run directory already holds a board, continue that run instead of starting
a new one. When the run ends for any reason, bring the board and `promotions.md`
up to date and write `summary.md` in the run directory: per problem, reference us,
incumbent us at the start of the run, best verified us, what changed, and what
remains.

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
substitute-kernels problems, only verified winners replace `<l2>.py`.

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
   or the workload's CUDA-graph harness; report the median and variability. For
   substitute-kernels problems, acceptance uses the performance acceptance criterion
   above.
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
