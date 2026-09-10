# RFC: Schedule-Aware Auto-Partitioning for Pipeline Parallelism in TorchTitan

## Summary

This RFC describes the auto-partitioning infrastructure used by the autopartition experiment in TorchTitan.

The current implementation has two related pieces:

- `infra/autopipe.py`: a smaller historical baseline that bakes in several assumptions that make it difficult to generalize: a single-stage `1F1B`-style schedule view, a fixed microbatch regime tied to stage count, a global `COMM_OVERHEAD` constant, and no support for split-backward schedules such as zero-bubble.
- `infra/_partition.py`: the current main implementation, which lifts those hardcoded assumptions into more explicit schedule-aware abstractions by handling multiple schedule types, decoupling stage construction from a single fixed microbatch formula, modeling communication through schedule simulation rather than one global constant, and supporting split-backward style schedules used by zero-bubble.

The main goal of this document is to make the implementation reviewable without requiring reviewers to reconstruct the design entirely from a large source file. In particular, this RFC explains the problem setting, the main terminology, the intended review entry points, and the current modeling assumptions.

## Motivation

Pipeline-parallel training performance is highly sensitive to how model layers are assigned to pipeline stages.

The autopartition experiment aims to improve over static or hand-tuned splits by automatically producing a contiguous partition from layer cost estimates. The current implementation follows a staged approach:

1. construct a simple compute-balanced baseline partition,
2. evaluate that partition under a concrete pipeline schedule,
3. identify the stage that dominates steady-state execution,
4. explore nearby repartitioning candidates that may reduce the bottleneck.

This division keeps the implementation easier to reason about than a single monolithic global optimizer while still allowing schedule-aware refinement beyond a pure compute-balance heuristic.

## Goals

The goals of this work are:

- provide an automatic contiguous layer partitioner for pipeline parallelism,
- support multiple schedule types already used in pipeline parallelism,
- expose explicit schedule generation and simulation logic so the resulting behavior is debuggable,
- keep compatibility with a lightweight API that accepts per-layer forward and backward costs,
- preserve a clear boundary between baseline partition construction and schedule-aware refinement.

Concretely, the implementation currently supports:

- `1F1B`,
- `Interleaved1F1B`,
- `InterleavedZeroBubble`.

## Where the Core Logic Lives

The relevant files are:

- `torchtitan/experiments/autopartition/infra/_partition.py`
- `torchtitan/experiments/autopartition/infra/autopipe.py`
- `torchtitan/experiments/autopartition/infra/pipeline_parallel.py`

At the moment, the active integration path uses `auto_partition(...)` from `infra/_partition.py`. The import in `pipeline_parallel.py` points to `_partition.py`, while the older `autopipe.py` import is commented out. In that sense:

- `autopipe.py` is best viewed as a compact historical baseline and reference,
- `_partition.py` is the current main implementation that is actually used by the autopartition integration path.

## Suggested Review Order

The implementation is easier to review in the following order:

1. `partition_model_layers(...)`
2. `optimize_partition_model_layers(...)`
3. `identify_steady_critical_stage(...)`
4. `simulate_pipeline_schedule(...)`
5. compute-schedule generation helpers
6. action-graph simulation helpers
7. the compatibility wrapper `auto_partition(...)`

This order mirrors the intended design:

- start with baseline partition construction,
- then move to schedule-aware evaluation,
- then examine the heuristic search that uses that evaluation,
- and only then inspect the lower-level schedule and simulation details.

## Terminology

The implementation uses the following terms:

### Physical rank

A physical rank is one pipeline worker in the pipeline-parallel group.

### Logical stage

A logical stage is one partition unit in the schedule. For simple schedules there may be one logical stage per rank. For interleaved schedules, multiple logical stages may be colocated on the same physical rank.

### Stage partition

A stage partition is the contiguous set of layer indices assigned to a logical stage.

### Microbatch

A microbatch is one unit of input data that flows through the stage schedule during an iteration.

### Warmup, steady phase, and cooldown

The simulated iteration is naturally divided into three conceptual regions:

- warmup: the pipeline is being filled, so forward work is accumulating before all overlapping dependencies become active,
- steady phase: forward and backward work overlap in a repeating pattern,
- cooldown: the remaining backward work drains after the last relevant forward steps have been issued.

### Critical stage

The critical stage is the stage that most strongly determines the modeled iteration time. In the current implementation, the identification logic focuses on the steady-state region rather than treating warmup and cooldown effects as equally informative for repartitioning.

## Inputs and Outputs

The partitioning and simulation code works with stage-level abstractions derived from layer-level inputs.

### Main inputs

The core inputs include:

- per-layer forward FLOPs,
- per-layer backward FLOPs,
- communication volume, either as a scalar or per-boundary list,
- device compute throughput,
- network bandwidth,
- pipeline-parallel group size,
- number of microbatches,
- optional backward-input and backward-weight FLOPs for schedules that split backward work more explicitly.

These inputs are intentionally expressed in hardware-meaningful units:

- computation in FLOPs,
- communication in bytes,
- hardware throughput in FLOP/s and bytes/s.

This keeps the public inputs interpretable and allows the simulator to convert them into modeled time values internally.

### Main outputs

The main outputs include:

- contiguous stage partitions.

## High-Level Design

The implementation is organized as a sequence of increasingly schedule-aware phases.

### 1. Input validation and normalization

The first layer of logic validates basic invariants:

- array lengths must match,
- counts must be positive where required,
- communication inputs must be well-formed,
- schedule names must map to a known schedule type.

The code also normalizes hardware-related inputs by converting per-layer FLOPs and communication volumes into modeled durations when needed.

This stage is intentionally strict because most later logic assumes aligned dimensions and consistent stage counts.

### 2. Compute-normalized DP baseline partition

The baseline partitioner constructs a contiguous stage split by balancing total compute time across stages.

At a high level:

1. per-layer forward and backward FLOPs are converted into compute times using device throughput,
2. per-layer total compute cost is formed by summing forward and backward cost,
3. a contiguous block-partition DP is used to split layers across the required number of logical stages,
4. the resulting stage partition is aggregated into stage-level metadata for later simulation.

The baseline objective is deliberately simple: minimize the slowest stage under a compute-only contiguous partition model.

This is not the final partition objective for the entire system. Instead, it acts as a stable and interpretable starting point that is:

- easy to validate,
- easy to compare against the older `autopipe.py` logic,
- suitable for feeding into a more detailed schedule-aware evaluation pass.

### 3. Schedule-aware simulation

Once a candidate partition has been built, the simulator evaluates it under a concrete pipeline schedule.

At a high level, the simulator does the following:

1. build a compute schedule for the selected schedule type,
2. place logical stages onto physical ranks,
3. insert the required send and receive actions,
4. build a timing dependency graph over the resulting action sequence,
5. simulate execution to recover iteration time and one critical path.

This phase is the first point where schedule-specific ordering and communication behavior materially affect the cost model.

The key design choice here is that evaluation happens by explicit schedule construction and dependency simulation rather than by a single closed-form formula. That makes the implementation heavier than a simple analytic estimate, but it also makes it easier to inspect and debug behavior for different schedules.

### 4. Steady-phase critical-stage identification

The repartitioning heuristic does not simply use the stage with the largest total compute cost as its target. Instead, it identifies a stage that dominates the steady-state portion of the simulated schedule.

The rationale is that warmup and cooldown can produce transient bottlenecks that are less informative for deciding how layers should be reassigned. The code therefore tries to isolate the steady portion of the schedule and determine which stage accumulates the most critical-path time there.

This stage acts as the bridge between:

- a full action-level timing simulation,
- and a compact decision signal that the local search can use.

### 5. Local heuristic repartitioning around the critical stage

The optimization pass starts from the compute-balanced baseline and performs a bounded local search.

At a high level:

1. simulate the current partition,
2. identify the steady-phase critical stage,
3. generate nearby candidates by moving a boundary layer around that stage,
4. rebuild a valid contiguous candidate partition,
5. re-simulate the candidate,
6. keep improved results and continue exploring under a pruning rule.

The current heuristic is intentionally local rather than global. It aims to answer the practical question:

"Given the current bottleneck stage, is there a small adjacent boundary change that improves iteration time?"

This is a pragmatic tradeoff:

- more expressive than a one-shot compute-balanced split,
- much cheaper and easier to review than a global search over all contiguous partitions.

## Relationship Between `autopipe.py` and `_partition.py`

`autopipe.py` represents the earlier and smaller version of the idea. It is useful because it shows the core structure of the original approach in a compact form:

- compute-only per-stage cost aggregation,
- a DP baseline split,
- a heuristic search around a critical stage.

`_partition.py` generalizes that approach into the current implementation. Compared with `autopipe.py`, it adds:

- decoupled logical-stage construction for schedules beyond a single-stage `1F1B` view,
- support for variable stage/rank layouts instead of assuming one fixed microbatch formula,
- communication modeling through explicit schedule simulation rather than a single global constant,
- support for interleaved schedules,
- support for split-backward execution via backward-input and backward-weight modeling,
- richer public debugging and inspection entry points.

## Public Entry Points

The most important public or review-facing entry points in `_partition.py` are:

- `partition_model_layers(...)`
- `optimize_partition_model_layers(...)`
- `identify_steady_critical_stage(...)`
- `generate_compute_schedule(...)`
- `simulate_pipeline_schedule(...)`
- `auto_partition(...)`

These functions serve slightly different purposes:

- `partition_model_layers(...)` constructs the DP baseline partition,
- `optimize_partition_model_layers(...)` performs the schedule-aware heuristic search,
- `identify_steady_critical_stage(...)` extracts a stage-level bottleneck signal from simulation,
- `generate_compute_schedule(...)` exposes schedule construction for inspection and tests,
- `simulate_pipeline_schedule(...)` is the main timing-model entry point,
- `auto_partition(...)` provides a simpler compatibility wrapper for the autopartition integration path.

## Modeling Assumptions

The current implementation makes a number of explicit modeling choices.

### Contiguous partitions only

Each stage receives a contiguous block of layers. This keeps the split easy to understand and aligned with the existing pipeline-parallel integration assumptions.

### Schedule-aware evaluation

Partition quality is measured by simulation under a selected schedule, not just by total compute per stage.

### Hardware-aware normalization

FLOPs and communication bytes are translated into modeled time through device throughput and network bandwidth, keeping the user-facing inputs in interpretable units.

## Design Rationale

The implementation makes a few deliberate architectural choices that are worth calling out explicitly.

### Separate baseline partitioning from detailed schedule simulation

Rather than embedding all schedule effects directly into the partitioning objective, the code constructs a simple baseline first and then evaluates candidates through a richer simulator. This keeps the design layered:

- the baseline explains how the initial partition is chosen,
- the simulator explains how schedule behavior affects iteration time,
- the heuristic search explains how those two pieces interact.

This separation is a large part of what makes the system reviewable.

### Prefer explicit simulation over opaque scoring

The current design pays some implementation complexity to expose:

- schedule generation,
- communication insertion,
- timing-graph dependencies,
- critical-path reconstruction.

That extra machinery is valuable for debugging and for reviewer confidence. A reviewer can inspect not only the final partition, but also why the simulator believes a given stage or action is on the critical path.

## Conclusion

The current auto-partitioner is built around a simple principle:

- start from an interpretable contiguous compute-balanced baseline,
- evaluate it under an explicit schedule-aware timing model,
- identify the stage that dominates steady-state execution,
- apply a bounded local search to improve the result.

It is intended to be a practical and inspectable design for pipeline-parallel partitioning in TorchTitan.
