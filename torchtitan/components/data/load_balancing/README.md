# Optimizer-Step Load Balancing

Packed text microbatches can contain the same number of tokens but require
different amounts of attention work. For document segments of lengths `L_i`,
the current cost model estimates work as:

```text
sum(L_i ** 2)
```

`LoadBalancingDataLoader` reduces DP stragglers by reassigning complete
microbatches within one optimizer step.

## Flow

```text
                         one optimizer step
                                  |
                       LoadBalancingDataLoader
                                  |
                 +----------------+----------------+
                 | coordinator.collect(...)        |
logical input -->| - read candidate payloads       |
streams          | - assign stable IDs             |
                 +----------------+----------------+
                                  |
                   payload + stable ID
                                  |
                       itemize callback
                 +----------------+----------------+
                 | packing adapter                 |
                 | - inspect concrete microbatch   |
                 | - produce packing metadata      |
                 |                                 |
                 | cost model                      |
                 | - estimate work from metadata   |
                 +----------------+----------------+
                                  |
                            PackableItem
                                  |
                 coordinator collects all items into
                         CoordinatedWindow
                 +----------------+----------------+
                 | PackableItem metadata and bins  |
                 | locally available payloads      |
                 +----------------+----------------+
                                  |
                           balancer.plan(...)
                            (metadata only)
                                  |
                             assignments
                                  |
                 +----------------+----------------+
                 | coordinator.distribute(...)     |
                 | - realize the assignments       |
                 | - order this rank's payloads    |
                 +----------------+----------------+
                                  |
                    local microbatches for the step
                                  |
                               Trainer
```

The four responsibilities are deliberately separate:

- The loader orchestrates one complete optimizer step and records metrics.
- The adapter and cost model turn each concrete microbatch into a small
  `PackableItem`. They inspect data but do not mutate it.
- The balancer sees only metadata and chooses assignments.
- The coordinator controls which inputs are visible and delivers the assigned
  payloads. It also owns child-loader state and cleanup.

The coordinator receives `itemize` as a callback because it owns input reads,
while the adapter owns knowledge of packed text. This also supports a future
coordinator that must inspect local payloads before exchanging their metadata.

## Coordinator choices

The current `ReplicatedInputCoordinator` gets a shared candidate window by
reconstructing every logical input stream in its balancing group. For a group
containing logical ranks 0 and 1:

```text
physical rank 0 reads logical streams 0 and 1
physical rank 1 reads logical streams 0 and 1
```

Both processes independently produce the same deterministic plan, then each
returns only the microbatches assigned to its physical rank. Reads are
replicated; training samples are not duplicated.

An exchange-based coordinator could use the same interface:

```text
each rank reads its own stream
        |
collect: exchange PackableItem metadata
        |
the same balancer produces assignments
        |
distribute: exchange only payloads that changed owner
```

That alternative would avoid replicated reads and candidate storage, but would
add collectives, variable-size payload transfer, and more complex checkpoint
semantics. It is not currently implemented.

## Semantics

The implementation balances whole, already-collated microbatches. Within each
balancing group and optimizer step, its output is exactly the same microbatch
multiset that the ordinary `GrainDataLoader` streams would have produced. Only
the DP-rank and execution-slot assignment may change.

It does not split documents, repack document segments, or move data between
optimizer steps. The original tensors and document boundaries are preserved.
Training is not guaranteed to be bitwise identical because changing rank or
execution order can change random-number consumption and floating-point
reduction order.

For effective DP degree `D`, group size `G`, and physical DP rank `r`, the
replicated coordinator uses this contiguous group of logical streams:

```text
group_start = (r // G) * G
logical ranks = group_start, ..., group_start + G - 1
```

The group size must divide `D`. With `G=1`, the same machinery performs local
reordering without cross-rank reassignment.

## Planning objective

`TokenizedTextPackingAdapter` derives document lengths from resets in
`positions` and reports token count, document count, and payload size. The
whole-microbatch planner groups similarly expensive work into the same gradient
accumulation step, then uses capacity-constrained LPT to balance the total PP
microbatch cost assigned to each DP rank. It compares the candidate and
ordinary schedules by, in order:

1. The sum, over accumulation steps, of the maximum DP-rank total PP cost.
2. The worst total-cost skew between DP ranks in one accumulation step.
3. Payload bytes assigned to a different rank.

For multiple ranks, an exact tie retains the ordinary assignment. A single-rank
group instead uses the deterministic heavy-first candidate so independently
reordered groups align expensive work in the same accumulation steps. PP
indices are assigned deterministically but are not treated as synchronization
boundaries.

## Example

Consider two DP ranks and two accumulation slots. Every microbatch has eight
tokens, but its document lengths differ:

| Microbatch | Ordinary location | Document lengths | Cost |
| --- | --- | --- | ---: |
| A | rank 0, slot 0 | `[8]` | 64 |
| B | rank 0, slot 1 | `[4, 4]` | 32 |
| C | rank 1, slot 0 | `[2, 2, 2, 2]` | 16 |
| D | rank 1, slot 1 | `[1, 1, 1, 1, 1, 1, 1, 1]` | 8 |

| Plan | Slot 0 `(rank 0, rank 1)` | Slot 1 `(rank 0, rank 1)` | Total synchronized cost |
| --- | --- | --- | ---: |
| Ordinary | `(A: 64, C: 16)` | `(B: 32, D: 8)` | 96 |
| Balanced | `(A: 64, B: 32)` | `(D: 8, C: 16)` | 80 |

The balanced plan swaps complete microbatches B and D. It still trains on
exactly A, B, C, and D, but reduces the sum of per-slot maximum costs from 96
to 80.

## Configuration

Wrap the recipe's existing `GrainDataLoader.Config`:

```python
from torchtitan.components.data import (
    GrainDataLoader,
    LoadBalancingDataLoader,
    ReplicatedInputCoordinator,
)


def llama3_varlen_balanced():
    config = llama3_debugmodel_varlen_attn()
    child = config.dataloader
    assert isinstance(child, GrainDataLoader.Config)

    config.dataloader = LoadBalancingDataLoader.Config(
        dataloader=child,
        mode="shadow",
        coordinator=ReplicatedInputCoordinator.Config(group_size=2),
    )
    return config
```

Start with `mode="shadow"` to measure the predicted benefit while returning
the ordinary assignment. Change it to `mode="balance"` to use the selected
plan. To disable the feature, configure `GrainDataLoader` directly.

Larger groups provide more balancing opportunities, but the replicated
coordinator performs approximately `G` times as many source reads and holds
approximately `G` times as much candidate data per process.

## Current support

Supported:

- A `GrainDataLoader.Config` child using the built-in `TextCollator`.
- Canonical `TokenizedTrainingMicrobatch` values on CPU.
- Packed text whose `positions` reset to zero at document boundaries.
- Gradient accumulation and pipeline-parallel microbatch layouts.
- Any positive `group_size` that divides the effective DP degree.
- `shadow` and `balance` modes.
- Checkpointing at completed optimizer-step boundaries.

Not currently supported:

- Splitting or repacking document segments.
- Moving candidates between groups or optimizer steps.
- Exchange-based coordination.
- Validation dataloaders.
- TorchFT, because dynamic membership changes logical-stream ownership.
- Finite, non-repeating data under DP, inherited from `GrainDataLoader`.
- Restoring with a different effective DP degree or group topology.

## Metrics

The loader reports one observation per optimizer step:

| Metric | Meaning |
| --- | --- |
| `data_load/source_fetch_ms` | Time spent reading candidate payloads. |
| `data_load/inspection_ms` | Time spent validating and extracting metadata. |
| `data_load/planner_ms` | Time spent constructing the plan. |
| `data_load/baseline_predicted_cost` | Predicted cost of the ordinary assignment. |
| `data_load/balanced_predicted_cost` | Predicted cost of the selected plan. |
| `data_load/moved_payload_bytes` | Bytes assigned to a different logical owner. |
| `data_load/candidate_host_bytes` | Candidate tensor bytes held by this process. |
| `data_load/unchanged_plans` | Plans that retained the ordinary assignment. |

In `shadow` mode, balanced cost and moved bytes describe the plan that would be
used in `balance` mode.

## Checkpoint behavior

The loader delegates state handling to its coordinator. The replicated
coordinator checkpoints every logical child-stream cursor, namespaced by
physical and logical DP rank. Restore currently requires the same effective DP
degree, physical rank, and balancing-group coordinates.

Only completed optimizer-step boundaries are checkpointed, so no partially
planned window is stored. The dataset, tokenizer, and packing configuration
are not fingerprinted; the caller must resume with a compatible input
pipeline, as with the ordinary loader.

## Extending

- Add a coordinator by subclassing `InputCoordinator` and its `Config`.
  `collect()` returns globally visible items and locally available payloads;
  `distribute()` realizes assignments and returns the local step in execution
  order. The coordinator also owns input state and communication resources.
- Add a cost model by subclassing `QuadraticAttentionCost`. `estimate()` must
  return a deterministic, nonnegative integer.
- Add a whole-microbatch planner by subclassing `WholeMicrobatchBalancer`.
  Keep it limited to `PackableItem`, `PackingBin`, costs, capacities, and stable
  IDs.
- Add a whole-payload adapter by subclassing
  `TokenizedTextPackingAdapter`. It defines how a concrete microbatch becomes
  planner metadata while remaining indivisible.

Segment-level repacking additionally needs a materializer that rebuilds output
tensors from assigned segments while preserving token/label alignment,
position resets, padding, document limits, and loss masking. Those mechanics
should stay outside the metadata-only planner.

## Tests

```bash
taskset -c 0-7 /home/jinsooihm/local/mywork/pytorch-env/bin/python -m pytest \
    tests/unit_tests/cpu/components/data/load_balancing -x
```
