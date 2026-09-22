# Load-Balanced Dataloader Design

Date: 2026-09-22

## Problem

Packed variable-length attention gives each data-parallel (DP) rank the same
token count but potentially different work. For document segment lengths
`L_i`, full-attention work is approximately `sum(L_i ** 2)`, so a rank with
longer segments can become a straggler.

## Scope

The initial implementation reassigns complete, already-collated microbatches.
It does not split or repack document segments. For every optimizer step, the
global microbatch multiset is therefore unchanged; only rank and execution-slot
assignment may change.

The implementation supports:

- local reordering with `group_size=1`;
- replicated balancing with any positive group size that divides the effective
  DP degree;
- `shadow` mode, which computes metrics but returns the ordinary assignment;
- `balance` mode, which returns the balanced assignment; and
- checkpointing at optimizer-step boundaries.

Validation loaders and TorchFT are not supported initially.

## Optimizer-step interface

`OptimizerStepLayout` describes the number of gradient-accumulation and
pipeline-parallel microbatches in one optimizer step. `OptimizerStepBatch`
contains the corresponding accumulation-major, PP-minor nested list.

`BaseDataLoader.iter_optimizer_steps()` adapts an ordinary flat dataloader by
reading and grouping the same microbatches the trainer previously grouped.
`LoadBalancingDataLoader` overrides this method to plan the complete step.

The trainer obtains one complete `OptimizerStepBatch`, validates its shape,
and passes each group to the existing forward/backward path. This changes where
grouping happens, not when the ordinary dataloader is consumed: the old trainer
also read the complete group before starting computation.

## Replicated coordinator

For effective DP degree `D`, group size `G`, and physical rank `r`:

```text
group_start = (r // G) * G
logical ranks = group_start, ..., group_start + G - 1
```

Every physical process constructs one child `GrainDataLoader` for every logical
rank in its group. Each child receives the original `dp_world_size=D` and its
original logical `dp_rank`, preserving the ordinary sharding decision.

All group members independently read the same candidate streams and run the
same deterministic planner. No data or metadata collectives are introduced.
The supplied source, transforms, and tokenizer must therefore be deterministic
for the configured seed and logical DP coordinates.

Replication multiplies source reads and local loader resources by `G`. Choosing
an appropriate group size is an operational decision left to the user.

## Cost extraction and planning

`TokenizedTextPackingAdapter` validates the supported packed-text microbatch
representation and derives reset-delimited segment lengths from `positions`.
`QuadraticAttentionCost` assigns cost `sum(L_i ** 2)`.

The tensor-independent planner receives `PackableItem` and `PackingBin`
metadata. For the current whole-microbatch strategy it:

1. Builds the ordinary assignment.
2. Sorts microbatches by descending estimated cost.
3. Places similarly expensive microbatches in the same synchronized slot.
4. Uses stable IDs for deterministic tie-breaking.
5. Keeps the ordinary assignment unless the candidate objective is better.
6. Validates exact item coverage and bin capacities.

The objective minimizes synchronized cost, then worst DP skew, moved payload
bytes, and finally a stable tie-breaker.

The adapter/planner boundary is intentionally generic so a future strategy can
operate on document segments without coupling the planner to text tensors.

## Checkpointing

The wrapper checkpoints only child-loader cursor state and the topology needed
to interpret it:

```text
schema_version
effective_dp_degree
physical_dp_rank_N:
  balance_group_coordinates
  physical_logical_dp_rank
  children[logical_dp_rank].state
```

Physical-rank namespacing prevents distributed checkpoint backends from
merging different ranks' local child dictionaries into one value.

The effective DP degree and balancing group must currently match on restore.
Mode and optimizer-step layout may change because checkpoints are taken between
steps and contain no partially buffered candidate window.

## Metrics

The wrapper reports:

- source-fetch, inspection, and planner time;
- ordinary and balanced predicted synchronized cost;
- candidate payload bytes and hypothetical moved bytes;
- replicated-read amplification; and
- unchanged-plan count.

The trainer drains these metrics once per optimizer step and includes them in
its normal metrics output.

## Testing

Unit tests cover optimizer-step grouping, planner determinism and exact
coverage, packed-text inspection, group sizes larger than two, shadow and
balance behavior, lifecycle cleanup, and checkpoint continuation. Integration
tests exercise two-rank DCP and `torch_checkpointing` round trips. Shadow and
balance modes have also completed ten-step, two-GPU training runs.

## Future work

- Segment-level balancing and materialization.
- Payload exchange so each process reads only its own logical stream.
- Topology-independent checkpoint state for changing group size on resume.
- Performance evaluation with representative production data.
