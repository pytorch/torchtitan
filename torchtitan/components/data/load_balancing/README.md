# Optimizer-Step Load Balancing

Packed text microbatches can contain the same number of tokens but require very
different amounts of attention work. If their document lengths are `L_i`, the
current estimate is:

```text
cost = sum(L_i ** 2)
```

`LoadBalancingDataLoader` collects all microbatches for one optimizer step and
reassigns complete microbatches to reduce data-parallel (DP) stragglers. It does
not split documents, repack tokens, or move data between optimizer steps.

## Components and responsibilities

| Component | Role | Why it is separate |
| --- | --- | --- |
| `LoadBalancingDataLoader` | Orchestrates one optimizer-step window and records metrics. | Keeps the trainer-facing dataloader API independent of coordination and planning strategies. |
| `InputCoordinator` | Decides which input streams each process reads, exposes a candidate window, and delivers the assigned local payloads. It also owns child-loader state and cleanup. | Input ownership and data movement differ fundamentally between replicated reads and a future collective-based implementation. |
| `TokenizedTextPackingAdapter` | Inspects concrete text tensors and produces small, tensor-independent metadata. | The planner should not know about `positions`, padding masks, labels, or model kwargs. A different input format can supply another adapter. |
| `QuadraticAttentionCost` | Converts packing metadata into an estimated amount of work. | Cost estimation can evolve independently of input parsing and assignment. |
| `WholeMicrobatchBalancer` | Assigns abstract items to output bins using only IDs, costs, sizes, and execution coordinates. | Planning remains deterministic and independent of tensor storage or communication. |

The coordinator abstraction is needed because it owns both sides of routing:
which payloads are visible before planning and how planned assignments become
local payloads afterward. The loader cannot own those details without assuming
a particular input topology.

## Data flow

```text
Trainer requests microbatches for one optimizer step
                         |
                         v
             LoadBalancingDataLoader
                         |
                         v
              coordinator.collect()
              - read candidate payloads
              - assign stable IDs
                         |
             itemize callback for each payload
                         |
          +--------------+---------------+
          | packing adapter              |
          | concrete tensors -> metadata |
          +--------------+---------------+
                         |
          +--------------+---------------+
          | cost model                   |
          | metadata -> estimated work   |
          +--------------+---------------+
                         |
         CoordinatedWindow(items, bins, payloads)
                         |
                         v
                balancer.plan()
                - metadata only
                - choose assignments
                         |
                         v
             coordinator.distribute()
             - realize assignments
             - return local execution order
                         |
                         v
            Trainer executes the full step
```

`collect()` receives the itemization callback because the coordinator owns the
input reads, while the loader owns the adapter and cost model. This arrangement
also fits a future coordinator that first inspects local payloads, exchanges
only metadata, and later exchanges reassigned tensors.

## Current coordinator: replicated input

`ReplicatedInputCoordinator` reconstructs every logical input stream in a
contiguous balancing group. For a group containing logical ranks 0 and 1:

```text
physical rank 0 reads logical streams 0 and 1
physical rank 1 reads logical streams 0 and 1
```

Both processes independently see the same candidate window and compute the
same plan. Each process returns only the microbatches assigned to its physical
rank. Source reads and candidate storage are replicated, but training samples
are not duplicated.

This requires the source, transforms, packing, and tokenizer to be deterministic
for a given logical rank and checkpoint state. Divergent replicas could silently
produce different plans and therefore duplicate or omit training samples.

For effective DP degree `D`, group size `G`, and physical DP rank `r`:

```text
group_start = (r // G) * G
logical ranks = group_start, ..., group_start + G - 1
```

`G` must divide `D`. With `G=1`, the same implementation performs local
optimizer-step reordering. Larger groups provide more balancing opportunities,
but multiply source reads, input-pipeline resources, and candidate host memory
by approximately `G`.

A future exchange-based coordinator can use the same interface:

```text
each process reads only its own logical stream
                         |
collect(): exchange PackableItem metadata
                         |
             the same planner runs
                         |
distribute(): exchange payloads that changed owner
```

That design avoids replicated reads but requires collectives, variable-size
payload transfer, and different checkpoint semantics. It is not implemented.

## Planning algorithm

The current planner treats every microbatch as indivisible:

1. Build the ordinary assignment as the baseline.
2. Sort candidate microbatches by descending estimated cost.
3. Form one heavy-to-light cohort for each gradient-accumulation iteration.
4. Within each cohort, use Longest Processing Time assignment: give the next
   heaviest microbatch to the currently lightest DP rank that still has a free
   PP slot.
5. Compare the candidate with the baseline by:
   1. sum of the slowest DP-rank cost in each accumulation iteration;
   2. worst DP-rank cost skew in one accumulation iteration;
   3. payload bytes assigned to a different DP rank.
6. Keep the locally better assignment. A single-rank group uses the candidate
   on a tie so it can form heavy-first cohorts.
7. Reorder the selected plan's complete accumulation iterations by descending
   synchronized cost.

The final ordering in step 7 gives independent balancing groups a common
heavy-first convention. It does not change ownership, moved bytes, or the local
score. Without it, one group could run heavy-first while another runs
light-first, creating global DP stragglers even though both local plans have the
same score.

Pipeline-parallel (PP) microbatches within one accumulation iteration are
balanced by their total cost per DP rank. Their individual PP positions are
assigned deterministically, but PP positions are not treated as synchronization
boundaries.

## Example

Consider two DP ranks, two accumulation iterations, and one PP microbatch per
iteration. Every microbatch has eight tokens:

| Microbatch | Ordinary location | Document lengths | Cost |
| --- | --- | --- | ---: |
| A | rank 0, accumulation 0 | `[8]` | 64 |
| B | rank 0, accumulation 1 | `[4, 4]` | 32 |
| C | rank 1, accumulation 0 | `[2, 2, 2, 2]` | 16 |
| D | rank 1, accumulation 1 | eight `[1]` documents | 8 |

```text
ordinary:
  accumulation 0: rank 0 = A:64, rank 1 = C:16 -> 64
  accumulation 1: rank 0 = B:32, rank 1 = D: 8 -> 32
  synchronized cost = 96

balanced:
  accumulation 0: rank 0 = A:64, rank 1 = B:32 -> 64
  accumulation 1: rank 0 = D: 8, rank 1 = C:16 -> 16
  synchronized cost = 80
```

The balanced step trains on exactly A, B, C, and D. Only their DP-rank and
accumulation-slot assignments change.

## Configuration

Wrap an existing `GrainDataLoader.Config` in a recipe:

```python
from torchtitan.components.data import (
    GrainDataLoader,
    LoadBalancingDataLoader,
    ReplicatedInputCoordinator,
)


config = llama3_debugmodel_varlen_attn()
child = config.dataloader
assert isinstance(child, GrainDataLoader.Config)

config.dataloader = LoadBalancingDataLoader.Config(
    dataloader=child,
    mode="shadow",
    coordinator=ReplicatedInputCoordinator.Config(group_size=2),
)
```

Use `mode="shadow"` first. It measures the proposed plan while returning the
ordinary assignment. Use `mode="balance"` to execute the selected plan. Configure
`GrainDataLoader` directly to disable the feature.

## Current support

Supported:

- A `GrainDataLoader.Config` child using the built-in `TextCollator`.
- Canonical CPU `TokenizedTrainingMicrobatch` values.
- Packed text whose real-document `positions` reset to zero.
- Gradient accumulation and pipeline-parallel microbatch layouts.
- Any positive replicated `group_size` that divides the effective DP degree.
- Shadow and balance modes.
- Checkpointing at completed optimizer-step boundaries.

Current limitations:

- The cost estimate includes real document segments but not executed padding
  segments.
- Replicated input requires a deterministic and stable input pipeline.
- Source reads, loader resources, and candidate memory scale with group size.
- Documents are not split or repacked.
- Candidates cannot move between groups or optimizer steps.
- Exchange-based coordination is not implemented.
- Validation dataloaders and TorchFT are not supported.
- Restore requires the same effective DP degree and balancing-group topology.
- Finite, non-repeating data under DP remains unsupported by `GrainDataLoader`.

Reassignment preserves the optimizer-step sample multiset and original document
boundaries. It is not guaranteed to be bitwise identical to ordinary loading:
changing rank or execution order can change random-number consumption and
floating-point reduction order.

## Checkpointing and metrics

The loader delegates checkpointing to the coordinator. The replicated
coordinator stores every logical child-stream cursor under its physical and
logical DP rank. Only completed optimizer-step boundaries are checkpointed; a
partially planned window is never stored.

The loader reports:

- source-fetch, inspection, and planner time;
- ordinary and selected predicted cost;
- payload bytes assigned to another logical rank;
- candidate host tensor bytes;
- whether the final assignment changed.

## Extending

- A new coordinator subclasses `InputCoordinator`. `collect()` defines input
  visibility and returns planner metadata plus locally available payloads;
  `distribute()` realizes the plan. The coordinator owns child loaders,
  communication, checkpoint state, and cleanup.
- A new adapter converts another concrete microbatch representation into
  planner metadata without changing its payload.
- A new cost model estimates work from adapter metadata.
- A new whole-payload planner consumes `PackableItem` and `PackingBin` values
  without depending on tensors or communication.

Segment-level balancing is a larger extension. It needs an adapter that emits
segments and a materializer that packs assigned segments back into fixed-length
microbatches while preserving token/label alignment, position resets, padding,
document limits, and loss masking.

## Tests

```bash
taskset -c 0-7 /home/jinsooihm/local/mywork/pytorch-env/bin/python -m pytest \
    tests/unit_tests/cpu/components/data/load_balancing -x
```
