# Optimizer-Step Load Balancing

Packed text microbatches can contain the same number of tokens but require
different amounts of attention work. For reset-delimited document segments of
lengths `L_i`, the current cost model estimates work as:

```text
sum(L_i ** 2)
```

If expensive microbatches repeatedly land on one data-parallel (DP) rank, that
rank becomes a straggler. `LoadBalancingDataLoader` reduces this skew by
reassigning complete microbatches within one optimizer step.

## Semantics and guarantees

The current implementation balances whole, already-collated microbatches. It
does not split documents, repack document segments, or move data across
optimizer-step boundaries.

For each balancing group and optimizer step, the output contains exactly the
same microbatch multiset as the ordinary `GrainDataLoader` streams would have
produced. Only the DP-rank and execution-slot assignment may change. This
preserves every microbatch's tensors and document boundaries, but it does not
promise bitwise-identical training: changing rank or execution order can change
random-number consumption and floating-point reduction order.

The planner is deterministic when the underlying dataset, transforms, and
tokenizer are deterministic for a given seed and logical DP rank.

## How it works

The trainer requests all microbatches for one optimizer step at once. The step
is a grid with gradient-accumulation iterations on one axis and
pipeline-parallel microbatches on the other.

For effective DP degree `D`, group size `G`, and physical DP rank `r`, the
coordinator selects this contiguous group of logical input streams:

```text
group_start = (r // G) * G
logical ranks = group_start, ..., group_start + G - 1
```

Every process in that group independently:

1. Constructs one child `GrainDataLoader` for each logical rank in the group.
2. Reads one optimizer-step window from every child.
3. Inspects packed-document lengths and estimates each microbatch's cost.
4. Runs the same deterministic planner over the complete candidate set.
5. Returns only the microbatches assigned to its physical DP rank.

### Coordinator role

The name `ReplicatedInputCoordinator` describes how group members obtain a
shared candidate window. Every physical process reconstructs every logical
input stream in its balancing group. For example, with logical ranks 0 and 1:

```text
physical rank 0 reads logical streams 0 and 1
physical rank 1 reads logical streams 0 and 1
```

The input reads are replicated, but the training samples are not: after both
processes independently compute the same plan, each microbatch is returned by
exactly one physical rank.

The coordinator turns `dp_world_size`, `dp_rank`, and `group_size` into:

- the contiguous logical-rank group whose candidates are visible; and
- the logical output rank owned by this physical process.

The coordinator does not read data, estimate cost, or communicate between
processes. The loader uses its mapping to build the child loaders and select
this process's bins from the shared deterministic plan. With `group_size=1`,
the mapping contains only the physical rank's ordinary logical stream.

This is a separate component because input visibility and ownership are a
coordination policy, not part of the balancing algorithm. In particular, it:

- keeps physical-process identity and logical dataset-shard identity out of
  the tensor-independent planner;
- centralizes which original `dp_rank` and `dp_world_size` are passed to each
  child loader, so rebalancing does not accidentally change ordinary dataset
  sharding;
- lets local reordering, subgroup balancing, and full-DP balancing use the same
  loader and planner; and
- gives checkpointing one explicit topology to record and validate.

One alternative is an exchange-based coordinator:

```text
physical rank 0 reads only logical stream 0 --\
                                             +--> exchange metadata --> plan
physical rank 1 reads only logical stream 1 --/
                                                       |
                                              exchange reassigned payloads
```

Each rank would read its ordinary stream once, all-gather only the adapter
metadata, run the same planner, and then transfer microbatch payloads whose
owner changed. This removes replicated source reads and reduces the number of
local child loaders, but adds a communication collective and payload movement
on every optimizer step. It also needs buffer management, variable-size
transfer support, and more complex checkpoint/failure semantics.

The current replicated design deliberately chooses extra reads and host memory
in exchange for no runtime communication and a simple exact-data argument.
Keeping coordination separate means an exchange-based implementation could
reuse the same adapter, cost model, and planner. It would still require a
broader coordinator/loader interface and is not currently implemented.

### Adapter role

The planner deliberately does not understand tensors or packed text. A packing
adapter is the boundary between a concrete `TrainingMicrobatch` and the
metadata-only planner. `TokenizedTextPackingAdapter`:

- validates that the microbatch matches the supported packed-text contract;
- derives document segment lengths from resets in `positions`;
- reports token count, document count, and payload size; and
- leaves the original tensors unchanged for the loader to return later.

The current adapter accepts only the four canonical tensors (`input`,
`labels`, `positions`, and `padding_mask`) plus `num_valid_tokens`. It rejects
nonempty `model_kwargs` because it does not know how to validate, size, or
interpret extra model inputs. Supporting such fields belongs in an adapter
that defines their invariants and includes their tensors in payload accounting.

No communication collective is needed because all group members reconstruct
the same logical streams and plan locally. The tradeoff is that source reads,
child-loader resources, and candidate host memory grow approximately with
`G`. Child prefetch depth is limited to one microbatch by the wrapper.

The whole-microbatch planner sorts microbatches by estimated cost and groups
similarly expensive work into the same synchronized execution slots. Its
lexicographic objective minimizes:

1. The sum of the maximum DP-rank cost in each synchronized slot.
2. The worst cost skew between DP ranks.
3. Payload bytes assigned to a different rank.
4. A stable deterministic tie-breaker.

The ordinary assignment is retained unless the candidate has a better
objective.

## Worked example

Consider two DP ranks, two accumulation slots, and one pipeline microbatch per
slot. Each microbatch has eight tokens but different document segmentation:

| Microbatch | Ordinary owner and slot | Segment lengths | Predicted cost |
| --- | --- | --- | ---: |
| A | rank 0, slot 0 | `[8]` | 64 |
| B | rank 0, slot 1 | `[4, 4]` | 32 |
| C | rank 1, slot 0 | `[2, 2, 2, 2]` | 16 |
| D | rank 1, slot 1 | `[1, 1, 1, 1, 1, 1, 1, 1]` | 8 |

The ordinary and balanced schedules are:

| Plan | Slot 0 `(rank 0, rank 1)` | Slot 0 max | Slot 1 `(rank 0, rank 1)` | Slot 1 max | Synchronized cost |
| --- | --- | ---: | --- | ---: | ---: |
| Ordinary | `(A: 64, C: 16)` | 64 | `(B: 32, D: 8)` | 32 | 96 |
| Balanced | `(A: 64, B: 32)` | 64 | `(D: 8, C: 16)` | 16 | 80 |

The balanced plan moves complete microbatch B to rank 1 and complete
microbatch D to rank 0. Expensive work is aligned in the same slot, reducing
the sum of synchronized slot maxima from 96 to 80. The optimizer step still
contains exactly A, B, C, and D; no document is split or repacked.

## Current support

Supported:

- A `GrainDataLoader.Config` child using the built-in `TextCollator`.
- Canonical `TokenizedTrainingMicrobatch` values on CPU.
- Packed text whose `positions` reset to zero at each document boundary.
- Gradient accumulation and pipeline-parallel microbatch layouts.
- `group_size=1` local reordering.
- Replicated balancing with any positive `group_size` that divides the
  effective DP degree.
- `shadow` and `balance` modes.
- Checkpointing at completed optimizer-step boundaries.

Not currently supported:

- Splitting or repacking document segments.
- Moving candidates between balancing groups or optimizer steps.
- Exchanging payloads so each logical stream is read only once.
- Validation dataloaders.
- TorchFT, because dynamic DP membership changes logical-stream ownership.
- Finite, non-repeating data under DP; this is inherited from
  `GrainDataLoader`.
- Restoring with a different effective DP degree or balancing-group topology.

## Configuration

Wrap the recipe's existing `GrainDataLoader.Config`. Configuration is done in
Python, like other TorchTitan components:

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

Run the resulting recipe normally. Start with `mode="shadow"` to measure the
predicted benefit without changing assignment, then change that constructor
argument to `mode="balance"`.

`group_size` controls the scope and replication cost:

- `1` sees only the physical rank's ordinary stream.
- A value greater than `1` balances among that many adjacent logical DP ranks.
- The effective DP degree balances across the entire DP group.

Larger groups expose more scheduling opportunities but replicate more input
work and hold more candidate microbatches in host memory. The group size must
divide the effective DP degree exactly.

The child loader's dataset, seed, shuffle, packing, and collator configuration
remain unchanged. Set `max_num_documents` on the child config when the packed
data path has a known document-count limit.

## Modes

`shadow` computes both ordinary and balanced plans, records their metrics, and
returns the ordinary assignment. Use it to validate deterministic replicated
reads and estimate whether balancing is worthwhile.

`balance` returns the selected balanced assignment. If the candidate objective
is not better than the ordinary assignment, it still returns the ordinary one.

To disable the feature completely, configure the original `GrainDataLoader`
directly rather than wrapping it.

## Metrics

The trainer drains and reports these metrics once per optimizer step:

| Metric | Meaning |
| --- | --- |
| `data_load/source_fetch_ms` | Time spent reading all replicated candidates. |
| `data_load/inspection_ms` | Time spent validating and extracting metadata. |
| `data_load/planner_ms` | Time spent constructing the plan. |
| `data_load/baseline_predicted_cost` | Predicted synchronized cost of the ordinary assignment. |
| `data_load/balanced_predicted_cost` | Predicted synchronized cost of the selected plan. |
| `data_load/moved_payload_bytes` | Bytes whose selected assignment changes logical owner. |
| `data_load/candidate_host_bytes` | Tensor payload bytes inspected by this process. |
| `data_load/replicated_read_amplification` | Configured balancing-group size. |
| `data_load/unchanged_plans` | Number of plans that retained the ordinary assignment. |

In `shadow` mode, balanced cost and moved bytes describe the plan that would be
used in `balance` mode.

## Checkpoint behavior

The wrapper checkpoints the cursor of every logical child stream, namespaced by
physical and logical DP rank. A restore requires the same effective DP degree,
physical rank, and balancing-group coordinates. The balancing mode and
optimizer-step layout may change because no partially planned window is stored
at an optimizer-step boundary.

The checkpoint does not fingerprint the dataset, tokenizer, or packing
configuration. As with the ordinary loader, the caller must resume with a
compatible input pipeline.

## Extending the implementation

The implementation separates tensor-specific inspection from tensor-independent
planning:

```text
TrainingMicrobatch
    -> packing adapter -> metadata
    -> cost model      -> PackableItem
    -> planner         -> LoadBalancePlan
    -> loader          -> original microbatch payloads
```

### Add a cost model

Subclass `QuadraticAttentionCost` and its nested `Config`, then override
`estimate()`. The result must be a deterministic, nonnegative integer and
should be additive across independently scheduled items. Pass the new config
as `LoadBalancingDataLoader.Config(cost_model=...)`.

### Add a whole-microbatch planner

Subclass `WholeMicrobatchBalancer` and its nested `Config`, then implement
`plan(items, bins)`. A plan must:

- assign every item and bin exactly once;
- respect token and document capacities;
- include the ordinary baseline and correct objective values;
- never be worse than the baseline under `LoadBalancePlanObjective`; and
- use stable IDs for deterministic tie-breaking.

Call `validate_plan()` before returning. Keeping the planner limited to
`PackableItem`, `PackingBin`, and stable IDs avoids dependencies on tensors,
datasets, or distributed runtime state.

### Add another whole-payload adapter

Subclass `TokenizedTextPackingAdapter` and its nested `Config` when another
`TrainingMicrobatch` representation can still be treated as one indivisible
item. `inspect_microbatch()` must return metadata providing `num_tokens`,
`num_documents`, and `payload_bytes`, and the configured cost model must accept
that metadata.

### Add segment-level repacking

Segment-level balancing is a larger extension, not just another planner. It
would require the adapter to expose multiple items per input microbatch and a
materializer to rebuild output tensors from multiple assigned segments. The
materializer must preserve token/label alignment, position resets, padding,
document limits, and loss masking. It must also define and test any changes to
sample order, random-number consumption, and numerical reproducibility.

Keep those mechanics outside the planner: the planner should continue to see
only abstract items, bins, costs, capacities, and stable IDs.

## Tests

The focused tests are under
`tests/unit_tests/cpu/components/data/load_balancing/`:

```bash
taskset -c 0-7 /home/jinsooihm/local/mywork/pytorch-env/bin/python -m pytest \
    tests/unit_tests/cpu/components/data/load_balancing -x
```
