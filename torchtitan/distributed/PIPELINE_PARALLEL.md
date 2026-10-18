# Pipeline Parallelism

Design notes and tuning knobs for `pipeline_parallel.py`. Pipeline parallelism
splits the model into stages placed on different ranks; this note covers how
TorchTitan models are written so stages are easy to cut, how stages get a
consistent initialization, and how FSDP and send scheduling interact with
multi-stage schedules.

## Writing pipeline-friendly models
When applying Pipeline Parallelism, you will have to construct nn.Module objects representing the portion of the model that runs on a given pipeline stage. Whether you plan to manually edit your model code, or use techniques like tracing to extract model chunks, a few changes to the original model code can go a long way to making this process easier.

### Simplifying the top-level model forward
Most likely, you can write your model in such a way that the top-level nn.Module owns a sequence of child modules that it calls during forward, delegating most of the complexity to the child module forwards.  If you can reduce your top level forward to mostly a for-loop over child module calls, then you'll simplify the pipeline-partitioning task to choosing the set of submodules to keep per stage.  If you have non-trivial logic in the top-level forward, you'll have to find a way to patch that logic back onto the resulting pipeline stage model, which can be annoying.

Example ([PR #321](https://github.com/pytorch/torchtitan/pull/321)):
We used to slice the `freqs_cis` buffer by `seq_len` in the top level forward, pass that into child modules, and expect that inside the child modules the `seq_len` would match up with the size of other local tensors.  But we don't know about whether TP was applied or not when we consider PP splitting and could create a mismatch.  Its just as easy to perform the `freqs_cis` slicing inside the child submodule, using the runtime-accurate local `seq_len`, and this sidesteps the issue at PP slicing time.

Example ([PR #322](https://github.com/pytorch/torchtitan/pull/322)):
We decided to actually reuse the top-level model object on every PP stage, just delete the layers we don't want, and make sure that the top-level forward would do the right thing.  This means we don't have to make a separate runtime pp_forward that glues together child modules per stage.  The first change was using a moduledict instead of modulelist to store layers. This preserves layer Fully Qualified Names (FQNs) even when deleting some layers - e.g. layers.1 stays layers.1 even if you remove layers.0, which isn't true for a list- this matters for checkpoint save/load.  Preserving FQNs is a requirement for using Distributed Checkpointing (DCP) since it uses FQNs as globally unique IDs for sharding metadata. The second change was making the input and output layers optional- if the layer exists, we run it, otherwise we feed the input through to bypass it.  With these two changes, we can just (meta)-initialize the whole model, delete the unused parts per stage, then materialize the remaining part on GPU before loading a checkpoint.

## Initializing stages from a seed checkpoint
Initializing the pipeline-parallel model is challenging because we assume the model could be so large as to not fit on local GPU (or possibly, even on CPU), and we also want to use the (bitwise) same initialization as we use for 1D or 2D parallel models, to ease debugging or comparisons between runs. It's not that easy to rewrite the original model's `init_weights` function to be tolerant of initializing only some layers, and also serializing initialization operations globally for consistent RNG order.

For now, we sidestep all these problems with a simple but brutal solution: Initialize the whole model on some CPU instance, save a checkpoint file, and then lean on Distributed Checkpointing's "load" functionality to initialize the FQNs that are present on a given PP stage after stage creation.  For future work, we consider adding a more elaborate initialization scheme to `torch.pipelining`.

## Controlling FSDP residency and unshard lookahead in pipeline schedules

Looped pipeline schedules can have several local stages backed by FSDP. Two
separate policies control their parameter lifetime:

- `pp_max_unsharded_active_stages` bounds how many local stages
  may remain resident. Lowering this value can reduce parameter memory, but it
  can also introduce additional reshard and later unshard cycles.
- `pp_num_unshard_lookahead_factor` controls how many upcoming distinct
  stages may issue their asynchronous all-gathers. It changes only where
  existing `UNSHARD` actions are issued; it does not change stage residency,
  `RESHARD` placement, collective counts, or the wait immediately before a
  stage first consumes its parameters.

The lookahead policy accepts three forms:

| Value | Resolved lookahead on PP rank `r` | Intended use |
| --- | --- | --- |
| `"full"` | `max_unsharded_stages` | Preserve PyTorch's full-window behavior by issuing every eligible unshard as early as the residency window permits. |
| `"auto"` | `min(r + 2, max_unsharded_stages)` | Avoid a full all-gather burst on early PP ranks while leaving progressively more lookahead on ranks with a pipeline warmup window. |
| `(k0, ..., kN)` | `kr` | Expert control for asymmetric stage cost, parameter size, or network behavior. |

An explicit tuple must contain one positive integer per PP rank. Every value
must be no larger than the resolved
`pp_max_unsharded_active_stages`. A lookahead of one issues an
unshard immediately before that stage; a lookahead of two keeps one future
stage in flight while the current stage computes.

PyTorch pipeline schedules default to `"full"` for backward compatibility.
TorchTitan defaults to `"auto"` because it owns the complete pipeline topology
and can select the rank-aware policy without recipe-specific tuning. Use
`"full"` when reproducing the original eager all-gather issue pattern, or an
explicit tuple when profiling establishes a better rank-specific distance:

```python
config.parallelism.pp_max_unsharded_active_stages = 4
config.parallelism.pp_num_unshard_lookahead_factor = "auto"

# Equivalent to "full" for a four-rank pipeline with a residency bound of four.
config.parallelism.pp_num_unshard_lookahead_factor = (4, 4, 4, 4)
```

## Bounding pending pipeline sends

`pipeline_parallel_max_outstanding_sends` limits the number of forward and
backward send actions that have not reached their wait on each pipeline rank.
Each action is identified by its direction, stage, and microbatch and may
contain several P2P operations. The limit does not count microbatches, tensors,
operations, or bytes. `None` keeps the schedule's normal causal waits without
adding a hard limit.

This option applies only to multi-stage schedules. PyTorch lowers those
schedules to a global communication-action sequence with explicit
`WAIT_SEND_F` and `WAIT_SEND_B` actions. Single-stage schedules issue
communication directly and do not expose this sequence for wait placement.

PyTorch's pipeline schedule lowering may move an existing wait earlier to meet
the limit. This releases the schedule's ownership of sent outputs sooner and
can reduce peak memory. A small limit can reduce communication overlap if the
matching receive has not completed.

The lowering pass rejects a move that would create a dependency cycle. For
example, assume it wants to move the wait for send `S` before local action `A`.
The wait depends on the peer receive `R`. If the schedule already orders `A`
before `R`, the move would add the reverse dependency:

```text
existing: A -> ... -> R
moved wait: R -> A
cycle: A -> ... -> R -> A
```

The pass can prove whether a placement is legal, but it cannot predict the
fastest limit. A wait may stall based on tensor sizes, topology, NCCL timing,
and runtime contention. The memory benefit also depends on tensor sizes and
other owners of the sent storage.

Start with `None`. When memory is constrained, profile smaller values and use
the smallest value that does not reduce throughput:

```python
config.parallelism.pipeline_parallel_max_outstanding_sends = 8
```
