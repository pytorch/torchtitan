# Silent data corruption

TorchTitan can detect silent data corruption (SDC) by re-executing a
deterministic forward/backward and comparing all of its observable effects.
The feature is disabled by default.

Replay-based detection is one of several SDC strategies: alternatives include
shadow computation on redundant hardware and algorithm-level checks such as
checksummed matmuls. Replay trades extra forward/backward time on checked
steps for an in-training, hardware-agnostic check, and requires fully
deterministic execution.

## Configuration

The feature has no CLI flags; enable it programmatically in a config/recipe:

```python
config.debug.deterministic = True
config.sdc_replay.enabled = True
config.sdc_replay.num_steps = 1    # optimizer steps checked after each (re)start; -1 checks every step
config.sdc_replay.num_replays = 1  # re-executions compared against the reference
```

`num_steps` counts optimizer steps from trainer start and restarts after every
checkpoint load, so the default checks the first step after every (re)start,
where corruption from a bad restore or initialization is most likely.
`num_replays` is the number of times the checked forward/backward is
re-executed and compared against the initial reference execution; it must be
at least one, and higher values catch intermittent corruption a single replay
can miss.

Replay requires `debug.deterministic=True`,
`debug.deterministic_warn_only=False`, and uses `torch.hash_tensor`.

## What is replayed

Only the first forward/backward of a checked optimizer step is replayed: one
full gradient-accumulation group or, with pipeline parallelism, one complete
pipeline schedule including all pipeline microbatches. State is restored
before every execution; the reference and intermediate executions are
discarded, and only the final execution's gradients, registered buffers, RNG
advancement, token counter, and loss are committed.

Eager execution, `torch.compile`, CUDA graphs, symmetric-memory FSDP,
distributed GEMM, async TP, DeepEP v2, HybridEP, and MinimalAsyncEP can
participate in replay. GraphTrainer uses the same replay boundary. Execution
backends do not need their internal scratch state restored as long as one
forward/backward invocation completes before it returns and later invocations
overwrite that state before reading it. Such scratch state is neither
snapshotted nor included in the replay signature.

## Limitations

CUDA graphs currently require `sdc_replay.num_replays=1`. Additional replays
would require restoring graph-owned gradient and optional-buffer storage
without changing its captured addresses.

Replay currently uses the existing XOR-based `torch.hash_tensor` mode. This
checksum is order-insensitive, so permutations and some repeated-value
corruptions can collide today.

## Failure reporting

A mismatch raises `SDCReplayMismatch` on every rank before gradient clipping,
the optimizer, the learning-rate scheduler, or checkpoint saving. The checked
signature includes the loss, local parameter gradients, registered buffers,
Python and torch RNG state, and the token counter. The exception identifies
the optimizer step, the step's position in the current check schedule
(`local_step`), the replay number, the originating rank, and the first
differing signature entry.

The expected checked-step cost is `1 + num_replays` forward/backward
executions. Unchecked steps do not compute replay signatures.
