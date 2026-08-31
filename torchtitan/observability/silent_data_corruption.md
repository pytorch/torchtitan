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

The feature has no CLI flags; enable it programmatically in a config/recipe
by assigning a config (`config.sdc_replayer` is `None` by default, which
disables replay):

```python
from torchtitan.observability.sdc_replayer import SDCReplayer

config.debug.deterministic = True
config.sdc_replayer = SDCReplayer.Config(
    num_steps=1,    # optimizer steps checked after each (re)start; -1 checks every step
    num_replays=1,  # re-executions compared against the reference
)
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

Only the first forward/backward call of a checked optimizer step is replayed:
one gradient-accumulation group, which under pipeline parallelism is one
complete pipeline schedule including all pipeline microbatches. Gradient
accumulation composes with pipeline parallelism; when a step has multiple
accumulation groups, the remaining groups run unchecked. This is a cost
choice rather than an engine limitation: the replay engine checks any
forward/backward callable, and the later groups of a step exercise the same
compute and communication paths as the first, so checking them as well would
multiply the checked-step overhead without covering new code paths. State is
restored before every execution; the reference and intermediate executions
are discarded, and only the final execution's gradients, registered buffers,
RNG advancement, token counter, and loss are committed.

Gradient values are never snapshotted. The checked forward/backward must
begin with no pending gradients (`None` or zeros, the post-`zero_grad`
state), and restore rebuilds that entry state directly: parameters that
entered without a gradient return to `None`, and entry gradient tensors are
zeroed in place, preserving their storage addresses for CUDA graphs, with no
gradient-sized clone or copy kernels on checked steps.

Eager execution, `torch.compile`, CUDA graphs, symmetric-memory FSDP,
distributed GEMM, async TP, DeepEP v2, HybridEP, and MinimalAsyncEP can
participate in replay. GraphTrainer uses the same replay boundary. Execution
backends do not need their internal scratch state restored as long as one
forward/backward invocation completes before it returns and later invocations
overwrite that state before reading it. Such scratch state is neither
snapshotted nor included in the replay signature.

## Limitations

CUDA graphs currently require `sdc_replayer.num_replays=1`. Additional replays
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
