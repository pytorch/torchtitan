# Silent data corruption

TorchTitan can detect silent data corruption (SDC) by replaying a deterministic
forward/backward unit and comparing its training state. The feature is disabled
by default.

Enable it with `--sdc-replay.enabled`. `num_steps` controls how many
attempt-local optimizer steps are checked; `-1` checks every step.
`num_replays` is the number of candidate executions compared with one reference
execution and must be at least one.

Replay requires `debug.deterministic=True`,
`debug.deterministic_warn_only=False`, and uses `torch.hash_tensor`. Eager
execution, `torch.compile`, CUDA graphs, symmetric-memory FSDP, distributed
GEMM, async TP, DeepEP v2, HybridEP, and MinimalAsyncEP can participate in
replay. GraphTrainer and TorchFT use the same replay boundary. Execution
backends do not need their internal scratch state restored as long as one
forward/backward invocation completes before it returns and later invocations
overwrite that state before reading it. Such scratch state is neither
snapshotted nor included in the replay signature.

CUDA graphs currently require `sdc_replay.num_replays=1`. Additional replays
would require restoring graph-owned gradient and optional-buffer storage without
changing its captured addresses.

Replay currently uses the existing XOR-based `torch.hash_tensor` mode. This
checksum is order-insensitive, so permutations and some repeated-value
corruptions can collide today.

Only the first gradient-accumulation unit in a checked optimizer step is
replayed. With pipeline parallelism, that unit is one complete pipeline schedule,
including all pipeline microbatches. State is restored before every execution;
the reference and intermediate candidates are discarded, and only the final
candidate's gradients, registered buffers, RNG advancement, token counter, and
loss are committed.

A mismatch raises `SDCReplayMismatch` on every rank before gradient clipping,
the optimizer, the learning-rate scheduler, or checkpoint saving. The checked
signature includes the loss, local parameter gradients, registered buffers,
Python and torch RNG state, and the token counter. The exception identifies the
optimizer step, attempt-local step, replay number, originating rank, and first
differing signature entry.

The expected checked-step cost is `1 + num_replays` forward/backward executions.
Unchecked steps do not compute replay signatures.
