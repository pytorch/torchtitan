## Human Generated Ideas

- Idea: Bootstrap runnable Qwen3 14B DP-sharded run
  - Current best source commit: 68f86f1
  - Source: human/setup
  - Expected mechanism for improving reported tokens/sec: The target command currently selects `dp_shard=-1` on 8 GPUs, but `parallelize_qwen3()` only applies replicated DDP and does not apply FSDP for the active `fsdp` mesh. A narrow bootstrap should make the inferred target command run correctly by applying TorchTitan FSDP on the configured shard mesh.
  - Supporting evidence: `qwen3_14b()` sets `data_parallel_shard_degree=-1`, which resolves to `dp_shard=8` for `NGPU=8`; the Qwen3 scaffold rejects TP/CP/PP/EP and otherwise only calls `replicate()` when `dp_enabled`.
  - Planned source/config changes: In `torchtitan/models/qwen3/parallelize.py`, replace the DP-only replicated wrapper for the active DP-shard case with the minimal FSDP wrapping needed for Qwen3 14B.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10`
  - Success criteria and expected risk: The 10-step run completes, logs finite losses and reported `tps`; risk is incorrect wrapper order or insufficient wrapping granularity causing OOM or slow all-gathers.

## Manager Generated Ideas

- Idea: Profile current best after first runnable result
  - Current best source commit: TBD after bootstrap
  - Source: agent-generated
  - Expected mechanism for improving reported tokens/sec: Use a profiled 10-step run to distinguish compute, HBM, communication, launch, and data-loading bottlenecks before changing TP/CP/PP or activation checkpointing.
  - Supporting evidence: No Qwen3 14B 8xB200 result has been measured yet on this branch.
  - Planned source/config changes: None for the profiling run.
  - Planned command or config overrides: Add TorchTitan profiler flags to the same 10-step command.
  - Success criteria and expected risk: A trace and structured metrics identify the next narrow hypothesis; profiling overhead means the profiled `tps` should not be ranked against unprofiled candidates.
