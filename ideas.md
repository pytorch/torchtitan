## Human Generated Ideas

- None recorded yet.

## Manager Generated Ideas

- Idea: bootstrap minimal baseline FSDP
  Current best source commit: 7c324f2
  Source: agent-generated setup finding
  Expected mechanism: The recorded Qwen3 14B baseline config requests data-parallel sharding across the full 8-GPU world, but the scaffold applies replicated DDP on the batch mesh. Replacing only that DP path with minimal per-layer FSDP should make the baseline workload fit memory and exercise the intended sharded optimizer/parameter path.
  Supporting evidence: `qwen3_14b()` sets `data_parallel_shard_degree=-1`, TP/CP/PP/EP all 1, local batch size 4, sequence length 4096, and activation checkpointing mode `full`. The setup hardware has 8 NVIDIA B200 GPUs with about 179 GiB each and full NVLink connectivity across GPUs.
  Planned source/config changes: Edit only `torchtitan/models/qwen3/parallelize.py` to apply `fully_shard` over Qwen3 transformer blocks and the root model on the existing `fsdp` mesh. Do not add activation checkpointing, compile, TP, CP, PP, batch-size, sequence-length, dtype, optimizer, or resharding changes in this first candidate.
  Planned command or config overrides: `NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run01-fsdp-bootstrap > run.log 2>&1`
  Success criteria and expected risk: Success is a completed 10-step run with finite loss and reported steady-state `tps`. Risk is that the run remains slow or OOMs because the configured full activation checkpointing is still a separate missing bootstrap capability and is intentionally not bundled into the first source diff.
  Result: kept as commit `01d1f8e` with 7,254 tps, 30.31% MFU, 173.9 GiB peak memory, and finite decreasing loss. The first attempt needed an in-idea fix to wrap `lm_head` because `ChunkedCELoss` calls it directly.

- Idea: apply configured full activation checkpointing
  Current best source commit: 01d1f8e
  Source: agent-generated setup finding
  Expected mechanism: The baseline config requests `activation_checkpoint.mode="full"`; applying it after the minimal FSDP bootstrap should reduce activation memory for the long sequence and may permit larger useful work in later iterations.
  Supporting evidence: The Qwen3 scaffold currently accepts `ac_config` but does not apply activation checkpointing. The FSDP-only run peaked at 173.91 GiB, 97.51% of the reported B200 capacity, which is risky.
  Planned source/config changes: Apply the existing TorchTitan activation checkpoint helper to Qwen3 without changing FSDP, compile, or batch knobs.
  Planned command or config overrides: Reuse the baseline command with only the dump folder/run log changed for artifact isolation.
  Success criteria and expected risk: Success is equal or better 10-step completion with lower peak memory and no convergence regression. Risk is extra recompute reducing tokens/sec if memory is not the bottleneck.

- Idea: disable full activation checkpointing after baseline AC is honored
  Current best source commit: pending activation-checkpoint result
  Source: manager-generated follow-up
  Expected mechanism: If applying configured full AC reduces memory too much while costing recompute, a later command-only override can test whether disabling AC improves tokens/sec while staying under the memory risk threshold.
  Supporting evidence: Manager review suggested this as a speed experiment, but the current best without AC already uses 97.51% memory, so it should not run until after the baseline AC hook is implemented and measured.
  Planned source/config changes: None if the AC hook exists; use only the CLI override `--activation_checkpoint.mode=none`.
  Planned command or config overrides: Baseline command plus `--activation_checkpoint.mode=none`.
  Success criteria and expected risk: Success is higher step-10 tps with finite loss and peak memory below the risky range. Main risk is OOM or unstable memory near capacity.

- Idea: full BF16 training dtype
  Current best source commit: 01d1f8e
  Source: manager-generated follow-up
  Expected mechanism: Setting `--training.dtype=bfloat16` may reduce memory traffic and full-precision parameter footprint, improving throughput or memory headroom on B200.
  Supporting evidence: The current config uses `training.dtype="float32"` plus FSDP mixed-precision params. Memory is near capacity, and B200 has strong BF16 Tensor Core support.
  Planned source/config changes: None initially; command-only override.
  Planned command or config overrides: Baseline command plus `--training.dtype=bfloat16`.
  Success criteria and expected risk: Success is higher tps or lower peak memory with finite decreasing loss. Risk is numerical behavior changes or no throughput gain if FSDP/collectives dominate.

- Idea: profile current best after runnable baseline
  Current best source commit: pending first runnable result
  Source: setup roofline plan
  Expected mechanism: A profiled 10-step run should identify whether the next optimization should attack compute kernels, HBM traffic, collectives, launch overhead, or data loading.
  Supporting evidence: No TorchTitan metrics or Kineto traces exist yet on the fresh branch.
  Planned source/config changes: None. Add only profiler CLI overrides to the current-best command.
  Planned command or config overrides: Add `--profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1` to a normal 10-step command.
  Success criteria and expected risk: Success is a trace under the run dump folder and clear follow-up bottleneck classification in `learnings.md`. Do not compare profiled throughput against unprofiled runs as the primary objective.
