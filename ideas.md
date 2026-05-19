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
  Result: discarded as commit `dc2765b`. Peak memory dropped to 47.0 GiB, but tps dropped to 5,564 from the current-best 7,254.

- Idea: disable full activation checkpointing after baseline AC is honored
  Current best source commit: 01d1f8e
  Source: manager-generated follow-up
  Expected mechanism: If applying configured full AC reduces memory too much while costing recompute, a later command-only override can test whether disabling AC improves tokens/sec while staying under the memory risk threshold.
  Supporting evidence: Full AC was discarded at 5,564 tps. The current best already effectively behaves like no AC and reaches 7,254 tps, so this idea is satisfied by the current best and should not be run as a separate experiment unless AC becomes part of a later kept source state.
  Planned source/config changes: None if the AC hook exists; use only the CLI override `--activation_checkpoint.mode=none`.
  Planned command or config overrides: Baseline command plus `--activation_checkpoint.mode=none`.
  Success criteria and expected risk: Success is higher step-10 tps with finite loss and peak memory below the risky range. Main risk is OOM or unstable memory near capacity.

- Idea: full AC with larger local batch
  Current best source commit: dc2765b for the AC source state, not the current best
  Source: result-driven follow-up
  Expected mechanism: Full AC lowers peak memory to 47.0 GiB, leaving enough memory to increase local batch size and potentially convert recompute-heavy memory savings into higher useful tokens/sec.
  Supporting evidence: `dc2765b` was slower at local batch 4 but used only 26.37% memory. The program values memory savings only when converted into more tokens/sec.
  Planned source/config changes: This should be considered only if deliberately testing from the AC source state; source AC plus larger batch are coupled for this hypothesis.
  Planned command or config overrides: Candidate command would add a single local batch-size increase to the AC source state, for example `--training.local_batch_size=8`, while preserving model, data, checkpoint, and 10-step cap.
  Success criteria and expected risk: Success is tps above 7,254 with finite decreasing loss and memory below the risk threshold. Risk is comparing a bundled source-and-command state unless the source state is clearly the previously measured AC candidate.

- Idea: full BF16 training dtype
  Current best source commit: 01d1f8e
  Source: manager-generated follow-up
  Expected mechanism: Setting `--training.dtype=bfloat16` may reduce memory traffic and full-precision parameter footprint, improving throughput or memory headroom on B200.
  Supporting evidence: The current config uses `training.dtype="float32"` plus FSDP mixed-precision params. Memory is near capacity, and B200 has strong BF16 Tensor Core support.
  Planned source/config changes: None initially; command-only override.
  Planned command or config overrides: Baseline command plus `--training.dtype=bfloat16`.
  Success criteria and expected risk: Success is higher tps or lower peak memory with finite decreasing loss. Risk is numerical behavior changes or no throughput gain if FSDP/collectives dominate.
  Result: discarded at source state `73e33ba`: 7,222 tps and 160.7 GiB, slightly slower than current best but much safer memory.

- Idea: BF16 training dtype with larger local batch
  Current best source commit: 73e33ba
  Source: result-driven follow-up
  Expected mechanism: BF16-only reduced peak memory from 173.9 GiB to 160.7 GiB at local batch 4. Increasing local batch size should raise tokens per step and may improve reported tokens/sec if the extra activation memory fits.
  Supporting evidence: FSDP-only local batch 4 is at 97.51% memory, but BF16 local batch 4 is at 90.11%. The next small batch increase uses that recovered headroom directly.
  Planned source/config changes: None; command-only coupled overrides.
  Planned command or config overrides: Baseline command plus `--training.dtype=bfloat16 --training.local_batch_size=5`.
  Success criteria and expected risk: Success is tps above 7,254 with finite decreasing loss and peak memory below the OOM/risky range. Risk is OOM because activation memory may scale enough to exceed the B200 capacity.
  Result: crashed at source state `c65a743` during first backward with about 177.6 GiB in use and a failed 1.45 GiB allocation.

- Idea: profile current best after runnable baseline
  Current best source commit: 73e33ba
  Source: setup roofline plan
  Expected mechanism: A profiled 10-step run should identify whether the next optimization should attack compute kernels, HBM traffic, collectives, launch overhead, or data loading.
  Supporting evidence: Current best is runnable at 7,254 tps but uses 97.51% memory. Full AC was slower, BF16-only was slightly slower, and BF16 local batch 5 OOMed. Profiling is needed before trying more knobs.
  Planned source/config changes: None. Add only profiler CLI overrides to the current-best command.
  Planned command or config overrides: Add `--profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1` to a normal 10-step command.
  Success criteria and expected risk: Success is a trace under the run dump folder and clear follow-up bottleneck classification in `learnings.md`. Do not compare profiled throughput against unprofiled runs as the primary objective.
  Result: completed at source state `38ac2c8`; trace shows GEMM largest but NCCL collectives a close second. Profiled throughput is not used for ranking.

- Idea: BF16 training dtype with FSDP reshard-after-forward disabled
  Current best source commit: 38ac2c8
  Source: profile and roofline
  Expected mechanism: The profile shows about 0.94 s of NCCL collective GPU time in the traced step, including substantial FSDP all-gather and reduce-scatter. Disabling FSDP reshard-after-forward can reduce repeated all-gathers; BF16 dtype is included to make the larger parameter residency more likely to fit.
  Supporting evidence: BF16-only reduced peak memory to 160.7 GiB while preserving almost the same fwd/bwd timing. Current-best NCCL time is large enough that a communication/memory tradeoff could beat the FSDP-only result if it fits.
  Planned source/config changes: None.
  Planned command or config overrides: Baseline command plus `--training.dtype=bfloat16 --parallelism.fsdp_reshard_after_forward=never`.
  Success criteria and expected risk: Success is reported tps above 7,254 with finite decreasing loss. Main risk is OOM because retained all-gathered parameters may exceed the remaining B200 memory headroom.
  Result: crashed at source state `a156590`; step 1 reached 176.49 GiB and first backward failed on a 2.90 GiB allocation.

- Idea: full activation checkpointing with FSDP reshard-after-forward disabled
  Current best source commit: a156590
  Source: profile and OOM follow-up
  Expected mechanism: Full AC lowers peak memory enough to make no-reshard feasible; no-reshard may reduce repeated FSDP all-gathers, offsetting some AC recompute cost.
  Supporting evidence: Full AC alone used only 47.0 GiB but was slower. BF16 plus no-reshard OOMed, proving no-reshard needs more memory relief. The profile shows NCCL collectives are a major time bucket.
  Planned source/config changes: Re-add the measured `apply_ac(model, ac_config)` source hook before FSDP wrapping.
  Planned command or config overrides: Baseline command plus `--parallelism.fsdp_reshard_after_forward=never`.
  Success criteria and expected risk: Success is reported tps above 7,254 with finite decreasing loss. Risk is still lower tps if AC recompute costs more than saved all-gathers, or OOM if no-reshard retains too much memory even with AC.
  Result: discarded as commit `9c5ebcc`; 5,582 tps and 72.9 GiB, so no-reshard did not offset AC recompute.

- Idea: enable per-block compile on current best
  Current best source commit: 01d1f8e / restored branch source
  Source: profile and roofline
  Expected mechanism: GEMM kernels are the largest profile bucket and there is substantial CPU launch overhead. Applying TorchTitan's existing compile helper to Qwen3 blocks may improve kernel scheduling and reduce launch overhead without changing the model or parallel layout.
  Supporting evidence: The rank 0 profile shows matmul/GEMM as the largest GPU bucket, about 1.24 s, and CPU launch/command-buffer overhead is visible. AC and no-reshard tradeoffs did not improve throughput.
  Planned source/config changes: Add the existing `apply_compile(model, compile_config)` hook in Qwen3 parallelization only.
  Planned command or config overrides: Baseline command plus `--compile.enable`.
  Success criteria and expected risk: Success is reported tps above 7,254 with finite decreasing loss and no compile-induced graph/runtime failure. Risk is compile overhead or graph breaks on the short 10-step run.
  Result: kept as commit `51369ea` with 7,545 tps, 31.53% MFU, and 153.7 GiB peak memory.

- Idea: compile with BF16 training dtype
  Current best source commit: 51369ea
  Source: result-driven follow-up
  Expected mechanism: Compile improved fwd/bwd and reduced memory. BF16 may further reduce memory traffic and model footprint under the compiled source, possibly preserving the compile speedup with safer memory.
  Supporting evidence: BF16 without compile was slightly slower but lowered memory. Compile changed the kernel mix and may alter the BF16 tradeoff.
  Planned source/config changes: None.
  Planned command or config overrides: Current-best command plus `--training.dtype=bfloat16`.
  Success criteria and expected risk: Success is tps above 7,545 with finite decreasing loss and memory below current best. Risk is that BF16 remains slightly slower or changes compile behavior unfavorably.
  Result: kept at source state `48c69e1` with 8,168 tps, 34.13% MFU, and 140.3 GiB.

- Idea: compile and BF16 with local batch 5
  Current best source commit: 48c69e1
  Source: result-driven follow-up
  Expected mechanism: The current best uses only 140.3 GiB at local batch 4. Raising local batch size to 5 may improve GPU utilization and reported tokens/sec if the added activation memory fits.
  Supporting evidence: Local batch 5 OOMed before compile, but compile+BF16 reduced peak memory by about 20.4 GiB versus BF16-only and by about 33.6 GiB versus FSDP-only.
  Planned source/config changes: None.
  Planned command or config overrides: Current-best command plus `--training.local_batch_size=5`.
  Success criteria and expected risk: Success is tps above 8,168 with finite decreasing loss and no OOM. Risk is memory still exceeding capacity during backward.
  Result: kept at source state `f6ae44e` with 8,391 tps, 35.06% MFU, and 168.7 GiB peak memory.

- Idea: profile compile+BF16 local batch 5 best
  Current best source commit: f6ae44e
  Source: result-driven profile follow-up
  Expected mechanism: The best command now differs materially from the previous profiled command. A new trace should show whether compile shifted the bottleneck toward FSDP collectives, attention, optimizer, or remaining GEMMs.
  Supporting evidence: Current best is 8,391 tps and 94.61% memory. Further memory-increasing ideas are risky without better bottleneck evidence.
  Planned source/config changes: None.
  Planned command or config overrides: Current-best command plus profiler overrides.
  Success criteria and expected risk: Success is trace generation and a new roofline note. Profiled throughput is diagnostic only, not ranked against unprofiled candidates.
  Result: completed at source state `da805d8`; profile shows mixed compute/communication with memory as the immediate limiter.

- Idea: compile and BF16 with local batch 6
  Current best source commit: da805d8
  Source: result-driven boundary test
  Expected mechanism: Local batch 5 improved reported tps by increasing useful work per step. Local batch 6 may further improve tps if it fits under the B200 memory limit.
  Supporting evidence: Local batch 5 peaked at 168.74 GiB, 94.61%. This is close to the risk threshold but not over it, so batch 6 is the next integer boundary.
  Planned source/config changes: None.
  Planned command or config overrides: Current-best command with `--training.local_batch_size=6`.
  Success criteria and expected risk: Success is tps above 8,391 with finite decreasing loss. Main risk is OOM during first backward.
  Result: crashed at source state `d26bddb`; OOM before completing step 1 in `lm_head(h_chunk)` inside `ChunkedCELoss`, with about 176.62 GiB already in use.

- Idea: compile only model blocks, not loss
  Current best source commit: d26bddb
  Source: result-driven narrowing after batch-size boundary
  Expected mechanism: The current best compiles both model blocks and the loss by default. Disabling loss compilation may reduce compile/runtime overhead or memory pressure in the output projection path while retaining the model-block speedup.
  Supporting evidence: Batch size 6 OOMed in the compiled loss path at `lm_head(h_chunk)`, and the best command is close to the memory limit. The Qwen3 source compile hook only controls model-block compile; the existing core loss compile remains controlled by `compile.components`.
  Planned source/config changes: None.
  Planned command or config overrides: Current-best command plus `--compile.components '["model"]'`.
  Success criteria and expected risk: Success is tps above 8,391 with finite decreasing loss and no memory regression. Risk is that compiled loss was beneficial and disabling it lowers throughput.
  Result: crashed at source state `620b4bb`; disabling loss compile made local batch size 5 OOM during first-step loss backward.

- Idea: TP=2 with FSDP=4, compile, BF16, and local batch 5
  Current best source commit: 620b4bb
  Source: memory-bound follow-up from loss-path OOMs
  Expected mechanism: Tensor parallelism should shard attention/FFN projections and the output projection. With loss parallel enabled, the loss path should no longer materialize the full vocabulary logits per rank, reducing the memory pressure that blocked local batch 6 and model-only compile.
  Supporting evidence: The latest OOMs happen in `lm_head`/loss backward before optimizer state matters. Existing common decoder sharding helpers already encode colwise, rowwise, sequence-parallel, inner-attention local-map, and loss-parallel placements for dense decoder models.
  Planned source/config changes: Implement Qwen3 dense sharding config for TP and call `model.parallelize(tp_mesh)` before compile and FSDP wrapping. Keep CP/PP/EP unsupported for this experiment.
  Planned command or config overrides: Current-best command plus `--parallelism.tensor_parallel_degree=2 --parallelism.data_parallel_shard_degree=4`.
  Success criteria and expected risk: Success is a 10-step run with finite decreasing loss and tps above 8,391. Risk is DTensor placement mismatch in Qwen3 qk norm, attention, or chunked loss.
  Result: discarded at source state `4512daa`; TP=2 ran successfully but reached only 7,418 tps at local batch size 5, with 99.7 GiB peak memory.

- Idea: TP=2 with local batch 10
  Current best source commit: 4512daa
  Source: result-driven follow-up from TP memory headroom
  Expected mechanism: TP=2 local batch 5 only used 99.7 GiB and had global batch size 20. Raising local batch size to 10 restores global batch size 40, uses the memory released by output/loss sharding, and may recover or exceed the no-TP best throughput.
  Supporting evidence: No-TP local batch 5 reached 8,391 tps at global batch size 40 but was memory-bound at 168.7 GiB. TP=2 local batch 5 was slower but had about 69 GiB more headroom.
  Planned source/config changes: None.
  Planned command or config overrides: TP=2/FSDP=4 compile+BF16 command with `--training.local_batch_size=10`.
  Success criteria and expected risk: Success is tps above 8,391 with finite decreasing loss and no OOM. Risk is activation memory scaling beyond the available headroom or TP communication still dominating.
  Result: crashed at source state `a544471`; step 1 completed at 173.69 GiB, then rank 7 aborted after repeated allocator OOM mapping warnings.

- Idea: TP=2 with local batch 8
  Current best source commit: a544471
  Source: midpoint after TP batch 10 memory abort
  Expected mechanism: Local batch 8 should use substantially more of the TP memory headroom than batch 5 while avoiding the 97%+ memory cliff observed at batch 10. The larger per-step token count may amortize TP communication enough to approach or beat the no-TP best.
  Supporting evidence: TP batch 5 used 99.7 GiB and ran at 7,418 tps; TP batch 10 reached 173.69 GiB and aborted after step 1. Linear interpolation suggests batch 8 has a better chance to stay below the allocator cliff.
  Planned source/config changes: None.
  Planned command or config overrides: TP=2/FSDP=4 compile+BF16 command with `--training.local_batch_size=8`.
  Success criteria and expected risk: Success is tps above 8,391 with finite decreasing loss and memory below the batch 10 cliff. Risk is still slower throughput from TP communication.
  Result: discarded at source state `30886e6`; stable at 7,850 tps and 148.4 GiB, but still below no-TP best.

- Idea: async TP on TP=2 local batch 8
  Current best source commit: 30886e6
  Source: TP communication follow-up
  Expected mechanism: Async TP can overlap TP communication with compiled compute. The stable TP=2 local batch 8 run is below the no-TP best but has enough memory headroom; improving TP overlap is more promising than pushing batch size closer to the batch 10 cliff.
  Supporting evidence: TP=2 runs log a DTensor redistribution warning and remain slower than no-TP despite lower memory. TorchTitan already exposes `parallelism.enable_async_tensor_parallel`, but the Qwen3 autoresearch TP path needs to call the existing helper.
  Planned source/config changes: Call `maybe_enable_async_tp(parallelism, compile_config, tp_mesh)` in the Qwen3 TP path before model compile.
  Planned command or config overrides: TP=2/FSDP=4 compile+BF16 local batch 8 command plus `--parallelism.enable_async_tensor_parallel`.
  Success criteria and expected risk: Success is tps above 8,391 with finite decreasing loss. Risk is compile failure or no throughput gain.
  Result: discarded at source state `46e6f5e`; async TP ran but Inductor found no matching fusion patterns and throughput was 7,827 tps.

- Idea: selective activation checkpointing with compile, BF16, and local batch 6
  Current best source commit: 5abcfd3
  Source: loss-path memory boundary follow-up
  Expected mechanism: Selective AC should reduce activation memory enough for no-TP local batch 6 to fit while avoiding the large recompute cost of full AC. Combining it with compile and BF16 may preserve enough throughput for the extra batch tokens to beat local batch 5.
  Supporting evidence: No-TP compile+BF16 local batch 6 OOMed in the loss path, while full AC previously reduced memory but was too slow before the compile+BF16 improvements. TP reduced memory but added too much communication. Activation checkpointing is an allowed knob and Qwen3 14B already has an AC config.
  Planned source/config changes: Add the existing `apply_ac(model, ac_config, model_compile_enabled=...)` hook before compile and FSDP wrapping in Qwen3 parallelize.
  Planned command or config overrides: Current-best command with `--training.local_batch_size=6 --activation_checkpoint.mode=selective`.
  Success criteria and expected risk: Success is tps above 8,391 with finite decreasing loss and no OOM. Risk is selective AC overhead still outweighing the larger batch.
  Result: discarded at source state `a593f3a`; 7,701 tps and 91.7 GiB. It fixed memory but was slower.

- Idea: selective activation checkpointing with local batch 10
  Current best source commit: a593f3a
  Source: follow-up from selective AC memory headroom
  Expected mechanism: Selective AC local batch 6 used only 91.7 GiB. Raising local batch size to 10 should use the memory headroom and may amortize selective AC recompute overhead enough to beat the no-AC local-batch-5 best.
  Supporting evidence: TP batch scaling improved from local batch 5 to 8 but was communication-limited; selective AC avoids TP communication and has more memory headroom than TP batch 8. The risk threshold is memory, not correctness, because selective AC batch 6 trained normally.
  Planned source/config changes: None beyond the existing AC hook source.
  Planned command or config overrides: Selective AC compile+BF16 command with `--training.local_batch_size=10`.
  Success criteria and expected risk: Success is tps above 8,391 with finite decreasing loss and memory below the OOM cliff. Risk is that recompute cost still dominates or batch 10 exceeds memory.
  Result: invalid at source state `9b87a63`; OOM was contaminated by external VLLM processes occupying GPUs 0-3. Retry after the node is free.
  Result: discarded on retry at source state `ad6265f`; 7,434 tps and loss increased from step 1 to step 10.

- Idea: memory-budget activation checkpointing with compile, BF16, and local batch 6
  Current best source commit: ad6265f
  Source: follow-up from eager selective AC overhead
  Expected mechanism: Compiler memory-budget AC may choose a less expensive rematerialization plan than eager selective AC while still reducing enough activation memory for local batch 6 to fit.
  Supporting evidence: No-AC local batch 6 OOMed, selective AC local batch 6 fit but was slower at 7,701 tps, and selective AC local batch 10 was worse. The source hook already passes `model_compile_enabled=True`, which memory-budget mode requires.
  Planned source/config changes: None beyond the existing AC hook source.
  Planned command or config overrides: Current-best compile+BF16 local batch 6 command with `--activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.8`.
  Success criteria and expected risk: Success is tps above 8,391 with finite decreasing loss and no OOM. Risk is that memory-budget mode does not reduce enough memory or adds compile overhead.
  Result: discarded at source state `b62adad`; 8,312 tps, 34.73% MFU, 146.2 GiB. Close, but still below best.

- Idea: memory-budget activation checkpointing budget 0.9 with compile, BF16, and local batch 6
  Current best source commit: b62adad
  Source: direct tuning of promising memory-budget AC
  Expected mechanism: Raising memory budget from 0.8 to 0.9 should save fewer activations for recompute, reducing overhead while still staying below the no-AC local-batch-6 OOM point.
  Supporting evidence: Budget 0.8 reached 8,312 tps and only 146.2 GiB, leaving meaningful memory headroom. The current best is only 79 tps higher.
  Planned source/config changes: None.
  Planned command or config overrides: Same memory-budget AC local-batch-6 command with `--activation_checkpoint.memory_budget=0.9`.
  Success criteria and expected risk: Success is tps above 8,391 with finite decreasing loss. Risk is OOM if 0.9 saves too little memory.
  Result: invalid at source state `14a0c41`; OOM was contaminated by external VLLM processes occupying GPUs 4-7. Retry after the node is free.
  Result: invalid again at source state `7ef6201`; OOM was contaminated by a transient external 151 GiB process on GPU 0. Retry on a clear node.
  Result: discarded on valid retry at source state `78002e0`; 8,331 tps, 34.81% MFU, 146.2 GiB, finite decreasing loss. Close, but still below the 8,391 tps best.

- Idea: memory-budget activation checkpointing budget 0.95 with compile, BF16, and local batch 6
  Current best source commit: 78002e0
  Source: direct tuning of memory-budget AC after valid 0.8 and 0.9 runs
  Expected mechanism: A 0.95 budget may reduce rematerialization overhead compared with 0.9 while preserving enough checkpointing to avoid the no-AC local-batch-6 loss-path OOM.
  Supporting evidence: Budget 0.8 reached 8,312 tps and budget 0.9 reached 8,331 tps, both at 146.2 GiB. The best is only 60 tps higher, and reported peak memory is still well below the no-AC local-batch-5 memory.
  Planned source/config changes: None.
  Planned command or config overrides: Same memory-budget AC local-batch-6 command with `--activation_checkpoint.memory_budget=0.95`.
  Success criteria and expected risk: Success is tps above 8,391 with finite decreasing loss. Risk is OOM if the compiler stores too many activations or no speed change if the selected rematerialization plan is unchanged.
  Result: discarded at source state `4cff4ed`; 8,329 tps, 34.80% MFU, 146.2 GiB. Throughput did not improve over budgets 0.8 or 0.9, and loss barely decreased from 12.26335 to 11.98264.

- Idea: disable NCCL flight recorder on current best
  Current best source commit: f6ae44e
  Source: profile and runtime-overhead follow-up
  Expected mechanism: TorchTitan enables the torch/NCCL flight recorder by default with `comm.trace_buf_size=20000`. The current-best profile still has a large NCCL collective bucket, so disabling recorder writes/bookkeeping may reduce communication-side overhead without changing model math or layout.
  Supporting evidence: The current best is mixed compute/communication bound, with the profiled compile+BF16 local-batch-5 run showing about 1.10 s of NCCL collectives. Recent memory-budget AC tuning plateaued below best, so a low-risk command-only runtime-overhead test is worthwhile.
  Planned source/config changes: None.
  Planned command or config overrides: Current-best command plus `--comm.trace_buf_size=0`.
  Success criteria and expected risk: Success is tps above 8,391 with finite decreasing loss and similar memory. Risk is little or no effect; the tradeoff is losing timeout flight-recorder diagnostics for that run.

- Idea: FP8 rowwise linear converter on current best
  Current best source commit: f6ae44e
  Source: profile and roofline
  Expected mechanism: GEMM/compiled matmul remains the largest profile bucket, and B200 supports hardware FP8. Applying the existing `Float8LinearConverter` to Qwen3 dense linear layers may reduce GEMM time and memory bandwidth while preserving the compile+BF16 FSDP layout.
  Supporting evidence: The current best's profile is mixed compute/communication, but compiled matmul kernels are still the largest bucket. Memory-budget AC improved memory but not speed, so attacking matmul efficiency is the next distinct bottleneck. The program explicitly allows converter changes in `model_registry("14B", ...)`.
  Planned source/config changes: Edit only `qwen3_14b()` to pass `converters=[Float8LinearConverter.Config(recipe_name="rowwise", filter_fqns=["auto_filter_small_kn"], model_compile_enabled=True)]` to `model_registry("14B", ...)`, with the required import from `torchtitan.components.quantization`.
  Planned command or config overrides: Current-best command unchanged: `--compile.enable --training.dtype=bfloat16 --training.local_batch_size=5`.
  Success criteria and expected risk: Success is tps above 8,391 with finite, non-exploding loss. Risks are torchao/compile incompatibility, FP8 numerical instability, or slower kernels if conversion includes layers that do not benefit.

- Idea: memory-budget AC 0.9 with local batch 7
  Current best source commit: f6ae44e
  Source: result-driven memory-headroom follow-up
  Expected mechanism: Memory-budget AC local batch 6 nearly matched the best while using only 146.2 GiB. Increasing only local batch size to 7 may convert that unused memory into enough additional work per step to overcome the rematerialization overhead.
  Supporting evidence: Budgets 0.8, 0.9, and 0.95 all used the same 146.2 GiB at local batch 6 and reached 8,312-8,331 tps. Budget 0.9 had the healthiest recent loss trend, while 0.95 barely decreased loss. There is roughly 22 GiB of headroom below the no-AC local-batch-5 best's 168.7 GiB peak.
  Planned source/config changes: Use the existing AC-hook source state needed for memory-budget AC; no new source changes.
  Planned command or config overrides: Memory-budget AC command with `--activation_checkpoint.memory_budget=0.9 --training.local_batch_size=7`, keeping compile and BF16 enabled.
  Success criteria and expected risk: Success is tps above 8,391 with finite decreasing loss and peak memory below the OOM cliff. Risk is that memory scales above capacity or rematerialization overhead still dominates.
  Result: discarded at source state `4cff4ed`; 8,329 tps, 34.80% MFU, 146.2 GiB. Budget 0.95 did not improve on 0.9 and stayed below the best.

- Idea: BF16 optimizer states on compile+BF16 local batch 5
  Current best source commit: f6ae44e
  Source: structured metrics follow-up after memory-budget AC plateau
  Expected mechanism: The current best spends about 42-45 ms in the optimizer on later steps. `fused_opt_states_bf16` keeps the fused AdamW implementation but uses BF16 optimizer state tensors, which may reduce optimizer memory traffic and slightly improve the measured step time while preserving the no-AC local-batch-5 path.
  Supporting evidence: Memory-budget AC could not beat the best, so the next idea should target a different part of the step. Structured logs for run10 show optimizer time is small but nonzero; the remaining gap to beat is only about 60 tps. This knob is isolated to optimizer implementation and does not change model, loss, data, or parallel layout.
  Planned source/config changes: Restore Qwen3 source to the no-AC best path, then no source change for the optimizer candidate.
  Planned command or config overrides: Best compile+BF16 local-batch-5 command plus `--optimizer.implementation=fused_opt_states_bf16`.
  Success criteria and expected risk: Success is tps above 8,391 with finite decreasing loss. Risk is no speedup or a small slowdown if optimizer state dtype conversion overhead outweighs memory savings.
  Result: invalid at source state `99a75e3`; OOM was contaminated by external VLLM processes occupying GPUs 4-7. Retry after the node is free.
