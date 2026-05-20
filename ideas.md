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
  Result: discarded at source state `cb767bf`; 8,378 tps, 35.00% MFU, 168.7 GiB. Close but still below the 8,391 tps best.

- Idea: FP8 rowwise linear converter on current best
  Current best source commit: f6ae44e
  Source: profile and roofline
  Expected mechanism: GEMM/compiled matmul remains the largest profile bucket, and B200 supports hardware FP8. Applying the existing `Float8LinearConverter` to Qwen3 dense linear layers may reduce GEMM time and memory bandwidth while preserving the compile+BF16 FSDP layout.
  Supporting evidence: The current best's profile is mixed compute/communication, but compiled matmul kernels are still the largest bucket. Memory-budget AC improved memory but not speed, so attacking matmul efficiency is the next distinct bottleneck. The program explicitly allows converter changes in `model_registry("14B", ...)`.
  Planned source/config changes: Edit only `qwen3_14b()` to pass `converters=[Float8LinearConverter.Config(recipe_name="rowwise", filter_fqns=["auto_filter_small_kn"], model_compile_enabled=True)]` to `model_registry("14B", ...)`, with the required import from `torchtitan.components.quantization`.
  Planned command or config overrides: Current-best command unchanged: `--compile.enable --training.dtype=bfloat16 --training.local_batch_size=5`.
  Success criteria and expected risk: Success is tps above 8,391 with finite, non-exploding loss. Risks are torchao/compile incompatibility, FP8 numerical instability, or slower kernels if conversion includes layers that do not benefit.
  Result: kept at source state `5681e36`; 8,429 tps, 35.22% MFU, 168.7 GiB, finite decreasing loss. This is the new best.

- Idea: disable NCCL flight recorder on FP8 best
  Current best source commit: 5681e36
  Source: follow-up from two close command-only/compute-efficiency results
  Expected mechanism: FP8 rowwise conversion is now the best source state. Disabling the NCCL flight recorder was close on the previous best and may still shave a small amount of communication-side overhead on the FP8 path without changing model math, converter choice, or parallel layout.
  Supporting evidence: Run28 reached 8,378 tps versus the old 8,391 best, while run29 raised the best to 8,429 tps. The current profile diagnosis remains mixed compute/communication, and this is a command-only single-knob test.
  Planned source/config changes: None.
  Planned command or config overrides: FP8 best command plus `--comm.trace_buf_size=0`.
  Success criteria and expected risk: Success is tps above 8,429 with finite decreasing loss and similar memory. Risk is no effect or a small slowdown; timeout flight-recorder diagnostics are disabled for the run.
  Result: kept at source state `477f662`; 8,469 tps, 35.39% MFU, 168.7 GiB, finite decreasing loss. This is the new best command.

- Idea: replace Float8 rowwise with MXFP8 linear converter
  Current best source commit: 5681e36
  Source: B200 quantization follow-up
  Expected mechanism: MXFP8 is targeted at Blackwell/B200 tensor cores and may accelerate the same dense linear GEMMs more than Float8 rowwise dynamic quantization while preserving the compile+FSDP layout.
  Supporting evidence: Float8 rowwise improved throughput from 8,391 to 8,429, and the current best with flight recorder disabled is 8,469. TorchTitan's MXFP8 docs state B200 support and expected dense-GEMM speedups; Qwen3 14B dimensions are multiples of 32 for the major linear layers.
  Planned source/config changes: Edit only `qwen3_14b()` to replace `Float8LinearConverter` with `MXFP8LinearConverter` using `model_compile_enabled=True`; keep the rest of the source and command unchanged.
  Planned command or config overrides: Current best command with `--comm.trace_buf_size=0`.
  Success criteria and expected risk: Success is tps above 8,469 with finite decreasing loss. Risks are torchao/MXFP8 compatibility, slower dynamic quantization overhead, or numerical instability.
  Result: cublas recipe crashed at source state `ed317a2`; installed TorchAO does not recognize `mxfp8_cublas`. Retry the same MXFP8 line with the valid default `mxfp8_rceil` recipe.
  Result: default rceil recipe crashed at source state `92e0502`; compiled forward failed in `torchao.mxfp8_quantize` with `RuntimeError: invalid argument`. Abandon MXFP8 on this stack for now.

- Idea: FP8 rowwise without auto-filter on current best
  Current best source commit: 5681e36
  Source: Manager review after FP8 win
  Expected mechanism: The H100-derived `auto_filter_small_kn` heuristic may skip Qwen3 linear layers that are still profitable on B200. Removing it converts every dimension-compatible linear layer to Float8 rowwise and may improve GEMM coverage.
  Supporting evidence: FP8 rowwise with auto-filter improved throughput to 8,429, and with flight recorder disabled to 8,469. MXFP8 is not runnable on this stack, so the next quantization search should stay within the working Float8 path.
  Planned source/config changes: Restore `Float8LinearConverter` in `qwen3_14b()` and remove `filter_fqns=["auto_filter_small_kn"]`.
  Planned command or config overrides: Current best command with `--comm.trace_buf_size=0`.
  Success criteria and expected risk: Success is tps above 8,469 with finite decreasing loss. Risk is slower training if extra small linear layers add dynamic quantization overhead or harm numerics.
  Result: discarded at source state `582d685`; 7,814 tps, MFU N/A, 129.0 GiB. Same-batch throughput is worse, but memory headroom is much larger.

- Idea: FP8 rowwise without auto-filter local batch size 8
  Current best source commit: 5681e36
  Source: memory-headroom follow-up from broad FP8 conversion
  Expected mechanism: Broad FP8 conversion is slower at local batch 5 but uses only 129.0 GiB. Raising local batch size to 8 may convert that memory headroom into enough extra tokens per step to overcome the dynamic quantization overhead.
  Supporting evidence: The current best auto-filter FP8 path uses 168.7 GiB at local batch 5. The broad FP8 path uses 128.96 GiB at the same batch, leaving about 39.8 GiB of headroom on B200.
  Planned source/config changes: None beyond the no-auto-filter FP8 source from `582d685`.
  Planned command or config overrides: Broad FP8 command with `--training.local_batch_size=8`, keeping `--comm.trace_buf_size=0`.
  Success criteria and expected risk: Success is tps above 8,469 with finite decreasing loss. Risks are OOM if activation scaling is nonlinear, or worse numerics/throughput from broader FP8 conversion.
  Result: invalid at source state `a04a025`; OOM was contaminated by external VLLM processes occupying GPUs 4-7. Retry after the node is free.
  Result: crashed on valid retry at source state `addc7f9`; local batch 8 reached about 175.5 GiB and OOMed in the float8 `lm_head` loss path.

- Idea: FP8 rowwise without auto-filter local batch size 7
  Current best source commit: 5681e36
  Source: memory-bound midpoint after broad-FP8 batch8 OOM
  Expected mechanism: Local batch 7 should fit below the batch8 loss-path memory cliff while using more of the broad-FP8 memory headroom than local batch 5. The larger batch may amortize dynamic quantization overhead enough to beat the current best.
  Supporting evidence: Broad FP8 local batch 5 used 129.0 GiB but was slow; broad FP8 local batch 8 reached about 175.5 GiB and OOMed needing only another 2.90 GiB. A midpoint batch should be below the allocator cliff.
  Planned source/config changes: None beyond the no-auto-filter FP8 source from `582d685`.
  Planned command or config overrides: Broad FP8 command with `--training.local_batch_size=7`, keeping `--comm.trace_buf_size=0`.
  Success criteria and expected risk: Success is tps above 8,469 with finite decreasing loss. Risk is still OOM or slow throughput from broad FP8 overhead.
  Result: invalid at source state `4c7d50e`; OOM was contaminated by external VLLM processes occupying GPUs 4-7. Retry after the node is free.
  Result: discarded on valid retry at source state `4549f88`; 8,413 tps, MFU N/A, 171.2 GiB. It fits but remains slower than the auto-filter FP8 best and is memory-risky.

- Idea: profile current FP8 best
  Current best source commit: 5681e36
  Source: Manager review and post-quantization bottleneck refresh
  Expected mechanism: The current best now includes FP8 rowwise auto-filtering and `--comm.trace_buf_size=0`, so the old profile no longer reflects the best kernel/communication mix. A new profile should identify whether the next idea should target attention, remaining GEMMs, communication, or memory.
  Supporting evidence: Quantization changed throughput and the broad-FP8 memory/throughput tradeoff was non-obvious. Program.md directs profiling when ideas run low or after materially changing the best command.
  Planned source/config changes: Restore FP8 auto-filter source; no source changes for the profile run.
  Planned command or config overrides: Current best command plus profiler overrides: `--profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1`.
  Success criteria and expected risk: Success is trace generation and updated roofline notes. Profiled throughput is diagnostic and not ranked against unprofiled candidates.
  Result: completed at source state `91f4e9d`; trace shows mixed compute/communication, with flash attention backward about 599 ms, NCCL kernels about 693 ms, and B200 nvjet GEMM kernels still large.

- Idea: FP8 best with flex_flash attention backend
  Current best source commit: 5681e36
  Source: profile follow-up
  Expected mechanism: The current FP8-best profile shows flash attention backward as the largest individual kernel family. Qwen3 supports `attn_backend="flex_flash"` through `model_registry`, which may use a different compiled FlexAttention flash path and improve attention time.
  Supporting evidence: Rank 0 profile for run38 shows flash attention backward around 599 ms and forward around 156 ms in the profiled step, while other lines such as TP and AC have not beaten the best. This is a single allowed `model_spec` keyword change.
  Planned source/config changes: Edit only `qwen3_14b()` to call `model_registry("14B", attn_backend="flex_flash", converters=[...])`.
  Planned command or config overrides: Current best command with `--comm.trace_buf_size=0`.
  Success criteria and expected risk: Success is tps above 8,469 with finite decreasing loss. Risks are compile overhead, FlexAttention incompatibility with this Qwen3 mask/shape, or slower attention kernels.
  Result: crashed at source state `ad43e66`; CUTE flash attention library is unavailable, so Inductor cannot lower `BACKEND="FLASH"`.

- Idea: FP8 best with varlen attention backend
  Current best source commit: 5681e36
  Source: attention-backend follow-up after `flex_flash` environment crash
  Expected mechanism: Qwen3 supports `attn_backend="varlen"`, which uses the varlen attention path and avoids FlexAttention's unavailable CUTE flash backend. It may improve or at least change the large attention-backward bucket seen in the profile.
  Supporting evidence: Run38 profile shows flash attention backward around 599 ms. `flex_flash` could not run because CUTE flash is missing; `varlen` is the other Qwen3 attention backend likely to exercise a different attention implementation on B200.
  Planned source/config changes: Edit only `qwen3_14b()` to call `model_registry("14B", attn_backend="varlen", converters=[...])`.
  Planned command or config overrides: Current best command with `--comm.trace_buf_size=0`.
  Success criteria and expected risk: Success is tps above 8,469 with finite decreasing loss. Risks are varlen metadata overhead, compile/runtime incompatibility, or slower attention.
  Result: crashed at source state `06f1c41`; `flash_attn_interface` is missing when varlen attention tries to activate FA3.

- Idea: FP8 best with flex attention backend
  Current best source commit: 5681e36
  Source: attention-backend follow-up after `flex_flash` and `varlen` library crashes
  Expected mechanism: Plain `attn_backend="flex"` uses compiled FlexAttention without forcing `BACKEND="FLASH"`, so it may run without the missing CUTE/FA3 libraries and provide a different attention implementation for the profile-visible attention bucket.
  Supporting evidence: Run38 profile shows attention backward is large. `flex_flash` and `varlen` are blocked by missing libraries, leaving plain `flex` as the only remaining Qwen3 attention backend in scope.
  Planned source/config changes: Edit only `qwen3_14b()` to call `model_registry("14B", attn_backend="flex", converters=[...])`.
  Planned command or config overrides: Current best command with `--comm.trace_buf_size=0`.
  Success criteria and expected risk: Success is tps above 8,469 with finite decreasing loss. Risks are slower attention, compile failure, or mask overhead.
  Result: discarded at source state `67967a3`; 8,544 tps but loss increased from 12.41705 to 13.88057, so it fails correctness sanity.

- Idea: FP8 rowwise_with_gw_hp recipe on current best
  Current best source commit: 5681e36
  Source: quantization recipe follow-up
  Expected mechanism: `rowwise_with_gw_hp` changes the Float8 training recipe and may preserve more high-precision gradient-weight behavior while still using FP8 for profitable linear kernels. It could improve correctness margin or kernel behavior relative to default rowwise.
  Supporting evidence: Float8 rowwise auto-filter is the best source line so far. Broad conversion and alternate attention backends did not produce a keepable improvement, so the next quantization search should stay close to the working converter but try the other supported recipe.
  Planned source/config changes: Restore FP8 auto-filter source, then change only `recipe_name` from `"rowwise"` to `"rowwise_with_gw_hp"`.
  Planned command or config overrides: Current best command with `--comm.trace_buf_size=0`.
  Success criteria and expected risk: Success is tps above 8,469 with finite decreasing loss. Risk is slower throughput from extra high-precision work.
  Result: crashed at source state `17c8bd9`; torchao's `_auto_filter_for_recipe` does not support `ROWWISE_WITH_GW_HP`, so this recipe cannot be tested with the kept `auto_filter_small_kn` coverage.

- Idea: disable structured trace logging on FP8 best
  Current best source commit: 477f662
  Source: overhead-reduction follow-up after source/backend variants
  Expected mechanism: `debug.enable_structured_logging` defaults to true and TorchTitan records many per-rank spans during training. Disabling structured logging may reduce Python/context-manager overhead without changing model math, parallelism, data, checkpoint, or communication settings.
  Supporting evidence: Disabling NCCL flight-recorder logging improved the FP8 best from 8,429 to 8,469 tps. The remaining source-level quantization and attention variants either crashed, lost throughput, or failed loss sanity, so the next single idea should target measurement/runtime overhead on the kept source line.
  Planned source/config changes: Restore the FP8 rowwise auto-filter source; no source changes for the candidate.
  Planned command or config overrides: Current best command plus `--debug.enable_structured_logging=False`.
  Success criteria and expected risk: Success is tps above 8,469 with finite decreasing loss. Risk is that structured logging overhead is negligible or the CLI spelling for disabling the boolean needs adjustment.
  Result: discarded at source state `748f640`; 8,439 tps, 35.26% MFU, 168.7 GiB, finite decreasing loss. Keep structured logging enabled.

- Idea: FP8 best with FSDP reshard-after-forward disabled
  Current best source commit: 477f662
  Source: communication-overhead follow-up from profile
  Expected mechanism: The FP8-best profile still showed large NCCL all-gather and reduce-scatter kernel time. Setting `--parallelism.fsdp_reshard_after_forward=never` may reduce repeated FSDP all-gathers and improve tokens/sec if the extra parameter residency fits.
  Supporting evidence: Run38 profile attributed about 693 ms of rank-0 CUDA kernel time to NCCL kernels, with all-gather and reduce-scatter both substantial. Previous no-reshard testing before the current FP8 source line OOMed, but the current best includes FP8 conversion and disabled NCCL flight recorder, so the exact best command has not tested this tradeoff.
  Planned source/config changes: None; use the restored FP8 rowwise auto-filter source.
  Planned command or config overrides: Current best command plus `--parallelism.fsdp_reshard_after_forward=never`.
  Success criteria and expected risk: Success is tps above 8,469 with finite decreasing loss. Main risk is OOM because the current best already reaches 168.7 GiB peak memory.
  Result: crashed at source state `401de15`; first-step loss backward OOMed while trying to allocate 2.90 GiB with the training process already at about 175.35 GiB.

- Idea: FP8 no-reshard with local batch size 4
  Current best source commit: 477f662
  Source: follow-up to no-reshard local batch 5 OOM
  Expected mechanism: Lowering local batch size by one should reduce activation memory enough for `fsdp_reshard_after_forward=never` to fit. If no-reshard removes enough FSDP all-gather overhead, it may compensate for the smaller global batch and beat the current tokens/sec best.
  Supporting evidence: Run44 OOMed while trying to allocate 2.90 GiB with about 175.35 GiB already held by the training process. Current best local batch 5 with resharding uses 168.7 GiB and the profile shows all-gather/reduce-scatter time is material. Local batch 4 is the nearest lower-memory no-reshard test.
  Planned source/config changes: None; use the restored FP8 rowwise auto-filter source.
  Planned command or config overrides: Current best command plus `--parallelism.fsdp_reshard_after_forward=never --training.local_batch_size=4`.
  Success criteria and expected risk: Success is tps above 8,469 with finite decreasing loss. Risks are slower throughput from smaller batch size or residual OOM if no-reshard residency dominates.
  Result: discarded at source state `d8ba324`; it fit but reached only 6,939 tps, 28.99% MFU, 165.7 GiB, with finite decreasing loss.

- Idea: flex attention without FP8 converter
  Current best source commit: 477f662
  Source: correctness isolation after fast flex run
  Expected mechanism: Run41 with flex attention plus FP8 was faster than the best but failed loss sanity. Removing the FP8 converter while keeping flex attention tests whether the bad loss came from the FP8+flex interaction rather than flex attention itself. If loss becomes healthy, flex may still beat the current best or identify a keepable attention path.
  Supporting evidence: Run41 reached 8,544 tps, above the 8,469 best, but loss increased from 12.41705 to 13.88057. The kept FP8 sdpa path trains normally, so the invalidity is specific to flex attention or its interaction with FP8.
  Planned source/config changes: In `qwen3_14b()`, set `attn_backend="flex"` and remove the Float8 converter; keep the rest of the current-best source and FSDP/compile path unchanged.
  Planned command or config overrides: Current best command shape with `--comm.trace_buf_size=0`, `--compile.enable`, `--training.dtype=bfloat16`, and `--training.local_batch_size=5`.
  Success criteria and expected risk: Success is tps above 8,469 with finite decreasing loss. Risk is that removing FP8 loses too much throughput or flex attention still fails loss sanity.
  Result: kept at source state `5801b0f`; 8,489 tps, 35.47% MFU, 169.0 GiB, finite decreasing loss. This is the new best and shows plain flex attention is correct enough for the short sanity check.

- Idea: flex attention with lower learning rate
  Current best source commit: 477f662
  Source: correctness recovery after fast flex run
  Expected mechanism: If flex attention's higher throughput is valid but its short-run loss is update-size sensitive, lowering the learning rate may restore the 10-step loss trend without changing kernel/runtime performance materially.
  Supporting evidence: Run41 had the best raw throughput at 8,544 tps but loss increased. The configured LR is 8e-4 with a 10-step adjusted warmup; a smaller LR such as 4e-4 changes optimizer update size while preserving the fast flex kernels.
  Planned source/config changes: Restore the run41 source shape: FP8 rowwise auto-filter converter with `attn_backend="flex"`.
  Planned command or config overrides: Run41 command plus `--optimizer.lr=4e-4`.
  Success criteria and expected risk: Success is tps above 8,469 with finite decreasing loss. Risk is that LR does not fix a semantic/masking issue, or that lower LR only masks an invalid attention behavior.
  Result: discarded at source state `be0f401`; lower LR recovered finite decreasing loss, but throughput was 8,432 tps, below the 8,489 flex-without-FP8 best.

- Idea: FP8 flex attention with intermediate learning rate
  Current best source commit: 5801b0f
  Source: follow-up after FP8 flex lower-LR recovery
  Expected mechanism: Run41 at LR 8e-4 was faster than the best but failed loss sanity, while run47 at LR 4e-4 recovered decreasing loss but was slower. Testing LR 6e-4 checks whether an intermediate update size keeps the FP8+flex loss trend healthy while preserving more of the faster kernel path.
  Supporting evidence: Run47 showed the FP8+flex loss issue is not an immediate semantic failure; it can pass the 10-step sanity check with a lower LR. The throughput gap may include run-to-run variance, so one intermediate LR is justified before abandoning FP8+flex.
  Planned source/config changes: Use FP8 rowwise auto-filter converter with `attn_backend="flex"`.
  Planned command or config overrides: Current best command shape plus `--optimizer.lr=6e-4`.
  Success criteria and expected risk: Success is tps above 8,489 with finite decreasing loss. Risk is that loss still increases or throughput remains below the flex-without-FP8 best.
  Result: discarded at source state `a5a7276`; loss decreased but throughput was only 8,325 tps, below the 8,489 flex-without-FP8 best.

- Idea: FP8 rowwise_with_gw_hp without auto-filter
  Current best source commit: 477f662
  Source: quantization recipe fallback after auto-filter crash
  Expected mechanism: `rowwise_with_gw_hp` crashed only because torchao's `_auto_filter_for_recipe` does not support that recipe. Removing `auto_filter_small_kn` tests the recipe itself on this stack while accepting broader conversion coverage.
  Supporting evidence: Run42 failed before training with `Unsupported recipe: ROWWISE_WITH_GW_HP`. Broad default-rowwise conversion without auto-filter was slower, so this is lower priority than flex correctness isolation, but it is the only remaining way to evaluate the high-precision-gradient FP8 recipe.
  Planned source/config changes: In `qwen3_14b()`, use `Float8LinearConverter.Config(recipe_name="rowwise_with_gw_hp", model_compile_enabled=True)` with no `filter_fqns`.
  Planned command or config overrides: Current best command shape with `--comm.trace_buf_size=0`, `--compile.enable`, `--training.dtype=bfloat16`, and `--training.local_batch_size=5`.
  Success criteria and expected risk: Success is tps above 8,469 with finite decreasing loss. Risk is slower throughput from broader conversion and high-precision gradient-weight work.
  Result: discarded at source state `bca5b86`; it ran without auto-filter but reached only 6,226 tps, used 172.4 GiB, emitted allocator retries, and warned about FSDPFloat8Linear returning a view tensor.

- Idea: profile current flex-attention best
  Current best source commit: 5801b0f
  Source: profile refresh after new best
  Expected mechanism: The current best changed the attention backend and removed FP8, so the previous FP8-best profile no longer describes the kept kernel/communication mix. Profiling should identify whether the next candidate should target attention, GEMMs, FSDP communication, loss/lm_head, or memory pressure.
  Supporting evidence: Run46 became the new best at 8,489 tps with flex attention and no FP8 converter. Earlier run38 profile showed attention and NCCL were both significant, but it used the old FP8 SDPA source line.
  Planned source/config changes: None; use the current flex-without-FP8 best source.
  Planned command or config overrides: Current best command plus `--profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1`.
  Success criteria and expected risk: Success is trace generation plus updated bottleneck notes. Profiled tps is diagnostic only and should not be ranked against unprofiled candidates.
  Result: completed at source state `9f96d09`; rank 0 trace shows about 3.75 s CUDA kernel time, dominated by nvjet GEMMs at about 1.80 s, NCCL kernels at about 0.94 s, and flex-attention kernels at about 0.66 s.

- Idea: flex attention best with local batch size 6
  Current best source commit: 5801b0f
  Source: memory boundary check after new best
  Expected mechanism: Increasing local batch size from 5 to 6 raises useful tokens per step and may improve reported tokens/sec if it fits. This is the direct memory-to-throughput conversion for the current best source.
  Supporting evidence: Run46 uses 168.96 GiB, 94.73% of B200 capacity, so local batch 6 is risky but not proven for the flex source. Earlier local-batch-6 attempts on older source lines OOMed, but the current best changed the attention backend and removed FP8.
  Planned source/config changes: None; use current flex-without-FP8 best source.
  Planned command or config overrides: Current best command with `--training.local_batch_size=6`.
  Success criteria and expected risk: Success is tps above 8,489 with finite decreasing loss. Main risk is OOM or allocator instability above the 95% memory-risk line.
  Result: crashed at source state `4e5b73f`; local batch size 6 reached about 177 GiB in the training process and OOMed during first-step loss/backward.

- Idea: flex attention with memory-budget AC and local batch size 7
  Current best source commit: 5801b0f
  Source: memory-relief follow-up after local batch 6 OOM
  Expected mechanism: Memory-budget activation checkpointing can trade recompute for activation memory, allowing a larger local batch. Local batch 7 may recover enough useful tokens per step to offset AC overhead while preserving the flex-attention speedup.
  Supporting evidence: Run51 proved direct local batch 6 OOMs. Earlier memory-budget AC on older source lines lowered memory to about 146 GiB at local batch 6 and got close to the then-best, but did not include the current flex-attention source.
  Planned source/config changes: Re-add the standard `apply_ac` hook in Qwen3 parallelize, before compile, using the existing TorchTitan helper.
  Planned command or config overrides: Current best command shape with `--activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.9 --training.local_batch_size=7`.
  Success criteria and expected risk: Success is tps above 8,489 with finite decreasing loss. Risks are recompute overhead, OOM at local batch 7, or compile/memory-budget incompatibility.
  Result: discarded at source state `7d49b01`; local batch 7 fit with 7,840 tps, 32.76% MFU, 167.3 GiB, and finite decreasing loss.

- Idea: flex attention with memory-budget AC and local batch size 8
  Current best source commit: 5801b0f
  Source: memory-budget AC batch scaling follow-up
  Expected mechanism: Local batch 7 with memory-budget AC fit below the memory risk line but was slower. Increasing to local batch 8 may better amortize recompute overhead and use the memory headroom.
  Supporting evidence: Run52 local batch 7 used only 167.27 GiB with AC 0.9, lower than the no-AC local batch 5 best. This leaves enough headroom to justify one larger-batch AC test.
  Planned source/config changes: Keep the `apply_ac` hook source from run52.
  Planned command or config overrides: Run52 command with `--training.local_batch_size=8`.
  Success criteria and expected risk: Success is tps above 8,489 with finite decreasing loss. Risks are OOM, allocator instability, or continued recompute overhead.
  Result: crashed at source state `2ec5fb2`; local batch 8 OOMed in compiled flex-attention backward. The local batch 7 AC result was already too slow, so abandon this AC branch for now.

- Idea: flex attention best with Inductor GEMM max autotune
  Current best source commit: 5801b0f
  Source: profile follow-up on GEMM bucket
  Expected mechanism: Rank 0 profile for the flex best shows B200 nvjet GEMM kernels as the largest CUDA kernel bucket. Enabling Inductor GEMM max autotune may select faster matmul kernels for the compiled blocks and improve reported tokens/sec.
  Supporting evidence: Run49 profile attributed about 1.80 s of the profiled step to nvjet GEMM kernels, larger than NCCL or flex-attention kernels. This is a command/environment-only tuning knob and does not change model math.
  Planned source/config changes: None; use current flex-without-FP8 best source.
  Planned command or config overrides: Current best command with `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1` in the environment.
  Success criteria and expected risk: Success is tps above 8,489 with finite decreasing loss. Risks are longer compile time, no kernel change, or worse selected kernels.
  Result: discarded at source state `f2eba58`; tps dropped to 4,816 and memory rose to 171.8 GiB with allocator retries.

- Idea: flex attention with context parallel degree 2
  Current best source commit: 5801b0f
  Source: attention/memory profile follow-up
  Expected mechanism: Context parallelism shards the sequence dimension across two ranks and wraps flex attention to all-gather K/V. With local batch size 10 and CP=2, the effective token count is comparable to DP-only local batch 5 while each rank sees a shorter query sequence, potentially reducing flex-attention memory pressure and improving attention time.
  Supporting evidence: Run49 profile shows flex-attention kernels are still about 659 ms and memory is near the risk line. Run51 showed direct batch increase OOMs. Qwen3 model config allows CP for FlexAttention, and TorchTitan has a standard `apply_cp_to_forward` wrapper.
  Planned source/config changes: Allow CP in Qwen3 parallelize and call `apply_cp_to_forward` on each layer's inner attention before compile.
  Planned command or config overrides: Current best command plus `--parallelism.context_parallel_degree=2 --parallelism.context_parallel_load_balancer=ptrr --training.local_batch_size=10`.
  Success criteria and expected risk: Success is tps above 8,489 with finite decreasing loss. Risks are CP wrapper incompatibility, load-balancer/mask issues, extra K/V all-gather overhead, or higher parameter memory because FSDP degree falls from 8 to 4.
  Result: discarded at source state `c36ca11`; it ran but reached only 4,531 tps, used 170.2 GiB with allocator retries, and loss increased from 12.30233 to 17.01370.

- Idea: flex attention best with structured logging disabled
  Current best source commit: 5801b0f
  Source: overhead check on new best source
  Expected mechanism: Disabling structured logging may reduce per-step Python trace overhead for the current flex source, even though it did not help the older FP8 SDPA source.
  Supporting evidence: Run46 became the new best after run43's structured-logging test on a different source line. The command-only knob is cheap to isolate.
  Planned source/config changes: None; use current flex-without-FP8 best source.
  Planned command or config overrides: Current best command plus `--debug.no-enable-structured-logging`.
  Success criteria and expected risk: Success is tps above 8,489 with finite decreasing loss. Risk is no effect or a small regression, as seen on the older FP8 source.
  Result: discarded at source state `b29e75a`; 7,455 tps and loss increased from 12.41723 to 19.26537.

- Idea: flex attention best with lm_head-only no-reshard
  Current best source commit: 5801b0f
  Source: Manager review after run56
  Expected mechanism: `ChunkedCELoss` calls `model.lm_head` directly for each hidden-state chunk. Keeping only the separately FSDP-wrapped `lm_head` unresharded after forward may avoid repeated loss-path all-gathers without retaining every transformer block's full parameters.
  Supporting evidence: Full no-reshard at local batch size 5 OOMed, and no-reshard at local batch size 4 was far too slow, so whole-model no-reshard is closed. The current flex profile still shows about 0.94 s NCCL kernels, and the current best has roughly 9 GiB of physical memory headroom. The 14B `lm_head` BF16 weight is about 1.55 GiB, so an `lm_head`-only residency tradeoff is much narrower than whole-model no-reshard.
  Planned source/config changes: In Qwen3 FSDP wrapping, keep transformer blocks and root on the existing reshard policy, but call `fully_shard(model.lm_head, ..., reshard_after_forward=False)` as the only policy exception.
  Planned command or config overrides: Current flex best command unchanged: `--compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0`.
  Success criteria and expected risk: Success is tps above 8,489 with finite decreasing loss. Risks are OOM/allocator retries if loss chunks retain more than expected, or no benefit if FSDP already avoids repeated `lm_head` all-gathers.
  Result: discarded at source state `75e484f`; loss decreased from 12.32550 to 7.94975, but step-10 throughput fell to 6,922 tps and MFU to 28.92%.

- Idea: flex attention best with explicit FSDP prefetch schedule
  Current best source commit: 5801b0f
  Source: Manager review after run56
  Expected mechanism: Explicitly setting one-module-ahead FSDP forward and backward prefetch on Qwen3 transformer blocks can overlap part of the all-gather work with compute. This attacks the 0.94 s NCCL bucket without changing model math, precision, batch size, or attention backend.
  Supporting evidence: Current Qwen3 parallelization wraps each block and `lm_head` but does not set `set_modules_to_forward_prefetch` or `set_modules_to_backward_prefetch`. The transformers-modeling backend has a reference prefetch pattern for decoder blocks. Batch growth, AC, CP, and full no-reshard are already slower or OOM, so overlap is the remaining narrow communication lever.
  Planned source/config changes: Add a Qwen3-specific prefetch chain after FSDP wrapping: each block prefetches the next block in forward, `lm_head` or the final wrapped module at the tail, and each block prefetches the previous block in backward. Do not change reshard policy, batch size, AC, TP, CP, or converters in the same run.
  Planned command or config overrides: Current flex best command unchanged.
  Success criteria and expected risk: Success is tps above 8,489 with finite decreasing loss. Risk is higher peak memory or allocator retries from earlier all-gathers; if that happens, discard rather than pairing with AC immediately.
  Result: kept at source state `7c1c351`; 8,835 tps, 36.91% MFU, 168.10 GiB, and loss decreased from 12.19318 to 6.47119.

- Idea: flex attention with FP8 rowwise auto-filter but BF16 lm_head
  Current best source commit: 5801b0f
  Source: Manager review after run56
  Expected mechanism: Run41 showed flex attention plus FP8 rowwise was the fastest raw path but failed loss sanity. Skipping only `lm_head` in the Float8 converter keeps the loss projection in BF16 while preserving FP8 for most transformer-block GEMMs, testing whether the invalid loss was driven by quantized logits rather than flex attention or block GEMMs.
  Supporting evidence: Flex without FP8 is the current best and trains normally. FP8+flex at lower learning rates trained but was slower, while high-precision FP8 without auto-filter was much slower and memory-risky. A `filter_fqns` skip for `lm_head` is a narrow allowed converter-coverage change, not a broad recipe sweep.
  Planned source/config changes: In `qwen3_14b()`, use `model_registry("14B", attn_backend="flex", converters=[Float8LinearConverter.Config(recipe_name="rowwise", filter_fqns=["auto_filter_small_kn", "lm_head"], model_compile_enabled=True)])`.
  Planned command or config overrides: Current flex best command shape with `--compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0`.
  Success criteria and expected risk: Success is tps above 8,489 with finite decreasing loss. Risks are still-invalid loss if attention/MLP FP8 is the issue, or slower throughput if `lm_head` FP8 supplied most of the speedup.
  Attempt: run60 at source state `e213be2` was invalid; external VLLM workers appeared on GPUs 4-7 and caused a contaminated OOM before any step metrics.
  Result: discarded on valid retry at source state `e213be2`; 8,762 tps with finite decreasing loss, below the 8,835 prefetch best.

- Idea: rerun exact prefetch flex best
  Current best source commit: 7c1c351
  Source: variance check after new FSDP prefetch best
  Expected mechanism: Repeating the exact current-best command measures run-to-run variance in both throughput and short loss trend without introducing another source or config variable.
  Supporting evidence: Run59 improved the best from 8,489 to 8,835 tps, while the selective-FP8 follow-up reached 8,762 and preserved loss sanity. The current search has shown non-trivial throughput variance, so a direct rerun can either raise the best or confirm the prefetch gain is robust.
  Planned source/config changes: None; use current prefetch source with flex attention and no FP8 converter.
  Planned command or config overrides: Exact run59 command with a new dump folder.
  Success criteria and expected risk: Keep if it beats 8,835 with finite decreasing loss; otherwise record as diagnostic variance/discard while preserving run59 as the best measured result.
  Result: discarded at source state `025f0c3`; 8,829 tps with finite decreasing loss, close to but below the 8,835 prefetch best.

- Idea: profile prefetch flex best
  Current best source commit: 7c1c351
  Source: bottleneck refresh after FSDP prefetch became the best
  Expected mechanism: Profiling the current best should show how much of the old NCCL bucket was hidden by explicit prefetch, and whether the next target should be GEMM, attention, loss/lm_head, remaining collectives, or CPU launch overhead.
  Supporting evidence: The pre-prefetch flex profile showed about 1.80 s nvjet GEMMs, 0.94 s NCCL kernels, and 0.66 s flex attention kernels. Explicit prefetch improved throughput from 8,489 to 8,835 and reran at 8,829, so the bottleneck mix has likely changed.
  Planned source/config changes: None; use current prefetch source with flex attention and no FP8 converter.
  Planned command or config overrides: Add `--profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1` to the exact prefetch best command.
  Success criteria and expected risk: Success is trace generation and a concrete bottleneck note in `learnings.md`. Profiled throughput is diagnostic only and should not be ranked against unprofiled candidates.
  Result: completed at source state `1e8f047`; profiled step-10 tps was 8,484. Rank0 trace shows ~5.61 s total CUDA kernel time, with ~2.40 s NCCL kernels, ~2.05 s nvjet GEMM kernels, and ~0.78 s flex-attention kernels.

- Idea: two-module FSDP prefetch window on prefetch flex best
  Current best source commit: 7c1c351
  Source: profile follow-up after run63
  Expected mechanism: The one-module prefetch schedule improved throughput by overlapping FSDP all-gathers with compute. Asking each module to prefetch the next two modules may hide more of the remaining all-gather and backward parameter-gather latency while keeping the same math, precision, batch size, and attention backend.
  Supporting evidence: Run63 still shows 0.63 s rank0 all-gather kernel time and 2.40 s total NCCL kernel time, while memory remains at the same reported 168.10 GiB. The B200 nvidia-smi allocation is close to capacity, so this is a narrow test rather than a broad no-reshard change.
  Planned source/config changes: In Qwen3 FSDP wrapping, change the explicit prefetch lists so each layer prefetches up to the next two FSDP modules in forward, and each layer/lm_head prefetches up to the previous two modules in backward. Do not change reshard policy, converters, AC, batch size, or command flags.
  Planned command or config overrides: Exact current best command with a new dump folder.
  Success criteria and expected risk: Success is tps above 8,835 with finite decreasing loss. Risk is valid OOM or lower tps from too much early parameter residency/communication contention.
  Result: discarded at source state `388fdfd`; 8,627 tps with finite decreasing loss, and peak memory rose to 169.33 GiB.

- Idea: forward-only FSDP prefetch on prefetch flex best
  Current best source commit: 7c1c351
  Source: asymmetric prefetch follow-up after run63 and run64
  Expected mechanism: Keep the forward one-module-ahead prefetch that can hide all-gathers before compute, but remove backward prefetch to reduce contention with FSDP reduce-scatter, now the largest visible NCCL bucket in the profile.
  Supporting evidence: Run64 showed wider prefetch is worse and raises memory. Run63 showed 1.76 s rank0 reduce-scatter kernel time versus 0.63 s all-gather kernel time, suggesting backward communication contention is a plausible remaining limiter.
  Planned source/config changes: In Qwen3 FSDP wrapping, keep the current one-module forward prefetch chain through `lm_head`, but remove `set_modules_to_backward_prefetch` calls. Do not change batch size, reshard policy, converters, AC, or command flags.
  Planned command or config overrides: Exact current best command with a new dump folder.
  Success criteria and expected risk: Success is tps above 8,835 with finite decreasing loss. Risk is lower tps if backward prefetch was hiding necessary parameter all-gathers.
  Result: discarded at source state `1996329`; 8,596 tps with finite slightly decreasing loss, below the 8,835 bidirectional prefetch best.

- Idea: backward-only FSDP prefetch on prefetch flex best
  Current best source commit: 7c1c351
  Source: asymmetric prefetch follow-up after run65
  Expected mechanism: Keep backward one-module prefetch while removing forward prefetch. This isolates whether the current best depends mainly on backward parameter-gather overlap, and it may reduce forward-pass communication contention.
  Supporting evidence: Forward-only prefetch regressed to 8,596 tps, so backward prefetch appears useful. The complementary backward-only test is the smallest remaining way to identify whether forward prefetch is also necessary.
  Planned source/config changes: In Qwen3 FSDP wrapping, remove `set_modules_to_forward_prefetch` calls and keep the one-module backward prefetch chain from `lm_head` through the transformer blocks. Do not change batch size, reshard policy, converters, AC, or command flags.
  Planned command or config overrides: Exact current best command with a new dump folder.
  Success criteria and expected risk: Success is tps above 8,835 with finite decreasing loss. Risk is lower tps if forward prefetch was hiding important forward all-gathers.
  Attempt: run66 at source state `579e621` was invalid; external VLLM workers appeared on GPUs 4-7 and caused a contaminated OOM before any step metrics.
  Result: discarded on valid retry at source state `579e621`; 8,387 tps with finite decreasing loss, below the 8,835 bidirectional prefetch best.

- Idea: BF16 FSDP reduce dtype on prefetch flex best
  Current best source commit: 7c1c351
  Source: Manager review after run65
  Expected mechanism: Change only the FSDP mixed-precision reduce dtype from FP32 to BF16 so reduce-scatter communicates half as many bytes. This directly targets the largest remaining profile bucket without changing batch size, attention backend, FSDP prefetch policy, or parameter dtype.
  Supporting evidence: Run63 profile on the current best shows about 2.40 s NCCL kernels on rank 0, dominated by about 1.76 s reduce-scatter. The config type currently exposes `training.mixed_precision_reduce` only as `"float32"`, so this should be an explicit Qwen3 `parallelize.py` experiment rather than a CLI-only override.
  Planned source/config changes: In Qwen3 `parallelize_qwen3()`, keep `param_dtype` from `training.mixed_precision_param` but set `MixedPrecisionPolicy(reduce_dtype=torch.bfloat16)` for this candidate only. Keep the run59 one-module prefetch schedule unchanged.
  Planned command or config overrides: Exact current best command with a new dump folder.
  Success criteria and expected risk: Success is tps above 8,835 with finite decreasing loss. Risks are short-run loss regression from BF16 gradient reductions or no speedup if NCCL is already overlapped enough that reduced payload does not affect step time.
  Result: discarded at source state `f67bbcb`; 8,612 tps, loss increased from 12.51039 to 12.52511, and grad_norm printed as 0.0000 at both logged steps.

- Idea: HSDP 2x4 on prefetch flex best
  Current best source commit: 7c1c351
  Source: Manager review after run65
  Expected mechanism: Use two data-parallel replicas and four-rank FSDP shards (`dp_replicate=2`, `dp_shard=4`) to reduce the per-shard FSDP collective group size while preserving the same 8-GPU workload. This may reduce the exposed reduce-scatter/all-gather cost now dominating the profile.
  Supporting evidence: Whole-model no-reshard and TP were too costly, but run63 shows communication is now the largest visible bucket. HSDP is less invasive than TP because it keeps Qwen3 tensor shapes, attention backend, compile path, and local batch size unchanged while changing only the data-parallel mesh.
  Planned source/config changes: Narrowly allow `parallel_dims.dp_replicate > 1` in Qwen3 `parallelize_qwen3()` and select `parallel_dims.get_mesh(["dp_replicate", "fsdp"])` for `fully_shard`; preserve the current one-module prefetch schedule. Do not add TP, CP, AC, FP8, or batch-size changes in the same run.
  Planned command or config overrides: Current best command plus `--parallelism.data_parallel_replicate_degree=2 --parallelism.data_parallel_shard_degree=4`.
  Success criteria and expected risk: Success is tps above 8,835 with finite decreasing loss and no memory regression. Risks are slower replica synchronization, incorrect gradient scaling if HSDP interacts badly with the current disabled FSDP gradient division, or lower MFU from smaller shard groups.
  Result: discarded at source state `316182a`; 1,984 tps, 8.29% MFU, 172.10 GiB, and 37 CUDA allocation retries.

- Idea: separately FSDP-wrap tok_embeddings with terminal backward prefetch
  Current best source commit: 7c1c351
  Source: Manager review after run65
  Expected mechanism: Separately wrapping `tok_embeddings` and adding it as the final backward-prefetch target may expose and overlap embedding/root FSDP communication instead of leaving it inside the root wrapper.
  Supporting evidence: The transformers-modeling backend separately shards `tok_embeddings` and uses terminal backward prefetch to embeddings. Run59 proved prefetch scheduling matters materially, while wider prefetch and forward-only variants were worse. This is a narrower source-structure refinement than reopening TP, CP, AC, or broad FP8.
  Planned source/config changes: In Qwen3 `parallelize_qwen3()`, call `fully_shard(model.tok_embeddings, **fsdp_config)` before the transformer blocks, keep `lm_head` as currently wrapped, and make the first transformer block prefetch `model.tok_embeddings` in backward as the terminal target. Keep the one-module prefetch window.
  Planned command or config overrides: Exact current best command with a new dump folder.
  Success criteria and expected risk: Success is tps above 8,835 with finite decreasing loss. Risks are no improvement if root communication is not material, or extra all-gather scheduling overhead from an additional FSDP unit.
  Result: kept at source state `b6ccf9c`; 8,847 tps, 36.96% MFU, 167.77 GiB, and loss decreased from 12.45754 to 8.00693.

- Idea: rerun exact embedding-prefetch best
  Current best source commit: b6ccf9c
  Source: variance check after new run70 best
  Expected mechanism: Repeating the exact new best command measures whether the 12 tps improvement over run59 is stable or just measurement variance.
  Supporting evidence: The prior prefetch best reran within 6 tps, and run70 improves by only 12 tps while also reducing peak memory slightly. A direct rerun is the fastest way to validate this as the current best source before profiling or adding more changes.
  Planned source/config changes: None; use the current embedding-prefetch source.
  Planned command or config overrides: Exact run70 command with a new dump folder.
  Success criteria and expected risk: Keep if it beats 8,847 with finite decreasing loss; otherwise record as diagnostic variance while preserving the best measured source if it remains close.
  Result: discarded at source state `d876c83`; 4,790 tps and loss increased from 12.23535 to 15.69855.
  Second result: discarded at source state `d876c83`; 8,524 tps with finite decreasing loss, below both run70 and the older run59/run62 one-module prefetch results.

- Idea: separate tok_embeddings FSDP wrap without embedding prefetch
  Current best source commit: b6ccf9c
  Source: variance follow-up after run70-run72
  Expected mechanism: Separately sharding `tok_embeddings` may reduce root-wrapper communication or residency, while the extra embedding-specific prefetch edges may add scheduling noise. Keeping the separate wrap but reverting to the proven transformer-block/lm_head prefetch chain isolates these effects.
  Supporting evidence: Run70 was a best measured run, but exact reruns were unstable. The stable run59/run62 source used only transformer-block/lm_head prefetch. This candidate tests whether the small memory reduction from separately wrapping embeddings is useful without changing the prefetch graph at the model ends.
  Planned source/config changes: Keep `fully_shard(model.tok_embeddings, **fsdp_config)`, remove `model.tok_embeddings.set_modules_to_forward_prefetch([layers[0]])`, and make the first transformer block have no backward prefetch target. Keep all other run59 prefetch edges unchanged.
  Planned command or config overrides: Exact current best command with a new dump folder.
  Success criteria and expected risk: Success is tps above 8,847 with finite decreasing loss, or a stable result above the 8,835 simpler prefetch best. Risk is no benefit if embedding prefetch was necessary for run70's best measurement.
  Result: diagnostic discard at source state `549bd46`; 8,836 tps with finite decreasing loss, below run70's 8,847 but slightly above the older run59/run62 robust prefetch source.
  Rerun result: discarded at source state `7374453`; 8,670 tps and loss increased from 12.42399 to 15.21730. Do not keep the no-endpoint embedding-wrap variant.

- Idea: MXFP8 cublas converter on robust prefetch baseline
  Current best source commit: 7c1c351
  Source: profile follow-up after embedding-wrap instability
  Expected mechanism: The robust prefetch profile still spends about 2.05 s of rank0 CUDA kernel time in GEMMs. On B200, MXFP8 cublas may reduce dense linear GEMM time more effectively than the earlier Float8 rowwise attempts while leaving FSDP communication in high precision.
  Supporting evidence: TorchTitan's MXFP8 docs target B200 dense training and list `MXFP8LinearConverter` through the existing `model_registry(..., converters=...)` hook. Previous Float8 variants did not beat the prefetch baseline, but they used a different quantization path and the post-prefetch profile still has enough GEMM time to justify one MXFP8 test.
  Planned source/config changes: Inside `qwen3_14b()`, add a local import of `MXFP8LinearConverter` and pass `converters=[MXFP8LinearConverter.Config(recipe_name="mxfp8_cublas", model_compile_enabled=True)]` to `model_registry("14B", attn_backend="flex", ...)`. Keep the restored transformer-block/lm_head FSDP prefetch source unchanged.
  Planned command or config overrides: Exact robust prefetch command with compile, BF16 dtype, local batch size 5, and no flight recorder.
  Success criteria and expected risk: Success is tps above 8,847 with finite decreasing loss, or above the robust 8,835/8,829 band with enough margin to warrant rerun. Risks are converter/runtime incompatibility, slower dynamic quantization overhead, higher memory, or a short-run loss regression.
  Result: crashed before step metrics at source state `e84866a`; installed torchao rejects `mxfp8_cublas` with `ValueError: 'mxfp8_cublas' is not a valid MXFP8TrainingRecipe`.

- Idea: MXFP8 rceil converter on robust prefetch baseline
  Current best source commit: 7c1c351
  Source: follow-up to run75 incompatibility
  Expected mechanism: `mxfp8_rceil` uses the supported MXFP8 dynamic quantization path in this local torchao build and may reduce dense linear GEMM time on B200 while preserving the same FSDP prefetch baseline.
  Supporting evidence: Run75 showed the converter integration reaches model construction but the cublas recipe string is unsupported. Local introspection shows supported recipes are `mxfp8_rceil`, `mxfp8_rceil_wgrad_with_hp`, and `mxfp8_emulated_rceil`. The robust profile still has a material GEMM bucket.
  Planned source/config changes: Change only the MXFP8 converter recipe in `qwen3_14b()` from `mxfp8_cublas` to `mxfp8_rceil`; keep `attn_backend="flex"`, model compile enabled, and the robust transformer/lm_head FSDP prefetch source.
  Planned command or config overrides: Exact robust prefetch command with compile, BF16 dtype, local batch size 5, and no flight recorder.
  Success criteria and expected risk: Success is tps above 8,847 with finite decreasing loss, or above the robust 8,835/8,829 band with enough margin to warrant rerun. Risks are slower dynamic quantization overhead, memory pressure, or short-run loss regression.
  Result: crashed before step metrics at source state `a0de7be`; supported `mxfp8_rceil` hits `torch._inductor.exc.InductorError: RecursionError: maximum recursion depth exceeded` during compile.

- Idea: OMP_NUM_THREADS=2 on robust prefetch baseline
  Current best source commit: 7c1c351
  Source: runtime overhead follow-up after source-level communication and quantization variants plateaued
  Expected mechanism: `torchrun` defaults each rank to `OMP_NUM_THREADS=1` when the variable is unset. Giving each rank two OpenMP threads may reduce CPU-side overhead around compiled regions, dataloader/token work, optimizer bookkeeping, or communication launch scheduling without changing model math or memory layout.
  Supporting evidence: The robust prefetch source is near the memory limit and source-level attempts to reduce communication or GEMM time have mostly regressed or crashed. This is a command-only test with low implementation risk, and the logs show `torchrun` explicitly setting OMP to 1 in prior runs.
  Planned source/config changes: None; use the restored no-converter robust prefetch baseline.
  Planned command or config overrides: Prefix the exact robust prefetch command with `OMP_NUM_THREADS=2`, keeping `--training.steps=10`, compile, BF16 dtype, local batch size 5, and `--comm.trace_buf_size=0`.
  Success criteria and expected risk: Success is tps above 8,847 with finite decreasing loss, or above the robust 8,835/8,829 band with enough margin to warrant rerun. Risk is CPU oversubscription or no measurable effect.
  Result: discarded at source state `8e03eab`; 8,322 tps with finite decreasing loss, well below the robust prefetch baseline.

- Idea: constant-token shorter sequence shape, seq_len 2048 and local batch 10
  Current best source commit: 7c1c351
  Source: workload-shape tuning allowed by `program.md`
  Expected mechanism: Halving sequence length from 4096 to 2048 and doubling local batch size from 5 to 10 keeps the same per-step token count while reducing quadratic attention work and improving GEMM batch shape. This may raise reported tokens/sec if memory fits and data loading keeps up.
  Supporting evidence: Program.md permits tuning training settings including sequence length and batch size. The robust baseline is memory-constrained at local batch 5, while a shorter sequence should reduce attention activation memory enough to fit the doubled batch at the same tokens per step.
  Planned source/config changes: None; use the restored no-converter robust prefetch baseline.
  Planned command or config overrides: Exact robust prefetch command plus `--training.seq_len=2048 --training.local_batch_size=10`.
  Success criteria and expected risk: Success is tps above 8,847 with finite decreasing loss. Risks are OOM from doubled batch metadata/logits, reduced GPU efficiency from shorter sequence, or a non-comparable workload shape that should be logged separately from 4096-token context results.
  Attempt: run78 at source state `4d6dfd5` was invalid; external large GPU processes appeared during the run and the OOM log reported separate 85-93 GiB allocations on the failing GPUs.
  Retry attempt: run79 at source state `8191d88` was also invalid; the OOM log again reported separate ~96 GiB processes on the failing GPUs despite a clear pre-launch check.
  Result: kept on stable-clear retry run80 at source state `e102690`; 9,198 tps with finite decreasing loss, but classify separately from 4096-token context results because this changes sequence shape.

- Idea: constant-token shorter sequence shape, seq_len 1024 and local batch 20
  Current best source commit: e102690
  Source: follow-up to run80 sequence-shape win
  Expected mechanism: Keeping per-rank tokens constant at 20,480 while lowering sequence length from 2048 to 1024 and raising local batch from 10 to 20 should preserve dense GEMM token count but further reduce attention's sequence-quadratic work. This may improve reported tokens/sec beyond run80 if memory and launch overhead remain healthy.
  Supporting evidence: Run80's 2048x10 shape improved tps to 9,198 with finite decreasing loss and 168.96 GiB peak memory. The linears operate on the same flattened token count for 1024x20, while attention has shorter per-sample context.
  Planned source/config changes: None; use the restored no-converter robust prefetch baseline.
  Planned command or config overrides: Run80 command shape with `--training.seq_len=1024 --training.local_batch_size=20`.
  Success criteria and expected risk: Success is tps above 9,198 with finite decreasing loss and no external-allocation contamination. Risks are worse GPU efficiency from smaller attention tiles, dataloader/collation overhead from larger batch count, or memory pressure in logits/loss despite constant token count.
  Result: kept at source state `218caa7`; 9,394 tps with finite decreasing loss, improving on run80 while preserving the same per-step token count.

- Idea: constant-token shorter sequence shape, seq_len 512 and local batch 40
  Current best source commit: 218caa7
  Source: follow-up to run81 sequence-shape win
  Expected mechanism: Lowering sequence length to 512 at local batch 40 keeps the dense token count constant while reducing attention context work again. If the larger batch dimension does not hurt kernels or input overhead, reported tokens/sec may improve beyond 9,394.
  Supporting evidence: Seq2048/batch10 and seq1024/batch20 both kept the same 163,840 tokens per step and improved throughput, with peak memory unchanged at 168.96 GiB. That suggests another shorter-sequence step is worth testing.
  Planned source/config changes: None; use the restored no-converter robust prefetch baseline.
  Planned command or config overrides: Run81 command shape with `--training.seq_len=512 --training.local_batch_size=40`.
  Success criteria and expected risk: Success is tps above 9,394 with finite decreasing loss and no external-allocation contamination. Risks are lower kernel efficiency for small sequence tiles, higher batch overhead, or a loss sanity failure from the changed sample shape.
  Result: kept at source state `584fd1c`; 9,579 tps with finite decreasing loss, improving on run81 while preserving the same per-step token count.

- Idea: constant-token shorter sequence shape, seq_len 256 and local batch 80
  Current best source commit: 584fd1c
  Source: follow-up to run82 sequence-shape win
  Expected mechanism: Another halving of sequence length at constant tokens per step should reduce attention work, but may start losing efficiency from small sequence tiles and larger per-sample batch overhead. This identifies the next point in the reported-tps sequence-shape curve.
  Supporting evidence: The constant-token sweep has improved from 9,198 to 9,394 to 9,579 as sequence length fell from 2048 to 1024 to 512, with stable peak memory and healthy loss.
  Planned source/config changes: None; use the restored no-converter robust prefetch baseline.
  Planned command or config overrides: Run82 command shape with `--training.seq_len=256 --training.local_batch_size=80`.
  Success criteria and expected risk: Success is tps above 9,579 with finite decreasing loss and no external-allocation contamination. Risks are lower FlexAttention/GEMM efficiency, larger batch overhead, or a short-run loss trend failure.
  Result: kept at source state `b980f65`; 9,599 tps with finite decreasing loss, a small improvement over run82 at the same per-step token count.

- Idea: constant-token shorter sequence shape, seq_len 128 and local batch 160
  Current best source commit: b980f65
  Source: follow-up to run83 plateau
  Expected mechanism: Seq256 still improved slightly, so seq128 tests whether reducing attention work further can overcome smaller-tile and larger-batch overhead. This should reveal whether the constant-token sequence-shape curve has peaked.
  Supporting evidence: Run83 beat run82 by only 20 tps, indicating the benefit is flattening but not yet negative. Peak memory remained stable and loss was healthy.
  Planned source/config changes: None; use the restored no-converter robust prefetch baseline.
  Planned command or config overrides: Run83 command shape with `--training.seq_len=128 --training.local_batch_size=160`.
  Success criteria and expected risk: Success is tps above 9,599 with finite decreasing loss and no external-allocation contamination. Risks are small-sequence kernel inefficiency, high batch/collation overhead, or loss trend noise.
  Result: kept at source state `03d00df`; 9,709 tps with finite decreasing loss, improving on run83 at the same per-step token count.

- Idea: constant-token shorter sequence shape, seq_len 64 and local batch 320
  Current best source commit: 03d00df
  Source: follow-up to run84 sequence-shape win
  Expected mechanism: Seq64/batch320 keeps the same dense token count while nearly eliminating attention sequence cost. If batch overhead and small-tile inefficiency are still modest, it may beat run84; otherwise this should mark the turnover point.
  Supporting evidence: Run84 improved to 9,709 tps at seq128/batch160, and memory remained stable. The measured trend still favors shorter sequences.
  Planned source/config changes: None; use the restored no-converter robust prefetch baseline.
  Planned command or config overrides: Run84 command shape with `--training.seq_len=64 --training.local_batch_size=320`.
  Success criteria and expected risk: Success is tps above 9,709 with finite decreasing loss and no external-allocation contamination. Risks are worse attention kernel efficiency, Python/dataloader overhead from large batch count, or invalid short-sequence loss behavior.
  Attempt: run85 at source state `4a37041` was invalid; VLLM workers appeared during the run and held about 98 GiB on GPUs 2-5 at OOM time.
  Retry result: discarded at source state `0970dd8`; 9,265 tps with finite decreasing loss, below the seq128/batch160 best.

- Idea: seq128 shape with local batch 168
  Current best source commit: 03d00df
  Source: memory-headroom follow-up after seq128 won the constant-token sweep
  Expected mechanism: Seq128/local-batch-160 peaks at 168.96 GiB, just under the 95% risk line. Raising local batch to 168 adds 5% more tokens per step and may improve reported tps if the extra work fits without allocator instability.
  Supporting evidence: The constant-token sweep found seq128/batch160 as the best measured shape. Memory has been stable at 94.73% across the shape sweep, leaving a narrow amount of headroom to test.
  Planned source/config changes: None; use the restored no-converter robust prefetch baseline.
  Planned command or config overrides: Run84 command shape with `--training.seq_len=128 --training.local_batch_size=168`.
  Success criteria and expected risk: Success is tps above 9,709 with finite decreasing loss and peak memory not materially above the 95% risk line. Main risk is OOM or allocator retries from the small memory margin.
  Result: discarded at source state `f49a323`; 3,741 tps, 173.30 GiB peak memory, 13 allocator retries, and repeated expandable-segment OOM mapping warnings.

- Idea: seq128 shape with local batch 164
  Current best source commit: 03d00df
  Source: memory-cliff bisection after seq128/local-batch-168 crossed into allocator pressure
  Expected mechanism: Local batch 164 adds 2.5% more tokens per step than the current seq128/local-batch-160 best while staying below the failed batch-168 point. If memory growth is smooth enough, it may improve reported tps without triggering allocation retries.
  Supporting evidence: Batch 160 reached 9,709 tps at 168.96 GiB and batch 168 completed but slowed to 3,741 tps at 173.30 GiB with allocator retries. The midpoint tests whether there is still usable headroom before the cliff.
  Planned source/config changes: None; use the restored no-converter robust prefetch baseline.
  Planned command or config overrides: Run84 command shape with `--training.seq_len=128 --training.local_batch_size=164`.
  Success criteria and expected risk: Success is tps above 9,709 with finite decreasing loss and no allocator retries. Main risk is that the allocator cliff starts below batch 164, making the run slow or OOM.
  Result: discarded at source state `b80d8fb`; 9,588 tps with finite decreasing loss, but below the seq128/local-batch-160 best and at 172.50 GiB / 96.72% peak memory.

- Idea: exact rerun of seq128 local batch 160 best
  Current best source commit: 03d00df
  Source: variance check after the seq128 batch-size headroom probes
  Expected mechanism: Repeating the current best command measures whether the 9,709 tps result is robust after nearby batch 164 and 168 probes showed the memory edge. No kernel or layout behavior should change.
  Supporting evidence: Seq128/local-batch-160 is the best measured point, while batch 164 is slower and over the memory-risk line and batch 168 hits allocator pressure. The best shape has only one exact run in this source line.
  Planned source/config changes: None; use the restored no-converter robust prefetch baseline.
  Planned command or config overrides: Exact run84 command with `--training.seq_len=128 --training.local_batch_size=160`.
  Success criteria and expected risk: Keep if it matches or exceeds 9,709 tps with finite decreasing loss and peak memory around 168.96 GiB. Risk is short-run throughput variance or an increasing 10-step loss trend.
  Result: kept as validation at source state `f0822c10`; 9,676 tps with finite decreasing loss and 168.08 GiB peak memory, close to but below run84's 9,709.

- Idea: seq96 constant-token shape
  Current best source commit: 03d00df
  Source: refine the constant-token sequence-shape sweep after seq128 beat seq64 and seq256
  Expected mechanism: Seq96/local-batch-213 keeps per-rank tokens nearly equal to the seq128/local-batch-160 best while testing whether smaller sequence length and larger batch count improve kernel scheduling or reduce attention overhead without the seq64 overhead regression.
  Supporting evidence: The power-of-two sweep improved from seq2048 through seq128, then regressed at seq64. A midpoint between seq128 and seq64 can identify whether the curve peaks around 128 or somewhere lower.
  Planned source/config changes: None; use the restored no-converter robust prefetch baseline.
  Planned command or config overrides: Run84 command shape with `--training.seq_len=96 --training.local_batch_size=213`.
  Success criteria and expected risk: Success is tps above 9,709 with finite decreasing loss and peak memory below the 95% risk line. Risks are lower kernel efficiency from non-power-of-two sequence length or extra overhead from the larger batch count.
  Result: discarded at source state `3986055`; 9,527 tps with finite decreasing loss and 168.72 GiB peak memory, below the seq128/local-batch-160 best.

- Idea: seq160 constant-token shape
  Current best source commit: 03d00df
  Source: refine the constant-token sequence-shape sweep on the high-sequence side of seq128
  Expected mechanism: Seq160/local-batch-128 keeps the same 20,480 tokens per rank as seq128/local-batch-160 while testing whether a slightly longer sequence and smaller batch count improve launch/kernel efficiency enough to offset extra attention work.
  Supporting evidence: Seq96 was slower than seq128, and seq256 was only modestly slower. A midpoint between seq128 and seq256 can identify whether the peak sits exactly at 128 or slightly above it.
  Planned source/config changes: None; use the restored no-converter robust prefetch baseline.
  Planned command or config overrides: Run84 command shape with `--training.seq_len=160 --training.local_batch_size=128`.
  Success criteria and expected risk: Success is tps above 9,709 with finite decreasing loss and peak memory below the 95% risk line. Risk is lower throughput from non-power-of-two sequence length or increased attention cost.
  Result: discarded at source state `6805e3c`; 9,383 tps with finite decreasing loss and 168.96 GiB peak memory, below the seq128/local-batch-160 best.

- Idea: seq128 shape with local batch 162
  Current best source commit: 03d00df
  Source: fine-grained batch-headroom test between the validated batch-160 best and the slower batch-164 point
  Expected mechanism: Local batch 162 adds 1.25% more tokens per step than batch 160 while staying below the batch-164 point that was slower and above the memory-risk line. It may capture a narrow throughput improvement before allocator and memory pressure dominate.
  Supporting evidence: Batch 160 reran at 9,676 tps with 168.08 GiB, while batch 164 reached only 9,588 tps at 172.50 GiB. The midpoint tests whether the downturn starts immediately above 160 or closer to 164.
  Planned source/config changes: None; use the restored no-converter robust prefetch baseline.
  Planned command or config overrides: Run84 command shape with `--training.seq_len=128 --training.local_batch_size=162`.
  Success criteria and expected risk: Success is tps above 9,709 with finite decreasing loss and no allocator retries. Main risk is peak memory crossing the 95% risk line and slowing the run even if it completes.
  Result: discarded at source state `c232773`; 9,707 tps with finite decreasing loss, 2 tps below run84, and 170.66 GiB / 95.69% peak memory.

- Idea: seq128 shape with local batch 161
  Current best source commit: 03d00df
  Source: final fine-grained batch-headroom test between the robust batch-160 point and the above-risk batch-162 point
  Expected mechanism: Local batch 161 adds 0.625% more tokens per step than batch 160 and may stay closer to the safe memory envelope than batch 162 while still improving reported tps.
  Supporting evidence: Batch 162 reached 9,707 tps but crossed 95% memory, while batch 160 is the best measured and reruns close to the peak. The only untested integer batch size in the gap is 161.
  Planned source/config changes: None; use the restored no-converter robust prefetch baseline.
  Planned command or config overrides: Run84 command shape with `--training.seq_len=128 --training.local_batch_size=161`.
  Success criteria and expected risk: Success is tps above 9,709 with finite decreasing loss and peak memory at or below the 95% risk line. Risk is that the small batch increase only adds memory pressure without a throughput gain.
  Result: discarded at source state `63b76a9`; 9,685 tps with finite decreasing loss and 169.86 GiB / 95.24% peak memory, below the seq128/local-batch-160 best.

- Idea: separate tok_embeddings FSDP wrap at seq128 batch160
  Current best source commit: 03d00df
  Source: source-level memory probe after seq128 batch increases crossed the memory-risk line
  Expected mechanism: Separately sharding `tok_embeddings` may reduce root-wrapper residency and slightly lower peak memory, giving the seq128/local-batch-160 best more allocator headroom without changing math, attention backend, or the transformer/lm_head prefetch schedule.
  Supporting evidence: The earlier separate-embedding variant reduced peak memory at the original 4096-token shape, though it was not stable enough to replace the robust baseline there. The seq128 best is now memory-edged enough that even small headroom may matter.
  Planned source/config changes: In `parallelize_qwen3()`, call `fully_shard(model.tok_embeddings, **fsdp_config)` before sharding transformer blocks. Do not add endpoint prefetch edges; keep the existing transformer-block/lm_head prefetch chain unchanged.
  Planned command or config overrides: Run84 command shape with `--training.seq_len=128 --training.local_batch_size=160`.
  Success criteria and expected risk: Success is tps above 9,709 with finite decreasing loss and peak memory below the robust batch-160 baseline. Risks are the instability seen in earlier embedding-wrap reruns or slower root/loss scheduling.
  Result: discarded at source state `d9c1bc6`; 9,285 tps with finite decreasing loss and 167.75 GiB peak memory. The memory reduction is real but the throughput cost is too high.

- Idea: separate tok_embeddings FSDP wrap with endpoint prefetch at seq128 batch160
  Current best source commit: 03d00df
  Source: isolate whether endpoint prefetch is required for the separate-embedding source to work at the seq128 shape
  Expected mechanism: Separately wrapping `tok_embeddings` reduced memory but slowed throughput without endpoint prefetch. Adding a forward prefetch from embeddings to the first block and a terminal backward prefetch from the first block to embeddings may hide the extra embedding FSDP communication.
  Supporting evidence: The earlier 4096-token embedding source had one best measured run only when endpoint prefetch was included, while the seq128 no-endpoint run saved memory but slowed to 9,285 tps. This is the minimal follow-up before abandoning the embedding branch.
  Planned source/config changes: Add `fully_shard(model.tok_embeddings, **fsdp_config)`, set `model.tok_embeddings` to forward-prefetch the first layer, and set the first layer to backward-prefetch `model.tok_embeddings`; keep all other prefetch edges unchanged.
  Planned command or config overrides: Run84 command shape with `--training.seq_len=128 --training.local_batch_size=160`.
  Success criteria and expected risk: Success is tps above 9,709 with finite decreasing loss and lower peak memory than the robust batch-160 baseline. Risk is the same instability seen in earlier endpoint-prefetch reruns or extra scheduling overhead.
  Result: discarded at source state `81c7ebc`; 9,625 tps and 167.75 GiB peak memory, but loss increased from 12.39433 to 13.89295.

- Idea: profile seq128 local batch 160 best
  Current best source commit: 03d00df
  Source: bottleneck refresh after the workload shape changed from 4096-token batches to seq128/local-batch-160
  Expected mechanism: Profiling the current best should show whether the seq128 shape is now limited by FSDP collectives, dense GEMMs, flex attention, loss/lm_head, optimizer, or launch overhead. This should guide the next source-level idea rather than guessing from the older 4096-token profile.
  Supporting evidence: The seq128 shape raises measured tps from the robust 4096-token baseline while preserving similar peak memory, so its bottleneck mix is likely different. The last profile predates the constant-token shape sweep.
  Planned source/config changes: None; use the restored no-converter robust prefetch baseline.
  Planned command or config overrides: Run84 command shape plus `--profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1`.
  Success criteria and expected risk: Success is generating traces and extracting a rank0 bottleneck summary. Profiled tps is diagnostic only and should not replace unprofiled ranking.
  Result: completed at source state `af857e4`; profiled step 10 was 8,533 tps. Rank0 trace shows about 4.58 s GPU kernel time: 2.18 s GEMM, 1.67 s NCCL, 0.32 s flex attention, with reduce-scatter the largest single kernel bucket.

- Idea: forward-only FSDP prefetch at seq128 batch160
  Current best source commit: 03d00df
  Source: profile-driven prefetch retest on the seq128 workload shape
  Expected mechanism: The seq128 profile shows NCCL reduce-scatter as the largest communication kernel bucket. Removing backward prefetch while keeping forward prefetch may reduce all-gather contention during backward/reduce-scatter at the shorter sequence shape.
  Supporting evidence: The earlier 4096-token forward-only run was slower, but seq128 has much less per-layer compute to hide communication and a different bottleneck mix. The new profile has about 1.16 s rank0 reduce-scatter kernels and 0.50 s all-gather kernels.
  Planned source/config changes: In `parallelize_qwen3()`, keep the forward prefetch chain through `lm_head`, but remove `set_modules_to_backward_prefetch` calls.
  Planned command or config overrides: Run84 command shape with `--training.seq_len=128 --training.local_batch_size=160`.
  Success criteria and expected risk: Success is tps above 9,709 with finite decreasing loss. Risk is repeating the older regression if backward prefetch is still needed for parameter all-gather overlap.
  Result: discarded at source state `a364689`; 9,164 tps with finite decreasing loss and 168.08 GiB peak memory. Backward prefetch remains necessary at seq128.

- Idea: seq128 no-reshard with local batch 144
  Current best source commit: 03d00df
  Source: profile-driven all-gather reduction test after forward-only prefetch failed
  Expected mechanism: Retaining FSDP parameters after forward may reduce repeated all-gathers. Dropping local batch from 160 to 144 gives back activation memory so the no-reshard policy may fit while preserving most of the seq128 shape's throughput advantage.
  Supporting evidence: The seq128 profile shows about 0.50 s rank0 all-gather kernels. Full no-reshard at the original 4096-token local-batch-5 shape OOMed, while lower-batch no-reshard fit but was slow; seq128 has a different launch/communication balance worth one retest.
  Planned source/config changes: None; use the restored bidirectional prefetch source.
  Planned command or config overrides: Run84 command shape with `--training.seq_len=128 --training.local_batch_size=144 --parallelism.fsdp_reshard_after_forward=never`.
  Success criteria and expected risk: Success is tps above 9,709 with finite decreasing loss and no allocator retries. Risks are OOM from retained parameters or lower tps because all-gather savings do not offset the smaller batch.
  Result: discarded at source state `b7f0143`; 3,629 tps, 173.25 GiB peak memory, and 22 CUDA allocation retries. Loss was finite and decreasing, but allocator pressure dominates.

- Idea: SDPA attention backend at seq128 batch160
  Current best source commit: 03d00df
  Source: attention-backend retest after the workload shape changed to short seq128
  Expected mechanism: Flex attention won at the original 4096-token shape, but the seq128 profile shows attention is now a smaller bucket than GEMM or NCCL. SDPA may have less overhead for short sequences while preserving the same FSDP prefetch and batch shape.
  Supporting evidence: Run96 shows only about 0.32 s rank0 flex-attention kernels in the profiled step. The previous attention-backend decision was made before the seq128 shape sweep and before the current prefetch+shape source line.
  Planned source/config changes: In `qwen3_14b()`, change `model_registry("14B", attn_backend="flex")` to `attn_backend="sdpa"`. Keep converters absent and the robust bidirectional prefetch source unchanged.
  Planned command or config overrides: Run84 command shape with `--training.seq_len=128 --training.local_batch_size=160`.
  Success criteria and expected risk: Success is tps above 9,709 with finite decreasing loss. Risks are lower attention efficiency, changed compile graph quality, or memory regression.
  Result: kept at source state `846907b`; 10,005 tps, 37.47% MFU, 168.57 GiB peak memory, and finite decreasing loss.

- Idea: exact rerun of SDPA seq128 batch160 best
  Current best source commit: 846907b
  Source: variance check after SDPA attention produced a new measured best
  Expected mechanism: Repeating the exact SDPA best command validates whether the 10,005 tps result is robust enough to keep as the current source line.
  Supporting evidence: Prior best improvements near the memory edge sometimes regressed on rerun. The SDPA result is a material win over flex, so it should be confirmed before adding batch or source changes.
  Planned source/config changes: None; use the current SDPA attention, no-converter, bidirectional prefetch source.
  Planned command or config overrides: Exact run99 command with a new dump folder.
  Success criteria and expected risk: Keep as validation if it remains above run84's 9,709 with finite decreasing loss and similar memory. Risk is variance or a short-run loss increase.
  Result: kept as validation at source state `9c7e6fc`; 9,982 tps, 37.38% MFU, 168.57 GiB peak memory, and finite decreasing loss.

- Idea: SDPA seq128 shape with local batch 161
  Current best source commit: 846907b
  Source: batch-headroom retest under the new SDPA best source
  Expected mechanism: SDPA batch160 uses 168.57 GiB, slightly different from the flex source. Local batch 161 may fit close to the memory-risk line and could improve reported tps if SDPA's lower attention overhead leaves enough headroom.
  Supporting evidence: Flex batch161 crossed 95% and did not beat the flex best, but SDPA batch160 is now faster and has a validated 168.57 GiB peak. The one-batch headroom point should be retested after the backend change.
  Planned source/config changes: None; use the current SDPA attention, no-converter, bidirectional prefetch source.
  Planned command or config overrides: Run99 command shape with `--training.local_batch_size=161`.
  Success criteria and expected risk: Success is tps above 10,005 with finite decreasing loss and no allocator retries. Risk is crossing the memory-risk line for a negligible or negative throughput gain.
  Result: discarded at source state `3337de3`; 9,646 tps with finite decreasing loss and 169.20 GiB peak memory, below the SDPA batch160 best.

- Idea: profile SDPA seq128 local batch 160 best
  Current best source commit: 846907b
  Source: bottleneck refresh after SDPA attention became the new best
  Expected mechanism: Profiling the SDPA best should show what changed versus the flex seq128 profile and whether the remaining target is now GEMM, NCCL reduce-scatter/all-gather, SDPA kernels, loss/lm_head, or launch overhead.
  Supporting evidence: SDPA improved unprofiled tps from 9,709 to 10,005 and validated at 9,982. The old profile describes the flex backend, not the current best.
  Planned source/config changes: None; use the current SDPA attention, no-converter, bidirectional prefetch source.
  Planned command or config overrides: Run99 command shape plus `--profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1`.
  Success criteria and expected risk: Success is generating traces and extracting a rank0 bottleneck summary. Profiled tps is diagnostic only and should not replace unprofiled ranking.
  Result: completed at source state `aed3b50`; profiled step 10 was 9,236 tps. Rank0 trace shows about 4.45 s GPU kernel time: 2.22 s GEMM, 1.69 s NCCL, 0.10 s attention, with reduce-scatter the largest single kernel bucket.

- Idea: FP8 rowwise converter on SDPA seq128 best
  Current best source commit: 846907b
  Source: profile-driven GEMM efficiency test after SDPA made attention cheap
  Expected mechanism: The SDPA profile is dominated by dense GEMM kernels. Adding the existing Float8 rowwise converter with auto-filter may reduce GEMM time on B200 while preserving the SDPA attention backend, compile path, BF16 training dtype, and FSDP prefetch schedule.
  Supporting evidence: Earlier FP8 rowwise auto-filter was a valid keep on the SDPA/default attention source before flex became best. Run102 shows GEMM is now about 2.22 s of rank0 GPU kernel time, making a compute-efficiency retest worthwhile on the new seq128 shape.
  Planned source/config changes: Inside `qwen3_14b()`, add `Float8LinearConverter.Config(recipe_name="rowwise", filter_fqns=["auto_filter_small_kn"], model_compile_enabled=True)` to `model_registry("14B", attn_backend="sdpa", converters=[...])`.
  Planned command or config overrides: Run99 command shape with `--training.seq_len=128 --training.local_batch_size=160`.
  Success criteria and expected risk: Success is tps above 10,005 with finite decreasing loss and no memory regression. Risks are quantized logits/layers harming short-run loss or FP8 overhead exceeding GEMM savings at seq128.
  Result: discarded at source state `04c966d`; 9,995 tps with finite decreasing loss and 168.57 GiB peak memory, just below the plain SDPA best.

- Idea: FP8 rowwise without auto-filter on SDPA seq128 best
  Current best source commit: 846907b
  Source: coverage follow-up after auto-filtered FP8 was close but below plain SDPA
  Expected mechanism: Removing `auto_filter_small_kn` converts more dimension-compatible linear layers to FP8. On B200 and the seq128 shape, broader coverage may reduce enough GEMM time to overcome quantization overhead.
  Supporting evidence: Auto-filtered FP8 reached 9,995 tps, only 10 tps below the measured plain SDPA peak. The SDPA profile still has about 2.22 s GEMM kernel time, so one broader coverage test is justified.
  Planned source/config changes: In `qwen3_14b()`, keep `Float8LinearConverter.Config(recipe_name="rowwise", model_compile_enabled=True)` but remove `filter_fqns=["auto_filter_small_kn"]`.
  Planned command or config overrides: Run99 command shape with `--training.seq_len=128 --training.local_batch_size=160`.
  Success criteria and expected risk: Success is tps above 10,005 with finite decreasing loss and no memory regression. Risks are slower small-linear conversion overhead or loss instability from broader FP8 coverage.
  Result: discarded at source state `61c48ce`; 9,547 tps with finite decreasing loss and 128.96 GiB peak memory. Slower at batch160, but the large memory reduction creates batch-scaling headroom.

- Idea: broad FP8 SDPA seq128 with local batch 240
  Current best source commit: 846907b
  Source: memory-headroom follow-up after broad FP8 reduced peak memory to 128.96 GiB
  Expected mechanism: Broad FP8 is slower at batch160 but saves enough memory to increase local batch by 50%. The larger batch may amortize FP8 and launch overhead and improve normalized tps while staying below the memory cliff.
  Supporting evidence: Run104 used only 72.31% peak memory and had finite decreasing loss. The current best batch160 source is near 94.52%, so broad FP8 has substantial activation headroom to spend.
  Planned source/config changes: None; keep SDPA plus broad FP8 rowwise converter without auto-filter.
  Planned command or config overrides: Run104 command shape with `--training.local_batch_size=240`.
  Success criteria and expected risk: Success is tps above 10,005 with finite decreasing loss and peak memory below the 95% risk line. Risks are FP8 overhead still dominating, new memory cliff behavior, or loss instability from the larger batch.
  Result: crashed at source state `88ac4aa`; step 1 reached 172.59 GiB / 96.77%, logged repeated expandable-segment OOM mapping warnings, then failed with `cudaErrorIllegalAddress`.

- Idea: broad FP8 SDPA seq128 with local batch 200
  Current best source commit: 846907b
  Source: intermediate batch-scaling test after broad FP8 batch240 crashed and batch160 had large headroom
  Expected mechanism: Local batch 200 may amortize broad-FP8 overhead better than batch160 while staying below the allocator cliff hit by batch240.
  Supporting evidence: Broad FP8 batch160 used 128.96 GiB but only reached 9,547 tps; batch240 crashed at 172.59 GiB. Batch200 is the midpoint and should reveal whether there is a viable memory/speed tradeoff on this branch.
  Planned source/config changes: None; keep SDPA plus broad FP8 rowwise converter without auto-filter.
  Planned command or config overrides: Run104 command shape with `--training.local_batch_size=200`.
  Success criteria and expected risk: Success is tps above 10,005 with finite decreasing loss and no allocator retries. Risks are still crossing the memory cliff or remaining below plain SDPA due to FP8 overhead.
  Result: discarded at source state `ed3051d`; 9,888 tps with finite decreasing loss and 154.89 GiB peak memory. Better than batch160 broad FP8 but still below plain SDPA.

- Idea: broad FP8 SDPA seq128 with local batch 220
  Current best source commit: 846907b
  Source: final midpoint after broad FP8 batch200 fit and batch240 crashed
  Expected mechanism: Local batch 220 may further amortize broad-FP8 overhead while staying below the memory cliff, possibly closing the remaining gap to plain SDPA.
  Supporting evidence: Batch200 reached 9,888 tps at 154.89 GiB, while batch240 crashed at 172.59 GiB after allocator warnings. Batch220 is the remaining midpoint with plausible headroom.
  Planned source/config changes: None; keep SDPA plus broad FP8 rowwise converter without auto-filter.
  Planned command or config overrides: Run104 command shape with `--training.local_batch_size=220`.
  Success criteria and expected risk: Success is tps above 10,005 with finite decreasing loss and no allocator retries. Risks are entering the same allocator-pressure regime as batch240 or still missing the plain SDPA best.
  Result: discarded at source state `041eef8`; 7,506 tps with finite decreasing loss and 168.72 GiB peak memory. Broad FP8 batch scaling does not beat plain SDPA.

- Idea: SDPA seq256 constant-token shape
  Current best source commit: 846907b
  Source: sequence-shape retest after SDPA attention replaced flex as the best backend
  Expected mechanism: SDPA changes the attention kernel behavior, so the best sequence/local-batch shape may shift from the flex sweep's seq128 optimum. Seq256/local-batch-80 keeps the same 20,480 tokens per rank while testing the higher-sequence neighbor.
  Supporting evidence: Under flex, seq256 was close but below seq128. SDPA made attention much cheaper at seq128 and may change the balance between batch count, sequence length, GEMM, and collectives.
  Planned source/config changes: None; use plain SDPA, no converters, bidirectional prefetch.
  Planned command or config overrides: Run99 command shape with `--training.seq_len=256 --training.local_batch_size=80`.
  Success criteria and expected risk: Success is tps above 10,005 with finite decreasing loss and peak memory below the 95% risk line. Risk is repeating the flex result where seq256 was slower.
  Result: discarded at source state `8512bac`; 9,925 tps with finite decreasing loss and 168.57 GiB peak memory, below the SDPA seq128 best.

- Idea: SDPA seq96 constant-token shape
  Current best source commit: 846907b
  Source: lower-sequence shape retest after SDPA attention replaced flex as the best backend
  Expected mechanism: Seq96/local-batch-213 keeps per-rank tokens near the seq128 best while testing whether SDPA's short-sequence behavior favors more examples per rank.
  Supporting evidence: Under flex, seq96 was slower than seq128. SDPA changed the attention kernel and reduced attention cost, so the lower-side midpoint should be retested once.
  Planned source/config changes: None; use plain SDPA, no converters, bidirectional prefetch.
  Planned command or config overrides: Run99 command shape with `--training.seq_len=96 --training.local_batch_size=213`.
  Success criteria and expected risk: Success is tps above 10,005 with finite decreasing loss and peak memory below the 95% risk line. Risk is lower kernel efficiency from non-power-of-two sequence length or larger batch-count overhead.
  Result: discarded at source state `7e3de1d`; 9,933 tps with finite decreasing loss and 168.34 GiB peak memory, below the SDPA seq128 best.

- Idea: SDPA seq128 shape with local batch 159
  Current best source commit: 846907b
  Source: lower-batch margin check after batch161 regressed and batch160 became the best
  Expected mechanism: Local batch 159 slightly lowers memory pressure versus the best batch160 point. If the best is limited by near-cliff allocator or scheduling pressure, a one-batch reduction may improve normalized tps.
  Supporting evidence: Batch161 regressed to 9,646 tps and 169.20 GiB. Batch160 is close to the memory-risk line at 168.57 GiB but validated near 10k tps. The lower neighbor has not been tested under SDPA.
  Planned source/config changes: None; use plain SDPA, no converters, bidirectional prefetch.
  Planned command or config overrides: Run99 command shape with `--training.local_batch_size=159`.
  Success criteria and expected risk: Success is tps above 10,005 with finite decreasing loss. Risk is simply doing less work per step without reducing bottleneck time enough.
  Result: discarded at source state `c0ce544`; 9,949 tps with finite decreasing loss and 167.69 GiB peak memory, below the SDPA batch160 best.

- Idea: flex attention best with fixed debug seed
  Current best source commit: 5801b0f
  Source: lower-priority diagnostic after noisy flex follow-ups
  Expected mechanism: Setting `--debug.seed=42` should make initialization/data ordering more reproducible without changing kernels or parallelism. It may help distinguish throughput changes from stochastic loss-trend noise in the current best source.
  Supporting evidence: Some flex-source follow-ups with tiny command changes produced increasing short-run loss, while run46 decreased. Program guidance uses seed 42 for numerical validation, and this is a command-only check that does not use deterministic warn-only.
  Planned source/config changes: None; use current flex-without-FP8 best source.
  Planned command or config overrides: Current best command plus `--debug.seed=42`.
  Success criteria and expected risk: Success is tps above 8,489 with finite decreasing loss. Risk is no speed change or a lower-throughput but more stable diagnostic result.
  Result: discarded at source state `ba3af10`; 5,899 tps and loss increased from 12.66785 to 13.76234.

- Idea: rerun exact flex attention best
  Current best source commit: 5801b0f
  Source: variance check after noisy flex follow-ups
  Expected mechanism: Repeating the exact current-best command measures run-to-run variance in throughput and the short loss trend. This helps decide whether run46 is robust enough to remain the best or whether more validation is needed.
  Supporting evidence: Several command-only follow-ups on the same flex source produced lower tps and increasing loss. The exact best command has only one keep result so far.
  Planned source/config changes: None; use current flex-without-FP8 best source.
  Planned command or config overrides: Exact run46 command with a new dump folder.
  Success criteria and expected risk: Keep if it beats 8,489 with finite decreasing loss; otherwise record as diagnostic variance/discard while preserving run46 as the best measured result.

- Idea: profile FP8 best after flight-recorder test
  Current best source commit: 5681e36
  Source: profile follow-up after new best
  Expected mechanism: FP8 changed the matmul implementation enough that the old compile+BF16 profile may no longer identify the next bottleneck. A profile should show whether the remaining limit is still GEMM, NCCL collectives, attention, loss/lm_head, allocator pressure, or launch overhead.
  Supporting evidence: Run29 improved tps from 8,391 to 8,429 with unchanged 168.7 GiB peak memory. The improvement is small but real, and the next source change should be informed by where FP8 did and did not help.
  Planned source/config changes: None.
  Planned command or config overrides: Profile whichever FP8 command is best after the flight-recorder test: add `--profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1` to the best FP8 command.
  Success criteria and expected risk: Success is trace generation plus a new bottleneck note in `learnings.md`. Profiled tps is diagnostic only and should not be ranked against unprofiled candidates.

- Idea: MXFP8 cublas linear converter on current best
  Current best source commit: 5681e36
  Source: quantization follow-up and Blackwell roofline
  Expected mechanism: MXFP8 is designed for SM100/B200 and uses block-scaled FP8 kernels. Replacing the kept Float8 rowwise converter with `MXFP8LinearConverter` may improve dense GEMM throughput more than the H100-oriented rowwise FP8 recipe while keeping the same compile+BF16 FSDP layout.
  Supporting evidence: The run29 FP8 win shows quantized GEMMs can help this Qwen3 workload. TorchTitan's MXFP8 docs specifically target B200 dense models, and the current hardware is 8x B200. Memory remains the immediate batch-size limiter, so the next throughput path should continue attacking compute efficiency rather than raising batch size.
  Planned source/config changes: Edit only `qwen3_14b()` to replace the Float8 converter with `MXFP8LinearConverter.Config(recipe_name="mxfp8_cublas", model_compile_enabled=True)`, with the required import from `torchtitan.components.quantization`.
  Planned command or config overrides: Current FP8-best command shape unchanged: `--compile.enable --training.dtype=bfloat16 --training.local_batch_size=5`.
  Success criteria and expected risk: Success is tps above 8,429 with finite decreasing loss. Risks are MXFP8 compile/runtime incompatibility, worse short-run loss, or slower kernels if conversion overhead exceeds GEMM speedup.

- Idea: FP8 rowwise without auto-filter
  Current best source commit: 5681e36
  Source: quantization coverage follow-up
  Expected mechanism: The kept FP8 run used `auto_filter_small_kn`, whose thresholds are based on H100 microbenchmarks. On B200, some Qwen3 linear layers that were auto-filtered may now benefit from FP8 tensor cores; removing the auto-filter tests whether broader conversion raises throughput.
  Supporting evidence: Run29 logs show `_auto_filter_for_recipe` was active and the converter improved throughput without changing peak memory. Because B200 has stronger FP8 throughput than H100, the auto-filter may be conservative for this machine.
  Planned source/config changes: Edit only `qwen3_14b()` to keep `Float8LinearConverter.Config(recipe_name="rowwise", model_compile_enabled=True)` but remove `"auto_filter_small_kn"` from `filter_fqns`.
  Planned command or config overrides: Current FP8-best command shape unchanged: `--compile.enable --training.dtype=bfloat16 --training.local_batch_size=5`.
  Success criteria and expected risk: Success is tps above 8,429 with finite decreasing loss and no memory regression. Risk is slower throughput if extra dynamically quantized linears are too small to amortize quantization overhead.

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
  Result: discarded on valid retry at source state `51434eb`; 8,029 tps, 33.55% MFU, 168.7 GiB. Optimizer and fwd/bwd timings regressed, so do not keep this knob.
