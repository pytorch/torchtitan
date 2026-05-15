## Human Generated Ideas

- ~~Idea: Bootstrap runnable Qwen3 14B DP-sharded run~~
  - Current best source commit: 68f86f1
  - Source: human/setup
  - Expected mechanism for improving reported tokens/sec: The target command currently selects `dp_shard=-1` on 8 GPUs, but `parallelize_qwen3()` only applies replicated DDP and does not apply FSDP for the active `fsdp` mesh. A narrow bootstrap should make the inferred target command run correctly by applying TorchTitan FSDP on the configured shard mesh.
  - Supporting evidence: `qwen3_14b()` sets `data_parallel_shard_degree=-1`, which resolves to `dp_shard=8` for `NGPU=8`; the Qwen3 scaffold rejects TP/CP/PP/EP and otherwise only calls `replicate()` when `dp_enabled`.
  - Planned source/config changes: In `torchtitan/models/qwen3/parallelize.py`, replace the DP-only replicated wrapper for the active DP-shard case with the minimal FSDP wrapping needed for Qwen3 14B.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10`
  - Success criteria and expected risk: Kept at `fc6629b` with 5,774 tps; later command-only no-reshard improved to 5,872 tps.

## Manager Generated Ideas

- ~~Idea: Profile current best after first runnable result~~
  - Current best source commit: TBD after bootstrap
  - Source: agent-generated
  - Expected mechanism for improving reported tokens/sec: Use a profiled 10-step run to distinguish compute, HBM, communication, launch, and data-loading bottlenecks before changing TP/CP/PP or activation checkpointing.
  - Supporting evidence: No Qwen3 14B 8xB200 result has been measured yet on this branch.
  - Planned source/config changes: None for the profiling run.
  - Planned command or config overrides: Add TorchTitan profiler flags to the same 10-step command.
  - Success criteria and expected risk: Completed at `3037e26`; rank 0 trace shows FSDP collectives are material, with reduce-scatter the largest kernel bucket in the captured step.

- ~~Idea: TP=2 with shared decoder sharding helpers~~
  - Current best source commit: TBD after bootstrap
  - Source: manager research
  - Expected mechanism for improving reported tokens/sec: Qwen3 14B has 40 attention heads and 8 KV heads, so TP=2 divides both cleanly and should reduce per-rank dense matmul/state memory while keeping TP collectives within the B200 NVLink domain. Enabling sequence parallel should keep post-attention and FFN activations sharded over sequence positions and may convert freed memory into a larger follow-up batch or lower recompute pressure.
  - Supporting evidence: `Qwen3Model.Config.update_from_config()` validates that TP divides `n_heads` and `n_kv_heads`; the 14B flavor satisfies this for TP=2. `torchtitan.models.common.decoder_sharding` already provides `set_decoder_sharding_config`, `set_gqa_attention_sharding`, `set_gqa_inner_attention_local_map`, `set_dense_ffn_sharding`, and norm placement helpers for GQA+dense-FFN decoder blocks, which match the Qwen3 14B non-MoE structure.
  - Planned source/config changes: In `set_qwen3_sharding_config()`, call shared decoder/root helpers and walk Qwen3 layers to set attention, inner-attention local-map, q/k norm, FFN, and block norm sharding. In `parallelize_qwen3()`, call `model.parallelize(tp_mesh)` when TP is enabled before FSDP wrapping. Keep the diff limited to the active dense Qwen3 path.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.tensor_parallel_degree=2 --parallelism.data_parallel_shard_degree=4 --parallelism.enable_sequence_parallel`
  - Success criteria and expected risk: Discarded at `e67f61c`; it ran correctly but fell to 4,971 tps, so TP collectives and smaller local matmuls outweighed the memory savings for this shape.

- ~~Idea: Local batch size 8 with FSDP no-reshard~~
  - Current best source commit: 9db79f7 result row / current source after TP revert
  - Source: profile and memory-headroom analysis
  - Expected mechanism for improving reported tokens/sec: The best run keeps FSDP parameters unresharded after forward and reaches only 72.2GiB peak on a 178.35GiB B200. Doubling local batch size should increase useful tokens per optimizer step and per FSDP collective while staying below the memory risk threshold.
  - Supporting evidence: Baseline profile rank 0 showed NCCL reduce-scatter as the largest kernel bucket at about 0.86s in the captured step and all-gather around 0.23s. The no-reshard candidate improved tps from 5,774 to 5,872 while using 40.49% memory, leaving room to increase batch before considering more mesh complexity.
  - Planned source/config changes: None; command-only candidate.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --training.local_batch_size=8`
  - Success criteria and expected risk: Discarded at `dfac656`; tps was essentially tied at 5,873 but loss increased from 12.42 to 16.38, so it failed the convergence sanity check.

- ~~Idea: Selective activation checkpointing with FSDP no-reshard~~
  - Current best source commit: 9db79f7 result row / current source after TP revert
  - Source: memory-headroom analysis
  - Expected mechanism for improving reported tokens/sec: Full activation checkpointing saves memory but recomputes every block. Selective checkpointing should keep enough memory headroom on B200 while reducing recomputation and improving matmul/attention useful work per step.
  - Supporting evidence: Full AC plus no-reshard used 72.2GiB, far below the 178.35GiB capacity, and the profile/MFU indicated compute headroom remained.
  - Planned source/config changes: None; command-only candidate.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=selective`
  - Success criteria and expected risk: Kept at `0b0796f` with 6,808 tps, 28.45% MFU, 113.7GiB peak memory, and falling finite loss. Later superseded by model-only per-block compile.

- ~~Idea: Disable activation checkpointing with FSDP no-reshard~~
  - Current best source commit: 0b0796f
  - Source: memory-headroom follow-up
  - Expected mechanism for improving reported tokens/sec: Removing recomputation entirely could improve throughput if memory stayed below the B200 limit.
  - Supporting evidence: Selective AC still used only 113.7GiB, leaving apparent headroom before the experiment.
  - Planned source/config changes: None; command-only candidate.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=none`
  - Success criteria and expected risk: Crashed at `dc56c35`; no-AC with no-reshard OOMed during RoPE with about 178.15GiB in use.

- ~~Idea: Selective checkpointing without RNG preservation~~
  - Current best source commit: 0b0796f
  - Source: activation-checkpoint overhead follow-up
  - Expected mechanism for improving reported tokens/sec: Avoiding RNG state preservation can reduce checkpoint wrapper overhead if dropout/RNG is irrelevant for this model path.
  - Supporting evidence: The selective-AC run was the best result, so reducing wrapper overhead was a narrow follow-up.
  - Planned source/config changes: None; command-only candidate.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation-checkpoint.mode=selective --activation-checkpoint.no-preserve-rng-state`
  - Success criteria and expected risk: Discarded at `9690a1`; throughput dropped to 6,775 tps and loss was flat/increasing by step 10.

- ~~Idea: Local batch size 6 with selective AC and no-reshard~~
  - Current best source commit: 0b0796f
  - Source: memory-headroom and batch-scaling analysis
  - Expected mechanism for improving reported tokens/sec: Selective AC is the current best and uses 113.7GiB. A smaller batch increase than the failed batch-8/full-AC run may increase useful tokens per collective and improve GPU occupancy while staying below the no-AC OOM boundary.
  - Supporting evidence: Full AC batch 8 had enough memory but failed loss sanity, while no-AC batch 4 OOMed. Selective AC batch 4 leaves roughly 64% peak memory use, so batch 6 is a narrower memory-headroom probe than batch 8.
  - Planned source/config changes: None; command-only candidate.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=selective --training.local_batch_size=6`
  - Success criteria and expected risk: Discarded at `50404e`; loss fell normally, but tps was 6,805, slightly below the 6,808 best, and memory rose to 141.9GiB.

- ~~Idea: Compile transformer blocks with selective AC and no-reshard~~
  - Current best source commit: 0b0796f
  - Source: selective-profile analysis
  - Expected mechanism for improving reported tokens/sec: The selective-AC profile reduced FSDP collective time enough that flash attention backward, dense matmuls, layer norm, and elementwise kernels dominate the captured GPU work. Per-block `torch.compile` may fuse or schedule block-level work better and reduce CPU/runtime overhead on the repeated 48-layer structure.
  - Supporting evidence: The selective profile rank 0 trace shows kernel time around 3.19s, NCCL around 0.44s, with top compute kernels including flash backward, several nvjet matmuls, layer norm, and elementwise kernels. CPU op time and CUDA runtime overhead are still visible in the trace.
  - Planned source/config changes: None if existing `parallelize_qwen3()` compile path handles the command; otherwise only the minimum Qwen3-local compile ordering fix if the command fails before training.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=selective --compile.enable`
  - Success criteria and expected risk: Kept at `a68a3c` with 7,898 tps, 33.00% MFU, 108.2GiB peak memory, and falling finite loss.

- ~~Idea: Local batch size 6 with model-only compile, selective AC, and no-reshard~~
  - Current best source commit: a68a3c
  - Source: memory-headroom follow-up
  - Expected mechanism for improving reported tokens/sec: A larger local batch might increase useful work per collective once per-block compile reduces compute overhead.
  - Supporting evidence: Model-only compile uses 108.2GiB, leaving some apparent memory headroom.
  - Planned source/config changes: None; command-only candidate.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=selective --compile.enable --compile.components model --training.local_batch_size=6`
  - Success criteria and expected risk: Crashed at `0461f6`; compiled forward OOMed before step 1, so the apparent memory headroom is not enough for batch 6 after compiler temporaries.

- ~~Idea: Compile model and loss with selective AC and no-reshard~~
  - Current best source commit: a68a3c
  - Source: compile follow-up
  - Expected mechanism for improving reported tokens/sec: Compiling the loss in addition to transformer blocks might reduce loss-side overhead and improve end-to-end tps.
  - Supporting evidence: The current best uses `--compile.components model`; the default compile components are model plus loss, so this is a narrow command-only follow-up.
  - Planned source/config changes: None; command-only candidate.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=selective --compile.enable`
  - Success criteria and expected risk: Discarded; it completed but reached only 7,139 tps, below the 7,898 tps model-only compile best.

- ~~Idea: Profile model-only compile current best~~
  - Current best source commit: a68a3c
  - Source: manager review after new best
  - Expected mechanism for improving reported tokens/sec: A profiled 10-step run of the current best will show whether the next bottleneck is now attention, dense matmul, FSDP collectives, loss, or compile/runtime overhead.
  - Supporting evidence: Model-only compile improved tps by 16.0% over selective AC without compile and changed peak memory from 113.7GiB to 108.2GiB. The previous profile no longer represents the current compiled execution path.
  - Planned source/config changes: None; diagnostic command-only run.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=selective --compile.enable --compile.components model --profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1`
  - Success criteria and expected risk: Completed at `64178839` as a diagnostic discard; the trace shows dense matmuls and flash attention backward dominate the compiled path, with NCCL a smaller slice than before.

- ~~Idea: MXFP8 linear converter with model-only compile~~
  - Current best source commit: a68a3c
  - Source: compiled profile and B200 quantization research
  - Expected mechanism for improving reported tokens/sec: The compiled profile is dominated by large dense linear GEMMs and attention kernels. MXFP8 dynamic quantization can move Qwen3's linear layers onto Blackwell FP8/MX kernels, reducing GEMM time while retaining high-precision communication.
  - Supporting evidence: The latest rank 0 compiled trace has about 5.68s kernel time in the profiled step, with top kernels mostly nvjet matmuls and flash attention backward; NCCL is about 0.51s. `torchao` is installed, B200 satisfies SM100, and `MXFP8LinearConverter` is available. The TorchTitan MXFP8 docs say it is compatible with `torch.compile` and FSDP2 and requires B200/SM100.
  - Planned source/config changes: In `torchtitan/models/qwen3/config_registry.py`, import `MXFP8LinearConverter` and set `qwen3_14b()` to call `model_registry("14B", converters=[MXFP8LinearConverter.Config(model_compile_enabled=True)])`. Keep the edit limited to this model_spec converter choice and revert if discarded.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=selective --compile.enable --compile.components model`
  - Success criteria and expected risk: Crashed at `a76c225`; conversion succeeded, but backward failed in `torchao.prototype.mx_formats.kernels.mxfp8_quantize_cuda` with an invalid argument from the MXFP8 quantize backward path. Discard the broad all-linear MXFP8 converter for now.

- Idea: FlexAttention flash backend with model-only compile
  - Current best source commit: a68a3c
  - Source: compiled profile and attention-backend research
  - Expected mechanism for improving reported tokens/sec: The compiled profile still has flash attention backward and forward work among the top kernels. Switching the Qwen3 14B `model_spec` from the default SDPA attention backend to `flex_flash` may use the FlexAttention FLASH backend with block-causal masks on Blackwell, improving attention kernel scheduling without changing the model flavor or parallel layout.
  - Supporting evidence: `get_attention_config("flex_flash")` is supported on Hopper/Blackwell, configures `FlexAttention.Config(block_size=(256, 128), kernel_options={"BACKEND": "FLASH"})`, and returns a block-causal mask. The current best profile is no longer primarily NCCL-bound, so attention-kernel changes are a plausible narrow next lever.
  - Planned source/config changes: In `torchtitan/models/qwen3/config_registry.py`, change only the `qwen3_14b()` `model_spec` call to `model_registry("14B", attn_backend="flex_flash")`. Revert if discarded.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=selective --compile.enable --compile.components model`
  - Success criteria and expected risk: Crashed before training at `330843d`; Inductor reported `BACKEND='FLASH' but flash attention cannot be used: CUTE flash attention library is not available`.

- ~~Idea: Plain FlexAttention backend with model-only compile~~
  - Current best source commit: a68a3c
  - Source: accidental follow-up after flex-flash crash
  - Expected mechanism for improving reported tokens/sec: Use FlexAttention with block-causal masks without the unavailable FLASH backend, possibly improving attention scheduling or mask handling versus SDPA.
  - Supporting evidence: The same attention-backend hook is in scope through `model_registry("14B", attn_backend="flex")`, and it avoids the CUTE flash lowering requirement that blocked `flex_flash`.
  - Planned source/config changes: In `qwen3_14b()`, set `model_spec=model_registry("14B", attn_backend="flex")`.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=selective --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded from the observed run in `run.log`; it completed with finite loss dropping from 12.46586 to 11.08583, but only reached 5,101 tps and 21.31% MFU, far below the 7,898 tps SDPA current best. The source should be restored to default SDPA.

- ~~Idea: Float8 rowwise linear converter with model-only compile~~
  - Current best source commit: a68a3c
  - Source: compiled profile and quantization follow-up after MXFP8 crash
  - Expected mechanism for improving reported tokens/sec: The current best compiled profile is dominated by dense linear GEMMs and flash attention. Float8 rowwise training can accelerate large linear layers on SM89+ hardware, and unlike the failed MXFP8 path it uses the more established `Float8LinearConverter` implementation.
  - Supporting evidence: B200 supports float8 tensor cores; TorchTitan's Llama configs already use `Float8LinearConverter` with model compile. Qwen3 14B's repeated FFN and attention projections are all dimensions divisible by 16. The small combined KV projection (`5120 x 1024`) and LM head are higher-risk/less-obvious wins, so filter them out and allow torchao's `auto_filter_small_kn` to skip any additional small shapes.
  - Planned source/config changes: In `qwen3_14b()` only, locally import `Float8LinearConverter` and set `model_spec=model_registry("14B", converters=[Float8LinearConverter.Config(recipe_name="rowwise", filter_fqns=["lm_head", "attention.qkv_linear.wkv", "auto_filter_small_kn"], model_compile_enabled=True)])`.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=selective --compile.enable --compile.components model`
  - Success criteria and expected risk: Kept at `ba580fde`; the completed run in `run.log` reached 8,399 tps, MFU `N/A`, 108.15GiB peak memory, and finite loss falling from 12.48223 to 11.18137. The worker dropped `auto_filter_small_kn` before committing because the installed torchao helper expects built `nn.Linear` modules and converted zero Qwen3 config-time linears; the measured source manually filters `lm_head` and `attention.qkv_linear.wkv`.

- ~~Idea: Profile Float8 rowwise current best~~
  - Current best source commit: ba580fde
  - Source: new-best diagnostic
  - Expected mechanism for improving reported tokens/sec: Profiling the new Float8 best should show whether the remaining bottleneck is now dynamic scaling/casting overhead, FFN/attention GEMMs, FSDP collectives, loss/logit projection, or runtime overhead before adding another quantization or parallelism change.
  - Supporting evidence: Float8 rowwise improved the previous best from 7,898 to 8,399 tps and TorchTitan reports MFU as `N/A` for this quantized candidate, so the previous compiled profile no longer describes the active best.
  - Planned source/config changes: None; diagnostic command-only run using the current best source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=selective --compile.enable --compile.components model --profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1`
  - Success criteria and expected risk: Completed as diagnostic discard; profiled tps was 7,812 with profiler overhead. The trace shows shorter overall kernel time than the bf16 compiled profile, visible Float8 scaling/casting kernels, and NCCL again close to 1s in the captured profile step.

- ~~Idea: MXFP8 feed-forward-only converter with model-only compile~~
  - Current best source commit: ba580fde
  - Source: lower-priority quantization follow-up after broad MXFP8 crash
  - Expected mechanism for improving reported tokens/sec: Qwen3 14B FFN linears are the largest repeated GEMMs (`5120 x 17408`, `17408 x 5120`, `5120 x 17408`) and are dimensions aligned for MXFP8. Restricting MXFP8 conversion to `feed_forward` may avoid the broad all-linear crash path involving the LM head, KV projection, or attention output/query linears while still accelerating the dominant dense matmul bucket.
  - Supporting evidence: The compiled profile is GEMM-heavy, TorchTitan MXFP8 is intended for B200/SM100 with compile/FSDP2, and the MXFP8 docs recommend filtering linears such as small KV projections. The all-linear MXFP8 candidate converted and started training before failing in MXFP8 quantize backward, so a narrower FQN set is a plausible root-cause isolation step rather than repeating the same broad crash.
  - Planned source/config changes: In `qwen3_14b()` only, locally import `MXFP8LinearConverter` and set `model_spec=model_registry("14B", converters=[MXFP8LinearConverter.Config(recipe_name="mxfp8_rceil", fqns=["feed_forward"], model_compile_enabled=True)])`.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=selective --compile.enable --compile.components model`
  - Success criteria and expected risk: Crashed before completing step 1; even FFN-only MXFP8 failed in `torch.ops.torchao.mxfp8_quantize.default` in the compiled backward path. Stop pursuing MXFP8 on this environment without changing torchao/runtime.

- ~~Idea: Float8 current best with compiler memory-budget activation checkpointing~~
  - Current best source commit: ba580fde / branch source at `bc825c47`
  - Source: Float8 profile and memory-headroom analysis
  - Expected mechanism for improving reported tokens/sec: The current best uses manual selective activation checkpointing, which still recomputes work inside each compiled block. Compiler memory-budget activation checkpointing may save more useful activations while staying under the B200 memory limit, reducing recompute on the Float8 path and improving steady-state tps.
  - Supporting evidence: Float8 current best uses about 108.2GiB peak on 178.35GiB B200s, leaving substantial headroom. No-AC no-reshard OOMed in the bf16 path, so a mid-high compiler memory budget is safer than disabling checkpointing entirely. The latest Float8 profile shows only about 2.8s kernel time but visible recompute/scaling work, so reducing recompute is a plausible next lever.
  - Planned source/config changes: None; command-only candidate on the current Float8 source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.75 --compile.enable --compile.components model`
  - Success criteria and expected risk: Kept at `9cf1a5ff`; memory budget 0.75 completed with finite loss falling from 12.26095 to 6.07721, 8,821 tps, MFU `N/A`, and 129.9GiB peak memory.

- ~~Idea: Float8 memory-budget activation checkpointing at 0.9~~
  - Current best source commit: ba580fde / branch source at `bc825c47`
  - Source: memory-budget follow-up
  - Expected mechanism for improving reported tokens/sec: The 0.75 compiler memory budget improved throughput by reducing recomputation while using 129.9GiB peak memory. Raising the budget to 0.9 may save more activations and further reduce recomputation while still keeping below the 178.35GiB B200 capacity.
  - Supporting evidence: The 0.75 run increased peak memory by about 21.7GiB over selective AC and improved tps from 8,399 to 8,821 with healthy loss. This leaves roughly 48GiB of headroom to the physical capacity and about 39GiB to the program's rough 95% risk line.
  - Planned source/config changes: None; command-only candidate on the current Float8 source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.9 --compile.enable --compile.components model`
  - Success criteria and expected risk: Mixed but best observed row is kept. The first recorded run completed with finite/falling loss but only 8,011 tps, below the 8,821 best. An unintended duplicate run of the exact same command completed with finite/falling loss from 12.34735 to 9.44670, 144.0GiB peak memory, and 8,877 tps. Treat 0.9 as the current best but note the large run-to-run variance.

- ~~Idea: Profile Float8 memory-budget 0.9 current best~~
  - Current best source commit: ba580fde / branch source at `bc825c47`
  - Source: accidental diagnostic follow-up
  - Expected mechanism for improving reported tokens/sec: A profile of the 0.9 current best should show whether the extra activation memory shifted the bottleneck back to FSDP communication, attention, Float8 scaling, or CPU/runtime overhead.
  - Supporting evidence: Memory-budget 0.9 is the best observed setting but noisy, and budget 0.75 already showed NCCL and Float8 scaling overhead as material.
  - Planned source/config changes: None; diagnostic command-only run.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.9 --compile.enable --compile.components model --profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1`
  - Success criteria and expected risk: Completed as diagnostic discard with profiled tps 7,885, loss falling from 12.52719 to 9.78586, and 144.0GiB peak memory. The rank-0 trace shows NCCL total around 1.26s, with reduce-scatter the largest single kernel bucket, so communication is again a primary bottleneck.

- ~~Idea: Float8 memory-budget activation checkpointing at 1.0~~
  - Current best source commit: ba580fde / branch source at `bc825c47`
  - Source: memory-budget follow-up
  - Expected mechanism for improving reported tokens/sec: Budget 0.9 improved slightly over 0.75 while using 144.0GiB, so budget 1.0 may further reduce recomputation by using the compiler's runtime-optimized activation strategy.
  - Supporting evidence: Budget 0.9 remains below B200 capacity and below the program's rough 95% memory-risk threshold. The throughput gain over 0.75 is small and noisy, but the run is stable and still has roughly 25GiB to the 95% line.
  - Planned source/config changes: None; command-only candidate on the current Float8 source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=1.0 --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded at `3c76b65c`; the run completed with finite/slightly falling loss from 12.59874 to 12.51579, 154.35GiB peak memory, and 8,851 tps. Extra activation memory did not beat the 8,877 tps budget 0.9 best.

- ~~Idea: HSDP 2x4 with Float8 memory-budget 0.75~~
  - Current best source commit: `99ad2926`; current best result row: 8,877 tps from Float8 rowwise plus memory-budget 0.9.
  - Source: Float8 memory-budget 0.9 profile and HSDP/FSDP2 mesh research.
  - Expected mechanism for improving reported tokens/sec: The 0.9 profile shows NCCL around 1.26s with reduce-scatter the largest single kernel bucket. HSDP with two replica groups of four GPUs should reduce each FSDP shard group's reduce-scatter/all-gather scope from 8 GPUs to 4 GPUs, trading some cross-replica all-reduce and higher parameter memory for less sharded-collective pressure.
  - Supporting evidence: `docs/fsdp.md` says FSDP2 uses a 2D mesh for HSDP with replication on mesh axis 0 and sharding on axis 1. `torchtitan/experiments/transformers_modeling_backend/parallelize.py` uses `parallel_dims.get_mesh(["dp_replicate", "fsdp"])` when `parallel_dims.dp_replicate_enabled`, matching that contract. Memory-budget 0.9 already uses 144.0GiB, so this lower-budget HSDP probe left room for the smaller shard group.
  - Planned source/config changes: In `torchtitan/models/qwen3/parallelize.py`, replace the HSDP `NotImplementedError` with a Qwen3-local mesh selection: `["dp_replicate", "fsdp"]` when `parallel_dims.dp_replicate_enabled`, otherwise `"fsdp"`. Keep the existing Float8 rowwise `qwen3_14b()` source unchanged.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.data_parallel_replicate_degree=2 --parallelism.data_parallel_shard_degree=4 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.75 --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded; the run completed with finite/falling loss from 12.62485 to 7.83562, MFU `N/A`, and 157.72GiB peak memory, but throughput was only 5,455 tps. The Qwen3 HSDP source change was reverted because it underperformed the 8,877 tps best.

- Idea: HSDP 2x4 with Float8 memory-budget 0.9
  - Current best source commit: `99ad2926`; current best result row: 8,877 tps from Float8 rowwise plus memory-budget 0.9.
  - Source: Float8 memory-budget 0.9 profile and HSDP/FSDP2 mesh research.
  - Expected mechanism for improving reported tokens/sec: The 0.9 profile shows NCCL around 1.26s with reduce-scatter the largest single kernel bucket. HSDP with two replica groups of four GPUs should reduce each FSDP shard group's reduce-scatter/all-gather scope from 8 GPUs to 4 GPUs, trading some cross-replica all-reduce and higher parameter memory for less sharded-collective pressure.
  - Supporting evidence: `docs/fsdp.md` says FSDP2 uses a 2D mesh for HSDP with replication on mesh axis 0 and sharding on axis 1. `torchtitan/experiments/transformers_modeling_backend/parallelize.py` uses `parallel_dims.get_mesh(["dp_replicate", "fsdp"])` when `parallel_dims.dp_replicate_enabled`, matching that contract. Memory-budget 0.9 already uses 144.0GiB and budget 1.0 used 154.35GiB, so a first 0.9 HSDP probe is plausible but memory-riskier than 0.75.
  - Planned source/config changes: In `torchtitan/models/qwen3/parallelize.py`, replace the HSDP `NotImplementedError` with a Qwen3-local mesh selection: `["dp_replicate", "fsdp"]` when `parallel_dims.dp_replicate_enabled`, otherwise `"fsdp"`. Keep the existing Float8 rowwise `qwen3_14b()` source unchanged.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.data_parallel_replicate_degree=2 --parallelism.data_parallel_shard_degree=4 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.9 --compile.enable --compile.components model`
  - Success criteria and expected risk: Keep only if the 10-step run completes with finite/falling loss and exceeds 8,877 tps. Risks are higher per-rank parameter memory from shard degree 4, added replica all-reduce, and noisy throughput; if it crashes or underperforms, revert the source change and record discard/crash.
