## Human Generated Ideas

- ~~Idea: Bootstrap runnable Qwen3 14B DP-sharded run~~
  - Current best source commit: 68f86f1
  - Source: human/setup
  - Expected mechanism for improving reported tokens/sec: The target command currently selects `dp_shard=-1` on 8 GPUs, but `parallelize_qwen3()` only applies replicated DDP and does not apply FSDP for the active `fsdp` mesh. A narrow bootstrap should make the inferred target command run correctly by applying TorchTitan FSDP on the configured shard mesh.
  - Supporting evidence: `qwen3_14b()` sets `data_parallel_shard_degree=-1`, which resolves to `dp_shard=8` for `NGPU=8`; the Qwen3 scaffold rejects TP/CP/PP/EP and otherwise only calls `replicate()` when `dp_enabled`.
  - Planned source/config changes: In `torchtitan/models/qwen3/parallelize.py`, replace the DP-only replicated wrapper for the active DP-shard case with the minimal FSDP wrapping needed for Qwen3 14B.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10`
  - Success criteria and expected risk: Kept at `fc6629b` with 5,774 tps; later command-only no-reshard improved to 5,872 tps.

- Compare observed performance in traces to roofline to guide optimization search.

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

- ~~Idea: FlexAttention flash backend with model-only compile~~
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

- ~~Idea: HSDP 2x4 with Float8 memory-budget 0.9~~
  - Current best source commit: `99ad2926`; current best result row: 8,877 tps from Float8 rowwise plus memory-budget 0.9.
  - Source: Float8 memory-budget 0.9 profile and HSDP/FSDP2 mesh research.
  - Expected mechanism for improving reported tokens/sec: The 0.9 profile shows NCCL around 1.26s with reduce-scatter the largest single kernel bucket. HSDP with two replica groups of four GPUs should reduce each FSDP shard group's reduce-scatter/all-gather scope from 8 GPUs to 4 GPUs, trading some cross-replica all-reduce and higher parameter memory for less sharded-collective pressure.
  - Supporting evidence: `docs/fsdp.md` says FSDP2 uses a 2D mesh for HSDP with replication on mesh axis 0 and sharding on axis 1. `torchtitan/experiments/transformers_modeling_backend/parallelize.py` uses `parallel_dims.get_mesh(["dp_replicate", "fsdp"])` when `parallel_dims.dp_replicate_enabled`, matching that contract. Memory-budget 0.9 already uses 144.0GiB and budget 1.0 used 154.35GiB, so a first 0.9 HSDP probe is plausible but memory-riskier than 0.75.
  - Planned source/config changes: In `torchtitan/models/qwen3/parallelize.py`, replace the HSDP `NotImplementedError` with a Qwen3-local mesh selection: `["dp_replicate", "fsdp"]` when `parallel_dims.dp_replicate_enabled`, otherwise `"fsdp"`. Keep the existing Float8 rowwise `qwen3_14b()` source unchanged.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.data_parallel_replicate_degree=2 --parallelism.data_parallel_shard_degree=4 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.9 --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded at `d0ab38cc`; the run completed with finite/falling loss from 12.48534 to 7.49267, but throughput was only 5,837 tps and peak memory was 172.79GiB, above the program's rough 95% risk line. The Qwen3 HSDP source change was reverted.

- ~~Idea: TP=2 revisit with Float8 memory-budget current best~~
  - Current best source commit: `cbb4aeba`; current best result row: 8,877 tps from 8-way FSDP Float8 rowwise plus memory-budget 0.9.
  - Source: human-requested bounded TP revisit after HSDP discard.
  - Expected mechanism for improving reported tokens/sec: The earlier TP=2+SP discard was measured before no-reshard, selective/model-only compile, Float8 rowwise, and memory-budget activation checkpointing changed the bottleneck. Reusing the known-good dense Qwen3 TP2 sharding contract may reduce local matmul and activation work under the current best stack while preserving the current no-reshard FSDP and Float8 setup.
  - Supporting evidence: Qwen3 14B has 40 query heads and 8 KV heads, both divisible by TP=2, and commit `e67f61c6` already established a working dense GQA + dense FFN sharding patch. HSDP is now closed, but TP evidence is stale relative to the current best.
  - Planned source/config changes: In `torchtitan/models/qwen3/parallelize.py`, remove the TP guard and call `model.parallelize(tp_mesh)` before FSDP while keeping HSDP disabled. In `torchtitan/models/qwen3/sharding.py`, restore the shared decoder sharding helpers for dense GQA attention, QK norm, dense FFN, and block norms.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.tensor_parallel_degree=2 --parallelism.data_parallel_shard_degree=4 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.9 --compile.enable --compile.components model`
  - Success criteria and expected risk: Crashed before step 1. The initial source ordering and a minimal compile-before-TP ordering fix both failed in Inductor's memory-budget partitioner with `Unknown metadata type FakeScriptObject`, so TP=2 plus model-only compile plus memory-budget activation checkpointing is blocked in this stack. The TP source was reverted.

- ~~Idea: Include Qwen3 KV projection in Float8 rowwise~~
  - Current best source commit: `cbb4aeba`; current best result row: 8,877 tps from 8-way FSDP Float8 rowwise plus memory-budget 0.9.
  - Source: unmeasured candidate commit `99171b1b` and Float8 coverage review.
  - Expected mechanism for improving reported tokens/sec: The current Float8 source leaves `attention.qkv_linear.wkv` in bf16 because it is smaller than the main FFN and query/output projections. It is still a repeated 40-layer `5120 x 1024` projection, so converting it may remove a remaining bf16 GEMM bucket and reduce dtype boundary overhead in the attention projection path.
  - Supporting evidence: The active best already uses Float8 rowwise successfully for 200 larger linears, and the current profile still shows Float8 scaling/casting plus communication rather than pure attention math. The old source-only commit removed only the `attention.qkv_linear.wkv` filter and was never recorded in `results.tsv`, so this is a narrow unmeasured candidate. Keep `lm_head` filtered because it is large, numerically sensitive, and only used at the loss boundary.
  - Planned source/config changes: In `qwen3_14b()` only, remove `"attention.qkv_linear.wkv"` from `Float8LinearConverter.Config(filter_fqns=...)`; keep `"lm_head"` filtered and keep `recipe_name="rowwise"` with `model_compile_enabled=True`.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.9 --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded; the 10-step run completed with finite/falling loss from 12.53812 to 7.92596, MFU `N/A`, and 137.35GiB peak memory, but throughput was only 8,190 tps versus the 8,877 tps best. The `qwen3_14b()` Float8 filter source change was reverted.

- ~~Idea: Float8 tensorwise recipe with current memory-budget best~~
  - Current best source commit: `0f036086`; current best result row: 8,877 tps from rowwise Float8, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.9.
  - Source: Float8 profile, KV coverage result, and TorchAO recipe inspection.
  - Expected mechanism for improving reported tokens/sec: The current rowwise recipe adds many axiswise scaling/casting kernels. TorchAO's `Float8LinearConfig.from_recipe_name("tensorwise")` is present in the installed environment and uses tensorwise dynamic scales, which may reduce scale-reduction overhead and improve compile/runtime efficiency on the large repeated FFN and attention projection linears.
  - Supporting evidence: The Float8 profiles show visible scale/cast kernels alongside NCCL, and expanding Float8 to smaller KV projections was slower, so the next narrow quantization lever should change scaling overhead for the already-good large-linears subset rather than convert more linears. Tensorwise is riskier numerically than rowwise but still keeps communication in high precision and leaves `lm_head` and `attention.qkv_linear.wkv` filtered.
  - Planned source/config changes: In `qwen3_14b()` only, change `Float8LinearConverter.Config(recipe_name="rowwise", ...)` to `recipe_name="tensorwise"` while keeping the existing filter list and `model_compile_enabled=True`.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.9 --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded; the 10-step run completed but loss rose from 12.28795 to 15.71423, throughput reached only 5,000 tps versus the 8,877 tps best, MFU was `N/A`, and peak memory was 145.67GiB. The `qwen3_14b()` Float8 recipe source change was reverted.

- ~~Idea: TP=2 root-cause with selective AC~~
  - Current best source commit: `0f036086`; current best result row: 8,877 tps from 8-way FSDP Float8 rowwise plus memory-budget 0.9.
  - Source: human-requested TP root-cause follow-up.
  - Expected mechanism for improving reported tokens/sec: The TP=2 memory-budget run failed in Inductor's memory-budget/min-cut partitioner before step 1. Switching only activation checkpointing back to selective removes that partitioner while keeping TP=2, Float8 rowwise, model-only compile, no-reshard FSDP, and the current source filters, so it tests whether TP itself remains runnable/useful under the current Float8+compile stack.
  - Supporting evidence: Qwen3 14B's 40 query heads and 8 KV heads are divisible by TP=2, and the earlier TP2 sharding patch is known to build. The previous failure happened in memory-budget partitioning, not in Qwen3 head divisibility or FSDP mesh construction.
  - Planned source/config changes: Reapply the Qwen3 TP2 source/sharding patch from `e67f61c6`, keep HSDP disabled, and keep compile before TP wrapping as the least-bad ordering from the memory-budget attempt.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.tensor_parallel_degree=2 --parallelism.data_parallel_shard_degree=4 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=selective --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded at `05a39792`; the run completed with finite/falling loss from 3.06316 to 2.32347 and 72.00GiB peak memory, but reached only 7,017 tps, below the 8,877 tps current best. TP+compile is runnable with selective AC, so the prior crash is specific to the memory-budget path, but TP remains slower here. The TP source was reverted.

- ~~Idea: Float8 memory-budget activation checkpointing at 0.95~~
  - Current best source commit: `524fe989`; current best result row: 8,877 tps from 8-way FSDP Float8 rowwise plus memory-budget 0.9.
  - Source: manager memory-budget sweep follow-up.
  - Expected mechanism for improving reported tokens/sec: Budget 0.9 is the best observed row but noisy, while budget 1.0 used 154.35GiB and was slightly slower. A 0.95 budget may retain more activations than 0.9 without taking the full extra memory/runtime cost of 1.0.
  - Supporting evidence: The 0.75, 0.9, and 1.0 rows are tightly clustered at 8,821, 8,877, and 8,851 tps, so the best point may sit between the measured 0.9 and 1.0 settings. The 1.0 run stayed below the rough memory risk line, so 0.95 is unlikely to OOM.
  - Planned source/config changes: None; command-only candidate on the current Float8 rowwise source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.95 --compile.enable --compile.components model`
  - Success criteria and expected risk: Kept at `524fe989`; the run completed with finite/falling loss from 12.28270 to 9.57252, 143.96GiB peak memory, and 8,897 tps. This narrowly beats the prior 8,877 tps memory-budget 0.9 best without increasing memory materially.

- ~~Idea: Float8 memory-budget activation checkpointing at 0.975~~
  - Current best source commit: `a9329d6c`; current best result row: 8,897 tps from 8-way FSDP Float8 rowwise plus memory-budget 0.95.
  - Source: manager high-side memory-budget threshold probe.
  - Expected mechanism for improving reported tokens/sec: Budget 0.95 matched the 0.9 memory footprint but beat it slightly, while budget 1.0 moved to a higher-memory, slightly slower point. A 0.975 budget tests whether there is an intermediate compiler partition that reduces recompute without paying the full 1.0 memory/runtime cost.
  - Supporting evidence: The 0.9 and 0.95 rows both used about 144GiB, suggesting the compiler may choose discrete activation partitions across part of the budget range. The 1.0 row used 154.35GiB and remained below the memory risk line, so 0.975 should be safe enough as a single 10-step probe.
  - Planned source/config changes: None; command-only candidate on the current Float8 rowwise source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.975 --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded at `a9329d6c`; the run completed with finite/falling loss from 12.47411 to 11.52573, 143.96GiB peak memory, and MFU `N/A`, but reached only 5,453 tps versus the 8,897 tps 0.95 best.

- ~~Idea: Repeat-confirm Float8 memory-budget activation checkpointing at 0.95~~
  - Current best source commit: `0c612874`; current best result row: 8,897 tps from 8-way FSDP Float8 rowwise plus memory-budget 0.95.
  - Source: manager repeat-confirmation request after noisy 0.975/0.9 adjacent rows.
  - Expected mechanism for improving reported tokens/sec: Re-running the exact current-best command tests whether the 8,897 tps observation is reproducible and whether normal variance can produce a higher max without changing source or configuration.
  - Supporting evidence: The earlier 0.9 command varied substantially, and 0.975 produced a much slower same-memory run, so a repeat is useful before treating the tiny 0.95 improvement as robust.
  - Planned source/config changes: None; command-only repeat on the current Float8 rowwise source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.95 --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded as repeat diagnostic at `0c612874`; the run completed with finite/falling loss from 12.49943 to 7.65341 and 143.96GiB peak memory, but reached 8,891 tps, slightly below the 8,897 tps best.

- ~~Idea: Float8 memory-budget activation checkpointing at 0.85~~
  - Current best source commit: `0c612874`; current best result row: 8,897 tps from 8-way FSDP Float8 rowwise plus memory-budget 0.95.
  - Source: manager lower-side memory-budget plateau probe.
  - Expected mechanism for improving reported tokens/sec: The 0.75 row used 129.9GiB and reached 8,821 tps, while 0.9 and 0.95 used about 144GiB and reached 8,877-8,897 tps. A 0.85 budget may select the same useful activation-save partition as 0.9/0.95, or a slightly cheaper one, without moving into the high-side instability seen at 0.975.
  - Supporting evidence: The 0.975 run kept the same 143.96GiB peak memory as 0.95 but collapsed to 5,453 tps, so high-side budgets are not reliably better. Testing 0.85 checks whether the current best partition starts below 0.9 and whether the lower side has less variance.
  - Planned source/config changes: None; command-only candidate on the current Float8 rowwise source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.85 --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded at `0c612874`; the run completed with finite but rising loss from 12.52796 to 17.56807, reached 8,834 tps with MFU `N/A`, and used 136.50GiB peak memory. It did not beat the 8,897 tps 0.95 best and failed the falling-loss sanity check.

- ~~Idea: Local batch size 5 with Float8 memory-budget 0.95~~
  - Current best source commit: `3cde416b`; current best result row: 8,897 tps from 8-way FSDP Float8 rowwise plus memory-budget 0.95 and local batch size 4.
  - Source: manager bounded batch-size probe after activation-budget saturation; this was the already-active run that superseded the pending 0.925 handoff for this iteration.
  - Expected mechanism for improving reported tokens/sec: Increasing local batch size from 4 to 5 raises global batch size from 32 to 40 tokens-per-step units, which could improve device utilization and amortize fixed per-step communication/runtime overhead if the extra activation memory remains below the rough risk line.
  - Supporting evidence: The current best uses 143.96GiB on 178.35GiB B200s, leaving roughly 25GiB to the 95% memory-risk line. Earlier batch-size probes were before Float8 and memory-budget activation checkpointing, so the evidence was stale.
  - Planned source/config changes: None; command-only candidate on the current Float8 rowwise source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.95 --compile.enable --compile.components model --training.local_batch_size=5`
  - Success criteria and expected risk: Discarded at `3cde416b`; the run completed with finite/falling loss from 12.55001 to 8.61428 and stayed below the rough memory risk line at 164.84GiB, but throughput reached only 5,924 tps, far below the 8,897 tps local-batch-size 4 best.

- ~~Idea: Float8 memory-budget activation checkpointing at 0.925~~
  - Current best source commit: `3cde416b`; current best result row: 8,897 tps from 8-way FSDP Float8 rowwise plus memory-budget 0.95.
  - Source: manager centered plateau probe.
  - Expected mechanism for improving reported tokens/sec: Budget 0.9 and 0.95 both use the high-throughput 144GiB memory region, and a repeat of 0.95 confirmed the setting at 8,891 tps. Budget 0.925 may select the same activation partition while avoiding the bad high-side 0.975 behavior.
  - Supporting evidence: The useful rows are tightly clustered at 0.9 and 0.95, while 0.85 used less memory and failed the loss trend check, and 0.975 used the same memory but was much slower. A centered point is the last cheap way to test whether 0.95 is truly the local optimum.
  - Planned source/config changes: None; command-only candidate on the current Float8 rowwise source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded at `460ce52c`; the run completed with finite/falling loss from 12.39717 to 9.11266, MFU `N/A`, and 143.96GiB peak memory, but reached only 8,718 tps versus the 8,897 tps 0.95 best.

- ~~Idea: Default compile components with Float8 memory-budget 0.95~~
  - Current best source commit: `0b83acb6`; current best result row: 8,897 tps from 8-way FSDP Float8 rowwise plus memory-budget 0.95 and model-only compile.
  - Source: manager compile-components revisit after the current Float8 memory-budget best.
  - Expected mechanism for improving reported tokens/sec: Earlier model-plus-loss compile was slower before Float8 and memory-budget activation checkpointing. Re-running without `--compile.components model` tests whether compiling the loss function now helps or remains overhead on the current best stack.
  - Supporting evidence: The current best uses model-only compile, while the default compile configuration logs `Compiling the loss function with torch.compile` in addition to compiling TransformerBlocks.
  - Planned source/config changes: None; command-only candidate on the current Float8 rowwise source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.95 --compile.enable`
  - Success criteria and expected risk: Discarded at `0b83acb6`; the run completed with finite/falling loss from 12.46363 to 7.79407 and acceptable 141.93GiB peak memory, but throughput reached only 8,882 tps, below the 8,897 tps model-only compile best.

- ~~Idea: Float8 rowwise with high-precision grad-weight path~~
  - Current best source commit: `bc4abd27`; current best result row: 8,897 tps from 8-way FSDP Float8 rowwise plus memory-budget 0.95 and model-only compile.
  - Source: manager final supported Float8 recipe check.
  - Expected mechanism for improving reported tokens/sec: TorchTitan and TorchAO support `rowwise_with_gw_hp`, which disables Float8 casting for the grad-weight GEMM inputs. This is primarily a numerical-safety variant, but it also removes some grad-weight scaling work; a 10-step run can verify whether the reduced casting overhead offsets the bf16 grad-weight GEMM cost.
  - Supporting evidence: TorchAO's recipe config shows `rowwise_with_gw_hp` keeps forward and grad-input rowwise Float8 while making the grad-weight GEMM high precision. The tensorwise recipe was slower and numerically bad, so this is the last supported Float8 recipe variant worth measuring before closing quantization recipe tuning.
  - Planned source/config changes: In `qwen3_14b()` only, change `Float8LinearConverter.Config(recipe_name="rowwise", ...)` to `recipe_name="rowwise_with_gw_hp"` while keeping the existing filters and `model_compile_enabled=True`.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.95 --compile.enable --compile.components model`
  - Success criteria and expected risk: Kept at `2c54749b`; the run completed with finite/falling loss from 12.29840 to 9.54931, MFU `N/A`, 145.05GiB peak memory, and 9,229 tps, beating the prior 8,897 tps best. The `rowwise_with_gw_hp` recipe source change was kept.

- ~~Idea: Repeat-confirm Float8 rowwise_with_gw_hp current best~~
  - Current best source commit: `2c54749b`; current best result row: 9,229 tps from `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.95.
  - Source: manager repeat-confirmation after unexpectedly large recipe win; an identical command is already active in the checkout.
  - Expected mechanism for improving reported tokens/sec: Re-running the exact new-best command checks whether the 9,229 tps result is stable or a favorable variance sample before spending runs on adjacent activation budgets.
  - Supporting evidence: Earlier memory-budget rows showed high variance, and `rowwise_with_gw_hp` unexpectedly beat rowwise by 332 tps despite using a high-precision grad-weight path. A repeat is the fastest way to distinguish a robust recipe win from noise.
  - Planned source/config changes: None; repeat on the current kept `rowwise_with_gw_hp` source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.95 --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded as repeat diagnostic at `a1f59bf`; the run completed with finite/falling loss from 12.37330 to 7.53270, MFU `N/A`, and the same 145.05GiB peak memory, but reached 9,213 tps versus the 9,229 tps best.

- Idea: Float8 rowwise_with_gw_hp memory-budget 1.0
  - Current best source commit: `90b4c8b6`; current best result row: 9,229 tps from `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.95.
  - Source: manager activation-budget retune after recipe change.
  - Expected mechanism for improving reported tokens/sec: The high-precision grad-weight recipe reduced Float8 scaling/casting overhead and shifted the current best to 9.2k tps. With that overhead reduced, spending more activation memory at budget 1.0 may reduce recompute enough to beat the 0.95 rows.
  - Supporting evidence: The `rowwise_with_gw_hp` 0.95 repeat was stable at 9,213-9,229 tps and used 145.05GiB, leaving headroom below the rough 95% B200 memory risk line. The old rowwise 1.0 row used 154.35GiB and did not beat rowwise 0.95, but that evidence is stale after the recipe changed the kernel mix.
  - Planned source/config changes: None; command-only candidate on the current kept `rowwise_with_gw_hp` source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=1.0 --compile.enable --compile.components model`
  - Success criteria and expected risk: Keep only if the run completes with finite/falling loss and beats 9,229 tps. Discard if it only raises memory without throughput gain or weakens the loss trend.

- ~~Idea: Float8 rowwise_with_gw_hp memory-budget 0.9~~
  - Current best source commit: `90b4c8b6`; current best result row: 9,229 tps from `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.95.
  - Source: already-active run discovered during the 1.0 handoff; it superseded the pending 1.0 command for this iteration.
  - Expected mechanism for improving reported tokens/sec: The recipe change slightly raised peak memory and changed casting overhead, so rechecking budget 0.9 tests whether a lower compiler activation budget can recover throughput while keeping the new Float8 recipe.
  - Supporting evidence: The current best and repeat both used 145.05GiB at budget 0.95, while older rowwise budget 0.9 evidence is stale because it used a different Float8 recipe.
  - Planned source/config changes: None; command-only candidate on the current kept `rowwise_with_gw_hp` source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.9 --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded at `90b4c8b6`; loss rose from 12.59183 to 16.25217 and throughput fell to 4,888 tps at 145.05GiB peak memory, so 0.95 remains the best `rowwise_with_gw_hp` budget.
