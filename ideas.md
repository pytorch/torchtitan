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

- ~~Idea: Float8 rowwise_with_gw_hp memory-budget 1.0~~
  - Current best source commit: `90b4c8b6`; current best result row: 9,229 tps from `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.95.
  - Source: manager activation-budget retune after recipe change.
  - Expected mechanism for improving reported tokens/sec: The high-precision grad-weight recipe reduced Float8 scaling/casting overhead and shifted the current best to 9.2k tps. With that overhead reduced, spending more activation memory at budget 1.0 may reduce recompute enough to beat the 0.95 rows.
  - Supporting evidence: The `rowwise_with_gw_hp` 0.95 repeat was stable at 9,213-9,229 tps and used 145.05GiB, leaving headroom below the rough 95% B200 memory risk line. The old rowwise 1.0 row used 154.35GiB and did not beat rowwise 0.95, but that evidence is stale after the recipe changed the kernel mix.
  - Planned source/config changes: None; command-only candidate on the current kept `rowwise_with_gw_hp` source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=1.0 --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded at `ff585c4e`; the run completed but loss rose from 12.14356 to 17.49169, throughput fell to 1,014 tps, peak memory reached 176.49GiB, and the allocator reported 13 memory allocation retries.

- ~~Idea: Float8 rowwise_with_gw_hp memory-budget 0.9~~
  - Current best source commit: `90b4c8b6`; current best result row: 9,229 tps from `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.95.
  - Source: already-active run discovered during the 1.0 handoff; it superseded the pending 1.0 command for this iteration.
  - Expected mechanism for improving reported tokens/sec: The recipe change slightly raised peak memory and changed casting overhead, so rechecking budget 0.9 tests whether a lower compiler activation budget can recover throughput while keeping the new Float8 recipe.
  - Supporting evidence: The current best and repeat both used 145.05GiB at budget 0.95, while older rowwise budget 0.9 evidence is stale because it used a different Float8 recipe.
  - Planned source/config changes: None; command-only candidate on the current kept `rowwise_with_gw_hp` source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.9 --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded at `90b4c8b6`; loss rose from 12.59183 to 16.25217 and throughput fell to 4,888 tps at 145.05GiB peak memory, so 0.95 remains the best `rowwise_with_gw_hp` budget.

- ~~Idea: Profile Float8 rowwise_with_gw_hp current best~~
  - Current best source commit: `ff585c4e`; current best result row: 9,229 tps from `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.95.
  - Source: already-completed diagnostic run in `run.log`.
  - Expected mechanism for improving reported tokens/sec: Profiling does not compete for best throughput, but it should identify whether the new `rowwise_with_gw_hp` best is dominated by NCCL, attention, bf16 grad-weight GEMMs, remaining Float8 casting, or runtime overhead.
  - Supporting evidence: `rowwise_with_gw_hp` unexpectedly improved throughput over rowwise, and the repeat confirmed a stable 9.2k tps range. A profile is useful before choosing another source-level candidate.
  - Planned source/config changes: None; diagnostic command-only profile on the current kept `rowwise_with_gw_hp` source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.95 --compile.enable --compile.components model --profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1`
  - Success criteria and expected risk: Discard diagnostic at `067fdf0b`; the completed run in `run.log` has finite/falling loss from 12.46392 to 10.33925, MFU `N/A`, 145.05GiB peak memory, and 8,540 profiled tps. The trace directory was logged as `./outputs/profiling/traces`; profiler overhead makes the tps non-ranking.

- ~~Idea: Include Qwen3 lm_head in Float8 rowwise_with_gw_hp~~
  - Current best source commit: `2c54749b`; current best result row: 9,229 tps from `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.95.
  - Source: manager profile review and chunked-loss inspection.
  - Expected mechanism for improving reported tokens/sec: `ChunkedCELoss` applies `lm_head` over eight chunks outside model compile, and Qwen3 14B's head is a large `5120 x 151936` projection. Converting it to Float8 may reduce a remaining large bf16 GEMM bucket and improve the chunked-loss tail without changing mesh shape or activation policy.
  - Supporting evidence: The current profile still has large GEMM/scaled-mm and attention buckets, while direct Float8 scale/cast is only a small bucket. The current best filters `lm_head`, so this large projection remains bf16 even though the converter can handle dimensions divisible by 16. Prior KV-projection expansion was slower, but that was a smaller repeated attention projection; `lm_head` is larger and sits in the chunked loss path.
  - Planned source/config changes: In `qwen3_14b()` only, remove `"lm_head"` from `Float8LinearConverter.Config(filter_fqns=...)`; keep `"attention.qkv_linear.wkv"` filtered, keep `recipe_name="rowwise_with_gw_hp"`, and keep `model_compile_enabled=True`. Revert the source change if the run is discarded or crashes.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.95 --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded at `a7bea58e`; the run completed but loss rose from 12.38387 to 16.43954, throughput reached only 8,399 tps, and peak memory increased to 149.01GiB. The source was reverted.

- ~~Idea: Default FSDP reshard policy with Float8 rowwise_with_gw_hp~~
  - Current best source commit: `eae444b9`; current best result row: 9,229 tps from `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.95.
  - Source: manager deferred source-free reshard policy check.
  - Expected mechanism for improving reported tokens/sec: Omitting the no-reshard override reduces parameter residency and may reduce memory pressure, but it can add all-gather overhead relative to the current explicit no-reshard best.
  - Supporting evidence: Earlier no-reshard helped before the `rowwise_with_gw_hp` recipe; the profile still shows material NCCL time, so default reshard needed a fresh command-only check on the active source.
  - Planned source/config changes: None; command-only candidate on the current kept `rowwise_with_gw_hp` source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.95 --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded at `eae444b9`; the run completed with finite/falling loss from 12.55290 to 10.28098 and lower 120.30GiB peak memory, but reached only 9,067 tps versus the 9,229 tps no-reshard best.

- ~~Idea: Default compile components with Float8 rowwise_with_gw_hp~~
  - Current best source commit: `2c54749b`; current best result row: 9,229 tps from `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.95.
  - Source: manager low-risk compile-components revisit after the `rowwise_with_gw_hp` recipe changed the kernel mix.
  - Expected mechanism for improving reported tokens/sec: Removing `--compile.components model` lets TorchTitan compile the loss function in addition to the model. `ChunkedCELoss` still does not compile `lm_head`, but compiling the per-chunk cross-entropy may reduce loss-side overhead under the current recipe.
  - Supporting evidence: The earlier default-compile check was measured before `rowwise_with_gw_hp`; it was only slightly slower than the old rowwise best. The current `lm_head` Float8 expansion was a discard, so the remaining loss-path lever that does not change numerics is to compile the CE function.
  - Planned source/config changes: None; command-only candidate on the current kept `rowwise_with_gw_hp` source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.95 --compile.enable`
  - Success criteria and expected risk: Discarded at `28d7f02e`; the run completed but loss rose from 12.52405 to 12.72060, throughput reached only 8,687 tps, and peak memory was 142.74GiB.

- ~~Idea: Default FSDP reshard policy with local batch size 5~~
  - Current best source commit: `1436b57f`; current best result row: 9,229 tps from `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, memory-budget 0.95, and local batch size 4.
  - Source: manager bounded follow-up using memory saved by default FSDP reshard.
  - Expected mechanism for improving reported tokens/sec: Default reshard reduced peak memory from 145.05GiB to 120.30GiB at local batch size 4. Increasing local batch size to 5 tests whether those saved bytes can carry more tokens per step and amortize reshard/all-gather overhead.
  - Supporting evidence: Default reshard alone was slower but left roughly 25GiB of extra headroom compared with no-reshard; earlier local-batch-size 5 with no-reshard was memory-heavy and slow.
  - Planned source/config changes: None; command-only candidate on the current kept `rowwise_with_gw_hp` source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.95 --compile.enable --compile.components model --training.local_batch_size=5`
  - Success criteria and expected risk: Discarded at `1436b57f`; the run completed but loss rose from 12.47422 to 14.69380, throughput reached only 8,543 tps, and peak memory rose to 144.04GiB, so it failed both the loss trend and throughput gates.

- ~~Idea: Qwen3 bfloat16 FSDP reduction precision~~
  - Current best source commit: `2c54749b`; current best result row: 9,229 tps from `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, memory-budget 0.95, and float32 gradient reduction.
  - Source: manager reduce-scatter bandwidth follow-up from the current-best profile.
  - Expected mechanism for improving reported tokens/sec: The profile showed NCCL around 26% of rank-0 GPU kernel time, dominated by reduce-scatter. Setting `mixed_precision_reduce="bfloat16"` for Qwen3 14B halves reduction payload size and may improve reduce-scatter bandwidth without adding HSDP or TP.
  - Supporting evidence: HSDP and TP reduced or reshaped communication but were slower/unstable, while this changes only FSDP reduction dtype in `qwen3_14b()`.
  - Planned source/config changes: In `qwen3_14b()` `TrainingConfig`, set `mixed_precision_reduce="bfloat16"`; keep `rowwise_with_gw_hp` and do not edit the global `TrainingConfig` type.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.95 --compile.enable --compile.components model`
  - Success criteria and expected risk: Kept at `ef51a052`; loss fell from 12.34481 to 9.12246, throughput reached 9,332 tps, and peak memory was 145.48GiB. The run emitted a tyro warning because `TrainingConfig.mixed_precision_reduce` is annotated as `Literal["float32"]`, but the command completed and Qwen3 parallelize consumed the value.

- ~~Idea: Disable structured trace logging on current best~~
  - Current best source commit: `ef51a052`; current best result row: 9,332 tps from `rowwise_with_gw_hp`, bfloat16 FSDP reduction, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.95.
  - Source: manager overhead audit after source and compile-coverage candidates were exhausted.
  - Expected mechanism for improving reported tokens/sec: The training loop records structured spans and scalars around step, data fetch, forward/backward, optimizer, and metric collection. Disabling the structured JSONL trace logger should remove that per-step logging overhead while preserving normal stdout metrics used for ranking.
  - Supporting evidence: `DebugConfig.enable_structured_logging` explicitly makes all `log_trace_span`, `log_trace_instant`, and `log_trace_scalar` calls no-ops when false. The active objective is only 10 steps, so fixed Python/logging overhead can move reported step-10 tps even if GPU kernels are unchanged.
  - Planned source/config changes: None; command-only candidate on the current kept `rowwise_with_gw_hp` source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.95 --compile.enable --compile.components model --debug.no-enable-structured-logging`
  - Success criteria and expected risk: Discarded at `b317775f`; the run completed and reached 9,355 tps, but loss rose from 12.19051 to 14.05692 at 145.48GiB peak memory, so it failed the strict loss trend gate despite higher tps.

- ~~Idea: Repeat-confirm bfloat16 FSDP reduction current best~~
  - Current best source commit: `ef51a052`; current best result row: 9,332 tps from `rowwise_with_gw_hp`, bfloat16 FSDP reduction, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.95.
  - Source: manager repeat-confirmation after a narrow communication-dtype win and a higher-tps structured-logging discard.
  - Expected mechanism for improving reported tokens/sec: Re-running the exact current-best command checks whether the bfloat16 reduce source change is reproducible and whether normal variance can produce a clean row above the structured-logging discard's throughput without disabling structured logging.
  - Supporting evidence: The bfloat16-reduce run is only 103 tps above the prior best, while the no-structured-logging run reached 9,355 tps but failed the loss trend gate. An exact repeat should clarify whether the new source is robust before trying additional knobs.
  - Planned source/config changes: None; repeat on the current kept bfloat16-reduce source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.95 --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded as repeat diagnostic at `f60c1124`; the run completed with finite/falling loss from 12.43899 to 9.72588, MFU `N/A`, and 145.48GiB peak memory, but reached 9,315 tps versus the 9,332 tps best. It remains above the prior 9,229 tps gw-hp best, so it confirms keeping the bf16-reduce source.

- ~~Idea: Fused AdamW with bfloat16 optimizer states~~
  - Current best source commit: `ef51a052`; current best result row: 9,364 tps from `rowwise_with_gw_hp`, bfloat16 FSDP reduction, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: manager optimizer-memory/step-overhead audit.
  - Expected mechanism for improving reported tokens/sec: `optimizer.implementation=fused_opt_states_bf16` pre-initializes AdamW momentum and variance buffers in bfloat16 while retaining the fused optimizer path. This can reduce optimizer-state memory and may reduce optimizer-step bandwidth on the 10-step workload without changing parallelism or Float8 coverage.
  - Supporting evidence: `docs/bf16_optimizer_states.md` says the option is compatible with FSDP2 and operates on local DTensor shards. The current best still spends a training-loop optimizer phase after gradient clipping, and memory is around 145.48GiB, so optimizer-state pressure remains relevant.
  - Planned source/config changes: None; command-only candidate using the current kept bfloat16-reduce source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model --optimizer.implementation fused_opt_states_bf16`
  - Success criteria and expected risk: Kept at `6252b73c`; the run completed with finite/falling loss from 12.57776 to 10.44660, peak memory fell to 137.47GiB, and throughput reached 9,382 tps, beating the 9,364 tps best.

- ~~Idea: bfloat16-reduce memory-budget 0.925 active probe~~
  - Current best source commit: `ef51a052`; current best result row: 9,332 tps from `rowwise_with_gw_hp`, bfloat16 FSDP reduction, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.95.
  - Source: already-active run discovered while preparing the optimizer-state handoff.
  - Expected mechanism for improving reported tokens/sec: The bfloat16 FSDP reduction change slightly shifted the current-best runtime and memory profile. A centered lower memory-budget value may select a similar activation partition with less recompute/overhead variance or slightly better loss behavior than the 0.95 setting.
  - Supporting evidence: The old rowwise and gw-hp memory-budget sweeps were measured before bfloat16 reduction. The current best and repeat are close, so a single adjacent budget probe is reasonable if it is already running.
  - Planned source/config changes: None; command-only candidate on the current kept bfloat16-reduce source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model`
  - Success criteria and expected risk: Kept at `14e2361c`; the active run completed with finite/falling loss from 12.38271 to 9.11822, MFU `N/A`, 145.48GiB peak memory, and 9,364 tps, beating the 9,332 tps bf16-reduce budget 0.95 best.

- ~~Idea: Repeat-confirm bfloat16-reduce memory-budget 0.925~~
  - Current best source commit: `ef51a052`; current best result row: 9,364 tps from `rowwise_with_gw_hp`, bfloat16 FSDP reduction, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: manager repeat-confirmation request after the 0.925 budget narrowly beat the 0.95 setting.
  - Expected mechanism for improving reported tokens/sec: Re-running the exact current-best command checks whether the 0.925 activation-budget win is reproducible or just short-run throughput noise.
  - Supporting evidence: The 0.925 row beat the 0.95 max by only 32 tps, and nearby activation-budget runs have shown variance, so a clean repeat is needed before using 0.925 as a strong baseline for later command-only probes.
  - Planned source/config changes: None; exact repeat on the current kept bfloat16-reduce source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded as repeat diagnostic at `ad4926b0`; the run completed with finite/falling loss from 12.29194 to 6.69980, MFU `N/A`, and 145.48GiB peak memory, but reached only 9,233 tps versus the 9,364 tps best.

- ~~Idea: bfloat16-reduce memory-budget 0.9375 midpoint~~
  - Current best source commit: `ef51a052`; current best result row: 9,364 tps from `rowwise_with_gw_hp`, bfloat16 FSDP reduction, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: manager midpoint bracket after 0.925 was high but noisy and 0.95 repeated more stably.
  - Expected mechanism for improving reported tokens/sec: A midpoint budget might choose a more stable activation partition between the high 0.925 max and the more reproducible 0.95 setting.
  - Supporting evidence: 0.925 reached 9,364 tps but repeated at 9,233 tps; 0.95 reached 9,332 tps and repeated at 9,315 tps. A single 0.9375 run tests whether the best partition sits between them.
  - Planned source/config changes: None; command-only activation-budget probe on the current kept bfloat16-reduce source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.9375 --compile.enable --compile.components model`
  - Success criteria and expected risk: Discarded at `f8ea91e3`; the run completed with finite/falling loss from 12.47883 to 10.69838, MFU `N/A`, 145.48GiB peak memory, and 9,354 tps, which is close but still below the 9,364 tps max.

- ~~Idea: Profile bf16-reduce memory-budget 0.925 current best~~
  - Current best source commit: `ef51a052`; current best result row: 9,364 tps from `rowwise_with_gw_hp`, bfloat16 FSDP reduction, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: manager diagnostic request after bfloat16 FSDP reduction and activation-budget retuning.
  - Expected mechanism for improving reported tokens/sec: Profiling does not compete for best throughput, but it should show whether bfloat16 reduction shrank reduce-scatter enough for GEMM, attention, Float8 scaling, optimizer, or runtime overhead to become the next bottleneck.
  - Supporting evidence: The bfloat16-reduce win was narrow, activation-budget differences are noisy, and the pre-bfloat16 profile still had NCCL/reduce-scatter as a major bucket.
  - Planned source/config changes: None; diagnostic profile on the current kept bfloat16-reduce source.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model --profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1`
  - Success criteria and expected risk: Discard diagnostic at `6252b73c`; the run completed with loss falling from 12.46033 to 11.89387, MFU `N/A`, 145.48GiB peak memory, and 8,565 profiled tps. Rank-0 trace buckets were dense GEMM/scaled-mm 1.115s/34.2%, NCCL reduce-scatter 0.854s/26.2%, flash attention 0.600s/18.4%, Float8 scale/cast/fused elementwise 0.253s/7.8%, and NCCL all-gather 0.212s/6.5%.

- ~~Idea: Repeat-confirm fused bfloat16 optimizer-state current best~~
  - Current best source commit: `6252b73c`; current best result row: 9,382 tps from `rowwise_with_gw_hp`, bfloat16 FSDP reduction, fused bfloat16 optimizer states, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: manager repeat-confirmation after a narrow optimizer-state throughput win and substantial memory drop.
  - Expected mechanism for improving reported tokens/sec: Re-running the exact current-best command tests whether the 9,382 tps row is reproducible or a favorable variance sample. It also confirms that lower-precision optimizer moments do not consistently hurt the 10-step loss trend.
  - Supporting evidence: The optimizer-state run beat the prior max by only 18 tps, but reduced peak memory from 145.48GiB to 137.47GiB. A repeat is needed before spending that memory headroom on higher activation budgets or local batch size.
  - Planned source/config changes: None; exact repeat on the current kept source and command.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model --optimizer.implementation fused_opt_states_bf16`
  - Success criteria and expected risk: Discarded at `32af3b81`; the repeat completed but loss rose from 12.51469 to 14.71142, throughput reached only 9,332 tps, and peak memory remained 137.47GiB. This flags the 9,382 tps fused optimizer-state row as suspect rather than confirmed.

- ~~Idea: Second repeat-confirm fused bfloat16 optimizer-state run~~
  - Current best source commit: `6252b73c`; current best clean row remains 9,364 tps without fused optimizer states, while the fused optimizer-state max is 9,382 tps but failed exact repeat loss trend.
  - Source: already-active run discovered after the first fused optimizer-state repeat discard.
  - Expected mechanism for improving reported tokens/sec: A second exact fused repeat can determine whether the first repeat's rising loss was variance or whether bfloat16 optimizer states are too unstable for this 10-step correctness gate.
  - Supporting evidence: The original fused run had falling loss and lower 137.47GiB memory, but the first repeat rose in loss. Since the same command is already running, finalize it before abandoning or building on fused optimizer states.
  - Planned source/config changes: None; exact repeat on the fused optimizer-state command.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model --optimizer.implementation fused_opt_states_bf16`
  - Success criteria and expected risk: Kept at `ab6f0226`; the second repeat completed with loss falling from 12.47566 to 10.07680, peak memory 137.47GiB, MFU `N/A`, and 9,384 tps. This is a new observed max and gives fused optimizer states two valid falling-loss runs versus one rising-loss repeat.

- ~~Idea: Disable structured logging on fused optimizer-state current best~~
  - Current best result row: 9,384 tps from `rowwise_with_gw_hp`, bfloat16 FSDP reduction, fused bfloat16 optimizer states, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: manager runtime-overhead revisit after fused optimizer states became the current best.
  - Expected mechanism for improving reported tokens/sec: Disabling structured trace logging removes per-step `log_trace_span`, `log_trace_instant`, and `log_trace_scalar` JSONL overhead while preserving normal stdout metrics. This may improve the short 10-step objective now that the current command has lower optimizer-state memory and a slightly different runtime mix.
  - Supporting evidence: The earlier structured-logging-disabled run reached 9,355 tps, above its then-current 9,332 tps baseline, but was discarded because loss rose. Structured logging should not change model math, so the loss failure may have been ordinary run variance; this is a bounded retry on the newer fused best.
  - Planned source/config changes: None; command-only candidate on the current kept source and fused optimizer command.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model --optimizer.implementation fused_opt_states_bf16 --debug.no-enable-structured-logging`
  - Success criteria and expected risk: Discarded at `37bee59`; the run completed but loss rose from 12.56440 to 19.02411, throughput reached only 9,363 tps, and peak memory stayed at 137.47GiB. Close structured-logging disablement for this stack.

- ~~Idea: Fused optimizer states with memory-budget 0.9375~~
  - Current best result row: 9,384 tps from `rowwise_with_gw_hp`, bfloat16 FSDP reduction, fused bfloat16 optimizer states, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: manager activation-budget retune after optimizer-state memory savings.
  - Expected mechanism for improving reported tokens/sec: Fused bfloat16 optimizer states lowered peak memory from 145.48GiB to 137.47GiB. Raising compiler memory budget from 0.925 to 0.9375 may spend some freed memory on saved activations and reduce recompute while staying well below prior risky memory points.
  - Supporting evidence: Before fused optimizer states, memory-budget 0.9375 reached 9,354 tps with falling loss, close to the 0.925 max. The fused optimizer-state path changed memory pressure and now has two falling-loss best rows, so a single adjacent budget probe is a bounded way to see if the optimizer memory savings can be converted into throughput.
  - Planned source/config changes: None; command-only candidate on the current kept source and fused optimizer command.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.9375 --compile.enable --compile.components model --optimizer.implementation fused_opt_states_bf16`
  - Success criteria and expected risk: Discarded at `fac529ed`; the run reached 9,393 tps and 137.47GiB peak memory, but loss rose from 12.45222 to 17.76889, so it fails the loss trend gate. Return to memory-budget 0.925 for the fused optimizer-state best.

- ~~Idea: Profile fused optimizer-state current best~~
  - Current best result row: 9,384 tps from `rowwise_with_gw_hp`, bfloat16 FSDP reduction, fused bfloat16 optimizer states, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: current-best profile gap after optimizer-state change.
  - Expected mechanism for improving reported tokens/sec: Profiling does not compete for best throughput, but it should show whether the fused optimizer-state command changed the remaining bottleneck mix or simply reduced memory. This can guide whether the next real candidate should target communication, attention/GEMM, activation policy, or optimizer/runtime overhead.
  - Supporting evidence: The latest non-fused profile still had reduce-scatter at 26.2% of rank-0 kernel time, while optimizer kernels were only about 1.0%. Fused optimizer states changed peak memory and produced the current best, so a fresh diagnostic profile is the safest next evidence step.
  - Planned source/config changes: None; diagnostic command-only profile on the current kept fused optimizer-state command.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model --optimizer.implementation fused_opt_states_bf16 --profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1`
  - Success criteria and expected risk: Discard diagnostic at `293ea90e`; the profile completed with loss falling from 12.18422 to 10.30645, peak memory 137.47GiB, MFU `N/A`, and 8,705 profiled tps. Rank-0 buckets were dense GEMM/scaled-mm 1.091s/38.1%, NCCL reduce-scatter 0.604s/21.1%, flash attention 0.582s/20.3%, Float8 scale/cast/fused elementwise 0.256s/8.9%, and NCCL all-gather 0.101s/3.5%.
  - Repeat diagnostic: A later single rerun of the same profile command also completed as a discard with loss falling from 12.35789 to 10.80815, peak memory 137.47GiB, MFU `N/A`, and 8,372 profiled tps. The trace at `outputs/profiling/traces/iteration_10` showed rank-0 buckets dense GEMM/scaled-mm 1.100s/37.5%, flash attention 0.604s/20.6%, NCCL reduce-scatter 0.550s/18.8%, Float8 scale/cast/fused elementwise 0.320s/10.9%, and NCCL all-gather 0.195s/6.6%.

- ~~Idea: HSDP 2x4 revisit with fused optimizer-state best~~
  - Current best result row: 9,384 tps from fused bfloat16 optimizer states, bfloat16 FSDP reduction, `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: fused optimizer-state profile and stale HSDP discard review.
  - Expected mechanism for improving reported tokens/sec: The fused profile still has NCCL reduce-scatter as a 21.1% rank-0 bucket, while fused optimizer states consistently lower peak memory to 137.47GiB. Re-testing HSDP shard groups of four GPUs may reduce reduce-scatter scope with more memory headroom than the earlier HSDP attempt.
  - Supporting evidence: The previous HSDP candidate was measured before fused bfloat16 optimizer states lowered memory by about 8GiB and before the fused command became current best. All-gather is now smaller, but reduce-scatter remains the largest single communication bucket.
  - Planned source/config changes: If Qwen3 HSDP support is not active, re-apply only the minimal `parallelize.py` mesh-selection change needed for `dp_replicate` + `fsdp`; no sharding or Float8 source changes.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.data_parallel_replicate_degree=2 --parallelism.data_parallel_shard_degree=4 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model --optimizer.implementation fused_opt_states_bf16`
  - Success criteria and expected risk: Discarded at `75fa19eb`; the run completed with finite/falling loss from 12.35089 to 9.74531, but throughput reached only 9,198 tps and peak memory rose to 166.04GiB. The minimal HSDP source change was reverted after the run.

- ~~Idea: Disable NCCL flight recorder buffer on fused optimizer-state best~~
  - Current best result row: 9,384 tps from fused bfloat16 optimizer states, bfloat16 FSDP reduction, `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: manager runtime-overhead review after fused profile.
  - Expected mechanism for improving reported tokens/sec: TorchTitan initializes `TORCH_FR_BUFFER_SIZE` from `comm.trace_buf_size`, defaulting to 20000. Setting `--comm.trace_buf_size=0` disables the NCCL flight recorder ring buffer and avoids per-collective trace bookkeeping while preserving the same model, optimizer, sharding, and activation policy.
  - Supporting evidence: The fused profile still spends roughly 18.8-21.1% of rank-0 kernel time in reduce-scatter and 3.5-6.6% in all-gather. Structured logging disablement was not useful, and HSDP was slower and memory-heavy, but the NCCL flight recorder is a separate communication-side trace path and is a command-only knob.
  - Planned source/config changes: None; command-only communication diagnostics knob on the current best.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model --optimizer.implementation fused_opt_states_bf16 --comm.trace_buf_size=0`
  - Success criteria and expected risk: Discarded at `2c30ff05`; disabling the NCCL flight recorder buffer reached only 8,907 tps and loss rose from 12.47707 to 16.04552, so it failed both the throughput and falling-loss gates. Peak memory stayed at 137.47GiB.

- ~~Idea: Disable activation-checkpoint RNG state preservation on fused optimizer-state best~~
  - Current best result row: 9,384 tps from fused bfloat16 optimizer states, bfloat16 FSDP reduction, `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: manager activation-checkpoint overhead review after communication-side probes failed.
  - Expected mechanism for improving reported tokens/sec: `preserve_rng_state=True` makes checkpointing stash and restore RNG state around recompute. The Qwen3 model code has no dropout references in the current path, so disabling RNG preservation should avoid that overhead without changing model placements, Float8 coverage, optimizer, or communication.
  - Supporting evidence: `docs/debugging.md` explicitly notes that preserving RNG state may be slower. The fused profile shows dense GEMM/attention/recompute-adjacent work now dominates over optimizer kernels, and higher activation budgets or HSDP did not produce valid wins, so a low-risk checkpoint-overhead knob is worth one run.
  - Planned source/config changes: None; command-only activation-checkpoint knob on the current best.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --activation_checkpoint.no-preserve-rng-state --compile.enable --compile.components model --optimizer.implementation fused_opt_states_bf16`
  - Success criteria and expected risk: Discarded at `5d8ba810`; the preferred CLI spelling was accepted and loss fell from 12.37173 to 8.02247, but throughput reached only 8,686 tps versus the 9,384 tps current best. Peak memory stayed at 137.47GiB.

- ~~Idea: Full bfloat16 training dtype with fused optimizer~~
  - Current best result row: 9,384 tps from float32 training storage, bfloat16 FSDP parameter/reduce dtypes, fused bfloat16 optimizer states, `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: manager precision/storage review after optimizer-state bf16 proved useful but follow-up overhead knobs failed.
  - Expected mechanism for improving reported tokens/sec: `training.dtype=bfloat16` builds parameters, gradients, and optimizer state in bfloat16 instead of retaining fp32 training storage. The docs for bf16 optimizer states say default `implementation="fused"` is the normal choice in this mode. This may reduce optimizer, clipping, and parameter-memory bandwidth enough to beat the current mixed-storage command.
  - Supporting evidence: The current best already computes FSDP parameters and reductions in bfloat16 and stores Adam moments in bfloat16, so full bf16 training is the next coherent precision/storage step. Fused optimizer states lowered peak memory by about 8GiB and produced the best valid rows; full bf16 may reduce remaining fp32 parameter/gradient pressure further.
  - Planned source/config changes: None; command-only precision/storage candidate.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --training.dtype=bfloat16 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model --optimizer.implementation fused`
  - Success criteria and expected risk: Discarded at `6d5971d5`; the run completed with finite/falling loss from 12.48515 to 6.99830, peak memory 129.79GiB, and MFU `N/A`, but throughput reached only 9,027 tps versus the 9,384 tps current best.

- ~~Idea: Full bfloat16 training dtype with memory-budget 0.95~~
  - Current best result row: 9,384 tps from float32 training storage, bfloat16 FSDP parameter/reduce dtypes, fused bfloat16 optimizer states, `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: manager activation-budget retune after full bf16 saved memory but was slow at 0.925.
  - Expected mechanism for improving reported tokens/sec: Full bf16 training reduced peak memory to 129.79GiB and kept loss falling, leaving more activation headroom than the current-best 137.47GiB mixed-storage command. Raising memory-budget to 0.95 may save more activations, reduce recompute, and recover enough throughput to make the full-bf16 precision/storage path competitive.
  - Supporting evidence: The full-bf16 0.925 row was numerically clean but 357 tps below the current best. Earlier non-full-bf16 budget sweeps showed 0.95 and 0.9375 can be near-best, so one adjacent higher-budget retune is justified before closing full bf16.
  - Planned source/config changes: None; command-only activation-budget retune on the full-bf16 candidate.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --training.dtype=bfloat16 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.95 --compile.enable --compile.components model --optimizer.implementation fused`
  - Success criteria and expected risk: Discarded at `bfe5289d`; the run reached only 8,952 tps, peak memory 129.79GiB, and MFU `N/A`, and loss rose from 12.28825 to 14.20371. Close full bf16 as a memory-saving but throughput-negative direction for this branch.

- ~~Idea: Context parallel degree 2 on fused optimizer-state best~~
  - Current best result row: 9,384 tps from float32 training storage, bfloat16 FSDP parameter/reduce dtypes, fused bfloat16 optimizer states, `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: manager attention/compute review after command-only overhead and precision/storage probes failed.
  - Expected mechanism for improving reported tokens/sec: CP=2 shards the sequence dimension for attention, targeting the fused profile's flash-attention bucket while preserving TP=1 and the current Float8/FSDP optimizer path. TorchTitan already has `apply_cp_to_forward()` for inner attention modules, and Qwen3's default SDPA attention path is CP-compatible.
  - Supporting evidence: Fused profiles show flash attention around 20% of rank-0 kernel time and dense compute dominating after communication-side probes failed. CP is more targeted than TP because previous TP=2 attempts were slow or compiler-fragile, while CP can be enabled by wrapping only `block.attention.inner_attention` and letting FSDP cover the `fsdp=dp_shard*cp` mesh.
  - Planned source/config changes: In `torchtitan/models/qwen3/parallelize.py`, remove the CP `NotImplementedError`, import `apply_cp_to_forward`, and when `parallel_dims.cp_enabled`, wrap `[block.attention.inner_attention for block in model.layers.values()]` with `apply_cp_to_forward(..., parallel_dims.get_mesh("cp"))`. Revert this source change after recording the result unless the run is kept.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.context_parallel_degree=2 --parallelism.data_parallel_shard_degree=4 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model --optimizer.implementation fused_opt_states_bf16`
  - Success criteria and expected risk: Crashed under source state `448a5dba` before step 1. CP wrapping was applied and the mesh was created, but model-only compile plus memory-budget AC failed in Inductor min-cut partitioning with `Unknown metadata type FakeScriptObject`; no loss, throughput, MFU, or peak training memory was emitted. The CP source change was reverted.

- ~~Idea: Varlen attention backend on fused optimizer-state best~~
  - Current best result row: 9,384 tps from SDPA attention, float32 training storage, bfloat16 FSDP parameter/reduce dtypes, fused bfloat16 optimizer states, `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: manager attention-backend review after CP crashed and profile showed flash attention around 20% of rank-0 kernel time.
  - Expected mechanism for improving reported tokens/sec: Qwen3 supports `attn_backend="varlen"`, and the C4 dataloader provides per-document positions needed for block-causal varlen masks. Varlen attention can use the dedicated varlen flash-attention path and may reduce wasted attention work on packed documents relative to plain causal SDPA.
  - Supporting evidence: `get_attention_config("varlen")` maps to `VarlenAttention.Config()` with `block_causal` masks, and the trainer already builds attention masks for `VarlenAttention`. Flex/flex_flash were unattractive, but varlen is a distinct kernel path and is an allowed `model_registry(...)` keyword in `qwen3_14b()`.
  - Planned source/config changes: In `torchtitan/models/qwen3/config_registry.py`, change only the `qwen3_14b()` `model_registry("14B", ...)` call to pass `attn_backend="varlen"` while preserving converters and filters. Revert this config change after recording the result unless the run is kept.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model --optimizer.implementation fused_opt_states_bf16`
  - Success criteria and expected risk: Crashed under source state `f51f144b` before step 1. `attn_backend="varlen"` preserved the existing Float8 converter/filter settings but tried to activate FA3 and import `flash_attn_interface`, which is not installed in this environment. No throughput, MFU, peak memory, or loss was emitted, so this is a discard and the varlen config change was reverted.

- ~~Idea: Final exact repeat of fused optimizer-state current best~~
  - Current best result row: 9,384 tps from SDPA attention, float32 training storage, bfloat16 FSDP parameter/reduce dtypes, fused bfloat16 optimizer states, `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: manager variance check after the main novel directions in scope failed or were closed.
  - Expected mechanism for improving reported tokens/sec: No mechanism change; this checks whether the current best command can produce a clean falling-loss row above 9,384 tps through ordinary run-to-run variance now that all discarded source probes have been reverted.
  - Supporting evidence: The fused optimizer-state command has two valid falling-loss best rows at 9,382 and 9,384 tps, plus one rising-loss repeat at 9,332 tps. Recent source-backed probes were reverted, so the stable source path is back to the current-best stack.
  - Planned source/config changes: None; exact command repeat.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model --optimizer.implementation fused_opt_states_bf16`
  - Success criteria and expected risk: Discard diagnostic at `f2343049`; the exact repeat completed with finite/falling loss from 12.38264 to 8.69525, peak memory 137.47GiB, MFU `N/A`, and only 4,988 tps, below the strict `>9,384` keep threshold. Current best remains the prior 9,384 tps fused optimizer-state row.

- ~~Idea: Selective activation checkpointing on fused optimizer-state best~~
  - Current best result row: 9,384 tps from SDPA attention, float32 training storage, bfloat16 FSDP parameter/reduce dtypes, fused bfloat16 optimizer states, `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: detailed trace review of `outputs/profiling/traces/iteration_10` after the final variance check.
  - Expected mechanism for improving reported tokens/sec: Replace the current min-cut memory-budget AC policy with per-op selective AC. Selective AC saves expensive compute and communication ops and recomputes only a controlled subset of matmuls, which may reduce recompute in the compiled transformer-block graphs while still fitting after fused optimizer states lowered peak memory to 137.47GiB.
  - Supporting evidence: Kernel-only trace means are dominated by scaled GEMM (~1.328s/rank) and flash attention (~0.604s/rank), while actual Float8 scale/cast kernels are small (~0.085s/rank). The profiler's remaining target is therefore recompute/compiled compute scheduling rather than filtering more Float8 modules. HSDP did not convert the reduce-scatter skew into a win, and the profile shows optimizer kernels are already negligible.
  - Planned source/config changes: None; command-only activation-checkpoint mode change on the current best.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=selective --compile.enable --compile.components model --optimizer.implementation fused_opt_states_bf16`
  - Success criteria and expected risk: Discarded at `56563fda`; the run completed with peak memory 101.71GiB but reached only 8,635 tps and loss rose from 12.46031 to 15.65681. Keep memory-budget AC 0.925 as the current best path.

- ~~Idea: Force NCCL Simple protocol for fused optimizer-state best~~
  - Current best result row: 9,384 tps from SDPA attention, float32 training storage, bfloat16 FSDP parameter/reduce dtypes, fused bfloat16 optimizer states, `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: detailed trace review of FSDP collectives after selective AC failed.
  - Expected mechanism for improving reported tokens/sec: The trace shows large bf16 reduce-scatter and all-gather operations running as `ncclDevKernel_*_RING_LL`. For the hundreds-of-MB FSDP payloads visible in the CPU op shapes, NCCL's Simple protocol may provide better bandwidth than LL. Forcing `NCCL_PROTO=Simple` is a command-only communication probe that preserves the model, optimizer, activation budget, Float8 coverage, and sharding layout.
  - Supporting evidence: Kernel-only rank means still include 0.668s reduce-scatter and 0.170s all-gather, with reduce-scatter skew from 0.220s to 1.086s across ranks. HSDP topology was slower and memory-heavy, and disabling the NCCL flight recorder was not useful, but protocol selection is a separate NCCL path and directly targets the observed kernel names.
  - Planned source/config changes: None; environment-only communication protocol probe.
  - Planned command or config overrides: `NCCL_PROTO=Simple NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model --optimizer.implementation fused_opt_states_bf16`
  - Success criteria and expected risk: Discarded at `1871c2fd`; the run completed at 9,382 tps with peak memory 137.47GiB, but it did not beat 9,384 tps and loss rose from 12.50709 to 15.01339. Do not force NCCL Simple for this stack.

- ~~Idea: Avoid NCCL record-stream handling on fused optimizer-state best~~
  - Current best result row: 9,384 tps from SDPA attention, float32 training storage, bfloat16 FSDP parameter/reduce dtypes, fused bfloat16 optimizer states, `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: detailed trace review of CUDA runtime calls around FSDP collectives after the Simple protocol override failed.
  - Expected mechanism for improving reported tokens/sec: Set `TORCH_NCCL_AVOID_RECORD_STREAMS=1` so ProcessGroupNCCL avoids calling `record_stream` on collective tensors and instead keeps references until `wait()`. This may reduce CUDA event/stream bookkeeping and allocator interaction around async FSDP all-gather and reduce-scatter while preserving the same NCCL kernels, model, optimizer, activation budget, and sharding layout.
  - Supporting evidence: The profile shows FSDP collectives on separate NCCL streams plus many CUDA event/stream calls. `docs/composability.md` recommends this env var for async NCCL collectives in TP; the mechanism is process-group level and applies to the same style of async NCCL stream/wait behavior used by FSDP collectives.
  - Planned source/config changes: None; environment-only runtime probe.
  - Planned command or config overrides: `TORCH_NCCL_AVOID_RECORD_STREAMS=1 NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model --optimizer.implementation fused_opt_states_bf16`
  - Success criteria and expected risk: Discarded at `311ca162`; the run completed with falling loss from 12.35315 to 6.97227, but throughput reached only 8,833 tps and the log reported this env var is deprecated/default. Do not pursue record-stream env tuning further.

- ~~Idea: Split root FSDP units for embeddings, norm, and lm_head~~
  - Current best result row: 9,384 tps from SDPA attention, float32 training storage, bfloat16 FSDP parameter/reduce dtypes, fused bfloat16 optimizer states, `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: detailed trace shape analysis after NCCL env/runtime probes failed.
  - Expected mechanism for improving reported tokens/sec: Fully shard `tok_embeddings`, `norm`, and `lm_head` as their own FSDP units before the existing per-layer and root FSDP wrapping. The current root wrapper appears to all-gather `tok_embeddings.weight + lm_head.weight` together during decoder forward, even though `ChunkedCELoss` skips `lm_head` in the decoder and applies it later. Splitting these modules should avoid a single 1.56B-element root all-gather and may defer the `lm_head` all-gather to the loss chunk loop where the loss code already manages `lm_head` resharding.
  - Supporting evidence: The rank-0 trace has an all-gather output shape of 1,555,829,760 bf16 elements and local shard shape of 194,478,720 elements. Qwen3 14B has `tok_embeddings.weight` and `lm_head.weight` shapes of `151936 * 5120 = 777,912,320` elements each, so the root FSDP payload is exactly their sum. Smaller per-layer FSDP all-gathers are around 330,311,936 elements, and communication remains a major rank-skew source.
  - Planned source/config changes: In `torchtitan/models/qwen3/parallelize.py`, before the per-layer loop, call `fully_shard` on non-`None` `model.tok_embeddings`, `model.norm`, and `model.lm_head` with the same `fsdp_config`; then keep the existing per-layer `fully_shard(layer, **fsdp_config)` and root `fully_shard(model, **fsdp_config)`. Revert this source change unless the result is kept.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model --optimizer.implementation fused_opt_states_bf16`
  - Success criteria and expected risk: Discarded at `e91f1b75`; the run completed with finite/falling loss from 12.46221 to 7.00204 and reduced peak memory to 134.31GiB, but throughput reached only 9,029 tps. The source change was reverted.

- ~~Idea: Split only lm_head into its own FSDP unit~~
  - Current best result row: 9,384 tps from SDPA attention, float32 training storage, bfloat16 FSDP parameter/reduce dtypes, fused bfloat16 optimizer states, `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: follow-up to the full root split discard and the same root all-gather trace shape analysis.
  - Expected mechanism for improving reported tokens/sec: Fully shard only `model.lm_head` before the existing per-layer and root FSDP wrapping. This should remove the unused `lm_head.weight` half from the decoder-forward root all-gather while avoiding the extra `tok_embeddings` and `norm` FSDP units that made the broader split slow. `ChunkedCELoss` already owns the `lm_head` loss path and explicitly disables/resets its resharding across chunks, so a standalone `lm_head` FSDP unit matches that intended use better than bundling it into the model root.
  - Supporting evidence: The full split reduced memory to 134.31GiB, confirming the root payload diagnosis, but was slower due to broader wrapper/collective overhead. The trace's 1.56B-element root all-gather is exactly `tok_embeddings + lm_head`; `tok_embeddings` is needed for decoder forward, whereas `lm_head` is skipped by `_skip_lm_head=True` and used later in the loss.
  - Planned source/config changes: In `torchtitan/models/qwen3/parallelize.py`, before the per-layer loop, call `fully_shard(model.lm_head, **fsdp_config)` if `model.lm_head is not None`; then keep the existing per-layer `fully_shard(layer, **fsdp_config)` and root `fully_shard(model, **fsdp_config)`. Revert this source change unless the result is kept.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model --optimizer.implementation fused_opt_states_bf16`
  - Success criteria and expected risk: Discarded at `252de076`; the run reduced peak memory to 134.48GiB but reached only 8,907 tps and loss rose from 12.49483 to 16.76770. The source change was reverted.

- Idea: Enable Inductor GEMM max autotune for fused optimizer-state best
  - Current best result row: 9,384 tps from SDPA attention, float32 training storage, bfloat16 FSDP parameter/reduce dtypes, fused bfloat16 optimizer states, `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925.
  - Source: compute-side trace review after communication, AC, precision/storage, and root-FSDP placement probes failed.
  - Expected mechanism for improving reported tokens/sec: Set `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1` so Inductor autotunes GEMM choices for the compiled transformer blocks. The trace's largest remaining kernel bucket is scaled GEMM, so a better GEMM kernel choice could improve throughput without changing parallelism, activation budget, optimizer, or model numerics.
  - Supporting evidence: Detailed kernel-only trace means show scaled GEMM around 1.328s/rank, larger than flash attention (~0.604s/rank) and all-gather (~0.170s/rank), and comparable to or larger than reduce-scatter on most ranks. The current PyTorch config reports `max_autotune_gemm=False` and `coordinate_descent_tuning=False` by default. FlexAttention already has its own max-autotune path, but this run targets dense/FP8 GEMM in the model compile path.
  - Planned source/config changes: None; environment-only compile/autotune probe.
  - Planned command or config overrides: `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1 NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.925 --compile.enable --compile.components model --optimizer.implementation fused_opt_states_bf16`
  - Success criteria and expected risk: Keep only if the run completes 10 steps, loss is finite/falling, and throughput beats 9,384 tps. Main risk is longer compilation, no effect on nvjet/scaled-mm kernels, or autotune noise that does not survive the strict loss and throughput gates.
