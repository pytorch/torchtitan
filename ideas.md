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

- Idea: Selective activation checkpointing with FSDP no-reshard
  - Current best source commit: 9db79f7 result row / current source after TP revert
  - Source: memory-headroom analysis
  - Expected mechanism for improving reported tokens/sec: Full activation checkpointing saves memory but recomputes every block. Selective checkpointing should keep enough memory headroom on B200 while reducing recomputation and improving matmul/attention useful work per step.
  - Supporting evidence: Full AC plus no-reshard used 72.2GiB, far below the 178.35GiB capacity, and the profile/MFU indicated compute headroom remained.
  - Planned source/config changes: None; command-only candidate.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=selective`
  - Success criteria and expected risk: Kept at `0b0796f` with 6,808 tps, 28.45% MFU, 113.7GiB peak memory, and falling finite loss.

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

- Idea: Compile transformer blocks with selective AC and no-reshard
  - Current best source commit: 0b0796f
  - Source: selective-profile analysis
  - Expected mechanism for improving reported tokens/sec: The selective-AC profile reduced FSDP collective time enough that flash attention backward, dense matmuls, layer norm, and elementwise kernels dominate the captured GPU work. Per-block `torch.compile` may fuse or schedule block-level work better and reduce CPU/runtime overhead on the repeated 48-layer structure.
  - Supporting evidence: The selective profile rank 0 trace shows kernel time around 3.19s, NCCL around 0.44s, with top compute kernels including flash backward, several nvjet matmuls, layer norm, and elementwise kernels. CPU op time and CUDA runtime overhead are still visible in the trace.
  - Planned source/config changes: None if existing `parallelize_qwen3()` compile path handles the command; otherwise only the minimum Qwen3-local compile ordering fix if the command fails before training.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --activation_checkpoint.mode=selective --compile.enable`
  - Success criteria and expected risk: Keep if step 10 tps exceeds 6,808 and loss stays finite/falling. Risks are compile overhead contaminating a 10-step run, graph breaks with checkpoint wrappers, compile-time failure, or higher memory use.
