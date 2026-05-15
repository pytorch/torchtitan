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

- Idea: TP=2 with shared decoder sharding helpers
  - Current best source commit: TBD after bootstrap
  - Source: manager research
  - Expected mechanism for improving reported tokens/sec: Qwen3 14B has 40 attention heads and 8 KV heads, so TP=2 divides both cleanly and should reduce per-rank dense matmul/state memory while keeping TP collectives within the B200 NVLink domain. Enabling sequence parallel should keep post-attention and FFN activations sharded over sequence positions and may convert freed memory into a larger follow-up batch or lower recompute pressure.
  - Supporting evidence: `Qwen3Model.Config.update_from_config()` validates that TP divides `n_heads` and `n_kv_heads`; the 14B flavor satisfies this for TP=2. `torchtitan.models.common.decoder_sharding` already provides `set_decoder_sharding_config`, `set_gqa_attention_sharding`, `set_gqa_inner_attention_local_map`, `set_dense_ffn_sharding`, and norm placement helpers for GQA+dense-FFN decoder blocks, which match the Qwen3 14B non-MoE structure.
  - Planned source/config changes: In `set_qwen3_sharding_config()`, call shared decoder/root helpers and walk Qwen3 layers to set attention, inner-attention local-map, q/k norm, FFN, and block norm sharding. In `parallelize_qwen3()`, call `model.parallelize(tp_mesh)` when TP is enabled before FSDP wrapping. Keep the diff limited to the active dense Qwen3 path.
  - Planned command or config overrides: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.tensor_parallel_degree=2 --parallelism.data_parallel_shard_degree=4 --parallelism.enable_sequence_parallel`
  - Success criteria and expected risk: The run completes with finite losses and improves steady-state `tps` over the DP-sharded baseline. Risks are QK norm placement mistakes, an inner-attention local-map mismatch, or TP collectives outweighing smaller matmuls at this batch/sequence shape.
