# TorchTitan Qwen3 Parallelism Autoresearch

## Setup Summary

- Run tag: `may15-qwen3-14b-8xb200`
- Branch: `autoresearch-parallelize/may15-qwen3-14b-8xb200`
- Source commit: `68f86f18f11531fc40cd58c5d46edb5538e80b70`
- Target command: `NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10`
- Objective: maximize TorchTitan-reported steady-state `tps` for the exact Qwen3 14B workload on this 8xB200 node while keeping the 10-step run numerically sane.
- Current best commit: none measured yet.

## Editable Scope

- Source/config files: `torchtitan/models/qwen3/parallelize.py`, `torchtitan/models/qwen3/sharding.py`, and allowed fields inside `qwen3_14b()` in `torchtitan/models/qwen3/config_registry.py`.
- Experiment logs: `ideas.md`, `learnings.md`, and `results.tsv`.
- Local run artifacts such as `run.log` and profiler traces are allowed but should not be committed.
- Fixed config fields inside `qwen3_14b()`: `loss`, `hf_assets_path`, `dataloader`, `checkpoint`; model flavor must remain Qwen3 14B.

## Environment

- GPUs: 8x NVIDIA B200.
- Visible memory: `nvidia-smi` reports 183359 MiB per GPU; PyTorch reports 182631 MiB per GPU.
- Driver: 580.82.07.
- Python: 3.12.13.
- PyTorch: `2.13.0a0+git85cb48a`.
- CUDA: available with 8 devices, compute capability 10.0.
- Topology: every GPU pair is connected by `NV18`; GPUs 0-3 are NUMA 0 and GPUs 4-7 are NUMA 1. NICs mlx5_0 through mlx5_15 are visible.
- Roofline inputs: nominal B200 class hardware provides high FP8/BF16 tensor-core throughput and HBM3e bandwidth; use TorchTitan's `Peak FLOPS used for computing MFU` line from measured runs as the authoritative MFU denominator for this software stack.

## Data And Checkpoint Availability

- Tokenizer path exists: `tests/assets/tokenizer/tokenizer.json` and `tests/assets/tokenizer/tokenizer_config.json`.
- `c4_test` is configured in `torchtitan/hf_datasets/text_datasets.py` as `tests/assets/c4_test`, and `tests/assets/c4_test/data.json` exists.
- The target config uses local test tokenizer/data and no required external checkpoint load was found during setup.

## Starting Config Notes

- `qwen3_14b()` uses Qwen3 14B, `local_batch_size=4`, `seq_len=4096`, `steps=3000`, `data_parallel_shard_degree=-1`, TP/CP/PP all 1, and full activation checkpointing.
- With `NGPU=8`, `dp_shard=-1` resolves to an 8-rank FSDP mesh.
- Current `parallelize_qwen3()` rejects TP/CP/PP/EP and applies replicated DDP only when DP is enabled; `set_qwen3_sharding_config()` is empty.

## Initial Interpretation

The first worker experiment should be a narrow bootstrap rather than a performance tuning change. The inferred command asks for an 8-way DP-sharded Qwen3 14B run, but the Qwen3 scaffold does not yet apply FSDP on the configured shard mesh. The minimal measurable hypothesis is that applying FSDP to the existing DP-shard layout makes the target command runnable and provides the first valid throughput row.

## Manager Research Notes

Shared config-based decoder sharding appears directly applicable to the dense Qwen3 14B path. The 14B registry builds a dense GQA decoder with 40 query heads, 8 KV heads, head dim 128, 40 layers, RMSNorm QK normalization, and no MoE. TP=2 divides both query and KV heads and preserves an 8-GPU layout as dp_shard=4 x tp=2. TP=4 also divides both head counts, but it is a larger communication change and should wait until TP=2 establishes that the Qwen3 sharding contract is correct.

The relevant shared helpers are in `torchtitan.models.common.decoder_sharding`: root decoder sharding, GQA projection sharding, inner-attention local-map for heads-sharded q/k/v tensors, dense FFN colwise/rowwise sharding, and RMSNorm sequence-parallel placement. The likely Qwen3-specific work is attaching these helpers to every `Qwen3TransformerBlock.Config`, including `attention.qk_norm`, `attention_norm`, and `ffn_norm`, then calling `model.parallelize(tp_mesh)` before FSDP in `parallelize_qwen3()`.

Roofline status is still unclear before the first run. If the bootstrap is memory-heavy or slow with low MFU, TP/SP is a reasonable next candidate because it reduces dense matmul state and activation footprints on B200 NVLink. If the bootstrap has high MFU and low communication overhead, profile before changing TP so the next idea is not a blind mesh sweep.

## Experiment Review: FSDP Bootstrap Through No-Reshard

The bootstrap FSDP implementation made the target command runnable. The first valid unprofiled row completed 10 steps with finite loss dropping from 12.45655 to 7.96787, 5,774 tps, 24.12% MFU, and 50.1GiB peak memory. This established the first correct source baseline.

TP=2 with sequence parallel was correct but slower: 4,971 tps and 39.5GiB peak memory. The likely cause is that this batch/sequence shape does not have enough per-rank work to pay for the added TP collectives and placement redistributions. The memory savings are not useful by themselves because the FSDP-only path already has ample headroom.

The profiled FSDP diagnostic should not be ranked against unprofiled rows, but it identifies communication as an important bottleneck. In the rank 0 trace for the captured step, kernel time was dominated by NCCL reduce-scatter at about 862 ms, with NCCL all-gather around 226 ms. Flash attention and dense matmul kernels were also substantial, but the largest single bucket was FSDP gradient communication.

The command-only no-reshard candidate improved the best unprofiled result from 5,774 to 5,872 tps and 24.53% MFU, at the cost of raising peak memory to 72.2GiB. Loss remained finite and dropped from 12.40488 to 6.59394. This supports the interpretation that avoiding repeated parameter reshards helps enough to justify extra memory on 8xB200.

Roofline conclusion: the run is not compute-roofline-bound; reported MFU is around 24.5% and the profile shows material FSDP collective time. The immediate opportunity is to convert memory headroom into more tokens per collective and larger GEMMs before adding more parallel axes. The next narrow experiment should increase local batch size with `fsdp_reshard_after_forward=never`.

## Experiment Review: Activation Checkpointing And Batch Size

Increasing local batch size from 4 to 8 while keeping full activation checkpointing and no-reshard did not produce a useful result. It reached 5,873 tps, effectively tied with the previous best, and memory rose to 84.5GiB, but the loss increased from 12.42239 at step 1 to 16.38219 at step 10. Per the program's convergence sanity check, this should stay discarded even though the run completed.

Switching from full to selective activation checkpointing with no-reshard is the strongest result so far: 6,808 tps, 28.45% MFU, 113.7GiB peak memory, and finite loss dropping from 12.30167 to 8.88175. This is a 15.9% improvement over the previous no-reshard/full-AC best and a 17.9% improvement over the initial FSDP bootstrap. The mechanism is likely reduced recomputation while still staying comfortably below B200 memory capacity.

Disabling activation checkpointing entirely with no-reshard crashed with OOM during RoPE in `apply_rotary_emb_cos_sin`, with about 178.15GiB in use and only about 188MiB free. This brackets the memory tradeoff: selective checkpointing is viable; no checkpointing is not viable at local batch size 4 when FSDP parameters remain unresharded after forward.

Disabling checkpoint RNG preservation was not useful. It produced 6,775 tps, slightly worse than the selective-AC best, and the loss was effectively flat/increasing from 12.36230 to 12.36555 with a high final grad norm. Keep RNG preservation for now.

Current roofline conclusion: selective AC no-reshard shifts the run toward better compute utilization but still only reaches 28.45% MFU. Memory is now 113.7GiB, leaving some but not unlimited headroom. The next batch-size probe should be smaller than 8, because batch 8 failed loss sanity and no-AC OOMed; local batch size 6 with selective AC is the next narrow memory-headroom experiment.

## Experiment Review: Batch 6 And Selective Profile

Local batch size 6 with selective AC and no-reshard was correct but not better. Loss dropped from 12.27970 to 7.50107 and peak memory rose to 141.9GiB, but tps was 6,805, slightly below the 6,808 best. This suggests the current local batch size 4 is already near the best per-token efficiency for this no-reshard/selective-AC layout; more batch primarily consumes memory without raising reported tps.

The selective-AC profile is now more compute-heavy than the original full-AC profile. Rank 0 trace totals for the captured step show about 3.19s kernel time, with NCCL around 0.44s. The largest kernels are flash attention backward, several nvjet dense matmuls, NCCL reduce-scatter, NCCL all-gather, layer norm, and elementwise kernels. Compared with the full-AC profile, communication is still relevant but no longer the dominant single story.

Roofline conclusion: after selective AC, the run is mixed compute/overhead/communication rather than primarily FSDP-collective limited. Since batch-size scaling failed and no-AC OOMed, the next narrow idea is to try the existing per-transformer-block `torch.compile` path with the current best command. The expected upside is better block-level fusion/scheduling and lower runtime overhead; the risk is compile overhead or graph issues inside the 10-step measurement.

## Experiment Review: Per-Block Compile

Per-transformer-block compile with selective activation checkpointing and no-reshard is the new best. The model-only compile command completed with loss dropping from 12.42103 to 7.74147, 7,898 tps, 33.00% MFU, and 108.2GiB peak memory. This is 16.0% faster than selective AC without compile and 36.8% faster than the initial FSDP bootstrap. The first step was slow from compile overhead, but the step-10 steady-state metric is clearly better.

Compiled local batch size 6 crashed before step 1 with an inductor allocation OOM for an 816MiB bf16 buffer while the GPU was nearly full. Even though model-only compile at batch size 4 reports 108.2GiB peak, compiler temporaries and larger activations make batch 6 unsafe. Do not keep exploring larger batch sizes around the compiled current best unless a later idea substantially reduces memory.

Compiling the loss in addition to transformer blocks was worse. The default `--compile.enable` command compiled both loss and model, completed with finite loss dropping from 12.47696 to 9.88424, but only reached 7,139 tps and 29.83% MFU. Keep `--compile.components model`; loss compile adds overhead or worse codegen for this 10-step workload.

Current roofline conclusion: the best has shifted again. Model-only compile raised MFU to 33.00%, but this is still far from the B200 peak denominator. The previous non-compiled profile is stale, so the next step should be a profiled diagnostic of the model-only compiled best before trying broader changes such as reshard/no-checkpoint tradeoffs or compile-related source edits.

## Experiment Review: Compiled Profile

The model-only compiled profile is slower due to profiler overhead and should not be ranked, but it shows a different bottleneck shape than the pre-compile profile. In the latest rank 0 trace, total kernel time is about 5.68s, with NCCL about 0.51s. The top kernels are mostly dense matmul kernels (`nvjet_sm100...`) and flash attention backward; reduce-scatter is significant but no longer dominant. CPU/runtime categories are also large under profiling, but the unprofiled compile result already showed compile can improve steady-state.

Default compile components were worse than model-only compile: compiling both model and loss reached only 7,139 tps, below the 7,898 best. Keep `--compile.components model`.

The next performance lever should target dense GEMM work. MXFP8 is available in this environment through `torchao` 0.17.0 and `torchtitan.components.quantization.MXFP8LinearConverter`, and the node is B200/SM100. The converter uses the `torchao.prototype.moe_training.config` MXFP8 API, which imports successfully. A narrow qwen3_14b `model_spec` converter change is in scope and directly attacks the largest current kernel bucket.

## Experiment Review: MXFP8 Converter

The broad MXFP8 linear converter path is not currently usable for this Qwen3 14B run. The candidate converted the model and started training, but failed in backward with `RuntimeError: invalid argument` from `torchao.prototype.mx_formats.kernels.mxfp8_quantize_cuda`, reached through the MXFP8 linear backward implementation. Because the failure is inside torchao's MXFP8 quantize backward path rather than a TorchTitan sharding issue, the source was reverted and the result was recorded as a crash.

This rules out the simplest all-linear MXFP8 lever for now. A narrower FQN-filtered MXFP8 attempt might still be possible later, but it would be a higher-risk debugging path and the next lower-risk idea is to target attention kernels. The compiled profile shows flash attention backward/forward still matter, and `qwen3_14b()` can legally switch the 14B `model_spec` to `attn_backend="flex_flash"` while keeping the same model flavor and parallel layout.

## Experiment Review: FlexAttention Backends

The `flex_flash` attention backend is blocked in this environment. The model built far enough to reach compile/lowering, but Inductor failed before training with `BACKEND='FLASH' but flash attention cannot be used: CUTE flash attention library is not available`. This is an environment/kernel-availability failure rather than a Qwen3 sharding failure, so `flex_flash` should remain discarded unless the CUTE flash attention dependency becomes available.

Plain `flex` attention completed the same compiled selective-AC no-reshard command, but it was much slower than the SDPA current best. The observed run logged loss falling from 12.46586 to 11.08583, peak memory 108.37GiB, 5,101 tps, and 21.31% MFU. That is below even the pre-compile selective-AC result. The block-mask/FlexAttention overhead and codegen path do not pay off for this 4096-token dense Qwen3 14B shape on the current stack.

## Next Quantization Direction

Attention backend changes are not useful on the current stack, so the search should return to the compiled profile's dominant linear GEMMs. The broad MXFP8 converter failed in a torchao MXFP8 backward kernel, but that does not rule out standard Float8 rowwise training. `Float8LinearConverter` is already used by TorchTitan Llama configs, supports model compile, and filters dimensions that are not multiples of 16. Qwen3 14B's FFN and attention output/query projections are large enough to be plausible candidates.

The conservative next experiment should filter out the LM head and the small combined KV projection (`attention.qkv_linear.wkv`, shape 5120 x 1024), and also include `auto_filter_small_kn` if the installed torchao exposes the auto-filter helper. This keeps the candidate focused on the large repeated FFN and attention projection GEMMs while avoiding the most likely low-benefit or numerically sensitive linears.

## Experiment Review: Float8 Rowwise

Float8 rowwise conversion of the large Qwen3 linears is the new best. The measured source filters out `lm_head` and `attention.qkv_linear.wkv`, converting 200 repeated large projection/FFN linears and leaving 41 linears in bf16. The 10-step run completed with loss falling from 12.48223 to 11.18137, peak memory 108.15GiB, and 8,399 tps. TorchTitan reported MFU as `N/A` for this quantized run, so MFU should be treated as a diagnostic gap rather than a ranking blocker.

The `auto_filter_small_kn` option was not used in the measured source because a pre-run config check showed it converted zero Qwen3 linears in this checkout: the torchao auto-filter helper expects built `nn.Linear` modules, while TorchTitan's Qwen3 converter applies to `Linear.Config` objects. Manual filtering of the LM head and combined KV projection kept the intended conservative scope and produced a real Float8 candidate.

Float8 improves the previous SDPA/model-only-compile best by about 6.3% (7,898 to 8,399 tps) with similar peak memory. Because the best has changed and MFU is omitted for quantized training, the next loop should profile the Float8 current best before trying another quantization or parallelism candidate. FFN-only MXFP8 remains a possible lower-priority follow-up, but it should not run before the new bottleneck is confirmed.

## Experiment Review: Float8 Profile And FFN MXFP8

The profiled Float8 current-best run should not be ranked against unprofiled rows, but it confirms that the active bottleneck changed again. The run completed with loss falling from 12.30372 to 11.29539, peak memory 108.15GiB, and profiled step-10 throughput of 7,812 tps. Rank 0 trace totals show about 2.81s of kernel time, around 0.99s of NCCL-named work, and many visible Float8 scaling/casting kernels such as `triton_red_fused_abs_amax...` and `aten::_scaled_mm`. Compared with the prior bf16 compiled profile, Float8 reduced the dense matmul burden but did not eliminate communication or quantization overhead.

FFN-only MXFP8 still crashed before completing step 1 with `RuntimeError: invalid argument` from `torch.ops.torchao.mxfp8_quantize.default` in the compiled backward path. Since both broad MXFP8 and FFN-only MXFP8 hit the same torchao quantize-backward failure, MXFP8 should be considered blocked in this environment unless the torchao/runtime stack changes.

The next performance lever should be activation-checkpointing policy on the Float8 source. The current best still uses only about 108.2GiB on 178.35GiB B200s, while disabling checkpointing entirely was unsafe in the earlier bf16 path. Compiler `memory_budget` mode is a narrower way to spend some memory headroom on fewer recomputations without jumping directly to no checkpointing.

## Experiment Review: Float8 Memory Budget 0.75

Compiler memory-budget activation checkpointing at budget 0.75 is the new best. On top of the Float8 rowwise source, it completed the 10-step run with loss falling from 12.26095 to 6.07721, peak memory rising to 129.9GiB, and throughput improving to 8,821 tps. TorchTitan still reports MFU as `N/A` for this quantized path.

This validates the hypothesis from the Float8 profile: spending memory to reduce recomputation is useful after the large linears are quantized. The memory increase from selective AC to budget 0.75 is about 21.7GiB, while the node still has roughly 48GiB to physical capacity and roughly 39GiB to the program's 95% risk threshold. The next narrow probe should raise the memory budget to 0.9 rather than jumping straight to no activation checkpointing.

## Experiment Review: Float8 Memory Budget 0.9

Memory budget 0.9 showed significant run-to-run variance. The first run completed normally with loss falling from 12.48586 to 7.42518, 144.0GiB peak memory, and 8,011 tps, so it was worse than the 0.75 best. A duplicate run of the exact same command completed with loss falling from 12.34735 to 9.44670, the same 144.0GiB peak memory, and 8,877 tps, which is the best observed throughput so far.

Because both 0.9 runs were finite and used the same command, keep the 8,877 tps repeat as the current best but treat the 0.9 setting as noisy. The next step should profile or repeat-confirm the 0.9 current-best path before making source changes that might be smaller than the observed variance.

The accidental profiled 0.9 run completed and confirms that the bottleneck has shifted back toward communication. The profiled run reached 7,885 tps with profiler overhead, loss falling from 12.52719 to 9.78586, and the same 144.0GiB peak memory. Rank 0 trace totals show about 2.84s kernel time and about 1.26s NCCL-named time; reduce-scatter is the largest single kernel bucket at about 571ms. Float8 scaling/casting kernels are still visible, but NCCL is now too large to ignore.

Even with communication reappearing, there is still enough memory headroom to test budget 1.0 before moving to source-level HSDP or sharding changes. If budget 1.0 does not improve materially, the next class of ideas should target FSDP communication, likely HSDP with a smaller shard group if the added memory stays below the risk threshold.

## Experiment Review: Float8 Memory Budget 1.0

Memory budget 1.0 completed but did not improve the current best. The run used the current Float8 rowwise source, finite loss fell only slightly from 12.59874 to 12.51579, peak memory rose to 154.35GiB, and throughput reached 8,851 tps. That is below the best 0.9 row at 8,877 tps and only marginally above the 0.75 row at 8,821 tps, while consuming about 10GiB more memory than 0.9.

The memory-budget lever now looks saturated for this workload. The 0.9 profile showed NCCL around 1.26s with reduce-scatter as the largest single kernel bucket, so the next useful experiment should target FSDP communication instead of spending more memory on activation retention. HSDP is the narrowest in-scope source change: FSDP2 accepts a 2D mesh with `dp_replicate` on axis 0 and `fsdp` on axis 1, and the existing TorchTitan transformer backend already uses `parallel_dims.get_mesh(["dp_replicate", "fsdp"])` for that case.

## Experiment Review: HSDP 2x4 Memory Budget 0.75

The HSDP 2x4 lower-budget probe completed but was much slower than the current best. With `dp_replicate=2`, `dp_shard=4`, no-reshard FSDP, Float8 rowwise, model-only compile, and memory budget 0.75, loss fell from 12.62485 to 7.83562 and peak memory reached 157.72GiB, but throughput was only 5,455 tps with MFU `N/A`. This is below the 8,877 tps memory-budget 0.9 best and below the prior 0.75 8-way FSDP row.

The smaller FSDP shard group did not pay for the extra HSDP replication/all-reduce cost on this workload, and the memory cost was high even with the lower activation budget. The Qwen3 HSDP source change was reverted after the discard. Treat same-shape HSDP as unattractive unless a later profile or topology-specific change gives a stronger reason to re-open it.

## Experiment Review: HSDP 2x4 Memory Budget 0.9

The requested HSDP 2x4 run with memory-budget 0.9 confirmed that this mesh direction is not useful here. Loss was finite and fell from 12.48534 to 7.49267, but throughput reached only 5,837 tps, far below the 8,877 tps 8-way FSDP best. Peak memory rose to 172.79GiB, about 96.88% of reported B200 capacity, which is above the program's rough memory risk line.

The higher activation budget improved over the accidental HSDP 0.75 row but not nearly enough to offset HSDP overhead, and it made memory risky. The Qwen3 HSDP source change was reverted. Current best remains Float8 rowwise with model-only compile, no-reshard FSDP, and compiler memory-budget 0.9.
