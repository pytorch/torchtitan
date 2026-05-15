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

Same-shape HSDP should be considered closed for this single-node fully NVLinked run. Both HSDP probes were far below the 8-way FSDP best, and the 0.9 budget case exceeded the memory risk line. The next lower-risk direction is to finish measuring narrow Float8 coverage candidates before changing mesh shape again. One source-only commit already tested the edit shape for including `attention.qkv_linear.wkv` in Float8 rowwise, but no result row exists, so that is the next bounded experiment.

Human direction supersedes the queued Float8 KV coverage idea for one bounded experiment: revisit TP=2 under the current best stack. The prior TP=2 result is stale because it was measured before the main winning changes, so this run will isolate only the TP mesh/sharding change on top of Float8 rowwise, no-reshard FSDP, model-only compile, and memory-budget 0.9. HSDP remains disabled.

## Experiment Review: TP=2 Current-Best Revisit

The bounded TP=2 revisit is blocked by the compiler stack before training starts. Restoring the known-good dense Qwen3 TP sharding contract and running the current best command with TP=2, Float8 rowwise, no-reshard FSDP, model-only compile, and memory-budget 0.9 failed before step 1 inside Inductor's min-cut memory-budget partitioner with `Unknown metadata type FakeScriptObject`. Moving compile before TP wrapping did not change the failure. This appears to be a TP/DTensor metadata interaction with the compiled memory-budget path, not a throughput result.

Do not spend more runs on same-shape TP=2 with `--compile.enable --compile.components model --activation_checkpoint.mode=memory_budget` unless the compiler/DTensor issue is fixed or the experiment intentionally changes away from the current-best stack. The Qwen3 TP source was reverted, and the best remains the 8-way FSDP Float8 memory-budget 0.9 row.

## Experiment Review: Float8 KV Projection Coverage

Including Qwen3's repeated `attention.qkv_linear.wkv` projection in Float8 rowwise did not improve the current best. The run converted 280 Float8 linear configs instead of the filtered baseline's 200 and completed with finite/falling loss from 12.53812 to 7.92596, MFU `N/A`, and 137.35GiB peak memory, but throughput reached only 8,190 tps versus the 8,877 tps best.

This closes the narrow KV coverage candidate for now. The extra small-projection Float8 work appears to add more scaling/casting or compile/runtime overhead than it removes from bf16 GEMM work under the current 8-way FSDP memory-budget 0.9 stack. The `qwen3_14b()` source filter was reverted, so `lm_head` and `attention.qkv_linear.wkv` remain filtered.

The remaining in-scope search space is mostly narrow quantization/config tuning. Mesh changes are either slower (HSDP) or compiler-blocked under the current best stack (TP=2), and converting additional smaller linears to Float8 was slower. The next bounded candidate is to change the Float8 scaling recipe for the same large-linear subset: installed TorchAO supports `Float8LinearConfig.from_recipe_name("tensorwise")`, which may reduce axiswise scaling overhead at the cost of higher numerical risk.

Next TP root-cause step: switch only activation checkpointing from memory-budget to selective while keeping TP=2, Float8 rowwise, model-only compile, and no-reshard FSDP. This isolates whether the previous TP crash was specifically the memory-budget min-cut partitioner or a broader TP/compile incompatibility.

## Experiment Review: TP=2 Selective AC Root Cause

TP=2 with Float8 rowwise, model-only compile, no-reshard FSDP, and selective AC is runnable, so the prior TP crash is specific to the compiler memory-budget partitioner rather than a general TP/compile incompatibility. The selective run completed with finite/falling loss from 3.06316 to 2.32347 and 72.00GiB peak memory.

It is still a discard for performance: 7,017 tps is below the 8,877 tps 8-way FSDP memory-budget best and below the 8,399 tps Float8 rowwise selective-AC non-TP row. The run also logged global batch size 16 because TP=2 leaves only four data-parallel shard ranks, so TP reduces the reported token throughput for this fixed local-batch command. The Qwen3 TP source was reverted.

## Experiment Review: Float8 Tensorwise Recipe

The Float8 tensorwise recipe is not useful under the current memory-budget best stack. Changing only `qwen3_14b()` from `recipe_name="rowwise"` to `recipe_name="tensorwise"` and running the 8-way FSDP no-reshard, model-only compile, memory-budget 0.9 command completed, but loss rose from 12.28795 to 15.71423. Throughput reached only 5,000 tps with MFU `N/A`, far below the 8,877 tps rowwise best, and peak memory was 145.67GiB.

This rules out the simple tensorwise scaling swap for the current large-linear Float8 subset. It did not crash or fail config validation, but it was both numerically worse over 10 steps and much slower than rowwise. The `qwen3_14b()` source change was reverted, so the active best remains rowwise Float8 with `lm_head` and `attention.qkv_linear.wkv` filtered.

## Next Memory-Budget Probe

The remaining low-risk command space is a fine sweep around the current memory-budget plateau. Budget 0.9 is the best observed result at 8,877 tps and 144.0GiB, but its duplicate run history shows meaningful variance. Budget 1.0 increased memory to 154.35GiB and was slightly slower at 8,851 tps. A 0.95 probe is a narrow command-only test between those two settings; it should be accepted only if it beats 8,877 tps with finite/falling loss, otherwise the memory-budget lever should be treated as saturated.

## Experiment Review: Float8 Memory Budget 0.95

Memory budget 0.95 is the new best, but only by a narrow margin. On the current Float8 rowwise source with model-only compile and no-reshard 8-way FSDP, the 10-step command completed with loss falling from 12.28270 to 9.57252, peak memory 143.96GiB, and 8,897 tps. MFU remains `N/A` for this quantized path.

This beats the prior 0.9 best by 20 tps with essentially the same memory, while 1.0 was slower and used about 10GiB more memory. Treat 0.95 as the current best row, but the gain is within the observed memory-budget noise band, so future candidates should clear 8,897 tps rather than reading this as a large activation-policy breakthrough.

## Next Memory-Budget High-Side Probe

Budget 0.95 is now the best row, but its peak memory is essentially the same as 0.9. This suggests the compiler's memory-budget partitioner may choose discrete activation-save sets rather than changing smoothly for each decimal value. Budget 1.0 picked a higher-memory point and slowed slightly. A single 0.975 run is the next narrow test: if it still uses about 144GiB, the result mostly measures variance around the current best; if it uses an intermediate or 1.0-like memory footprint, it tells whether the extra saved activations can beat the 8,897 tps threshold.

## Experiment Review: Float8 Memory Budget 0.975

Memory budget 0.975 did not improve the current best. The command completed normally on the current Float8 rowwise source with model-only compile and no-reshard 8-way FSDP, and loss fell from 12.47411 to 11.52573. Peak memory stayed at 143.96GiB, matching the 0.95 row rather than moving toward the 1.0 memory footprint, but throughput was only 5,453 tps with MFU `N/A`.

This is a discard. The result is far below the 8,897 tps 0.95 best despite identical peak memory, reinforcing that this memory-budget region is noisy and that 0.975 did not select a useful higher-throughput partition.

## Experiment Review: Float8 Memory Budget 0.95 Repeat

The 0.95 repeat supports the current best but did not improve it. The exact current-best command completed with loss falling from 12.49943 to 7.65341, peak memory 143.96GiB, and 8,891 tps. This is only 6 tps below the 8,897 tps best and far above the 0.975 same-memory discard, so the 0.95 setting appears reproducible near the observed maximum.

Record this repeat as a discard diagnostic rather than a new best. Future candidates should still clear 8,897 tps, and adjacent memory-budget probes should be interpreted against the confirmed variance around this plateau.

## Next Memory-Budget Lower-Side Probe

The high-side 0.975 result argues against spending more runs above 0.95 for now. It used the same 143.96GiB peak memory as 0.95 but fell to 5,453 tps, so the budget value is not a smooth proxy for useful saved activations or runtime. The lower side is still under-sampled: 0.75 used 129.9GiB and 0.9 used 144.0GiB. A 0.85 run can show whether the current 144GiB partition starts below 0.9 or whether there is a cheaper stable point that can match or beat the 8,897 tps best.

## Experiment Review: Float8 Memory Budget 0.85

Memory budget 0.85 did not improve the current best. The command completed on the current Float8 rowwise source with model-only compile and no-reshard 8-way FSDP, but loss rose from 12.52796 to 17.56807. Peak memory was 136.50GiB, between the 0.75 and 0.9/0.95 rows, and throughput reached 8,834 tps with MFU `N/A`.

This is a discard. It is below the 8,897 tps 0.95 best and fails the falling-loss sanity check, so the lower-side memory-budget probe does not provide a better stable point.

## Experiment Review: Local Batch Size 5

Local batch size 5 is a discard on the current best stack. This was the already-active run in the checkout, so it superseded the pending 0.925 handoff for this iteration; the 0.925 idea remains recorded as pending. The command completed with global batch size 40, finite/falling loss from 12.55001 to 8.61428, and peak memory 164.84GiB, which is below the rough 95% B200 risk line but much closer to it than the local-batch-size 4 best.

Throughput reached only 5,924 tps, far below the 8,897 tps best at local batch size 4. The extra per-step tokens did not overcome the memory/runtime cost of the larger batch under Float8 rowwise, model-only compile, no-reshard FSDP, and memory-budget 0.95. Treat local batch size increases as unattractive unless another change materially lowers memory or per-step overhead first.

## Next Memory-Budget Center Probe

The memory-budget search is now nearly exhausted. Budget 0.95 has the best row at 8,897 tps and a repeat at 8,891 tps, both with 143.96GiB peak memory. Budget 0.9 is slightly behind at 8,877 tps with the same memory region, while 0.85 used less memory but failed loss trend and 0.975 was much slower. A final 0.925 command-only probe is reasonable because it sits inside the stable 0.9-0.95 band. If it does not beat 8,897 tps with falling loss, treat memory-budget tuning as saturated.

## Experiment Review: Default Compile Components

Default compile components remain a discard on the current best stack. Running Float8 rowwise, no-reshard 8-way FSDP, memory-budget 0.95, and `--compile.enable` without `--compile.components model` compiled the loss function as well as each TransformerBlock. The run completed with loss falling from 12.46363 to 7.79407, peak memory 141.93GiB, and 8,882 tps.

This is below the 8,897 tps model-only compile best and below the 8,891 tps 0.95 repeat, so loss compilation still does not pay for itself here. Keep `--compile.components model` as part of the current best command.

## Experiment Review: Float8 Memory Budget 0.925

The centered 0.925 memory-budget probe completed normally but did not improve the current best. On the current Float8 rowwise source with model-only compile and no-reshard 8-way FSDP, loss fell from 12.39717 to 9.11266, peak memory matched the 0.95 plateau at 143.96GiB, and throughput reached 8,718 tps with MFU `N/A`.

This is a discard against the 8,897 tps 0.95 best. Since 0.9, 0.925, 0.95, and 0.975 all sit in roughly the same memory region but vary widely in throughput, memory-budget tuning looks saturated and noisy rather than smoothly optimizable.

## Next Float8 Recipe Probe

Memory-budget tuning and local batch size are saturated on the current best stack. The only remaining supported Float8 recipe variant not measured is `rowwise_with_gw_hp`. TorchAO config inspection shows it keeps forward and grad-input rowwise Float8 but disables Float8 casting for the grad-weight path, making grad-weight high precision. This is likely slower because the grad-weight GEMM becomes bf16, but it is a bounded `qwen3_14b()` converter change and closes the official Float8 recipe search space if it does not beat 8,897 tps.

## Experiment Review: Float8 Rowwise Grad-Weight High Precision

The `rowwise_with_gw_hp` Float8 recipe is the new best on the current stack. Changing only the `qwen3_14b()` `Float8LinearConverter.Config` recipe and running 8-way FSDP no-reshard with model-only compile and memory-budget 0.95 completed with finite/falling loss from 12.29840 to 9.54931, MFU `N/A`, and 145.05GiB peak memory.

Throughput reached 9,229 tps, beating the previous 8,897 tps best by 332 tps while using roughly the same memory region. Despite moving the grad-weight GEMM inputs back to high precision, the reduced Float8 casting/scaling work appears to pay off for this 10-step workload. Keep the source change and treat `rowwise_with_gw_hp` plus memory-budget 0.95 as the current best.

## Next Rowwise GW-HP Repeat

An identical `rowwise_with_gw_hp` memory-budget 0.95 command is already active after the new-best commit. Finalize it as a repeat-confirmation run rather than starting a different experiment. If it stays near 9,229 tps with falling loss, the recipe win is likely robust; if it falls back toward the old 8.9k plateau, treat the first row as a favorable variance sample before trying adjacent budgets.

## Experiment Review: rowwise_with_gw_hp Repeat

The exact `rowwise_with_gw_hp` current-best command repeated successfully and supports the new recipe as a real improvement, but it did not set a new best. The run completed with finite loss falling from 12.37330 to 7.53270, peak memory matching the prior row at 145.05GiB, MFU `N/A`, and 9,213 tps.

Record this as a discard diagnostic because it is 16 tps below the 9,229 tps best. The repeat is still close enough to confirm that `rowwise_with_gw_hp` is consistently in the 9.2k tps range on this stack, rather than only a one-off spike.

## Next GW-HP Memory Budget Probe

The `rowwise_with_gw_hp` recipe changed the performance surface enough that the old rowwise memory-budget sweep should not be treated as final. The new best uses 145.05GiB at budget 0.95 and repeats within 16 tps of the max. A budget 1.0 probe is safe enough on B200 memory and tests whether the reduced Float8 scaling overhead makes additional activation retention useful. If it does not beat 9,229 tps, the next retune point should be lower-side 0.9 rather than further high-side budget sweeps.

## Experiment Review: rowwise_with_gw_hp Memory Budget 0.9

The handoff requested a budget 1.0 command, but an already-active run in the checkout was using `--activation_checkpoint.memory_budget=0.9`. Per worker coordination, that actual active 0.9 command was finalized instead of starting another run, so it superseded the pending 1.0 handoff for this iteration while leaving the 1.0 idea pending.

The lower-side `rowwise_with_gw_hp` memory-budget 0.9 retune is a discard. The command completed with the current high-precision grad-weight Float8 source, no-reshard 8-way FSDP, and model-only compile, but loss rose from 12.59183 to 16.25217 instead of falling.

Throughput reached only 4,888 tps with 145.05GiB peak memory, far below both the 9,229 tps budget 0.95 best and its 9,213 tps repeat. The lower budget did not reduce peak memory relative to 0.95 in this run and made both throughput and loss trend worse, so keep memory-budget 0.95 as the active `rowwise_with_gw_hp` setting.

## Experiment Review: rowwise_with_gw_hp Memory Budget 1.0

The high-side `rowwise_with_gw_hp` memory-budget 1.0 probe is a discard. It completed the 10-step command, but step 1 already used 163.86GiB and the allocator repeatedly warned that expandable segment mappings failed with OOM-level free memory.

Step 10 reached only 1,014 tps with peak memory 176.49GiB, 13 CUDA allocation retries, and loss rising from 12.14356 to 17.49169. Budget 1.0 is both slower and too close to the B200 memory ceiling, so the activation-budget retune is closed with 0.95 as the active setting for `rowwise_with_gw_hp`.

## Experiment Review: rowwise_with_gw_hp Profile

The completed profile run for the current `rowwise_with_gw_hp` best command is a discard diagnostic, not a ranked throughput result. The rank-0 training log shows finite/falling loss from 12.46392 to 10.33925, MFU `N/A`, 145.05GiB peak memory, and 8,540 profiled tps at step 10. The trace directory was logged as `./outputs/profiling/traces`.

Because the profiler is active around the measured step, this row should not displace the 9,229 tps unprofiled best. Use the trace to choose any remaining in-scope source changes; keep `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.95 as the current best stack.

Trace bucket summary from `outputs/profiling/traces/iteration_10` shows rank-0 GPU kernel time at about 2.97s: matmul/scaled-mm kernels 1.33s (44.9%), NCCL kernels 0.77s (26.1%), attention kernels 0.59s (20.0%), and direct Float8 scale/cast/elemwise kernels 0.17s (5.7%). Across ranks, NCCL averaged 0.87s but ranged from 0.35s to 1.37s, with reduce-scatter dominating the communication bucket.

Interpretation: the current best is primarily compute-bound on GEMM/attention, but FSDP reduce-scatter remains the largest non-compute bucket and is rank-imbalanced. Direct Float8 scaling/casting is too small to justify more recipe-only tuning unless a candidate also changes GEMM behavior. The next useful idea should target communication/reduce-scatter or a matmul/attention efficiency lever, with extra caution because prior HSDP and TP attempts were slower or unstable.

## Next Float8 Coverage Probe: lm_head

The next bounded source candidate is to remove only the `lm_head` Float8 filter while keeping `rowwise_with_gw_hp`, the `attention.qkv_linear.wkv` filter, model-only compile, no-reshard 8-way FSDP, and memory-budget 0.95. `ChunkedCELoss` applies `lm_head` over eight chunks and explicitly notes that the head is not compiled; for Qwen3 14B this is a large `5120 x 151936` projection that remains bf16 in the current best.

This is higher risk than another command-only probe because it changes the output projection numerics and may add scaling overhead in the chunked-loss loop. It is still in scope and isolated to `qwen3_14b()` converter filtering. Keep it only if loss is finite/falling and throughput clears the current 9,229 tps best; otherwise revert the source and record the result as a coverage discard.

## Experiment Review: Default FSDP Reshard Policy

The source-free default FSDP reshard policy check is a discard on the active `rowwise_with_gw_hp` stack. Omitting `--parallelism.fsdp_reshard_after_forward=never` completed with finite/falling loss from 12.55290 to 10.28098, MFU `N/A`, and much lower peak memory at 120.30GiB.

Throughput reached 9,067 tps, below both the 9,229 tps no-reshard best and its 9,213 tps repeat. The memory savings do not compensate for the added reshard/all-gather overhead on the 10-step objective, so keep explicit no-reshard in the current best command.

## Experiment Review: Default FSDP Reshard with Local Batch Size 5

The default-reshard local-batch-size 5 follow-up is a discard. It fit in memory on the active `rowwise_with_gw_hp` source, with peak memory reaching 144.04GiB, but the extra batch consumed almost all of the memory saved by default reshard and did not recover throughput.

Step 10 reached 8,543 tps and loss rose from 12.47422 to 14.69380, so this fails both the 9,229 tps throughput gate and the finite/falling-loss sanity gate. Default reshard remains useful only as a memory-saving diagnostic, not as a path to higher throughput via batch-size 5.

## Experiment Review: lm_head Float8 Coverage

Including Qwen3 14B `lm_head` in `rowwise_with_gw_hp` Float8 is a discard. The candidate removed only the `lm_head` filter while keeping `attention.qkv_linear.wkv` filtered, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.95. The run completed, but loss rose from 12.38387 to 16.43954, peak memory increased to 149.01GiB, and throughput reached only 8,399 tps.

This closes the broad Float8 coverage expansion path for now. Both the smaller KV projection and the large chunked-loss `lm_head` projection were slower than the filtered current best, and `lm_head` also failed the falling-loss sanity check. Keep the current `filter_fqns=["lm_head", "attention.qkv_linear.wkv"]` source.

## Next Compile Coverage Check

The only remaining low-risk command-only lever near the current stack is default compile coverage under `rowwise_with_gw_hp`: run the current best command without `--compile.components model`, so TorchTitan compiles the loss function as well as the model. This does not compile `lm_head`, but it can test whether per-chunk CE compilation now helps after the recipe change. Keep it only if it beats 9,229 tps with finite/falling loss; prior rowwise evidence makes a small slowdown more likely than a large win.

## Experiment Review: Default Compile Coverage Under GW-HP

Default compile coverage is a discard on the active `rowwise_with_gw_hp` stack. Running the current best command without `--compile.components model` compiled the loss function in addition to the model, but the 10-step run reached only 8,687 tps with MFU `N/A`, 142.74GiB peak memory, and loss rising from 12.52405 to 12.72060.

This confirms the earlier rowwise result: compiling the CE loss does not help this chunked-loss workload, and may destabilize the short-run loss trend. Keep model-only compile in the current best command.

## Next Runtime-Overhead Probe

A final low-risk command-only probe is to disable structured trace logging with `--debug.no-enable-structured-logging` while keeping normal stdout metrics. `DebugConfig.enable_structured_logging=False` makes `log_trace_span`, `log_trace_instant`, and `log_trace_scalar` no-ops; the training loop calls these around step, batch fetch, forward/backward, optimizer, and metric collection. The expected gain is small, but for a 10-step objective this overhead can matter, and it does not change model math.

## Experiment Review: Default Compile Components with rowwise_with_gw_hp

The default compile-components revisit is a discard. Running the active `rowwise_with_gw_hp` stack without `--compile.components model` completed, but loss rose from 12.52405 to 12.72060, peak memory was 142.74GiB, and throughput reached only 8,687 tps.

Compiling the loss function still does not pay off on this stack. Keep `--compile.components model` in the current best command, and treat compile-coverage tuning as closed unless a future source change materially changes the loss path.

## Experiment Review: bfloat16 FSDP Reduction

The Qwen3-local bfloat16 FSDP reduction candidate is the new best. Adding `mixed_precision_reduce="bfloat16"` only to the `qwen3_14b()` `TrainingConfig` kept the active `rowwise_with_gw_hp` recipe, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.95 stack. The run completed with loss falling from 12.34481 to 9.12246, peak memory 145.48GiB, MFU `N/A`, and 9,332 tps.

This narrowly beats the prior 9,229 tps best and directly targets the reduce-scatter bandwidth bucket from the profile. The run emitted a tyro warning because the global `TrainingConfig.mixed_precision_reduce` type annotation is currently `Literal["float32"]`; per scope, the type was not edited. Since Qwen3 parallelize uses `getattr(torch, training.mixed_precision_reduce)` and the command completed with falling loss, keep the Qwen3-local source change while noting the type annotation mismatch as a cleanup concern if this graduates beyond autoresearch.

Current best is now `rowwise_with_gw_hp` with bfloat16 FSDP reduction, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.95.

## Experiment Review: Structured Logging Disabled

Disabling structured trace logging on the bf16-reduce current-best stack is a discard. The command completed and reached 9,355 tps with 145.48GiB peak memory, but loss rose from 12.19051 to 14.05692.

Because this was meant to be a low-risk runtime-overhead probe, the rising loss is enough to reject it even though throughput was higher than the 9,332 tps bf16-reduce row. Keep structured logging enabled for the current best until a clean repeat proves otherwise.

## Next Repeat Confirmation: bfloat16 Reduce

An exact current-best command is already active in the checkout: bfloat16 FSDP reduction, `rowwise_with_gw_hp`, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.95. Finalize it as a repeat-confirmation run rather than starting another experiment. If it stays above 9,332 tps with falling loss, the communication-dtype win is stronger; if it falls back or rises in loss, keep the original bfloat16-reduce row as a narrow best and treat the source as needing more confirmation.

## Experiment Review: bfloat16 Reduce Repeat

The clean repeat of the bfloat16-reduce current-best command is a discard diagnostic, not a new best. It completed with loss falling from 12.43899 to 9.72588, peak memory 145.48GiB, MFU `N/A`, and 9,315 tps.

This is below the 9,332 tps bf16-reduce max but still above the prior 9,229 tps `rowwise_with_gw_hp` best. Keep the bf16-reduce source as current best, but treat the win as narrow and somewhat noisy rather than a large step change.

## Next Optimizer Probe

The next bounded candidate is `--optimizer.implementation fused_opt_states_bf16` on the bf16-reduce current-best source. The documented behavior keeps the fused AdamW path while storing Adam moments in bfloat16 and is FSDP2-compatible on local DTensor shards. This may reduce optimizer-state memory or optimizer-step bandwidth enough to matter for the 10-step objective. Use the new memory-budget 0.925 best stack and keep it only if loss is finite/falling and throughput clears 9,364 tps; otherwise record it as an optimizer-state discard.

## Active Run Supersedes Optimizer Handoff

An already-active command is running the bf16-reduce current-best source at memory-budget 0.925. Finalize that active run before starting the optimizer-state probe. Treat it as a one-off adjacent memory-budget retune after bfloat16 reduction; keep it only if it beats 9,332 tps with finite/falling loss, otherwise return to the queued optimizer-state idea.

## Experiment Review: bfloat16 Reduce Memory Budget 0.925

The centered lower memory-budget retune on the confirmed bf16-reduce source is the new best. Running `rowwise_with_gw_hp`, bfloat16 FSDP reduction, no-reshard 8-way FSDP, model-only compile, and memory-budget 0.925 completed with loss falling from 12.38271 to 9.11822, peak memory 145.48GiB, MFU `N/A`, and 9,364 tps.

This beats the 9,332 tps memory-budget 0.95 max and the 9,315 tps repeat while using the same peak memory. Keep budget 0.925 as the active activation-checkpoint setting for the bf16-reduce stack, and use 9,364 tps as the next comparison threshold.

## Experiment Review: bfloat16 Reduce Memory Budget 0.925 Repeat

The exact repeat of the 0.925 budget current-best command is a discard diagnostic. It completed with loss falling from 12.29194 to 6.69980, peak memory 145.48GiB, MFU `N/A`, and 9,233 tps.

This does not reproduce the 9,364 tps max and lands below the 9,315 tps 0.95 repeat, though loss is cleanly falling and memory is unchanged. Keep 9,364 tps as the observed max/current best for the autoresearch objective, but treat the 0.925-over-0.95 advantage as noisy rather than robust.

## Experiment Review: bfloat16 Reduce Memory Budget 0.9375

The midpoint activation-budget probe is a discard. The 0.9375 run completed with loss falling from 12.47883 to 10.69838, peak memory 145.48GiB, MFU `N/A`, and 9,354 tps.

This is above both clean repeats at 0.95 and 0.925, but it does not beat the 9,364 tps observed max from 0.925. Keep 0.925 as the current-best max for ranking, while noting that 0.9375 provides another near-best point and the activation-budget differences are within short-run noise.
