# Setup

Branch: `autoresearch-parallelize/may19-qwen3-14b`

Source commit: `7c324f2532265456853d84393304d6fdf9afa26f`

Current best commit: none yet. No measured `results.tsv` rows exist on this fresh experiment branch.

Objective: maximize TorchTitan-reported steady-state `tps` / `throughput(tps)` for one 10-step Qwen3 14B training command on this machine while keeping loss finite and training normally. Use reported MFU and peak memory only as tie-breakers and diagnostics.

## Baseline Command

Recorded baseline command template:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run01-fsdp-bootstrap > run.log 2>&1
```

Run-specific `--dump_folder` values and redirected log filenames may change only to isolate artifacts. They are not optimization knobs.

Evidence for this command:

- `run_train.sh` is the repo launcher and defaults to 8 local processes when `NGPU=8`.
- The config manager exposes the Qwen3 14B workload as `MODULE=qwen3 CONFIG=qwen3_14b`; the lower-level model registry flavor is `14B`.
- `qwen3_14b()` keeps the target model, tokenizer, dataset, and checkpoint behavior local and available: `hf_assets_path="./tests/assets/tokenizer"`, `dataloader.dataset="c4_test"`, and checkpoint loading disabled.
- The program requires every experiment-loop run to cap training at exactly 10 steps, so `--training.steps=10` is part of every candidate command.

Fixed baseline requirements:

- Launcher and world: `NGPU=8`, single-node `torchrun` through `./run_train.sh`.
- Model/workload: `qwen3`, `qwen3_14b()`, model registry flavor `14B`, dense 40-layer Qwen3 with `dim=5120`, 40 attention heads, 8 KV heads, head dim 128, vocab 151936, no weight tying.
- Data/checkpoint: `c4_test`, `./tests/assets/tokenizer`, no initial checkpoint load, checkpoint saving disabled by default unless explicitly changed later as an idea.
- Training: local batch size 4, sequence length 4096, `training.dtype="float32"`, mixed-precision FSDP param dtype `bfloat16`, mixed-precision reduce dtype `float32`, AdamW fused optimizer, LR 8e-4, warmup 600, max norm 1.0.
- Parallelism config: `data_parallel_shard_degree=-1` resolving to 8 on this machine, `data_parallel_replicate_degree=1`, TP=1, CP=1, PP=1, EP=1, sequence parallel enabled but inactive without TP.
- Other knobs: activation checkpointing configured as `full`, compile disabled by default, CPU offload disabled, FSDP reshard policy `default`.

First experiment lock:

- First idea: bootstrap minimal baseline FSDP in `torchtitan/models/qwen3/parallelize.py`.
- Single changed capability: replace the scaffold's replicated-DP behavior with minimal `fully_shard` behavior for the already-recorded FSDP-only command.
- Intentionally unchanged for the first source diff: TP, CP, PP, EP, compile, quantization/converters, attention backend, dtype, optimizer, batch size, sequence length, CPU offload, FSDP resharding policy, and launch/runtime environment.
- Important pending bootstrap: the baseline config requests full activation checkpointing, but the scaffold currently ignores `ac_config`. Applying activation checkpointing is recorded as the next separate idea rather than bundled with FSDP.

Editable files for source/config experiments:

- `torchtitan/models/qwen3/parallelize.py`
- `torchtitan/models/qwen3/sharding.py`
- `torchtitan/models/qwen3/config_registry.py`, only within `qwen3_14b()` and only for allowed fields
- `ideas.md`, `learnings.md`, `results.tsv`
- Untracked local run artifacts such as `run.log` and dump folders

## Hardware And Environment

Detected hardware:

- 8 x NVIDIA B200 GPUs.
- `nvidia-smi` reports 183359 MiB memory per GPU, about 179.1 GiB.
- Driver version 580.82.07, system CUDA version 13.0.
- PyTorch reports `torch 2.13.0a0+git85cb48a`, CUDA 13.1, compute capability 10.0, and 8 CUDA devices.
- GPU topology shows `NV18` connectivity between every GPU pair, with GPUs 0-3 on NUMA node 0 and GPUs 4-7 on NUMA node 1.
- B200 public-spec expectation for roofline notes: roughly multi-PFLOP BF16/FP16 Tensor Core peak per GPU and about 8 TB/s HBM3e bandwidth per GPU. Prefer TorchTitan's own `Peak FLOPS used for computing MFU` line after the first run for exact MFU calculations.

Available local data/assets:

- `tests/assets/tokenizer/tokenizer.json`
- `tests/assets/tokenizer/tokenizer_config.json`
- `tests/assets/c4_test/data.json`

## Source Findings

`torchtitan/models/qwen3/parallelize.py` is a strategy-free scaffold. It rejects TP, CP, PP, and EP, and when DP is enabled it calls `replicate(model, device_mesh=parallel_dims.get_mesh("batch"), static_graph=True)`. That does not honor the `qwen3_14b()` FSDP-only baseline.

`torchtitan/models/qwen3/sharding.py` currently leaves all config-based DTensor sharding unset. This is acceptable for the first FSDP-only bootstrap because TP is 1 and no tensor placements are needed yet.

`Qwen3Model.Config.update_from_config()` validates TP divisibility and calls `set_qwen3_sharding_config()`. The 14B model has 40 attention heads and 8 KV heads, so future TP degrees must divide both; plausible TP degrees are 2, 4, and 8.

`ParallelDims` resolves `data_parallel_shard_degree=-1` as `world_size / (dp_replicate * cp * tp * pp)`, which is 8 for the baseline. The relevant FSDP mesh name is `fsdp`; the `batch` mesh is used for dataloading and is not the right source of FSDP behavior.

## Initial Roofline Notes

No measured TorchTitan metrics exist yet. The first runnable unprofiled run must establish throughput, memory, loss behavior, and TorchTitan's printed peak FLOPS for MFU. Until then, the bottleneck is unclear.

Likely early risks:

- Replicated full-model training is expected to exceed memory for a 14B model with optimizer states, so FSDP is the highest-priority bootstrap.
- Full activation checkpointing is configured but not yet applied by the Qwen3 scaffold. That may inflate activation memory and should be measured separately after FSDP.
- With only FSDP over 8 NVLinked B200s, collectives may become a significant cost. A later TP or FSDP resharding idea should be driven by measured throughput, memory, and profiler evidence rather than guessed.

## Experiment 1: Bootstrap Minimal Baseline FSDP

Commit: `01d1f8e`

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run01-fsdp-bootstrap-retry1 > run.log 2>&1
```

What changed:

- `parallelize_qwen3()` now rejects unsupported HSDP and CPU offload for this baseline.
- It applies composable FSDP on the `fsdp` mesh to each transformer block, `lm_head`, and the root model.
- It uses `MixedPrecisionPolicy` from the recorded training config: param dtype `bfloat16`, reduce dtype `float32`.
- It resolves `fsdp_reshard_after_forward="default"` to `True` for the non-PP baseline.
- It disables FSDP gradient division so TorchTitan's token-count loss scaling remains responsible for gradient scaling.

Diagnostic retry:

- The first attempt at the same idea crashed in `ChunkedCELoss` because `lm_head` is called directly outside `Qwen3Model.forward()` and its weight was a sharded DTensor while the hidden chunk was a local tensor.
- Fix: FSDP-wrap `model.lm_head` as its own unit before wrapping the root, so the direct loss call all-gathers `lm_head` parameters correctly.

Result:

- Status: keep, first runnable best.
- Step 10 `tps`: 7,254.
- Step 10 MFU: 30.31%.
- Step 10 peak memory: 173.91 GiB, 97.51% of the reported 178.35 GiB capacity.
- Loss moved from 12.53191 at step 1 to 12.04510 at step 10. It stayed finite and decreased across the short run.
- Step 10 `tflops`: 681.97.
- TorchTitan peak FLOPS for MFU on this B200 host: `2.250e+15`.
- Structured timing around later steps shows fwd/bwd around 1.60-1.63 s, optimizer around 21-22 ms, fetch around 9-13 ms, and step time around 2.20-2.79 s on logged rank 0.

Interpretation:

- The FSDP bootstrap is correct enough to train for the target 10-step run and becomes the current best.
- Memory is already above the program's rough 95% risk threshold. The run completed, but this leaves little room for allocator variation or later optimizations that raise activation/parameter residency.
- The current source still does not apply the configured `activation_checkpoint.mode="full"`. Given the 97.51% peak memory, the next narrow idea should apply the baseline full activation checkpointing before any speed-focused memory-increasing idea.
- Roofline status is still unclear without a profiler trace. The low model memory at initialization (8.62 GiB) but high step peak implies activations/FSDP all-gather residency dominate peak memory. Later per-step timings suggest fwd/bwd dominates wall time, but we need a trace to separate compute from FSDP collectives and recompute.

## Experiment 2: Apply Configured Full Activation Checkpointing

Commit: `dc2765b`

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run02-full-ac > run.log 2>&1
```

What changed:

- Added `apply_ac(model, ac_config)` before FSDP wrapping in `parallelize_qwen3()`.
- Left FSDP policy, launch command, batch size, sequence length, dtype, optimizer, TP/CP/PP/EP, compile, and resharding unchanged.

Result:

- Status: discard for primary objective.
- Step 10 `tps`: 5,564, down from 7,254 for `01d1f8e`.
- Step 10 MFU: 23.25%, down from 30.31%.
- Step 10 peak memory: 47.04 GiB, down from 173.91 GiB.
- Loss moved from 12.49552 at step 1 to 10.12602 at step 10; finite and decreasing.
- Later-step fwd/bwd time rose to roughly 2.27-2.32 s from roughly 1.60-1.63 s without AC.

Interpretation:

- Full AC successfully attacks activation memory, so the high peak in the FSDP-only best is activation-driven rather than optimizer-state-driven.
- The recomputation cost is large enough that full AC loses the primary objective at the existing local batch size.
- Memory headroom from full AC is real and may be useful only if converted into more work per step, such as a larger local batch, but that should be tested as a separate narrow command/config idea from an explicitly AC-enabled source state.
- Current best remains `01d1f8e` because it has higher reported tps and completed the 10-step run. Its 97.51% memory remains a risk.

## Experiment 3: Full BF16 Training Dtype

Source state: `73e33ba` with source restored to the `01d1f8e` FSDP-best implementation.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --training.dtype=bfloat16 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run03-bf16-training > run.log 2>&1
```

What changed:

- Command-only override `--training.dtype=bfloat16`.
- No source, FSDP, AC, compile, batch-size, sequence-length, optimizer, or parallelism changes.

Result:

- Status: discard for primary objective.
- Step 10 `tps`: 7,222, slightly below 7,254 for `01d1f8e`.
- Step 10 MFU: 30.17%, slightly below 30.31%.
- Step 10 peak memory: 160.71 GiB, down from 173.91 GiB.
- Model memory after init dropped from 8.62 GiB to 4.18 GiB.
- Loss moved from 12.15456 at step 1 to 9.81666 at step 10; finite and decreasing.
- Later-step fwd/bwd timing was similar to FSDP-only, around 1.61-1.65 s.

Interpretation:

- BF16 full training materially reduces parameter/model memory and lowers peak step memory to about 90.11% capacity, but it does not improve throughput at local batch size 4.
- Because throughput is almost tied and memory is much safer, BF16 is a good ingredient only if the saved memory can be converted into more useful tokens/sec with a larger local batch or another memory-using speed knob.
- Current best for primary objective remains `01d1f8e` / restored source `73e33ba`.

## Experiment 4: BF16 Training Dtype With Local Batch 5

Source state: `c65a743` with source still matching the FSDP-best implementation.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --training.dtype=bfloat16 --training.local_batch_size=5 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run04-bf16-lbs5 > run.log 2>&1
```

What changed:

- Command-only coupled overrides: `--training.dtype=bfloat16` and `--training.local_batch_size=5`.
- No source, FSDP, AC, compile, sequence-length, optimizer, or parallelism changes.

Result:

- Status: crash.
- The run built the model and initialized the trainer with local batch size 5, global batch size 40, and sequence length 4096.
- It OOMed during the first backward inside `ChunkedCELoss`, while calling `chunk_loss.backward()`.
- Error reported about 177.63 GiB in use on GPUs with 178.35 GiB capacity, and a failed 1.45 GiB allocation.

Interpretation:

- BF16's recovered memory is not enough to raise local batch from 4 to 5 without activation checkpointing or another memory reduction.
- The batch-size path is blocked unless paired with a stronger memory-saving source state such as AC or TP. Since full AC at batch 4 was much slower, blind batch-size search is not currently attractive.
- The search should pivot to profiling the current best to see whether compute kernels, FSDP collectives, or launch/data overhead dominate before making another source change.

## Experiment 5: Profile Current Best

Source state: `38ac2c8` with source still matching the FSDP-best implementation.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run05-profile-best --profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1 > run.log 2>&1
```

Result:

- Status in `results.tsv`: discard, because profiled throughput is not comparable to unprofiled candidates.
- Step 10 profiled `tps`: 6,854.
- Step 10 profiled MFU: 28.64%.
- Step 10 peak memory: 173.91 GiB, 97.51%.
- Trace files were written under `outputs/autoresearch/may19-qwen3-14b/run05-profile-best/profiling/traces/iteration_10/`.

Profile notes from rank 0 trace:

- GPU traced duration excluding the `ProfilerStep` wrapper was about 3.51 s.
- Matmul/GEMM kernels: about 1.24 s across 1,073 events.
- NCCL collectives: about 0.94 s across 293 events.
- Flash attention kernels: about 0.52 s across 205 events.
- Elementwise/activation/optimizer kernels: about 0.32 s.
- Copy/cat/split kernels: about 0.25 s.
- Top communication kernels included reduce-scatter and all-gather. The trace had 58 reduce-scatter device kernels totaling about 307 ms plus 42 `nccl:_reduce_scatter_base` events totaling about 223 ms, and 98 all-gather device kernels totaling about 215 ms plus 84 `nccl:_all_gather_base` events totaling about 196 ms.
- Later structured timings still show fwd/bwd around 1.62-1.65 s and optimizer around 22-24 ms; data fetch remains around 9-13 ms and is not the bottleneck.

Roofline conclusion:

- The run is not purely compute-bound. GEMMs are the largest bucket, but NCCL collectives are close behind and large enough to motivate a communication/memory tradeoff experiment.
- Memory pressure prevents simple larger-batch exploitation. Communication reduction should be attempted only with a memory-saving ingredient such as BF16 dtype or AC.
- Next narrow hypothesis: combine BF16 training dtype with `fsdp_reshard_after_forward=never` to spend the BF16 memory savings on fewer FSDP all-gathers. This may OOM; if it fits, success requires tps above 7,254.

## Experiment 6: BF16 Training Dtype With FSDP Reshard Disabled

Source state: `a156590` with source still matching the FSDP-best implementation.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --training.dtype=bfloat16 --parallelism.fsdp_reshard_after_forward=never --dump_folder=outputs/autoresearch/may19-qwen3-14b/run06-bf16-reshard-never > run.log 2>&1
```

Result:

- Status: crash.
- The run initialized with `reshard_after_forward=False` and reached step 1.
- Step 1 reported 881 tps, 3.68% MFU, and 176.49 GiB peak memory before crashing later in first backward.
- The OOM happened in FSDP post-backward when converting an unsharded accumulated gradient to the reduce dtype. The failed allocation was 2.90 GiB with about 175.75 GiB already in use.

Interpretation:

- BF16 alone does not create enough room for `fsdp_reshard_after_forward=never`.
- The profile's communication bottleneck remains relevant, but any no-reshard attempt needs stronger activation memory reduction than BF16 provides.
- A coupled AC plus no-reshard experiment is justified: AC alone was slower because recompute cost dominated, but no-reshard may recover some all-gather time while AC supplies the memory headroom required for retained parameters.

## Experiment 7: Full AC With FSDP Reshard Disabled

Commit: `9c5ebcc`

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --parallelism.fsdp_reshard_after_forward=never --dump_folder=outputs/autoresearch/may19-qwen3-14b/run07-ac-reshard-never > run.log 2>&1
```

What changed:

- Re-added `apply_ac(model, ac_config)` before FSDP wrapping.
- Used command override `--parallelism.fsdp_reshard_after_forward=never`.
- No dtype, batch-size, compile, TP/CP/PP/EP, optimizer, or sequence-length changes.

Result:

- Status: discard.
- Step 10 `tps`: 5,582, far below the current best 7,254 and essentially tied with full AC alone at 5,564.
- Step 10 MFU: 23.32%.
- Step 10 peak memory: 72.91 GiB.
- Loss moved from 12.41961 at step 1 to 8.34080 at step 10; finite and decreasing.
- Later-step fwd/bwd remained about 2.25-2.33 s, close to full AC alone.

Interpretation:

- Disabling FSDP resharding does not recover enough communication time to offset full AC recompute.
- The communication bucket in the profile is meaningful, but this particular memory/communication tradeoff is not a path to the current objective.
- Current best remains the FSDP-only source at `01d1f8e` / restored branch state.

## Experiment 8: Enable Per-Block Compile

Commit: `51369ea`

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --dump_folder=outputs/autoresearch/may19-qwen3-14b/run08-compile > run.log 2>&1
```

What changed:

- Added a guarded `apply_compile(model, compile_config)` call in `parallelize_qwen3()`.
- The hook only runs when `compile_config.enable` is true and `"model"` is in `compile_config.components`.
- The command enabled the existing TorchTitan compile config; it also compiled the loss because the default compile components are `["model", "loss"]`.
- No FSDP, AC, dtype, batch-size, sequence-length, optimizer, or parallelism changes.

Result:

- Status: keep, new current best.
- Step 10 `tps`: 7,545, up from 7,254.
- Step 10 MFU: 31.53%, up from 30.31%.
- Step 10 peak memory: 153.72 GiB, down from 173.91 GiB.
- Loss moved from 12.44588 at step 1 to 10.53577 at step 10; finite and decreasing.
- Later-step fwd/bwd time dropped to about 1.12-1.16 s from about 1.60-1.63 s in the uncompiled best.
- Optimizer time rose to about 35-41 ms from about 21-24 ms, but the fwd/bwd gain dominates.

Interpretation:

- Compile directly attacks the largest profile bucket and improves both throughput and memory. This is the new best source state.
- Lower memory headroom reopens command-only follow-ups that were too risky from the uncompiled best, especially BF16 dtype or a small local batch increase.

## Experiment 9: Compile With BF16 Training Dtype

Source state: `48c69e1`, using the `51369ea` compile-hook source.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run09-compile-bf16 > run.log 2>&1
```

What changed:

- Command-only override `--training.dtype=bfloat16` on top of the compiled best.
- No source, FSDP, AC, batch-size, sequence-length, optimizer, or parallelism changes.

Result:

- Status: keep, new current best.
- Step 10 `tps`: 8,168, up from 7,545.
- Step 10 MFU: 34.13%, up from 31.53%.
- Step 10 peak memory: 140.34 GiB, down from 153.72 GiB.
- Model memory after init: 4.18 GiB.
- Loss moved from 12.49057 at step 1 to 8.50811 at step 10; finite and decreasing.
- Later-step fwd/bwd remains around 1.13-1.16 s, similar to compiled float32 training, while memory is lower.

Interpretation:

- BF16 becomes throughput-positive once the model blocks are compiled. This is the best result so far and has much safer memory than the original FSDP-only path.
- The new memory headroom makes a local batch-size increase worth testing again, now under compile+BF16 where local batch 5 may fit.

## Experiment 10: Compile And BF16 With Local Batch 5

Source state: `f6ae44e`, using the `51369ea` compile-hook source.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run10-compile-bf16-lbs5 > run.log 2>&1
```

What changed:

- Command-only override `--training.local_batch_size=5` on top of compile+BF16.
- No source, FSDP, AC, sequence-length, optimizer, or parallelism changes.

Result:

- Status: keep, new current best.
- Step 10 `tps`: 8,391, up from 8,168.
- Step 10 MFU: 35.06%, up from 34.13%.
- Step 10 peak memory: 168.74 GiB, 94.61% of reported capacity.
- Loss moved from 12.51130 at step 1 to 10.05051 at step 10; finite and decreasing.
- Global valid tokens rose to 163,840 at step 10.
- Later-step fwd/bwd time was about 1.37-1.41 s versus about 1.13-1.16 s for local batch 4, but extra tokens per step improved reported tps.

Interpretation:

- Compile+BF16 created enough memory headroom to make local batch 5 viable and throughput-positive.
- Peak memory is close to the program's rough 95% risk threshold but still below it for this 10-step run.
- Local batch 6 is likely risky without another memory reduction. Before pushing memory further, profile the new best to see whether the bottleneck shifted after compile+BF16+larger batch.
