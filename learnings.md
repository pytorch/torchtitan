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

## Experiment 11: Profile Compile+BF16 Local Batch 5 Best

Source state: `da805d8`, using the `51369ea` compile-hook source.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run11-profile-compile-bf16-lbs5 --profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1 > run.log 2>&1
```

Result:

- Status in `results.tsv`: discard, because profiled throughput is diagnostic only.
- Step 10 profiled `tps`: 8,067.
- Step 10 profiled MFU: 33.70%.
- Step 10 peak memory: 168.74 GiB, 94.61%.
- Trace files were written under `outputs/autoresearch/may19-qwen3-14b/run11-profile-compile-bf16-lbs5/profiling/traces/iteration_10/`.

Profile notes from rank 0 trace:

- GPU traced duration excluding the `ProfilerStep` wrapper was about 6.09 s.
- Matmul/compiled kernels bucket: about 1.95 s.
- NCCL collectives: about 1.10 s.
- Flash attention: about 0.78 s.
- Copy/cat/split: about 0.16 s.
- Top events include compiled graph calls, flash attention backward, GEMM kernels, and FSDP reduce-scatter/all-gather.
- Later structured timings show fwd/bwd around 1.38-1.39 s and optimizer around 42-48 ms.

Roofline conclusion:

- The current best is still mixed compute/communication bound, but memory is now the immediate constraint: 94.61% leaves little room for larger batches or no-reshard.
- Local batch 6 is a reasonable boundary test because batch 5 improved throughput and remains barely below the memory warning threshold. It is expected to OOM unless compile memory behavior scales sublinearly.

## Experiment 12: Compile+BF16 Local Batch 6

Source state: `d26bddb`, using the `51369ea` compile-hook source.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=6 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run12-compile-bf16-lbs6 > run.log 2>&1
```

Result:

- Status: crash.
- The run initialized with local batch size 6, global batch size 48, and compile enabled for both loss and model blocks.
- It OOMed before completing step 1 in `ChunkedCELoss`, at `lm_head(h_chunk)`.
- The failing allocation was 892 MiB. The first-observed rank reported about 176.62 GiB already in use by the process, with only 117 MiB free on GPU 2.
- No throughput, MFU, loss, or peak-memory metric was emitted.

Interpretation:

- Local batch 5 is the practical memory boundary for the current FSDP+compile+BF16 setup.
- The OOM happens in the loss projection rather than in the optimizer, so increasing batch size further needs a real memory reduction or a different sharding strategy for the output projection/loss path.
- Current best remains the compile+BF16 local-batch-5 command at 8,391 tps and 35.06% MFU.

## Experiment 13: Compile Model Blocks Only

Source state: `620b4bb`, using the `51369ea` compile-hook source.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --compile.components '["model"]' --training.dtype=bfloat16 --training.local_batch_size=5 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run13-compile-model-only-bf16-lbs5 > run.log 2>&1
```

Result:

- Status: crash.
- The command parsed correctly and initialized with local batch size 5, global batch size 40, and model-block compile enabled.
- Loss compile was disabled, and the run OOMed before completing step 1 during `ChunkedCELoss` backward.
- The failing allocation was 1.45 GiB. Rank 0 reported about 177.63 GiB already in use, with 617 MiB free.
- No throughput, MFU, loss, or peak-memory metric was emitted.

Interpretation:

- Compiling the loss is not optional for the current best local-batch-5 setup; it appears to reduce memory enough for the first backward to fit.
- This also confirms that the loss/output projection path is the memory-critical boundary, not optimizer state.
- The next useful memory-reduction idea should shard the output projection/loss path rather than remove loss compile.

## Experiment 14: TP=2, FSDP=4, Compile+BF16 Local Batch 5

Source state: `4512daa`.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --parallelism.tensor_parallel_degree=2 --parallelism.data_parallel_shard_degree=4 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run14-tp2-fsdp4-compile-bf16-lbs5 > run.log 2>&1
```

What changed:

- Added Qwen3 dense TP sharding via the common decoder sharding helpers.
- Command used TP=2 and FSDP=4, with compile, BF16, loss parallel, and sequence parallel enabled.

Result:

- Status: discard for this command, because throughput is below the current best.
- Step 10 `tps`: 7,418, below the 8,391 current best.
- Step 10 MFU: 30.99%.
- Step 10 peak memory: 99.67 GiB, 55.88%.
- Loss moved from 3.10673 at step 1 to 2.08566 at step 10; finite and decreasing.
- Global batch size was 20 because TP does not contribute data parallelism.

Interpretation:

- TP=2 is functionally viable and substantially reduces memory.
- At the same local batch size, the extra TP communication and lower global batch lose throughput.
- The memory headroom is large enough to test local batch size 10 under TP=2, matching the no-TP best's global batch size of 40 while retaining output/loss sharding.

## Experiment 15: TP=2, FSDP=4, Compile+BF16 Local Batch 10

Source state: `a544471`, using the `4512daa` TP source.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=10 --parallelism.tensor_parallel_degree=2 --parallelism.data_parallel_shard_degree=4 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run15-tp2-fsdp4-compile-bf16-lbs10 > run.log 2>&1
```

Result:

- Status: crash.
- Step 1 completed with loss 3.09632 and peak memory 173.69 GiB, 97.38%.
- After step 1, the allocator repeatedly logged `expandable_segments: memory mapping failed with OOM` while trying to map 20 MiB segments.
- The run exited with rank 7 `SIGABRT`; other ranks were terminated by elastic.
- No step 10 metric was emitted.

Interpretation:

- TP=2 local batch 10 is above the practical memory boundary despite sharding the output/loss path.
- The crash happens after a completed first step, so it is likely allocator/headroom instability from running at 97%+ memory rather than a placement error.
- Local batch 8 is the next reasonable midpoint: it should use materially more TP headroom than batch 5 while staying below the batch 10 edge.

## Experiment 16: TP=2, FSDP=4, Compile+BF16 Local Batch 8

Source state: `30886e6`, using the `4512daa` TP source.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=8 --parallelism.tensor_parallel_degree=2 --parallelism.data_parallel_shard_degree=4 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run16-tp2-fsdp4-compile-bf16-lbs8 > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 7,850, below the 8,391 current best.
- Step 10 MFU: 32.80%.
- Step 10 peak memory: 148.39 GiB, 83.20%.
- Loss moved from 3.07666 at step 1 to 1.74183 at step 10; finite and decreasing.
- Global batch size was 32.

Interpretation:

- Increasing TP=2 from local batch 5 to 8 improved throughput from 7,418 to 7,850 and remained stable.
- The trend is not steep enough for local batch 9 to likely beat 8,391 before hitting the batch 10 memory cliff.
- TP communication/redistribution overhead remains too high without another TP-specific optimization.

## Experiment 17: Async TP on TP=2 Local Batch 8

Source state: `46e6f5e`.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=8 --parallelism.tensor_parallel_degree=2 --parallelism.data_parallel_shard_degree=4 --parallelism.enable_async_tensor_parallel --dump_folder=outputs/autoresearch/may19-qwen3-14b/run17-tp2-async-compile-bf16-lbs8 > run.log 2>&1
```

What changed:

- Added `maybe_enable_async_tp(parallelism, compile_config, tp_mesh)` to the Qwen3 TP path.
- Enabled `--parallelism.enable_async_tensor_parallel` on the stable TP=2 local-batch-8 command.

Result:

- Status: discard.
- Step 10 `tps`: 7,827, slightly below the non-async TP batch-8 run.
- Step 10 MFU: 32.70%.
- Step 10 peak memory: 148.39 GiB, 83.20%.
- Loss moved from 3.15246 at step 1 to 1.14390 at step 10; finite and decreasing.
- Inductor logged `async TP found no matching all-gather/reduce-scatter patterns for fusion`.

Interpretation:

- The async TP knob was enabled but did not find useful fusion patterns for this Qwen3 DTensor graph.
- TP=2 remains useful as a memory-reduction path but is not the throughput winner on this 8xB200, 10-step objective.
- Current best remains no-TP FSDP with compile, BF16, and local batch size 5: 8,391 tps, 35.06% MFU, 168.7 GiB.

## Experiment 18: Selective AC, Compile+BF16 Local Batch 6

Source state: `a593f3a`.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=6 --activation_checkpoint.mode=selective --dump_folder=outputs/autoresearch/may19-qwen3-14b/run18-selective-ac-compile-bf16-lbs6 > run.log 2>&1
```

What changed:

- Added the existing `apply_ac` hook to Qwen3 parallelization.
- Overrode Qwen3 14B's default full AC mode to selective AC.
- Raised local batch size from the current best's 5 to 6.

Result:

- Status: discard.
- Step 10 `tps`: 7,701, below the 8,391 current best.
- Step 10 MFU: 32.17%.
- Step 10 peak memory: 91.66 GiB, 51.40%.
- Loss moved from 12.34556 at step 1 to 8.54983 at step 10; finite and decreasing.

Interpretation:

- Selective AC fixes the local-batch-6 OOM and gives large memory headroom.
- At batch size 6, the recompute overhead dominates the extra tokens.
- The remaining question is whether a much larger local batch can amortize the selective AC overhead before hitting the memory limit.

## Experiment 19: Selective AC Local Batch 10, Invalid Due External GPU Use

Source state: `9b87a63`, using the `a593f3a` AC-hook source.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=10 --activation_checkpoint.mode=selective --dump_folder=outputs/autoresearch/may19-qwen3-14b/run19-selective-ac-compile-bf16-lbs10 > run.log 2>&1
```

Result:

- Status: invalid, not a candidate result.
- The run initialized with local batch size 10 and selective AC, then OOMed before step 1.
- The OOM log showed external `VLLM::Worker_TP*` processes from another user holding about 107 GiB on GPUs 0-3.
- Rank 2 failed trying to allocate 1.33 GiB while only 579 MiB was free on its GPU.

Interpretation:

- This run does not measure the candidate because the 8-GPU target was not available.
- Do not use it to conclude that selective AC local batch 10 is infeasible.
- Retry the same candidate after the node is free.

## Experiment 20: Selective AC Local Batch 10 Retry

Source state: `ad6265f`, using the `a593f3a` AC-hook source.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=10 --activation_checkpoint.mode=selective --dump_folder=outputs/autoresearch/may19-qwen3-14b/run20-selective-ac-compile-bf16-lbs10-retry > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 7,434, below the 8,391 current best.
- Step 10 MFU: 31.06%.
- Step 10 peak memory: 135.75 GiB, 76.11%.
- Loss moved from 12.37296 at step 1 to 14.42308 at step 10; this fails the short-run convergence sanity check.

Interpretation:

- Increasing local batch under eager selective AC did not amortize recompute overhead.
- The larger batch also produced an unhealthy short-run loss trend.
- Eager AC should not be kept as the best path. A compiler memory-budget AC mode may still be worth one test because it can choose a different rematerialization tradeoff under compile.

## Experiment 21: Memory-Budget AC 0.8, Compile+BF16 Local Batch 6

Source state: `b62adad`, using the `a593f3a` AC-hook source.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=6 --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.8 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run21-memory-budget-ac08-compile-bf16-lbs6 > run.log 2>&1
```

Result:

- Status: discard, but close to current best.
- Step 10 `tps`: 8,312, just below the 8,391 current best.
- Step 10 MFU: 34.73%.
- Step 10 peak memory: 146.23 GiB, 81.99%.
- Loss moved from 12.42556 at step 1 to 9.56584 at step 10; finite and decreasing.

Interpretation:

- Compiler memory-budget AC is much better than eager selective AC for this workload.
- Budget 0.8 makes local batch 6 fit and nearly matches the no-AC local-batch-5 best.
- Since memory is only 82%, budget 0.9 is the next direct test: less rematerialization may recover speed while still fitting.

## Experiment 22: Memory-Budget AC 0.9, Invalid Due External GPU Use

Source state: `14a0c41`, using the `a593f3a` AC-hook source.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=6 --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.9 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run22-memory-budget-ac09-compile-bf16-lbs6 > run.log 2>&1
```

Result:

- Status: invalid, not a candidate result.
- The run initialized memory-budget AC 0.9 and local batch size 6, then OOMed before step 1.
- The OOM log showed external `VLLM::Worker_TP*` processes from another user holding about 108-111 GiB on GPUs 4-7.
- The failing ranks reported this run using about 69 GiB, so the node was not measuring the candidate's real memory boundary.

Interpretation:

- Do not use this run to conclude budget 0.9 is infeasible.
- Retry memory-budget AC 0.9 after large external GPU allocations clear.

## Experiment 23: Memory-Budget AC 0.9 Retry, Invalid Due External GPU Use

Source state: `7ef6201`, using the `a593f3a` AC-hook source.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=6 --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.9 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run23-memory-budget-ac09-compile-bf16-lbs6-retry > run.log 2>&1
```

Result:

- Status: invalid, not a candidate result.
- The run OOMed before step 1.
- The OOM log showed a transient external process `3474354` holding 151.35 GiB on GPU 0, plus the small 616 MiB `train_perf_model` process.
- The external large process was gone by the post-run `nvidia-smi`.

Interpretation:

- This is another contaminated run. Do not use it to judge budget 0.9.
- Retry budget 0.9 once the node is clear.

## Experiment 24: Memory-Budget AC 0.9 Retry 2, Compile+BF16 Local Batch 6

Source state: `78002e0`, using the `a593f3a` AC-hook source.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=6 --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.9 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run24-memory-budget-ac09-compile-bf16-lbs6-retry2 > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 8,331, below the 8,391 current best.
- Step 10 MFU: 34.81%.
- Step 10 peak memory: 146.23 GiB, 81.99%.
- Loss moved from 12.23870 at step 1 to 6.82095 at step 10; finite and decreasing.

Interpretation:

- Memory-budget AC remains close but does not beat the no-AC local-batch-5 best.
- Budget 0.9 is only 19 tps faster than budget 0.8, and both report the same peak memory.
- The next direct test is budget 0.95: if the compiler saves fewer activations, it may recover the remaining overhead while still fitting.

## Experiment 25: Memory-Budget AC 0.95, Compile+BF16 Local Batch 6

Source state: `4cff4ed`, using the `a593f3a` AC-hook source.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=6 --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.95 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run25-memory-budget-ac095-compile-bf16-lbs6 > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 8,329, below the 8,391 current best.
- Step 10 MFU: 34.80%.
- Step 10 peak memory: 146.23 GiB, 81.99%.
- Loss moved from 12.26335 at step 1 to 11.98264 at step 10; finite and slightly decreasing.

Interpretation:

- Budgets 0.8, 0.9, and 0.95 all land around 146.2 GiB and 8.31-8.33k tps, so this knob is likely selecting the same or effectively equivalent rematerialization plan.
- The memory-budget AC line should be abandoned for now; its extra local batch does not overcome recompute/compile overhead.
- Restore source to the no-AC best path before testing non-AC ideas, because the AC hook is not part of the current best implementation.

## Experiment 26: BF16 Optimizer States, Invalid Due External GPU Use

Source state: `99a75e3`, with Qwen3 source restored to the no-AC compile+FSDP best path.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --optimizer.implementation=fused_opt_states_bf16 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run26-compile-bf16-lbs5-bf16-optimizer-states > run.log 2>&1
```

Result:

- Status: invalid, not a candidate result.
- The run OOMed during the first forward FSDP all-gather.
- The OOM log and post-run `nvidia-smi` showed external `VLLM::Worker_TP*` processes holding about 102 GiB on GPUs 4-7.

Interpretation:

- This does not evaluate BF16 optimizer states because the target 8-GPU node was not available.
- Retry the same optimizer-state candidate after large external GPU allocations clear.

## Manager Review After Experiment 26

Current best:

- Source state `f6ae44e`.
- Command shape: compile enabled, BF16 training dtype, local batch size 5, TP/CP/PP/EP disabled, FSDP across 8 B200s.
- Metrics: 8,391 tps, 35.06% MFU, 168.7 GiB peak memory, finite decreasing loss.

Search state:

- No-AC local batch 6 is blocked by loss-path OOM.
- TP=2 is functional and saves memory but is throughput-negative even with local batch 8 and async TP.
- Eager selective/full AC are too slow for the objective.
- Compiler memory-budget AC is close, but local-batch-6 budget tuning alone does not beat the current best; budgets 0.8, 0.9, and 0.95 all land around 146.2 GiB and 8.31-8.33k tps.
- The profile still supports a mixed compute/communication diagnosis: matmul kernels are the largest bucket, NCCL collectives are also material, and memory is the immediate limiter for larger no-AC batches.
- The BF16 optimizer-state run is invalid because external GPU users contaminated the memory result; it remains untested.

Next implications:

- Retry the BF16 optimizer-state candidate on a clear node before drawing a conclusion.
- Add one low-risk command-only runtime-overhead test on the current best by disabling NCCL flight recorder.
- Add one distinct compute-efficiency idea using the allowed FP8 linear converter, because AC and TP have not beaten the current best and GEMMs remain the largest profile bucket.

## Experiment 27: BF16 Optimizer States Retry

Source state: `51434eb`, with Qwen3 source restored to the no-AC compile+FSDP best path.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --optimizer.implementation=fused_opt_states_bf16 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run27-compile-bf16-lbs5-bf16-optimizer-states-retry > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 8,029, below the 8,391 current best.
- Step 10 MFU: 33.55%.
- Step 10 peak memory: 168.74 GiB, 94.61%.
- Loss moved from 12.35819 at step 1 to 11.80107 at step 10; finite and slightly decreasing.

Interpretation:

- BF16 optimizer states did not reduce reported peak memory on the steady-state step and slowed the run.
- Structured rank 0 timings regressed versus run10: average `fwd_bwd_end` rose from 1,419.67 ms to 1,533.56 ms, and average `optim_end` rose from 44.95 ms to 47.62 ms.
- The optimizer-state knob should be discarded. The next narrow no-source test is disabling the NCCL flight recorder with `--comm.trace_buf_size=0`.

## Experiment 28: Disable NCCL Flight Recorder

Source state: `cb767bf`, with Qwen3 source restored to the no-AC compile+FSDP best path.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run28-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 8,378, just below the 8,391 current best.
- Step 10 MFU: 35.00%.
- Step 10 peak memory: 168.74 GiB, 94.61%.
- Loss moved from 12.26960 at step 1 to 10.22765 at step 10; finite and decreasing.

Interpretation:

- Disabling the flight recorder is close but does not beat the best.
- The result suggests flight-recorder overhead is not the main limiter for this DP/FSDP run.
- Move to a distinct compute-efficiency idea: FP8 rowwise linear conversion on the current best command.

## Experiment 29: FP8 Rowwise Linear Converter

Source state: `5681e36`.

Source/config change:

- `qwen3_14b()` now passes `Float8LinearConverter.Config(recipe_name="rowwise", filter_fqns=["auto_filter_small_kn"], model_compile_enabled=True)` to `model_registry("14B", ...)`.
- Qwen3 `parallelize.py` remains on the no-AC compile+FSDP best path.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run29-fp8-rowwise-compile-bf16-lbs5 > run.log 2>&1
```

Result:

- Status: keep, new best.
- Step 10 `tps`: 8,429, above the previous 8,391 best.
- Step 10 MFU: 35.22%.
- Step 10 peak memory: 168.74 GiB, 94.61%.
- Loss moved from 12.22943 at step 1 to 11.01705 at step 10; finite and decreasing.

Interpretation:

- FP8 rowwise conversion gives a small but real throughput win without changing the memory boundary.
- Structured rank 0 timings show a modest end-to-end improvement versus run10: average `step_end` decreased from 2,440.60 ms to 2,429.64 ms, and last `step_end` decreased from 3,360.20 ms to 3,319.71 ms.
- Since memory is still the immediate limiter and FP8 did not reduce peak memory, local-batch-6 remains risky. The next low-risk follow-up is to add the previously close `--comm.trace_buf_size=0` command-only knob on top of the FP8 best.

## Experiment 30: Disable NCCL Flight Recorder on FP8 Best

Source state: `477f662`, using the FP8 rowwise converter source from `5681e36`.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run30-fp8-rowwise-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: keep, new best.
- Step 10 `tps`: 8,469, above the previous 8,429 FP8 best.
- Step 10 MFU: 35.39%.
- Step 10 peak memory: 168.74 GiB, 94.61%.
- Loss moved from 12.56713 at step 1 to 9.04841 at step 10; finite and decreasing.

Interpretation:

- Disabling the flight recorder is beneficial on the FP8 path even though it was just below best before FP8.
- Structured rank 0 timings improved versus run29: average `step_end` decreased from 2,429.64 ms to 2,418.11 ms, and last `step_end` decreased from 3,319.71 ms to 3,279.57 ms.
- The current best command is FP8 rowwise, compile, BF16 training dtype, local batch size 5, and `--comm.trace_buf_size=0`.
- The next distinct compute-efficiency variant is MXFP8 linear conversion on B200, replacing Float8 rowwise with the B200-targeted MXFP8 converter.

## Experiment 31: MXFP8 Cublas Recipe

Source state: `ed317a2`.

Source/config change:

- `qwen3_14b()` replaced `Float8LinearConverter` with `MXFP8LinearConverter.Config(recipe_name="mxfp8_cublas", model_compile_enabled=True)`.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run31-mxfp8-cublas-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: crash.
- The run failed during model build before training.
- Error: `ValueError: 'mxfp8_cublas' is not a valid MXFP8TrainingRecipe`.

Interpretation:

- This installed TorchAO build does not support the docs-mentioned `mxfp8_cublas` recipe.
- The MXFP8 line is not exhausted; retry with the valid default `mxfp8_rceil` recipe.

## Experiment 32: MXFP8 Rceil Recipe

Source state: `92e0502`.

Source/config change:

- `qwen3_14b()` used `MXFP8LinearConverter.Config(model_compile_enabled=True)`, which selects the default `mxfp8_rceil` recipe.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run32-mxfp8-rceil-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: crash.
- The run built the model and reached compiled forward, then failed before step 1 completed.
- Error: `RuntimeError: invalid argument` from `torch.ops.torchao.mxfp8_quantize.default(..., 'rceil')`.

Interpretation:

- MXFP8 is not currently runnable for this Qwen3 14B command and installed TorchAO/PyTorch stack.
- Restore to the FP8 rowwise best before continuing. The next FP8 variant is to remove `auto_filter_small_kn` and convert all dimension-compatible linear layers.

## Experiment 33: FP8 Rowwise Without Auto-Filter, Local Batch 5

Source state: `582d685`.

Source/config change:

- `qwen3_14b()` restored `Float8LinearConverter.Config(recipe_name="rowwise", model_compile_enabled=True)` without `filter_fqns=["auto_filter_small_kn"]`.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run33-fp8-rowwise-no-auto-filter-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 7,814, below the 8,469 current best.
- Step 10 MFU: N/A.
- Step 10 peak memory: 128.96 GiB, 72.31%.
- Loss moved from 12.42634 at step 1 to 10.49628 at step 10; finite and decreasing.
- The run emitted an FSDP warning that `FSDPFloat8Linear` returned a view tensor, which can be risky for in-place users.

Interpretation:

- Broad FP8 conversion is slower at the same batch size, so auto-filtering was important for throughput.
- However, broad conversion also reduces peak memory by about 39.8 GiB versus the current best, leaving room to test a larger local batch.
- The next direct follow-up is local batch size 8 on the no-auto-filter FP8 path to see whether extra tokens per step can amortize the broader-conversion overhead.

## Experiment 34: FP8 Rowwise Without Auto-Filter, Local Batch 8, Invalid Due External GPU Use

Source state: `a04a025`, using the no-auto-filter FP8 source from `582d685`.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=8 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run34-fp8-rowwise-no-auto-filter-compile-bf16-lbs8-no-flight-recorder > run.log 2>&1
```

Result:

- Status: invalid, not a candidate result.
- The run OOMed before step 1 completed.
- Post-run `nvidia-smi` showed external `VLLM::Worker_TP*` processes holding about 99 GiB on GPUs 4-7.
- The OOM messages reported the external process memory alongside the TorchTitan rank memory.

Interpretation:

- Do not use this to conclude local batch 8 is infeasible.
- Retry the same no-auto-filter FP8 local-batch-8 candidate after large external GPU allocations clear.

## Experiment 35: FP8 Rowwise Without Auto-Filter, Local Batch 8 Retry

Source state: `addc7f9`, using the no-auto-filter FP8 source from `582d685`.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=8 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run35-fp8-rowwise-no-auto-filter-compile-bf16-lbs8-no-flight-recorder-retry > run.log 2>&1
```

Result:

- Status: crash.
- The run OOMed on a clear node before step 1 completed.
- Failure occurred in the float8 `lm_head` loss path while converting the weight to float8, trying to allocate 2.90 GiB with about 2.86 GiB free.
- The failing ranks reported about 175.48 GiB in use.

Interpretation:

- Broad FP8 local batch 8 is just over the memory cliff, so it is infeasible without another memory-saving change.
- Local batch 7 is the next boundary to test because it may fit while using much more of the memory headroom than local batch 5.

## Experiment 36: FP8 Rowwise Without Auto-Filter, Local Batch 7, Invalid Due External GPU Use

Source state: `4c7d50e`, using the no-auto-filter FP8 source from `582d685`.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=7 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run36-fp8-rowwise-no-auto-filter-compile-bf16-lbs7-no-flight-recorder > run.log 2>&1
```

Result:

- Status: invalid, not a candidate result.
- The run OOMed before step 1 completed.
- Post-run `nvidia-smi` showed external `VLLM::Worker_TP*` processes holding about 99 GiB on GPUs 4-7.
- The OOM messages reported both the TorchTitan rank memory and the external VLLM process memory.

Interpretation:

- Do not use this to conclude local batch 7 is infeasible.
- Retry the same no-auto-filter FP8 local-batch-7 candidate on a clear node.

## Experiment 37: FP8 Rowwise Without Auto-Filter, Local Batch 7 Retry

Source state: `4549f88`, using the no-auto-filter FP8 source from `582d685`.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=7 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run37-fp8-rowwise-no-auto-filter-compile-bf16-lbs7-no-flight-recorder-retry > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 8,413, below the 8,469 current best.
- Step 10 MFU: N/A.
- Step 10 peak memory: 171.18 GiB, 95.98%.
- Loss moved from 12.47352 at step 1 to 11.22023 at step 10; finite and decreasing.

Interpretation:

- Local batch 7 fits, but it is slower than the auto-filter FP8 best and above the 95% memory-risk threshold.
- The broad FP8 path should be abandoned: batch 5 is slow, batch 7 is still slower and memory-risky, and batch 8 OOMs.
- Restore source to the FP8 auto-filter best before continuing. The next useful step is to profile the current best command.

## Experiment 38: Profile Current FP8 Best

Source state: `91f4e9d`, using the FP8 rowwise auto-filter source from `5681e36`.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run38-profile-fp8-rowwise-best --profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1 > run.log 2>&1
```

Result:

- Status: diagnostic discard; profiled throughput is not compared against unprofiled candidates.
- Step 10 `tps`: 7,863.
- Step 10 MFU: 32.85%.
- Step 10 peak memory: 168.74 GiB, 94.61%.
- Loss moved from 12.26053 at step 1 to 7.70341 at step 10; finite and decreasing.
- Traces were written under `outputs/autoresearch/may19-qwen3-14b/run38-profile-fp8-rowwise-best/profiling/traces/iteration_10/`.

Profile notes from rank 0 trace:

- CUDA kernel total: about 3.57 s for the profiled step.
- Flash attention backward is the largest single kernel family at about 599 ms; flash attention forward adds about 156 ms.
- NCCL kernels total about 693 ms (`ReduceScatter` about 351 ms, `AllGather` about 342 ms); NCCL annotations total about 526 ms.
- B200 nvjet GEMM kernels remain a large bucket, with several kernels totaling well over 1.6 s.
- Optimizer is small in the trace, about 7 ms of GPU annotation.

Interpretation:

- The best is still mixed compute/communication-bound, with attention now large enough to justify one attention-backend experiment.
- Since TP and AC lines failed to beat the best and memory remains near the risk line, the next narrow source/config test is `attn_backend="flex_flash"` while keeping FP8 auto-filtering and the best command.

## Experiment 39: FP8 Best With Flex Flash Attention Backend

Source state: `ad43e66`.

Source/config change:

- `qwen3_14b()` passed `attn_backend="flex_flash"` to `model_registry("14B", ...)`, keeping the FP8 rowwise auto-filter converter.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run39-fp8-flex-flash-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: crash.
- The run failed during Inductor lowering before step 1 completed.
- Error: `BACKEND='FLASH' but flash attention cannot be used: CUTE flash attention library is not available`.

Interpretation:

- The `flex_flash` backend is not runnable in this environment because the needed CUTE flash attention library is unavailable.
- Attention remains a profile-visible bottleneck, so the next attention-backend test should avoid the CUTE flash path. Try Qwen3 `attn_backend="varlen"`.

## Experiment 40: FP8 Best With Varlen Attention Backend

Source state: `06f1c41`.

Source/config change:

- `qwen3_14b()` passed `attn_backend="varlen"` to `model_registry("14B", ...)`, keeping the FP8 rowwise auto-filter converter.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run40-fp8-varlen-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: crash.
- The run failed during model build before training.
- Error: `ModuleNotFoundError: No module named 'flash_attn_interface'` while `VarlenAttention` tried to activate FA3.

Interpretation:

- The `varlen` backend is not runnable in this environment because FA3's Python interface is missing.
- `flex_flash` and `varlen` are both blocked by missing attention libraries. The remaining attention backend test is plain `attn_backend="flex"`, which avoids explicitly requesting the unavailable flash backend.

## Experiment 41: FP8 Best With Flex Attention Backend

Source state: `67967a3`.

Source/config change:

- `qwen3_14b()` passed `attn_backend="flex"` to `model_registry("14B", ...)`, keeping the FP8 rowwise auto-filter converter.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run41-fp8-flex-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 8,544, above the 8,469 current best.
- Step 10 MFU: 35.70%.
- Step 10 peak memory: 168.96 GiB, 94.73%.
- Loss moved from 12.41705 at step 1 to 13.88057 at step 10; finite but increasing.

Interpretation:

- Plain flex attention is faster, but it fails the short-run convergence sanity check.
- Do not keep it despite the higher throughput. The loss regression may reflect mask/backend numerical or semantic differences.
- Restore source to the FP8 auto-filter `sdpa` best before continuing.

## Experiment 42: FP8 Rowwise With High-Precision Grad Recipe

Source state: `17c8bd9`.

Source/config change:

- `qwen3_14b()` changed the Float8 converter recipe from `rowwise` to `rowwise_with_gw_hp`, keeping `filter_fqns=["auto_filter_small_kn"]`.
- The source was also restored from the prior flex-attention candidate back to the default Qwen3 attention backend.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run42-fp8-rowwise-gw-hp-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: crash.
- The run failed during config/model-spec construction before training.
- Error: `_auto_filter_for_recipe` raised `NotImplementedError: Unsupported recipe: Float8LinearRecipeName.ROWWISE_WITH_GW_HP`.

Interpretation:

- The installed torchao auto-filter path only supports the default rowwise recipe here.
- Since `rowwise_with_gw_hp` cannot be combined with the profitable `auto_filter_small_kn` coverage, do not keep this source line. Restore the current best FP8 rowwise auto-filter recipe before continuing.
- With quantization coverage, MXFP8, and attention-backend variants now exhausted or blocked, the next narrow search area is overhead reduction in the current best command. Structured trace logging is enabled by default and is a command-only knob worth testing separately from NCCL flight-recorder logging.

## Experiment 43: FP8 Best With Structured Logging Disabled

Source state: `748f640`.

Source/config change:

- None. Source was the restored FP8 rowwise auto-filter best.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --debug.no-enable-structured-logging --dump_folder=outputs/autoresearch/may19-qwen3-14b/run43-fp8-no-structured-logging-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 8,439, below the 8,469 current best.
- Step 10 MFU: 35.26%.
- Step 10 peak memory: 168.74 GiB, 94.61%.
- Loss moved from 12.51420 at step 1 to 11.45786 at step 10; finite and decreasing.
- Log confirmed structured logging was disabled via `DebugConfig.enable_structured_logging=False`.

Interpretation:

- Structured trace logging is not the source of the remaining gap; disabling it slightly regressed reported throughput.
- Keep structured logging at the default enabled value for future runs.
- The current best remains FP8 rowwise auto-filter with NCCL flight recorder disabled.

## Experiment 44: FP8 Best With FSDP Reshard-After-Forward Disabled

Source state: `401de15`.

Source/config change:

- None. Source was the restored FP8 rowwise auto-filter best.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --parallelism.fsdp_reshard_after_forward=never --dump_folder=outputs/autoresearch/may19-qwen3-14b/run44-fp8-no-reshard-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: crash.
- The run OOMed during the first-step loss backward before any step metrics were emitted.
- Error: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.90 GiB`.
- Rank 0 reported 175.35 GiB in the training process plus the known 616 MiB small `train_perf_model` process.
- Rank 4 also reported 175.35 GiB in the training process plus a 1.73 GiB external Python process. No external process exceeded the 5 GiB contamination threshold.

Interpretation:

- Disabling FSDP reshard-after-forward at local batch size 5 is not viable; the extra parameter/gradient residency exhausts the B200 memory margin.
- The failure happened close to fitting on ranks with only small external allocations, so a local-batch-size-4 follow-up is a reasonable communication/memory tradeoff. It must overcome the smaller global batch to beat 8,469 tps.

## Experiment 45: FP8 No-Reshard With Local Batch Size 4

Source state: `d8ba324`.

Source/config change:

- None. Source was the restored FP8 rowwise auto-filter best.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=4 --comm.trace_buf_size=0 --parallelism.fsdp_reshard_after_forward=never --dump_folder=outputs/autoresearch/may19-qwen3-14b/run45-fp8-no-reshard-compile-bf16-lbs4-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 6,939, far below the 8,469 current best.
- Step 10 MFU: 28.99%.
- Step 10 peak memory: 165.71 GiB, 92.91%.
- Loss moved from 12.46997 at step 1 to 7.50736 at step 10; finite and decreasing.

Interpretation:

- Lowering local batch size gives no-reshard enough memory to run, but the communication savings do not offset the smaller global batch and no-reshard overhead.
- Do not pursue no-reshard variants unless a later source change creates substantially more memory headroom at local batch size 5 or higher.
- The next best-supported branch is flex-attention correctness isolation because run41 was faster than the current best but failed loss sanity.

## Experiment 46: Flex Attention Without FP8 Converter

Source state: `5801b0f`.

Source/config change:

- `qwen3_14b()` changed `model_registry("14B", ...)` to pass `attn_backend="flex"`.
- Removed the FP8 rowwise auto-filter converter from the Qwen3 14B model spec for this isolation test.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run46-flex-no-fp8-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: keep.
- Step 10 `tps`: 8,489, above the previous 8,469 best.
- Step 10 MFU: 35.47%.
- Step 10 peak memory: 168.96 GiB, 94.73%.
- Loss moved from 12.36098 at step 1 to 11.82755 at step 10; finite and decreasing.

Interpretation:

- Flex attention itself is compatible with the 10-step loss sanity check.
- The invalid run41 behavior came from the FP8+flex combination or its update dynamics, not from plain flex attention.
- The new current best is flex attention without the FP8 converter, with compile+BF16, local batch size 5, and NCCL flight recorder disabled.
- Since run41's FP8+flex path was still faster at 8,544 tps but invalid, the next narrow follow-up is to test whether lowering LR recovers the loss trend while preserving the faster FP8+flex kernels.

## Experiment 47: FP8 Flex Attention With Lower Learning Rate

Source state: `be0f401`.

Source/config change:

- `qwen3_14b()` restored the FP8 rowwise auto-filter converter while keeping `attn_backend="flex"`.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --optimizer.lr=4e-4 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run47-fp8-flex-lr4e-4-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 8,432, below the 8,489 current best.
- Step 10 MFU: 35.23%.
- Step 10 peak memory: 168.10 GiB, 94.25%.
- Loss moved from 12.55363 at step 1 to 11.81441 at step 10; finite and decreasing.

Interpretation:

- Lowering LR recovered the short-run loss trend for FP8+flex, so run41's loss failure is update-size-sensitive or numerically marginal rather than an immediate semantic failure.
- The recovered FP8+flex run is slower than the flex-without-FP8 best, so do not keep this source/command.
- Restore the flex-without-FP8 source as the current best before continuing.

## Experiment 48: FP8 Flex Attention With Intermediate Learning Rate

Source state: `a5a7276`.

Source/config change:

- `qwen3_14b()` used `attn_backend="flex"` plus the FP8 rowwise auto-filter converter.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --optimizer.lr=6e-4 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run48-fp8-flex-lr6e-4-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 8,325, below the 8,489 current best.
- Step 10 MFU: 34.78%.
- Step 10 peak memory: 168.10 GiB, 94.25%.
- Loss moved from 12.54173 at step 1 to 9.93097 at step 10; finite and decreasing.

Interpretation:

- LR 6e-4 also recovers the FP8+flex loss trend, but throughput is even lower than LR 4e-4 in this run.
- The faster invalid run41 does not translate into a keepable FP8+flex configuration under the LR recovery tests tried so far.
- Restore the flex-without-FP8 source as the current best.

## Experiment 49: Profile Current Flex-Attention Best

Source state: `9f96d09`.

Source/config change:

- None. Source was the current flex-without-FP8 best.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run49-profile-flex-best --profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1 > run.log 2>&1
```

Result:

- Status: diagnostic discard; profiled throughput is not ranked against unprofiled candidates.
- Step 10 `tps`: 8,246.
- Step 10 MFU: 34.45%.
- Step 10 peak memory: 168.10 GiB, 94.25%.
- Loss moved from 12.39286 at step 1 to 7.64441 at step 10; finite and decreasing.
- Traces were written under `outputs/autoresearch/may19-qwen3-14b/run49-profile-flex-best/profiling/traces/iteration_10/`.

Profile notes from rank 0 trace:

- CUDA kernel total: about 3.75 s for the profiled step.
- B200 nvjet GEMM kernels total about 1.80 s and remain the largest bucket.
- NCCL kernels total about 0.94 s: reduce-scatter about 491 ms and all-gather about 450 ms. NCCL GPU annotations total about 731 ms.
- Flex-attention kernels total about 659 ms, dominated by `triton_tem_fused_flex_attention_backward_transpose_view_3` at about 563 ms and a forward/transpose flex kernel at about 92 ms.
- Optimizer remains small: optimizer GPU annotation about 7.6 ms and optimizer-like kernels about 39 ms.

Interpretation:

- Compared with the prior FP8-best profile, flex attention improved the kept throughput but did not remove the main compute/communication limits.
- No-reshard is not viable at local batch size 5 and too slow at local batch size 4, so the NCCL bucket is difficult to attack with the already-tested FSDP policy knob.
- GEMM remains the biggest profile bucket. The remaining in-scope source/config ideas should focus on compute kernels or memory headroom rather than logging or optimizer overhead.

## Experiment 50: Flex Attention With Rowwise High-Precision FP8 Recipe Without Auto-Filter

Source state: `bca5b86`.

Source/config change:

- `qwen3_14b()` used `attn_backend="flex"` plus `Float8LinearConverter.Config(recipe_name="rowwise_with_gw_hp", model_compile_enabled=True)`.
- No `auto_filter_small_kn` was used, avoiding the unsupported auto-filter path from run42.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run50-flex-fp8-rowwise-gw-hp-no-filter-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 6,226, far below the 8,489 current best.
- Step 10 MFU: N/A.
- Step 10 peak memory: 172.36 GiB, 96.64%.
- Loss moved from 12.26059 at step 1 to 8.27428 at step 10; finite and decreasing.
- Runtime emitted an FSDP2 warning that `FSDPFloat8Linear` returned a view tensor, which can drop pre-backward hooks under in-place ops.
- Runtime also emitted six CUDA allocator retry warnings.

Interpretation:

- The high-precision FP8 recipe can run without auto-filter, but it is much slower and memory-risky.
- The view-tensor FSDP warning makes this source line unattractive even though the short loss trend decreases.
- Restore the flex-without-FP8 source as the current best.

## Experiment 51: Flex Attention Best With Local Batch Size 6

Source state: `4e5b73f`.

Source/config change:

- None. Source was the current flex-without-FP8 best.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=6 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run51-flex-compile-bf16-lbs6-no-flight-recorder > run.log 2>&1
```

Result:

- Status: crash.
- The run OOMed during first-step loss/backward before a step metric was emitted.
- Rank 0 failed in `lm_head(h_chunk)` trying to allocate 892 MiB with only 683.5 MiB free.
- Other ranks failed in FSDP backward all-gather trying to allocate 1.45 GiB with about 1.11 GiB free.
- The training process held about 176.95-177.22 GiB, with only the known 616 MiB small `train_perf_model` process visible externally.

Interpretation:

- Current flex best cannot directly raise local batch size to 6; the memory headroom at local batch size 5 is not enough.
- Any further batch-size increase needs memory relief, for example activation checkpointing or a different sharding/parallelism layout.

## Experiment 52: Flex Attention With Memory-Budget AC And Local Batch Size 7

Source state: `7d49b01`.

Source/config change:

- Reintroduced the standard TorchTitan `apply_ac` hook in Qwen3 parallelization before model compile.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=7 --comm.trace_buf_size=0 --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.9 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run52-flex-memory-budget-ac09-compile-bf16-lbs7-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 7,840, below the 8,489 current best.
- Step 10 MFU: 32.76%.
- Step 10 peak memory: 167.27 GiB, 93.79%.
- Loss moved from 12.39671 at step 1 to 10.41591 at step 10; finite and decreasing.

Interpretation:

- Memory-budget AC allows a larger batch to fit, but local batch size 7 is still slower than the no-AC flex best.
- Since peak memory remains below the no-AC best and below the 95% risk line, test local batch size 8 once before abandoning this AC branch.

## Experiment 53: Flex Attention With Memory-Budget AC And Local Batch Size 8

Source state: `2ec5fb2`.

Source/config change:

- Same `apply_ac` hook source as run52.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=8 --comm.trace_buf_size=0 --activation_checkpoint.mode=memory_budget --activation_checkpoint.memory_budget=0.9 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run53-flex-memory-budget-ac09-compile-bf16-lbs8-no-flight-recorder > run.log 2>&1
```

Result:

- Status: crash.
- The run OOMed during first-step backward in compiled flex-attention backward.
- Error: `RuntimeError: CUDA driver error: out of memory`.
- Only small external allocations were visible after the run: 616 MiB `train_perf_model` and a 1.5 GiB Python process on one GPU.

Interpretation:

- Memory-budget AC 0.9 cannot scale the flex best to local batch size 8.
- The local batch 7 result was already far below the best, so do not keep the AC hook or continue this AC batch-scaling branch for now.
- Restore the no-AC flex-without-FP8 source as the current best.

## Experiment 54: Flex Attention Best With Inductor GEMM Max Autotune

Source state: `f2eba58`.

Source/config change:

- None. Source was the current flex-without-FP8 best.

Command:

```bash
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1 NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run54-flex-inductor-max-autotune-gemm-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 4,816, far below the 8,489 current best.
- Step 10 MFU: 20.12%.
- Step 10 peak memory: 171.79 GiB, 96.32%.
- Loss moved from 12.34380 at step 1 to 9.00300 at step 10; finite and decreasing.
- Runtime emitted nine CUDA allocator retry warnings.
- Autotune logs show many GEMM choices were benchmarked, often selecting `mm` over Triton choices.

Interpretation:

- Inductor GEMM max autotuning is counterproductive for this workload and increases memory risk.
- Do not continue nearby Inductor GEMM autotune sweeps unless a later profile or source change materially changes the compiled matmul shapes.

## Experiment 55: Flex Attention With Context Parallel Degree 2

Source state: `c36ca11`.

Source/config change:

- Allowed CP in Qwen3 parallelization and called `apply_cp_to_forward` on each layer's inner attention before compile.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=10 --comm.trace_buf_size=0 --parallelism.context_parallel_degree=2 --parallelism.context_parallel_load_balancer=ptrr --dump_folder=outputs/autoresearch/may19-qwen3-14b/run55-flex-cp2-ptrr-compile-bf16-lbs10-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 4,531, below the 8,489 current best.
- Step 10 MFU: 18.93%.
- Step 10 peak memory: 170.21 GiB, 95.44%.
- Loss moved from 12.30233 at step 1 to 17.01370 at step 10; finite but increasing.
- Runtime emitted nine CUDA allocator retry warnings.

Interpretation:

- CP=2 with flex attention is slower, more memory-risky, and fails the short loss sanity check.
- Do not keep this CP source line. Restore the no-CP flex-without-FP8 source as the current best.

## Experiment 56: Flex Attention Best With Structured Logging Disabled

Source state: `b29e75a`.

Source/config change:

- None. Source was the current flex-without-FP8 best.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --debug.no-enable-structured-logging --dump_folder=outputs/autoresearch/may19-qwen3-14b/run56-flex-no-structured-logging-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 7,455, below the 8,489 current best.
- Step 10 MFU: 31.15%.
- Step 10 peak memory: 168.10 GiB, 94.25%.
- Loss moved from 12.41723 at step 1 to 19.26537 at step 10; finite but increasing.
- Log confirmed structured logging was disabled via `DebugConfig.enable_structured_logging=False`.

Interpretation:

- Disabling structured logging does not improve the flex best and fails the loss sanity check in this run.
- Keep structured logging enabled for future candidates.

## Experiment 57: Flex Attention Best With Fixed Debug Seed

Source state: `ba3af10`.

Source/config change:

- None. Source was the current flex-without-FP8 best.

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --debug.seed=42 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run57-flex-seed42-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 5,899, below the 8,489 current best.
- Step 10 MFU: 24.65%.
- Step 10 peak memory: 168.10 GiB, 94.25%.
- Loss moved from 12.66785 at step 1 to 13.76234 at step 10; finite but increasing.

Interpretation:

- Fixed seed 42 does not make the flex best a better command; it fails the short loss sanity check and is much slower in this run.
- Recent flex runs show high measurement and loss-trend variance. Rerun the exact current-best command once to estimate variance before adding more speculative knobs.

## Manager Review After Experiment 29

Current best:

- Source state `5681e36`.
- Command shape: FP8 rowwise linear converter, compile enabled, BF16 training dtype, local batch size 5, TP/CP/PP/EP disabled, FSDP across 8 B200s.
- Metrics: 8,429 tps, 35.22% MFU, 168.74 GiB peak memory, finite decreasing loss.

Search state:

- FP8 rowwise is the first successful compute-efficiency source change after compile+BF16. It improves throughput by 38 tps over run10 without changing the memory boundary.
- Peak memory remains 94.61%, so no-AC local batch 6 is still expected to be risky. FP8 did not create meaningful batch-size headroom.
- Run28 showed disabling NCCL flight recorder was close on the pre-FP8 best but did not win. Retesting it on the FP8 best is still valid as a single command-only follow-up, but if it does not win, flight-recorder overhead should be deprioritized.
- The profile diagnosis should be refreshed after FP8 because the kernel mix changed. A profile should be diagnostic only and should not be ranked against unprofiled runs.

Next implications:

- After the queued FP8 + `--comm.trace_buf_size=0` run, profile whichever FP8 command is best unless the result clearly points to a different immediate bottleneck.
- Try a B200-native MXFP8 linear converter as a distinct quantization hypothesis; docs claim MXFP8 is intended for Blackwell and can outperform plain FP8 on dense GEMMs.
- If MXFP8 is not viable or is slower, refine the kept FP8 rowwise converter by changing conversion coverage rather than changing unrelated training knobs.

## Manager Review After Experiment 43

Current best:

- Source state: run30 FP8 rowwise auto-filter best.
- Command shape: compile enabled, BF16 training dtype, local batch size 5, `--comm.trace_buf_size=0`, TP/CP/PP/EP disabled, FSDP across 8 B200s.
- Metrics: 8,469 tps, 35.39% MFU, 168.7 GiB peak memory, finite decreasing loss.

Search state:

- Quantization: FP8 rowwise auto-filter is the kept source line. MXFP8 is blocked on this installed stack, broad FP8 conversion is slower, and `rowwise_with_gw_hp` is unsupported with auto-filter.
- Attention: `flex_flash` and `varlen` are blocked by missing libraries. Plain `flex` is faster but failed convergence sanity.
- Batch/memory: memory remains near the 95% risk line, and prior local-batch increases either OOMed or lost throughput.
- Overhead: disabling NCCL flight recorder helped and is part of the best; disabling structured logging did not help.

Next implications:

- Do not continue logging-overhead or nearby FP8 recipe sweeps.
- Prioritize one narrow follow-up on the run41 flex attention branch, because it is the only measured path above the current best.
- If flex remains correctness-bad after isolating FP8 and LR/update-size effects, return to profiling or communication-layout ideas rather than keeping a fast invalid candidate.

## Manager Review After Experiment 57

Current best:

- Source state: `5801b0f`.
- Command shape: flex attention, no FP8 converter, compile enabled, BF16 training dtype, local batch size 5, `--comm.trace_buf_size=0`, no AC/TP/CP/PP/EP, FSDP across 8 B200s.
- Metrics: 8,489 tps, 35.47% MFU, 168.96 GiB peak memory, finite decreasing loss.

Search state:

- The current flex profile is still mixed compute/communication: about 1.80 s nvjet GEMMs, 0.94 s NCCL kernels, and 0.66 s flex-attention kernels on rank 0.
- Direct memory-to-batch conversion is closed for now: no-AC local batch size 6 OOMed, memory-budget AC local batch size 7 was too slow, and local batch size 8 OOMed.
- Whole-model communication tradeoffs are closed for now: full no-reshard OOMed at local batch size 5 and was much slower at local batch size 4; CP=2 was slower, memory-risky, and failed loss sanity.
- Nearby runtime overhead and compiler knobs are not promising: structured logging disabled regressed badly, and Inductor GEMM max autotune was much slower with allocator retries.
- FP8+flex remains interesting only as a selective-coverage idea. Lowering LR made loss decrease but did not beat the BF16 flex best, and high-precision FP8 without auto-filter was both slow and memory-risky.

Next implications:

- First priority: isolate the loss-path communication issue with an `lm_head`-only no-reshard FSDP policy. This is much narrower than whole-model no-reshard because `ChunkedCELoss` calls the separately wrapped `lm_head` repeatedly.
- Second priority: test explicit FSDP module prefetch on the current flex best to overlap all-gathers with compute without changing the batch, AC, or converter state.
- Third priority: revisit FP8+flex only by excluding `lm_head` from Float8 conversion while keeping `auto_filter_small_kn`; this tests whether quantized logits caused the run41 loss failure.
- The fixed-seed diagnostic has now been tried and discarded at 5,899 tps with increasing loss. The queued exact-best rerun is a reasonable variance check before the next source change.

## Experiment 58: Flex Attention With `lm_head`-Only No-Reshard

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run58-flex-lm-head-no-reshard-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 6,922, below the 8,489 current best.
- Step 10 MFU: 28.92%.
- Step 10 peak memory: 168.10 GiB, 94.25%.
- Loss moved from 12.32550 at step 1 to 7.94975 at step 10; finite and decreasing.

Interpretation:

- Keeping only `lm_head` unresharded after forward is materially slower than the flex best, despite preserving loss sanity.
- Do not pursue `lm_head` no-reshard variants on this source line. The next communication idea should use overlap/prefetch rather than extra parameter residency.

## Experiment 59: Flex Attention With Explicit FSDP Prefetch

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run59-flex-fsdp-prefetch-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: keep; new best.
- Step 10 `tps`: 8,835, above the prior 8,489 best.
- Step 10 MFU: 36.91%.
- Step 10 peak memory: 168.10 GiB, 94.25%.
- Loss moved from 12.19318 at step 1 to 6.47119 at step 10; finite and decreasing.

Interpretation:

- Explicit one-module-ahead FSDP prefetch is the first communication-overlap change to beat the flex best.
- Current best source is now `7c1c351`, with flex attention, no FP8 converter, compile enabled, BF16 training dtype, local batch size 5, `--comm.trace_buf_size=0`, and Qwen3 FSDP prefetch.
- Next tests should keep this prefetch source line and isolate either variance, selective FP8 coverage, or a narrower prefetch refinement.

## Experiment 60: Selective FP8 With BF16 `lm_head` On Prefetch Source

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run60-prefetch-flex-fp8-bf16-lm-head-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: invalid.
- The run OOMed before step metrics.
- External `VLLM::Worker_TP*` processes appeared on GPUs 4-7 during the run and held about 127 GiB each at failure time.

Interpretation:

- This is a contaminated OOM, not evidence against selective FP8. Retry the same source when GPUs are clear.

Retry command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run61-prefetch-flex-fp8-bf16-lm-head-compile-bf16-lbs5-no-flight-recorder-retry > run.log 2>&1
```

Retry result:

- Status: discard.
- Step 10 `tps`: 8,762, below the 8,835 current best.
- Step 10 MFU: 36.61%.
- Step 10 peak memory: 168.10 GiB, 94.25%.
- Loss moved from 12.58068 at step 1 to 7.49573 at step 10; finite and decreasing.

Retry interpretation:

- Keeping `lm_head` in BF16 fixes the short-run loss issue seen in the faster FP8+flex run41, but selective FP8 still does not beat the prefetch-only source.
- Restore the no-FP8 prefetch source and avoid further nearby FP8 coverage tweaks unless a profile shows a specific dense-GEMM bottleneck.

## Experiment 62: Exact Rerun Of Prefetch Flex Best

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run62-rerun-prefetch-flex-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard as a variance check; it did not beat run59.
- Step 10 `tps`: 8,829, below the 8,835 current best by 6 tps.
- Step 10 MFU: 36.89%.
- Step 10 peak memory: 168.10 GiB, 94.25%.
- Loss moved from 12.31141 at step 1 to 6.95173 at step 10; finite and decreasing.

Interpretation:

- The prefetch best is reproducible within a narrow range: run59 at 8,835 tps and run62 at 8,829 tps.
- Keep source state `7c1c351` as the current best. The next useful step is profiling this prefetch source or testing a narrower prefetch variant, not repeating the exact command again immediately.

## Experiment 63: Profile Prefetch Flex Best

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run63-profile-prefetch-flex-best --profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1 > run.log 2>&1
```

Result:

- Status: diagnostic discard; profiling overhead means this is not ranked against unprofiled candidates.
- Step 10 `tps`: 8,484.
- Step 10 MFU: 35.45%.
- Step 10 peak memory: 168.10 GiB, 94.25%.
- Loss moved from 12.46043 at step 1 to 6.99815 at step 10; finite and decreasing.
- Traces were generated under `outputs/autoresearch/may19-qwen3-14b/run63-profile-prefetch-flex-best/profiling/traces/iteration_10/`.

Rank 0 trace summary:

- Profiled wall step: 5.66 s.
- Total CUDA kernel time: 5.61 s.
- NCCL kernels: 2.40 s total, including 1.76 s reduce-scatter and 0.63 s all-gather.
- nvjet GEMM kernels: 2.05 s.
- Flex attention kernels: 0.78 s.
- Compared with run49 before explicit prefetch, rank0 profiled wall step fell from 5.77 s to 5.66 s even though visible NCCL kernel time rose from 0.94 s to 2.40 s, which indicates more communication is being exposed/overlapped rather than simply removed.

Interpretation:

- The remaining profile is mixed GEMM plus communication. Reduce-scatter is now the largest visible NCCL bucket, while all-gather is smaller and likely partly hidden by prefetch.
- Next source experiments should refine prefetch scheduling or reduce gradient communication exposure; nearby FP8 coverage did not beat the best.

## Experiment 64: Two-Module FSDP Prefetch Window

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run64-flex-two-module-prefetch-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 8,627, below the 8,835 current best.
- Step 10 MFU: 36.04%.
- Step 10 peak memory: 169.33 GiB, 94.94%.
- Loss moved from 12.38179 at step 1 to 5.68745 at step 10; finite and decreasing.

Interpretation:

- A two-module prefetch window is too aggressive: it increases peak memory and slows throughput.
- Restore the one-module prefetch schedule. If refining prefetch further, prefer narrower/asymmetric schedules rather than wider windows.

## Experiment 65: Forward-Only FSDP Prefetch

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run65-flex-forward-only-prefetch-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 8,596, below the 8,835 current best.
- Step 10 MFU: 35.92%.
- Step 10 peak memory: 168.10 GiB, 94.25%.
- Loss moved from 12.51793 at step 1 to 12.50715 at step 10; finite and slightly decreasing.

Interpretation:

- Backward prefetch is contributing to the current best despite the visible reduce-scatter bucket. Removing it regresses throughput.
- Restore the one-module bidirectional prefetch schedule as the best known prefetch policy.

## Manager Review After Experiment 65

Current best:

- Source state: `7c1c351`.
- Command shape: flex attention, no FP8 converter, compile enabled, BF16 training dtype, local batch size 5, `--comm.trace_buf_size=0`, no AC/TP/CP/PP/EP, FSDP across 8 B200s, one-module FSDP prefetch.
- Metrics: run59 at 8,835 tps, 36.91% MFU, 168.10 GiB peak memory, finite decreasing loss. Exact rerun run62 was 8,829 tps with finite decreasing loss, so the improvement is reproducible.

Search state:

- Run59's prefetch schedule is the first strong communication-overlap win after flex attention.
- Run63 profile shows the remaining bottleneck is now dominated by NCCL reduce-scatter plus GEMM: about 2.40 s NCCL kernels on rank 0, including 1.76 s reduce-scatter and 0.63 s all-gather, versus 2.05 s nvjet GEMM and 0.78 s flex attention.
- Wider two-module prefetch is too aggressive: run64 slowed to 8,627 tps and raised memory to 169.33 GiB.
- Forward-only prefetch is worse at 8,596 tps, so backward prefetch is important and should remain part of the best policy unless a narrower complementary test proves otherwise.
- Selective FP8 with BF16 `lm_head` fixed the short-run loss trend but still reached only 8,762 tps, so FP8 coverage is lower priority than communication reduction.

Next implications:

- Highest-priority next idea: BF16 FSDP reduce dtype on the run59 prefetch source. This directly attacks the 1.76 s reduce-scatter bucket and is more likely to help than another prefetch-window width change.
- Second priority: HSDP 2x4 on the prefetch source. This changes FSDP collective topology without changing model math, attention, compile, local batch size, TP, CP, AC, or FP8.
- Third priority: separately wrap `tok_embeddings` and add terminal backward prefetch to reduce root-wrapper communication/scheduling opacity.
- Avoid more broad FP8, AC batch-scaling, CP, or Inductor autotune sweeps unless a later profile materially changes the bottleneck mix.

## Experiment 66: Backward-Only FSDP Prefetch

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run66-flex-backward-only-prefetch-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: invalid.
- The run OOMed before step metrics.
- External `VLLM::Worker_TP*` processes appeared on GPUs 4-7 during the run and held about 128 GiB each at failure time.

Interpretation:

- This is a contaminated OOM, not evidence against backward-only prefetch. Retry the same source when GPUs are clear.

Retry command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run67-flex-backward-only-prefetch-compile-bf16-lbs5-no-flight-recorder-retry > run.log 2>&1
```

Retry result:

- Status: discard.
- Step 10 `tps`: 8,387, below the 8,835 current best.
- Step 10 MFU: 35.04%.
- Step 10 peak memory: 168.10 GiB, 94.25%.
- Loss moved from 12.47595 at step 1 to 7.55834 at step 10; finite and decreasing.

Retry interpretation:

- Forward prefetch is also necessary. Both asymmetric variants lose to the one-module bidirectional prefetch policy.
- Restore the bidirectional prefetch schedule and stop pursuing narrower prefetch removals for now.

## Experiment 68: BF16 FSDP Reduce Dtype

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run68-flex-prefetch-bf16-reduce-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 8,612, below the 8,835 current best.
- Step 10 MFU: 35.98%.
- Step 10 peak memory: 163.18 GiB, 91.49%.
- Loss moved from 12.51039 at step 1 to 12.52511 at step 10; finite but increasing.
- `grad_norm` printed as 0.0000 at both logged steps.

Interpretation:

- BF16 FSDP reductions reduce memory, but they break the short-run training sanity signal and do not improve throughput.
- Restore FP32 reduce dtype from `training.mixed_precision_reduce`; do not pursue lower-precision gradient reductions without a numerics-focused redesign.

## Experiment 69: HSDP 2x4 On Prefetch Source

Command:

```bash
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh --training.steps=10 --compile.enable --training.dtype=bfloat16 --training.local_batch_size=5 --comm.trace_buf_size=0 --parallelism.data_parallel_replicate_degree=2 --parallelism.data_parallel_shard_degree=4 --dump_folder=outputs/autoresearch/may19-qwen3-14b/run69-flex-prefetch-hsdp2x4-compile-bf16-lbs5-no-flight-recorder > run.log 2>&1
```

Result:

- Status: discard.
- Step 10 `tps`: 1,984, far below the 8,835 current best.
- Step 10 MFU: 8.29%.
- Step 10 peak memory: 172.10 GiB, 96.50%.
- Loss moved from 12.50010 at step 1 to 10.01193 at step 10; finite and decreasing.
- The run logged 37 CUDA memory allocation retries and repeated expandable-segment mapping OOM warnings.

Interpretation:

- HSDP 2x4 is not viable for this Qwen3 14B workload at local batch size 5. It increases model memory residency and allocator pressure enough to dominate any collective-topology benefit.
- Restore the DP-only FSDP source and do not pursue nearby HSDP variants unless a later memory-saving source change creates substantial headroom.
