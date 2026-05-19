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
