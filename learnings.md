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

Shared config-based decoder sharding appears directly applicable to the dense Qwen3 14B path. The 14B registry builds a dense GQA decoder with 40 query heads, 8 KV heads, head dim 128, 48 layers, RMSNorm QK normalization, and no MoE. TP=2 divides both query and KV heads and preserves an 8-GPU layout as dp_shard=4 x tp=2. TP=4 also divides both head counts, but it is a larger communication change and should wait until TP=2 establishes that the Qwen3 sharding contract is correct.

The relevant shared helpers are in `torchtitan.models.common.decoder_sharding`: root decoder sharding, GQA projection sharding, inner-attention local-map for heads-sharded q/k/v tensors, dense FFN colwise/rowwise sharding, and RMSNorm sequence-parallel placement. The likely Qwen3-specific work is attaching these helpers to every `Qwen3TransformerBlock.Config`, including `attention.qk_norm`, `attention_norm`, and `ffn_norm`, then calling `model.parallelize(tp_mesh)` before FSDP in `parallelize_qwen3()`.

Roofline status is still unclear before the first run. If the bootstrap is memory-heavy or slow with low MFU, TP/SP is a reasonable next candidate because it reduces dense matmul state and activation footprints on B200 NVLink. If the bootstrap has high MFU and low communication overhead, profile before changing TP so the next idea is not a blind mesh sweep.

## Experiment Review: FSDP Bootstrap Through No-Reshard

The bootstrap FSDP implementation made the target command runnable. The first valid unprofiled row completed 10 steps with finite loss dropping from 12.45655 to 7.96787, 5,774 tps, 24.12% MFU, and 50.1GiB peak memory. This established the first correct source baseline.

TP=2 with sequence parallel was correct but slower: 4,971 tps and 39.5GiB peak memory. The likely cause is that this batch/sequence shape does not have enough per-rank work to pay for the added TP collectives and placement redistributions. The memory savings are not useful by themselves because the FSDP-only path already has ample headroom.

The profiled FSDP diagnostic should not be ranked against unprofiled rows, but it identifies communication as an important bottleneck. In the rank 0 trace for the captured step, kernel time was dominated by NCCL reduce-scatter at about 862 ms, with NCCL all-gather around 226 ms. Flash attention and dense matmul kernels were also substantial, but the largest single bucket was FSDP gradient communication.

The command-only no-reshard candidate improved the best unprofiled result from 5,774 to 5,872 tps and 24.53% MFU, at the cost of raising peak memory to 72.2GiB. Loss remained finite and dropped from 12.40488 to 6.59394. This supports the interpretation that avoiding repeated parameter reshards helps enough to justify extra memory on 8xB200.

Roofline conclusion: the run is not compute-roofline-bound; reported MFU is around 24.5% and the profile shows material FSDP collective time. The immediate opportunity is to convert memory headroom into more tokens per collective and larger GEMMs before adding more parallel axes. The next narrow experiment should increase local batch size with `fsdp_reshard_after_forward=never`.
