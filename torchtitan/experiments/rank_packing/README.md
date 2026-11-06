# Native rank-packing experiment

Run TorchTitan's real eight-rank DeepSeek-V3 debug training program on one
physical NVIDIA GPU. This is a numerical test, not LocalTensor, FakePG, or a
performance simulator. Each rank owns an independent process, autograd graph,
RNG state, and optimizer, while all processes share one GPU.

```text
8 logical ranks: dense FSDP4 x TP2, sparse EP8
       |                |
       +---- one B200 --+   native NCCL and peer buffers
                  |
     actual attention / expert / backward / optimizer kernels
                  |
     per-rank state snapshots <-> eight-physical-GPU reference
```

## Qualified example

The debug model has six layers, hidden dimension 256, eight routed experts,
learned top-k-three routing, and an auxiliary loss. The example uses sequence
length 128, 256 token slots per DP rank, and three complete optimizer updates.
Both standard MoE and actual BF16 Dist-MoE matched their respective eight-B200
reference runs bitwise, including initialization, losses, gradients, weights,
optimizer/scheduler state, persistent and nonpersistent buffers, RNG state, and
routing-assignment/score hashes. The default varlen attention priority selected
cuDNN in those runs, despite FA4 being registered. Profiler traces—not registered
implementation names—establish actual kernel execution.

The original qualification used TorchTitan revision
`40d524a1388ae76128af1cd8329510117a69f714`. This review branch starts from
`0980988794b34e195118dda18907db668e9bebdd`, whose only intervening change was
README documentation. The portable current-branch launcher also completed three
native Dist-MoE updates on one B200; the recorded full-state one-versus-eight
comparison refers to the original qualification.

This review experiment adds no core configuration fields, kernel changes,
reference curves, or CI jobs. `worker.py` uses the ordinary `Trainer.train_step`
path and captures state around the optimizer update. The instrumentation adds
synchronization, so it does not qualify race behavior or performance.

## Requirements

- A matching NVIDIA GPU with capacity for all ranks and supported recipe kernels.
  This experiment was qualified on B200, not H100.
- A PyTorch build with NCCL >= 2.30 experimental multi-rank-per-GPU support.
- For Dist-MoE, a PyTorch build containing
  `TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES` support, the optional public
  `dist_moe` package, and `nvidia-cuda-mps-control`.
- Normal TorchTitan dependencies and a compatible attention implementation.

The qualified environment used torch `2.16.0.dev20261030+cu130`, NCCL 2.32.3,
Triton 3.9.0, torchao `20ffbcad`, attention-gym 0.0.17, FA4 4.0.0b38,
Dist-MoE `285763b5`, and CUTLASS DSL/base/core/CUDA13 libraries all at 4.8.0.
The harness uses the invoking Python interpreter; it does not activate or modify
an environment.

## Run from the repository root

Select an idle physical GPU. Output directories must be new.

```bash
python -m torchtitan.experiments.rank_packing.launch \
  --output-dir outputs/rank_packing/one \
  --gpus 0 \
  --dist-moe \
  --mps \
  --profile

python -m torchtitan.experiments.rank_packing.launch \
  --output-dir outputs/rank_packing/eight \
  --gpus 0,1,2,3,4,5,6,7 \
  --dist-moe \
  --profile

python -m torchtitan.experiments.rank_packing.compare \
  outputs/rank_packing/one \
  outputs/rank_packing/eight \
  --output outputs/rank_packing/comparison.json
```

Omit `--dist-moe --mps` for standard MoE. `--steps`, `--seq-len`, and `--timeout`
only size and bound this debug example. The model recipe uses Titan's existing
config system; there is no new training configuration system.

The launcher sets `NCCL_MULTI_RANK_GPU_ENABLE=1`, disables NVLS, and limits NCCL
to one CTA per rank in **both** comparison arms. Dist-MoE additionally enables
real overlapping-device symmetric memory and disables multicast. All logical
ranks and mesh placements remain unchanged. Private MPS is scoped to a selected
GPU UUID with unique pipe/log directories; it is shut down after child cleanup.
The launcher does not change compute mode or global MPS services.

A short private temporary directory is used for IPC. PyTorch's symmetric-memory
IPC can overflow `sockaddr_un.sun_path` with long TMPDIR spellings; buffers and
kernel computation are not changed to work around that native bug. If cleanup
fails, the temporary directory is retained and its location recorded in
`outcome.json` for diagnosis.

## Existing integration tests

The launcher can supervise another existing command under the same rank-packing
environment. For example, a sequential existing numerical case keeps its logical
GPU count while inheriting one visible device:

```bash
python -m torchtitan.experiments.rank_packing.launch \
  --output-dir outputs/rank_packing/integration \
  --gpus 0 -- \
  python -m tests.integration_tests.run_tests \
  outputs/rank_packing/integration/suite \
  --test_suite b200 --test_name kimi_k3_mm_muon \
  --gpu_arch b200 --execution_mode real_pg --ngpu 8 --no-parallel
```

Check the runner's current `--help`, exact test name, and hardware/golden header
before using this command. Passing an A10G or MI350X golden on B200 is not assumed;
exporting or rebaselining a golden does not count as passing it. Full hardware
numerics qualification is separate from this example's matched B200 replay.

## Evidence and limits

`metadata_rank_*.json` records logical topology, local device binding, selected
modules, and parameter layouts. `initial_rank_*.pt` and `step_*_rank_*.pt` contain
full state; `steps_rank_*.json` records losses, gradient norms and routing hashes.
`launch.json` records tuning/transport settings; `outcome.json` records process
completion and cleanup. With `--profile`, raw GPU traces are captured per rank.
The comparator rejects failed/incomplete runs, mismatched layouts or tuning,
nonfinite values, missing gradients, or mismatching numerical state, returning
nonzero on any mismatch.

The example does not qualify native multicast, every shape, arbitrary aggregate
memory capacity, full-step graph capture, MXFP8, CP/PP, transparent CPU offload,
default communicator tuning, or all race schedules. Equal state to a real run
is evidence for the tested execution, not an independent proof of the algorithm.

Experiment-local CPU tests:

```bash
python -m unittest \
  torchtitan.experiments.rank_packing.test_compare \
  torchtitan.experiments.rank_packing.test_launch -v
```
