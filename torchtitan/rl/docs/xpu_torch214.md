# GRPO on Intel XPU with torch 2.14: reproduction guide

This branch runs end-to-end GRPO on Aurora (Intel Data Center GPU Max
1550) on the **torch 2.14.0+xpu / oneCCL 2022** stack: Monarch actors, TorchStore
weight sync, vLLM generation, FSDP2 training. It is written for a validation team
that wants to rebuild the stack and reproduce the reference runs below.

The torch 2.12 guide, [xpu.md](xpu.md), still describes the architecture, the
XPU-specific settings and the history of the multi-node fixes; read it for the
"why". This guide covers only what to build, what to run, and what a correct
run looks like on torch 2.14, plus everything that changed from 2.12.

Two training arms, controlled twins (same batch, lr, sampling and generator
settings; only the `LoRAConverter` differs):

| Arm | Config | Trains |
|-----|--------|--------|
| LoRA GRPO | `rl_grpo_lora_qwen3_0_6b` | LoRA adapters (rank 32) |
| Full GRPO | `rl_grpo_full_qwen3_0_6b_flex` | every parameter |

Both live in `torchtitan_recipes/rl/alphabet_sort_xpu.py` (Qwen3-0.6B,
alphabet-sort task, 0/1 reward).

## Reference results

All on Aurora, launched with the scripts in `torchtitan/rl/scripts/xpu/` of this
branch. "Byte-exact" means every TorchStore weight transfer was checksummed on
both sides and matched (study-only hook, not part of the scripts).

| Run | Nodes | Steps | Result |
|-----|-------|-------|--------|
| Full, job 8904699 | 4 | 10 | 10/10, 352/352 transfers byte-exact |
| LoRA, job 8904708 | 4 | 10 | 10/10, 697 transfers byte-exact |
| Full, job 8905380 (`CCL_OP_SYNC` unset) | 8 | 200 | 200/200, 12819 transfers byte-exact, `fwd_bwd` 22063 tok/s, 12.6 s/step |
| LoRA, job 8905381 (`CCL_OP_SYNC` unset) | 8 | 200 | 200/200, 51213 transfers byte-exact, `fwd_bwd` 45032 tok/s, 16.5 s/step |
| Full, job 8904955 | 1 | 200 | 200/200, `fwd_bwd` 8716 tok/s, 13.3 s/step |
| LoRA, job 8904956 | 1 | 200 | 200/200, `fwd_bwd` 11214 tok/s, 14.6 s/step |

A run is healthy when all of these hold:

- every requested step logs a `Train | Step: N` line;
- no `untrainable batches` warning and reward is non-zero from step 1 (the
  zero-weight failure below shows up as reward 0 and untrainable batches);
- LoRA `grad_norm` stays in 0.04-0.15. Full GRPO's grad_norm is 3-5x higher
  because it is taken over all parameters; compare it only with full runs;
- the `Mesh split` log line shows the mesh you expect (read the mesh from that
  line, not from the requested flags).

## What changed from the torch 2.12 stack

| | torch 2.12 (`xpu.md`) | torch 2.14 (this guide) |
|---|---|---|
| torch / oneCCL / SYCL | 2.12.0+xpu / 2021.17.2 / 2025.3 | 2.14.0+xpu / 2022.1.1 / 2026.1.0 |
| oneAPI env | oneAPI 2025.3 | oneAPI 2026.1 module + `ZE_FLAT_DEVICE_HIERARCHY=FLAT` (`env_torch214.sh`) |
| RMSNorm `LD_PRELOAD` interposer | required for full GRPO | **not needed**: the stock kernel passes the 65536-row weight-grad backward |
| vLLM | pinned old commit | upstream `main` |
| `CCL_WORKER_AFFINITY` | set explicitly | **unset**: oneCCL 2022 aborts comm init on an explicit list (`unexpected worker affinity length`) |
| `SYCL_UR_USE_LEVEL_ZERO_V2=1` | not set | set in all four launchers (see below) |
| `CCL_OP_SYNC` | `=1` (part of the CXI tuning block) | **unset**: on oneCCL 2022 it makes cross-node all_reduce a silent no-op (see below) |
| TorchStore storage volumes | inside the trainer processes | in their own processes (`storage_volumes_in_own_procs`, see below) |
| first weight pull | all generators at once | one generator at a time (`sequential_initial_pull`) |

### The two oneCCL bugs and the workarounds in this branch

Both are filed upstream with standalone reproducers that use only the oneCCL C
API (no PyTorch, TorchStore or torchtitan): [#222](https://github.com/uxlfoundation/oneCCL/issues/222) and [#225](https://github.com/uxlfoundation/oneCCL/issues/225).
The first is a regression from oneCCL 2021.17.2; the second also happens on
2021.17.2.

1. **Cross-node broadcast silently moves no data** ([#222](https://github.com/uxlfoundation/oneCCL/issues/222)). oneCCL fixes
   a process's position on the node when the process creates its first
   communicator, and oneCCL 2022 now routes cross-node broadcasts through a
   sub-group built from that position. A TorchStore storage volume living in an FSDP trainer process
   inherits the FSDP group's position, so its later cross-node broadcasts from
   tiles 1-3 return success and deliver nothing: the generators get zero
   weights, reward drops to 0, and the controller logs untrainable batches.
   **Workaround:** `Controller.Config.storage_volumes_in_own_procs=True` spawns
   the volumes in a twin of the trainer mesh (same hosts, same tiles), whose
   first communicator is a pull communicator.
2. **Concurrent communicator creation fails** ([#225](https://github.com/uxlfoundation/oneCCL/issues/225)) in oneCCL's topology discovery
   (`zesFabricPortGetConfig` -> `ZE_RESULT_ERROR_NOT_AVAILABLE`) when a volume
   process builds several communicators at once. TorchStore creates its
   communicators on a generator's first pull and caches them.
   **Workaround:** `InterGeneratorRouter.Config.sequential_initial_pull=True`
   runs that first pull one generator at a time. It costs tens of seconds once
   (26-50 s at 4 nodes, 50-160 s at 8); later pulls stay concurrent.

Both flags default off in core torchtitan and are on in the XPU recipe. A third
oneCCL 2022 bug (a SIGSEGV once one process holds more than ~9 communicators on
the scheduler path, [#224](https://github.com/uxlfoundation/oneCCL/issues/224)) is avoided because the default SYCL path is used; do not set
`CCL_SYCL_BROADCAST_SCALEOUT_THRESHOLD=0` or `CCL_ENABLE_SYCL_KERNELS=0`.

**`CCL_OP_SYNC` must not be set.** On oneCCL 2022, `CCL_OP_SYNC=1` turns every
cross-node `all_reduce` above 4 MiB into a no-op that reports success: each rank
gets its own input back. Under FSDP2 HSDP (the full arm at 8 nodes runs `dp_replicate=4 x
dp_shard=4`) the replicas then train independently on their own node's gradients,
while loss, reward and grad_norm look normal. It also makes the gradient sync ~6x
slower. Without it the reduction is exact and 8-node `fwd_bwd` is 23.5k tok/s
(full) and 47.6k (LoRA), against 7.3k and 40.3k with it. A 2-rank standalone
reproducer is in [#223](https://github.com/uxlfoundation/oneCCL/issues/223).

**`SYCL_UR_USE_LEVEL_ZERO_V2=1`.** With the default (v1) Level Zero adapter,
long runs abort intermittently with `ur_die: urEventWait must not be called for
an internal event`, about once per 100-150 steps, in trainer and generator
processes alike. The v2 adapter ran 200 steps clean in both arms at 1 and 8 nodes.
This one is not root-caused; treat it as a workaround to remove once the v1
adapter is fixed.

## Prerequisites

```bash
export TORCHTITAN_DIR=$HOME/git/torchtitan
export MONARCH_DIR=$HOME/git/monarch214
export TORCHSTORE_DIR=$HOME/git/torchstore
export VLLM_DIR=$HOME/git/vllm
export CONDA_PREFIX_BASE=$HOME/miniforge3
export HF_ASSETS_PATH=/flare/Aurora_deployment/intel/models/Qwen3-0.6B
```

The launchers read `TORCHTITAN_DIR`, `CONDA_PREFIX_BASE`, `CONDA_ENV` (default
`monarch214`) and `ONEAPI_ENV_SCRIPT` (default
`torchtitan/rl/scripts/xpu/env_torch214.sh`), all with the defaults above. `HF_ASSETS_PATH` is overridable the
same way.

- Aurora access: the UAN (login node) for builds, compute nodes for runs.
- Conda (miniforge3).
- Qwen3-0.6B weights on a filesystem the compute nodes can read.
- The alphabet-sort dataset pre-cached: runs set `HF_HUB_OFFLINE=1`, so on the
  UAN (which has internet) run once:
  `python -c "import datasets; datasets.load_dataset('kalomaze/alphabetic-arxiv-authors-it1', split='train')"`.

## Repositories

| Repo | Branch | Commit | Notes |
|------|--------|--------|-------|
| torchtitan | this branch | see branch | this guide |
| torchstore | `xpu-all` (github.com/songhappy/torchstore, not yet pushed) | `1ebced4` | PR #171 (XCCL transport) + replica dedup |
| monarch | `xpu-upstream` `7761764a` (PR #4307) + upstream `d9fc48862` cherry-picked | see note | `d9fc48862` makes `default_bootstrap_cmd` public, which torchtitan imports |
| vLLM | upstream `main` | `083060d04` | no local patches |

## Environment setup

Build on the UAN, never on a compute node. This recipe was reconstructed from
the build logs of the reference environment; please report any step that does
not reproduce.

### 1. Conda env and toolchain

```bash
source $TORCHTITAN_DIR/torchtitan/rl/scripts/xpu/env_torch214.sh   # oneAPI 2026.1 + gcc 14.3
$CONDA_PREFIX_BASE/bin/conda create -n monarch214 python=3.12 -y
eval "$($CONDA_PREFIX_BASE/bin/conda shell.bash hook)"
conda activate monarch214
conda install -y -c conda-forge rust=1.97.1
```

Never source a oneAPI 2025.x environment in the same shell: torch 2.14 ships
the 2026.1 SYCL runtime, and two SYCL runtimes in one process segfault.

### 2. torch, Triton and the vLLM XPU kernels (from vLLM's pins)

vLLM `main` pins the matching set in `requirements/xpu.txt` (`torch==2.14.0`,
`triton==3.8.0+xpu`, `vllm_xpu_kernels==0.1.15.4`):

```bash
cd $VLLM_DIR && git checkout 083060d04
pip install -r requirements/xpu.txt \
    --extra-index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://wheels.vllm.ai/xpu/
python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
```

On the UAN `torch.xpu.is_available()` is False (no GPUs there); the version must
be `2.14.0+xpu`.

If `import triton.language` later fails with `module 'triton' has no attribute
'language'`, another package's `triton` install deleted files that `triton-xpu`
owns: `pip install --force-reinstall --no-deps triton-xpu==3.8.0`.

### 3. vLLM

```bash
cd $VLLM_DIR
VLLM_TARGET_DEVICE=xpu VLLM_REQUIRE_RUST_FRONTEND=1 \
    pip install -e . --no-deps --no-build-isolation -v
```

`VLLM_REQUIRE_RUST_FRONTEND=1` (needs `cargo` from step 1) makes a missing Rust
toolchain an error instead of silently dropping `_rust_tool_parser`.

### 4. Monarch

```bash
cd $MONARCH_DIR && git checkout 7761764a && git cherry-pick d9fc48862   # PR #4307 + upstream #4327
GCC=$(dirname $(which gcc))
RUSTC_BOOTSTRAP=1 USE_TENSOR_ENGINE=0 \
PROTOC=$CONDA_PREFIX/lib/python3.12/site-packages/torch/bin/protoc \
CC=$GCC/gcc CXX=$GCC/g++ CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1" \
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=$GCC/gcc \
    pip install -e . --no-deps --no-build-isolation
pip install pyzmq pyarrow requests numpy pyre-extensions "typing-extensions>=4.12" \
    cloudpickle lark tabulate opentelemetry-api clusterscope "flask>=2.0" \
    xxhash py-spy aiohttp
python -c "from monarch.actor import Actor, default_bootstrap_cmd; import monarch._rust_bindings"
```

Why each variable: `USE_TENSOR_ENGINE=0` builds only the actor runtime the RL
stack uses; `RUSTC_BOOTSTRAP=1` lets stable rustc accept the nightly-only flags
`setup.py` passes; `PROTOC` points at the `protoc` torch bundles; `CC/CXX` must
be GNU gcc (the oneAPI env sets `CC=icx`, whose intrinsics GNU `ld` cannot
link); and the cargo linker override replaces conda's
`x86_64-conda-linux-gnu-cc`, whose old sysroot lacks the glibc 2.38 symbols
`aws-lc-sys` emits.

### 5. TorchStore

```bash
cd $TORCHSTORE_DIR && git checkout xpu-all
pip install -e . --no-deps --no-build-isolation
pip install pygtrie portpicker
```

`--no-deps` because TorchStore pins an older `torchmonarch`.

### 6. TorchTitan and training deps

```bash
cd $TORCHTITAN_DIR   # this branch checked out
pip install torchdata==0.11.0 spmd_types==0.2.5 tyro==1.0.15 einops tensorboard \
    "git+https://github.com/PrimeIntellect-ai/renderers.git@main" \
    --constraint <(echo "torch==2.14.0+xpu")
```

TorchTitan runs from the source tree (the launchers `cd` into
`$TORCHTITAN_DIR`); it does not need to be pip-installed. The `--constraint`
keeps pip from replacing the XPU torch.

### Version summary (reference environment)

| Package | Version |
|---------|---------|
| python | 3.12.13 |
| torch | 2.14.0+xpu |
| triton / triton-xpu | 3.8.0+xpu / 3.8.0 |
| oneccl | 2022.1.1 |
| intel-sycl-rt | 2026.1.0 |
| vllm | 0.30.1rc1.dev523+g083060d04.xpu (editable) |
| vllm-xpu-kernels | 0.1.15.4 |
| torchmonarch | 0.6.0.dev0 (editable) |
| torchstore | 0.0.0.dev0 (editable) |
| transformers | 5.18.0 |
| datasets | 4.7.0 |
| renderers | 0.1.11 |
| tyro | 1.0.15 |
| spmd_types | 0.2.5 |
| torchdata | 0.11.0 |

## Running

All four launchers are PBS scripts with their own headers. Overrides are
environment variables; under `qsub` forward them with `-v`. `qsub` forwards no
trailing arguments, so reach the config CLI through `EXTRA_ARGS`.

### Single node (2 trainer + 2 generator tiles)

```bash
cd $TORCHTITAN_DIR
qsub torchtitan/rl/scripts/xpu/run_grpo_lora_sn.sh        # LoRA, 10 steps
qsub torchtitan/rl/scripts/xpu/run_grpo_sn.sh             # full, 10 steps
NUM_STEPS=50 qsub -v NUM_STEPS torchtitan/rl/scripts/xpu/run_grpo_lora_sn.sh
```

### Multiple nodes (half trainer, half generator)

```bash
cd $TORCHTITAN_DIR
qsub -l select=4 -q debug-scaling torchtitan/rl/scripts/xpu/run_grpo_multinode.sh       # full
qsub -l select=4 -q debug-scaling torchtitan/rl/scripts/xpu/run_grpo_lora_multinode.sh  # LoRA
# 8 nodes, 200 steps (the reference runs)
NUM_STEPS=200 qsub -l select=8 -l walltime=04:00:00 -q capacity -v NUM_STEPS \
    torchtitan/rl/scripts/xpu/run_grpo_multinode.sh
```

Multinode overrides: `NUM_NODES` (use the first N of the allocation), `PPN`,
`TP`, `DP_REPLICATE`, `CONFIG`, `NUM_STEPS`, `VAL_SAMPLES`, `HF_ASSETS_PATH`,
`DUMP_FOLDER`, `MPIEXEC`, `EXTRA_ARGS`. The fabric block in the multinode
scripts (CXI provider, Cray PALS `mpiexec` by absolute path, `--cpu-bind none`,
`TMPDIR=/tmp`, `unset HOSTNAME`) is load-bearing; see xpu.md "Multi-node fixes"
for what each line prevents.

Logs: `$DUMP_FOLDER/train_<arm>_<N>n.log` under `$TORCHTITAN_DIR`.

## Known issues

- **Teardown hangs after the last step.** `qdel` the job once the final
  `Train | Step` line appears; PBS then reports a walltime exit for a run that
  succeeded. Teardown also aborts with core dumps of ~10 GB per process, which
  is why the launchers set `ulimit -c 0`.
- **Monarch startup flake**: `cast.service ... actor does not exist` before any
  step. Intermittent; resubmit.
- **vLLM "not enough free memory" at startup with 0 steps** is a phantom from
  Level Zero's lagging memory report; resubmit before changing
  `gpu_memory_limit`.
- **Checkpoint saving is off** (`load_only=True` in the recipe): the DCP save
  ran out of device memory inside oneCCL at 8 nodes. Never disable the
  checkpointer to avoid saves; that also skips loading the pretrained weights.
- **vLLM TP=2 SIGSEGVs at process exit** on this stack, after finishing its
  work. Not hit by the TP=1 generators these configs use.
- **Inductor cache corruption** (`CompiledFxGraph has no attribute
  compiled_fn_runner`): `rm -rf $TORCHINDUCTOR_CACHE_DIR`.
