# DeepSeek V3 671B DistMoE 256-GPU Runs

This handoff runs two 256-GPU target configurations and one eager control
directly from one self-contained Conda fbpkg. Do not clone TorchTitan or
install PyTorch, TorchAO, FlashAttention-4, DistMoE, or any Python dependency
separately. The fbpkg contains the complete runtime, TorchTitan source,
recipes, tokenizer, and analysis tools required by the runs.

## Runtime artifact and source reports

Use this exact immutable package:

```text
torchtitan_conda_ivankobzarev_dist_moe_256gpu_sm103a_20261002_v2:76a9a9df4c29443386dfbca02a05dd77
```

It has a 16.65 GiB published blob and expires on 2026-10-30 at 12:22 Pacific
time. Stop and request a newly published and reverified UUID if that expiry
does not cover the runs and artifact collection. Do not substitute `LATEST`.

The matching MAST launcher workspace is:

```text
torchtitan_muse_spark_launcher_ivankobzarev_20261002_v32:0e3b2eb2551f4f619b304c39a6e86e82
```

It expires on 2026-10-30 at 22:07 Pacific time. It contains the 64-host MAST
component, topology constraint, runtime preflight, Chien-Chin capture-headroom
wrapper, Sanket dataset binding, and the validated target/control wrappers.
The scheduler-independent configuration source is also committed as
[`scripts/dsv3_671b_dist_moe_256gpu/mast_configs.py`](../scripts/dsv3_671b_dist_moe_256gpu/mast_configs.py),
adapted to the latest public configuration and DistMoE APIs. The immutable v32
launcher remains the source of truth for reproducing the archived measurements;
the branch copy is the source of truth for a new package and post-fix rerun.

The durable source checkout and its complete diff from the upstream base are:

- [Final reproduction branch](https://github.com/pytorch/torchtitan/tree/dist-moe-256gpu-repro-final-20261005)
- [Full source diff from current main](https://github.com/pytorch/torchtitan/compare/main...dist-moe-256gpu-repro-final-20261005)

To inspect the exact source without relying on the original workstation:

```bash
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
git checkout dist-moe-256gpu-repro-final-20261005
git diff origin/main...HEAD
```

The immutable runtime below contains the pre-rebase executable source at
`0ddea1d44`. To
publish a replacement after expiry, use `scripts/publish_dist_moe_fbpkg.sh`
from this branch with clean checkouts of the revisions recorded below. The
script requires `RUNTIME_PREFIX`, `PYTORCH_SOURCE`, `DIST_MOE_SOURCE`,
`HF_TOKENIZER_SOURCE`, `TORCHTITAN_SOURCE`, and a new `FBPKG_NAME`; it verifies
the revisions, materializes editable sources, records provenance and
checksums, validates the GB300 runtime, and publishes only after every check
passes. Set `TORCHTITAN_SOURCE` to a clean checkout of `0ddea1d44` only for a
bit-for-bit reproduction of the archived measurements. Use this final branch
to test the rebased Cat/layout-tail optimization and current upstream DistMoE
integration.

The package contains TorchTitan revision
`0ddea1d44631413368fdca4f58b82c23f4962772`, PyTorch revision
`dc0efe32b9ee884dfe0b56e24b3f49a954cfc543`, TorchAO revision
`6ded493033ebba47e2631b087cb17f01992d51f0`, and DistMoE runtime source
`f208894313ad76c2`. PyTorch was built for `sm_103a`, and the packaged runtime
was verified on NVIDIA GB300. A Git checkout is not needed to fetch, verify,
or run the package.

- Chien-Chin source: [P2483940224](https://www.internalfb.com/phabricator/paste/view/P2483940224?view=markdown)
- Sanket source: [P2527793677](https://www.internalfb.com/phabricator/paste/view/P2527793677?view=markdown)

This v2 package replaces the package reviewed in
[P2532239023](https://www.internalfb.com/phabricator/paste/view/P2532239023?view=markdown)
and
[P2532238335](https://www.internalfb.com/phabricator/paste/view/P2532238335?view=markdown).
A fresh fetch of this exact UUID was verified on four GB300 GPUs: `pip check`
was clean; only `sm_103a` was present; MXFP8 quantization and an actual BF16
grouped GEMM executed; `lib/python3.1` was absent; `conda-unpack-fb` repaired
`torchrun` and `pip`; a four-rank NCCL all-reduce passed; both 256-GPU recipe
tests passed; and one streaming `allenai/c4` training sample was read. The
EP4/DP4 GraphTrainer workload completed four steps with finite loss and
gradient norm. Its outer full-step CUDA graph recorded on step 3 and replayed
on profiled step 4. The rank-zero trace contains one `cudaGraphLaunch` and
95,200 measured GPU slices; GraphTrainer's inner `cuda_graph_pass` was not in
the applied pass lists.

MTP is disabled in both validated configurations. The Chien-Chin recipe is the
current-stack expression of ladder1 R4 and inherits the model builder's default
of zero MTP layers. The
Sanket-derived recipe keeps PP2/VPP8, DP128, EP64, expert-FSDP2, MBS1/LBS32,
RAF-never, and the asymmetric 16-stage split. It intentionally omits MTP1 and
eager in-place WGrad accumulation: current public TorchTitan rejects MTP with
pipeline parallelism and reserves in-place WGrad accumulators for
GraphRuntime. Therefore, compare the second run's topology and stability with
P2527793677, but do not treat its throughput as an exact compute-parity
reproduction of the historical MTP1 run.

## Fetch and verify the fbpkg

Fetch the same immutable UUID into a shared path visible at the same absolute
location on every worker:

```bash
export FBPKG_ID='torchtitan_conda_ivankobzarev_dist_moe_256gpu_sm103a_20261002_v2:76a9a9df4c29443386dfbca02a05dd77'
export FBPKG_DEST=/absolute/shared/path/torchtitan_dist_moe_runtime
mkdir -p "$FBPKG_DEST"
fbpkg fetch --dest "$FBPKG_DEST" --extract --verify \
  --unexpected-fails-verify "$FBPKG_ID"
export RUNTIME_ROOT="$FBPKG_DEST/conda"
"$RUNTIME_ROOT/bin/python" "$RUNTIME_ROOT/bin/conda-unpack-fb"
export PATH="$RUNTIME_ROOT/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin"
export LD_LIBRARY_PATH="$RUNTIME_ROOT/lib:/usr/local/cuda-13.0/lib64"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$RUNTIME_ROOT/src/torchtitan"
cat "$RUNTIME_ROOT/torchtitan_fbpkg_provenance/revisions.txt"
test ! -e "$RUNTIME_ROOT/lib/python3.1"
test ! -L "$RUNTIME_ROOT/lib/python3.1"
head -1 "$RUNTIME_ROOT/bin/torchrun"
"$RUNTIME_ROOT/bin/torchrun" --help >/dev/null
"$RUNTIME_ROOT/bin/pip" --version
sha256sum -c "$RUNTIME_ROOT/torchtitan_fbpkg_provenance/tokenizer_sha256.txt"
```

Run `conda-unpack-fb` once at the final absolute location, before invoking any
packaged console script. Do not activate or move the runtime first.

Run the dependency and recipe preflight on a GB300 worker:

```bash
"$RUNTIME_ROOT/bin/python" - <<'PY'
from pathlib import Path
import importlib.metadata as metadata
import sys

import dist_moe
import dist_moe._blockscaled  # noqa: F401
import functorch
import torch
import torchao
import torchtitan
from dist_moe import BlockScaledConfig, BlockScaledFormat
from torchao.prototype.mx_formats.kernels import mxfp8_quantize_cuda
from torch.utils.checkpoint import _is_cacheable_effect

prefix = Path(sys.prefix).resolve()
for module in (functorch, torch, torchao, dist_moe, torchtitan):
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(prefix), (module.__name__, path, prefix)

assert torch.version.git_version == "dc0efe32b9ee884dfe0b56e24b3f49a954cfc543"
assert torch.cuda.get_arch_list() == ["sm_103a"]
assert torch.cuda.get_device_capability() == (10, 3)
assert torch.cuda.get_device_properties(0).total_memory >= 250 * 1024**3
assert hasattr(torch.ops.aten, "_scaled_addmm_")
assert hasattr(torch.ops.dist_moe, "block_scaled_backward_accumulate")
assert hasattr(torch.ops.dist_moe, "bf16_backward_accumulate")
assert torch._C._dispatch_has_kernel_for_dispatch_key(
    "torchao::mxfp8_quantize", "CUDA"
)
assert BlockScaledFormat.MXFP8_E4M3
assert BlockScaledConfig
assert _is_cacheable_effect

x = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)
mxfp8_quantize_cuda(x, rowwise=True, colwise=True)

a = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)
b = torch.randn((2, 64, 64), device="cuda", dtype=torch.bfloat16)
offsets = torch.tensor([32, 64], device="cuda", dtype=torch.int32)
actual = torch._grouped_mm(a, b, offs=offsets, out_dtype=torch.bfloat16)
expected = torch.cat((a[:32] @ b[0], a[32:] @ b[1]))
torch.cuda.synchronize()
torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

print("torch", torch.__version__, torch.version.git_version)
print("cuda_arches", torch.cuda.get_arch_list())
print("torchao", metadata.version("torchao"), torchao.__file__)
print("dist_moe", dist_moe.__file__)
print("torchtitan", torchtitan.__file__)
print("bf16_grouped_mm", tuple(actual.shape))
PY

"$RUNTIME_ROOT/bin/python" -I -m pip check
"$RUNTIME_ROOT/bin/pytest" -q tests/unit_tests/cpu/test_dist_moe.py
env PYTHONPATH="$PWD/scripts/dsv3_671b_dist_moe_256gpu" \
  "$RUNTIME_ROOT/bin/python" - <<'PY'
import mast_configs

names = (
    "graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu_profile",
    "deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu_profile",
    "deepseek_v3_671b_dist_moe_mxfp8_sanket_topology_256gpu_profile",
)
for name in names:
    config = getattr(mast_configs, name)()
    assert config.training.disable_cuda_graphs is False
    assert config.training.steps >= 41
    print(name, "ok")
PY
"$RUNTIME_ROOT/bin/tlparse" --version

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_RAS_ENABLE=0
export NCCL_DEBUG=WARN
"$RUNTIME_ROOT/bin/torchrun" --standalone --nproc-per-node=4 --no-python \
  "$RUNTIME_ROOT/bin/python" -c '
import os
import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
value = torch.tensor([dist.get_rank() + 1.0], device="cuda")
dist.all_reduce(value)
torch.cuda.synchronize()
assert value.item() == 10.0, value.item()
print(f"rank={dist.get_rank()} nccl_all_reduce={value.item()}")
dist.destroy_process_group()
'
```

Stop if any check fails.

## Common launch contract

Launch exactly 256 workers, one process per GB300 GPU. The launcher must set
valid `RANK`, `WORLD_SIZE=256`, `LOCAL_RANK`, `MASTER_ADDR`, and `MASTER_PORT`
for every worker. The reference packing is 64 hosts with four GPUs per host;
report any different packing and keep each EP64 group within a supported
high-bandwidth fabric domain.

Use one shared output root and a unique node-local cache root on every host:

```bash
export OUTPUT_ROOT=/absolute/shared/path/to/run_outputs
export LOCAL_CACHE_ROOT=/absolute/node_local/path/to/cache
export TRITON_CACHE_DIR="$LOCAL_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE_ROOT/torchinductor"
export CUTE_DSL_CACHE_DIR="$LOCAL_CACHE_ROOT/cute_dsl"
export CUDA_CACHE_PATH="$LOCAL_CACHE_ROOT/cuda"
mkdir -p "$OUTPUT_ROOT" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$CUTE_DSL_CACHE_DIR" "$CUDA_CACHE_PATH"
```

The Chien-Chin GraphTrainer and eager-control jobs run 60 steps, log every ten
steps, and profile step 41. Exclude the profile-contaminated step-50 logging
interval. The Sanket job runs 41 steps and also profiles step 41. Only rank
zero writes `TORCH_TRACE`.

### Recommended MAST launch

The validated launcher uses Normal priority in the
`MuseSpark_1_2_Safety_DCT` tenant and requests 64 four-GPU GB300 hosts. Fetch
the immutable launcher, then explicitly override its workspace package ID so
the package embedded in the job is the same UUID that was fetched:

```bash
export LAUNCHER_ID='torchtitan_muse_spark_launcher_ivankobzarev_20261002_v32:0e3b2eb2551f4f619b304c39a6e86e82'
export LAUNCHER_DIR=/absolute/local/path/torchtitan_muse_spark_launcher
mkdir -p "$LAUNCHER_DIR"
fbpkg fetch --dest "$LAUNCHER_DIR" --extract --verify \
  --unexpected-fails-verify "$LAUNCHER_ID"
cd "$LAUNCHER_DIR"

torchx run -s mast_conda \
  -cfg workspace_fbpkg_id="$LAUNCHER_ID" \
  mast.py:train \
  --module_name mast_configs \
  --config_name graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu_profile \
  --name torchtitan-cc-full-step-cg

torchx run -s mast_conda \
  -cfg workspace_fbpkg_id="$LAUNCHER_ID" \
  mast.py:train \
  --module_name mast_configs \
  --config_name deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu_profile \
  --name torchtitan-cc-eager-full-step-cg

torchx run -s mast_conda \
  -cfg workspace_fbpkg_id="$LAUNCHER_ID" \
  mast.py:train \
  --module_name mast_configs \
  --config_name deepseek_v3_671b_dist_moe_mxfp8_sanket_topology_256gpu_profile \
  --name torchtitan-sanket-pp2-vpp8
```

Do not add `TORCH_SYMM_MEM_DISABLE_MULTICAST=1`. The 256-GPU ablation changed
the selected barrier kernel but did not improve throughput.

For a scheduler other than this MAST component, put the durable configuration
module on `PYTHONPATH` on every worker:

```bash
export REPRO_SOURCE=/absolute/path/to/torchtitan
export REPRO_CONFIG_DIR="$REPRO_SOURCE/scripts/dsv3_671b_dist_moe_256gpu"
printf '%s  %s\n' \
  31c3e18b8d0cfae5f06fb8dc4c31a793ff89b9cfd342eb5c1310cde66c4d208c \
  "$REPRO_CONFIG_DIR/mast_configs.py" | sha256sum -c -
export PYTHONPATH="$REPRO_CONFIG_DIR"
```

The commands below use this module because it contains the capture-headroom
shim and exact target/control overrides. Static pipeline metadata and DistMoE
pipeline-slot handling now come from current upstream TorchTitan.

## Run 1: Test GraphTrainer Chien-Chin

This run uses PP1, DP/FSDP256, EP64, expert-FSDP4, TP1, CP1, local batch 1,
gradient accumulation 16, RAF-never, dense symmetric-memory FSDP, deferred
gradient reduction, first-microbatch unshard, last-microbatch reduce-grad, and
GraphTrainer WGrad producer fusion.

Launch all 256 workers with:

```bash
export RUN_OUTPUT="$OUTPUT_ROOT/chien_chin_r4"
if [[ "$RANK" -eq 0 ]]; then
  export TORCH_TRACE="$RUN_OUTPUT/tlparse_raw"
else
  unset TORCH_TRACE
fi
"$RUNTIME_ROOT/bin/python" -u -m torchtitan.train \
  --output-dir "$RUN_OUTPUT" \
  --module mast_configs \
  --config graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu_profile
```

The historical target from P2483940224 is `5,880.6 +/- 13.8` tokens/s/GPU,
about 1.51 million aggregate tokens/s. The new runtime may differ.

## Run 2: Test Sanket

This run uses PP2 with eight virtual stages per rank, DP/FSDP128, EP64,
expert-FSDP2, TP1, CP1, 32 pipeline microbatches, MBS1/LBS32/GBS4096,
RAF-never, no symmetric-memory FSDP, automatic unshard lookahead, and at most
eight active unsharded stages per pipeline rank.

The validated wrapper binds C4 to `/mnt/mffuse/c4`. Before allocating 256 GPUs,
run the following on every worker host. It must read a real training sample,
not merely construct the streaming dataset object:

```bash
"$RUNTIME_ROOT/bin/python" - <<'PY'
import datasets

dataset = datasets.load_dataset(
    "/mnt/mffuse/c4",
    "en",
    split="train",
    streaming=True,
)
sample = next(iter(dataset))
assert isinstance(sample["text"], str) and sample["text"]
print("c4_streaming_sample_ok", sorted(sample), len(sample["text"]))
PY
```

Stop if any worker cannot read the sample. Launch all workers with:

```bash
export RUN_OUTPUT="$OUTPUT_ROOT/sanket_pp2_vpp8"
if [[ "$RANK" -eq 0 ]]; then
  export TORCH_TRACE="$RUN_OUTPUT/tlparse_raw"
else
  unset TORCH_TRACE
fi
"$RUNTIME_ROOT/bin/python" -u -m torchtitan.train \
  --output-dir "$RUN_OUTPUT" \
  --module mast_configs \
  --config deepseek_v3_671b_dist_moe_mxfp8_sanket_topology_256gpu_profile
```

P2527793677 reported about 5,611 tokens/s/GPU for its MTP1 fused candidate.
That number is context only because this current-stack adaptation omits MTP1
and eager in-place WGrad accumulation.

## Run 3: Chien-Chin eager full-step CUDA-graph control

This control uses the same PP1/FSDP256/EP64 topology, batch, MXFP8 experts,
deferred gradient reduction, outer full-step CUDA graph, zero MTP layers, and
logging/profile cadence as Run 1, but does not use GraphTrainer.

```bash
export RUN_OUTPUT="$OUTPUT_ROOT/chien_chin_eager_control"
if [[ "$RANK" -eq 0 ]]; then
  export TORCH_TRACE="$RUN_OUTPUT/tlparse_raw"
else
  unset TORCH_TRACE
fi
"$RUNTIME_ROOT/bin/python" -u -m torchtitan.train \
  --output-dir "$RUN_OUTPUT" \
  --module mast_configs \
  --config deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu_profile
```

## Process and report artifacts

Render rank-zero `TORCH_TRACE` data for each experiment:

```bash
"$RUNTIME_ROOT/bin/tlparse" "$RUN_OUTPUT/tlparse_raw" \
  -o "$RUN_OUTPUT/tlparse_html" \
  --no-browser --overwrite
```

The profiler writes traces under:

```text
$RUN_OUTPUT/profiling/traces/iteration_41/rank<RANK>_trace.json.gz
```

Compact rank zero for visual inspection and check the printed summary. A run
that claims full-step CUDA graphs must report a nonzero `cuda_graph_launches`
count; zero is a failed CUDA-graph gate even if training completed.

```bash
"$RUNTIME_ROOT/bin/python" \
  .claude/skills/cuda_graph_trace_compaction/scripts/compact_cuda_graph_trace.py \
  "$RUN_OUTPUT/profiling/traces/iteration_41/rank0_trace.json.gz"
```

Share rank 0 from both experiments and rank 128 from the PP2 experiment. From
a machine with fbsource, internal `fbpython`, and Manifold access:

```bash
export FBSOURCE_ROOT=/absolute/path/to/fbsource
export SHARE_TRACE="$FBSOURCE_ROOT/arvr/scripts/perfetto/share_trace.py"
test -x "$SHARE_TRACE"

for TRACE_PATH in \
  "$OUTPUT_ROOT/chien_chin_r4/profiling/traces/iteration_41/rank0_trace.json.gz" \
  "$OUTPUT_ROOT/sanket_pp2_vpp8/profiling/traces/iteration_41/rank0_trace.json.gz" \
  "$OUTPUT_ROOT/sanket_pp2_vpp8/profiling/traces/iteration_41/rank128_trace.json.gz"; do
  test -f "$TRACE_PATH"
  "$SHARE_TRACE" "$TRACE_PATH" | tee "$TRACE_PATH.share.txt"
done
```

`share_trace.py` uses a 28-day TTL by default and is intentionally not bundled
in the portable fbpkg because it requires an internal fbsource environment.

For each run, report:

1. The exact fbpkg UUID, provenance files, commands, GPU count/model, and
   host/GPU packing.
2. Per-GPU tokens/s and MFU for every measured step 11-40, plus mean and
   standard deviation. Report aggregate tokens/s as the per-GPU mean times
   256. If MFU is unavailable, report TFLOP/s.
3. Peak allocated and reserved GPU memory, preferably min/median/max across
   all ranks, plus the step-41 memory snapshots.
4. All logged loss and gradient-norm values; they must remain finite and the
   gradient norm must remain nonzero.
5. Complete stdout/stderr from every rank, including unabridged error logs on
   failure, raw and rendered `tlparse`, all raw profiler traces, and the three
   `share_trace.py` URLs above.

Both runs must finish all 41 steps without OOM, NaN, graph recapture, or
distributed errors.

## Validated 256-GPU results

All three test runs completed at Normal priority on 64 four-GPU GB300 hosts
without a worker restart. MFU below is derived from the logged TFLOP/s using a
2.5 PFLOP/s BF16 peak per GB300. Baseline lines come from the historical source
pastes; Test lines come from this TorchTitan stack.

| Line | MTP | Tokens/s/GPU | MFU | Comparison |
| --- | --- | ---: | ---: | --- |
| Baseline Chien-Chin | Disabled | 5,880.60 +/- 13.80 | 66.13% | Historical R4 target baseline |
| Test GraphTrainer Chien-Chin r33 | Disabled | 5,378.45 +/- 18.56 | 60.49% | 8.54% below Baseline Chien-Chin |
| Test Eager Control Chien-Chin r35 | Disabled | 5,628.57 +/- 29.69 | 63.30% | 4.29% below Baseline Chien-Chin |
| Baseline Sanket | MTP1 | 5,610.91 | 63.10% | Historical reference configuration |
| Test Sanket r16 | Disabled | 5,615.50 +/- 3.54 | 63.15% | Not compute parity because MTP differs |

Test Eager Control Chien-Chin r35 is our run. Historical R3 is not a second
baseline; it is a no-in-place-WGrad intermediate reference at 5,762.80 +/-
25.50 tokens/s/GPU. Test Eager Control is 2.33% below that matching reference.

The Sanket comparison uses steps 20 and 30, matching the cadence quoted in
P2527793677. Across all 30 post-warmup samples, it measured 5,585.43 +/-
120.34 tokens/s/GPU; filtering the single startup tail below 5,200 gives
5,603.76 +/- 67.55. The Chien-Chin rows use uncontaminated logged intervals
20, 30, 40, and 60; interval 50 includes profiler overhead and is excluded.

Validated runs:

- Test GraphTrainer Chien-Chin:
  [MS-Ivan-cc-log10-r33-256-ivankobzarev](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-cc-log10-r33-256-ivankobzarev)
- Test Sanket:
  [MS-Ivan-sanket-max8-expand-r16-256-ivankobzarev](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-sanket-max8-expand-r16-256-ivankobzarev)
- Test Eager Control Chien-Chin:
  [MS-Ivan-cc-eager-r35-256-ivankobzarev](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-cc-eager-r35-256-ivankobzarev)

The GraphTrainer run logged exact step-1 loss `12.26255` and gradient norm
`17.5919` and recorded one
`cudaGraphLaunch` for the complete 16-microbatch forward/backward step. The
Sanket run logged step-1 loss `12.25781` and gradient norm `16.8609`; at step
41 they were `3.22177` and `2.4106`. Its rank-zero and rank-128 peaks were
263.25 GiB and 243.27 GiB.

Raw profiler and memory-snapshot artifacts are in Manifold:

```text
torchtrain_datasets/tree/outputs/MS-Ivan-cc-log10-r33-256-ivankobzarev/profiling/traces/iteration_41/rank0_trace.json.gz
torchtrain_datasets/tree/outputs/MS-Ivan-cc-log10-r33-256-ivankobzarev/profiling/memory_snapshot/step_000000000041/000000_step_41.pickle
torchtrain_datasets/tree/outputs/MS-Ivan-sanket-max8-expand-r16-256-ivankobzarev/profiling/traces/iteration_41/rank0_trace.json.gz
torchtrain_datasets/tree/outputs/MS-Ivan-sanket-max8-expand-r16-256-ivankobzarev/profiling/traces/iteration_41/rank128_trace.json.gz
torchtrain_datasets/tree/outputs/MS-Ivan-sanket-max8-expand-r16-256-ivankobzarev/profiling/memory_snapshot/step_000000000041/000000_step_41.pickle
torchtrain_datasets/tree/outputs/MS-Ivan-sanket-max8-expand-r16-256-ivankobzarev/profiling/memory_snapshot/step_000000000041/000128_step_41.pickle
```

The archived Chien-Chin gap is not a failed outer CUDA graph or a multicast selection
issue. The outer graph recorded and replayed correctly, and disabling
symmetric-memory multicast did not improve throughput. The GraphTrainer trace
contains 3,904 instances of a selected `CatArrayBatchedCopy` layout-copy
variant, exactly 244 per microbatch, while the eager control contains 244 per
optimizer step. Its repeated backward graph contains 122 `aten.cat` operations
from DTensor parameter-gradient redistribution and 61 required MLA backward
concatenations. GraphTrainer therefore repeats the DTensor gradient layout
reconstruction for all 16 microbatches even though FSDP reduce-scatter runs
only on the last one. The first commit on the rebased branch now moves the
FSDP boundary through split/pad/concatenate layout reconstruction and
accumulates the earlier local gradients in place. This should remove those 122
layout concatenations from each repeated graph while retaining the 61 MLA
concatenations. It has CPU structural coverage but no new 256-GPU measurement;
rerun Test GraphTrainer Chien-Chin before reporting an updated throughput or
MFU result.
