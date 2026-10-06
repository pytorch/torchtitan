# GraphTrainer performance run instructions

Benchmark DeepSeek V3 671B with MXFP8, one MTP layer and CUDA graphs for the
complete forward/backward pass. Optimizer updates run outside those graphs.
The matrix is **SPMD / PP2 / PP4 x natural / balanced routing x GT / Eager**:
12 training runs using public C4 and weights initialized from seed 42.

## 1. Setup and tokenizer

Use Linux ARM64, Python 3.12 and 256 GB300 GPUs. The example uses 64 nodes with
four GPUs per node. Each contiguous group of 64 global ranks must share a
supported NVLink direct-access domain for DistMoE's symmetric-memory peer
access. Place nodes 0..15, 16..31, 32..47 and 48..63 within those domains,
respectively; domains may contain additional GPUs. NCCL must also communicate
across domains for FSDP and PP. Have the cluster operator enable the NVLink
fabric and verify this placement before launch.

Install the same public source and packages on every node. This pins the
recipe revision and the public packages used in the configuration preflight:

```bash
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
git fetch origin 3515ae4430890bbb77007b2a072bfef29bfbd642
git checkout --detach FETCH_HEAD
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --pre "torch==2.16.0.dev20261013+cu132" \
  --index-url https://download.pytorch.org/whl/nightly/cu132
python -m pip install -r requirements.txt
python -m pip install "datasets==4.7.0" "grain==0.2.18"
USE_CPP=0 python -m pip install --pre --upgrade "torchao==0.19.0.dev20261014+cu132" \
  --index-url https://download.pytorch.org/whl/nightly/cu132
python -m pip install "nvidia-cutlass-dsl[cu13]==4.8.0" "apache-tvm-ffi==0.1.14.post1"
python -m pip install "flash-attn-4[cu13]==4.0.0b35" "quack-kernels==0.6.5"
python -m pip install \
  "git+https://github.com/meta-pytorch/dist_moe.git@6f0c3d4216aa9d2ef2eec2fa4215fa42b73af314"
```

TorchAO nightly is required for the 32x32 swizzled MXFP8 cast kernels; release
0.18.0 cannot construct these configurations. The snapshot above passed all
12 configuration/CLI checks on 2026-10-14; it has not been validated by a full
256-GPU public-data training run. The audit used Python 3.12.13, Triton
3.9.0+gitfc8d31a5, datasets 4.7.0 and grain 0.2.18.

Install a CUDA 13 toolkit for JIT compilation and a driver compatible with the
cu132 wheel. The toolkit and wheel runtime are separate requirements; see the
[upstream DistMoE installation and runtime tests](https://github.com/meta-pytorch/dist_moe/tree/6f0c3d4216aa9d2ef2eec2fa4215fa42b73af314#install)
and [MXFP8 requirements](../torchtitan/quantization/mxfp8/README.md#requirements).
Run the upstream kernel tests on supported GPUs before the full allocation.
Record the actual source, packages, driver and JIT toolkit for each comparison:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export GT_RUN_ENV_DIR="/shared/outputs/environment/$(hostname)"
mkdir -p "$GT_RUN_ENV_DIR"
git rev-parse HEAD > "$GT_RUN_ENV_DIR/torchtitan-revision.txt"
python -m pip freeze > "$GT_RUN_ENV_DIR/requirements.freeze.txt"
nvidia-smi > "$GT_RUN_ENV_DIR/nvidia-smi.txt"
nvcc --version > "$GT_RUN_ENV_DIR/nvcc-version.txt"
```

Download only the tokenizer; the benchmark initializes model weights:

```bash
export GT_RUN_TOKENIZER_PATH=/shared/assets/DeepSeek-V3.1-Base
python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="deepseek-ai/DeepSeek-V3.1-Base",
    revision="d3d4eafdc470de44bbf6f0a74f852eb522357be8",
    allow_patterns=["tokenizer.json", "tokenizer_config.json"],
    local_dir=os.environ["GT_RUN_TOKENIZER_PATH"],
)
PY
```

Make this path visible on every worker. C4 streams from `allenai/c4`, English
training split; every node needs dataset access. Record its resolved Hub
revision and keep the data fixed between paired runs. For a pinned local copy,
the following downloads all 1,024 English training shards: 304.3 GiB compressed
at this revision. Provision shared storage for them; the command prints the
compressed size before downloading:

```bash
export GT_RUN_C4_SOURCE=/shared/data/c4
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

revision = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
info = HfApi().dataset_info("allenai/c4", revision=revision, files_metadata=True)
shards = [f for f in info.siblings if f.rfilename.startswith("en/c4-train.")]
print(f"{len(shards)} shards, {sum(f.size for f in shards) / 2**30:.1f} GiB", flush=True)
root = Path(os.environ["GT_RUN_C4_SOURCE"])
snapshot_download(
    repo_id="allenai/c4", repo_type="dataset", revision=revision,
    allow_patterns=["en/c4-train.*.json.gz"], local_dir=root,
)
(root / "README.md").write_text("""---
configs:
  - config_name: en
    data_files:
      - split: train
        path: en/c4-train.*.json.gz
---
""")
(root / "revision.txt").write_text(revision + "\n")
PY
```

`GT_RUN_C4_SOURCE` replaces only the source path. A local directory must expose
the `en` configuration and `train` split to `datasets.load_dataset(...,
streaming=True)`, as the layout above does. `Dataset.save_to_disk()` output is
unsupported. Keep all shards in the same order for both engines. Unset
`GT_RUN_C4_SOURCE` to use public streaming; that default follows the current
Hub revision.

## 2. Configurations and parallelism

Every configuration uses sequence length 4096, microbatch size 1, global batch
size 15,360, 60 optimizer steps, no outer activation checkpointing and no CPU
offload. Both engines use FP32 persistent parameters, BF16 gathered parameters
and gradients, and the same public model/data/optimizer recipe.

The loader uses `max_num_documents=1`: short documents are padded to 4,096
slots instead of filling the row with another document. Reported TPS counts
these configured token slots, including padding. The 62,914,560 slots per
update are not a count of non-padding training tokens.

| Topology | PP | Dense FSDP/DP | EP | Expert FSDP | Virtual stages/rank | Microbatches/DP rank |
|---|---:|---:|---:|---:|---:|---:|
| SPMD | 1 | 256 | 64 | 4 | 1 | 60 |
| PP2 | 2 | 128 | 64 | 2 | 8 | 120 |
| PP4 | 4 | 64 | 64 | 1 | 4 | 240 |

TP, CP and DP replication are 1. The GPU count is `PP x DP = 256`.
Dense weights are sharded over the DP axis. Routed experts split that same
mesh as `DP = Expert FSDP x EP`: EP64 distributes 256 experts, while expert
weights are sharded over 4, 2 or 1 ranks. EP is already included in DP.

SPMD runs the whole model on each DP coordinate with sharded parameters.
PP2/PP4 split it into 16 logical stages using Interleaved1F1B: 15 stages with
four decoder layers each, then the final decoder layer, MTP1 and output head.
Each pipeline rank owns eight/four virtual stages, assigned round-robin:

```text
PP2: rank 0 -> stages 0,2,...,14     rank 1 -> stages 1,3,...,15
PP4: rank r -> stages r,r+4,r+8,r+12, for r = 0,1,2,3
```

The global batch stays fixed: `DP x microbatches x microbatch_size = 15,360`.
Natural routing uses learned grouped top-k with CF8 scratch. Balanced routing
uses an exact round-robin override with CF1 scratch. Both retain CF1 saved
activation capacity and the auxiliary objective. Scratch capacity reserves
memory; it does not rebalance natural routing.

Select a factory from module `scripts.graph_trainer_runs.perf_configs`:

| Topology | Routing | GT factory | Eager factory |
|---|---|---|---|
| SPMD | natural | `spmd_256gpu_graph_trainer_natural` | `spmd_256gpu_eager_natural` |
| SPMD | balanced | `spmd_256gpu_graph_trainer_force_balanced` | `spmd_256gpu_eager_force_balanced` |
| PP2 | natural | `pp2_256gpu_graph_trainer_natural` | `pp2_256gpu_eager_natural` |
| PP2 | balanced | `pp2_256gpu_graph_trainer_force_balanced` | `pp2_256gpu_eager_force_balanced` |
| PP4 | natural | `pp4_256gpu_graph_trainer_natural` | `pp4_256gpu_eager_natural` |
| PP4 | balanced | `pp4_256gpu_graph_trainer_force_balanced` | `pp4_256gpu_eager_force_balanced` |

Pass one of these configurations to `torchtitan.train` with
`--module scripts.graph_trainer_runs.perf_configs`:

```text
--config spmd_256gpu_graph_trainer_natural
--config spmd_256gpu_eager_natural
--config spmd_256gpu_graph_trainer_force_balanced
--config spmd_256gpu_eager_force_balanced
--config pp2_256gpu_graph_trainer_natural
--config pp2_256gpu_eager_natural
--config pp2_256gpu_graph_trainer_force_balanced
--config pp2_256gpu_eager_force_balanced
--config pp4_256gpu_graph_trainer_natural
--config pp4_256gpu_eager_natural
--config pp4_256gpu_graph_trainer_force_balanced
--config pp4_256gpu_eager_force_balanced
```

The factories set these `Trainer.Config` fields. Where three values are shown,
their order is SPMD / PP2 / PP4:

| Config field | Value |
|---|---|
| `parallelism.pipeline_parallel_degree` | 1 / 2 / 4 |
| `parallelism.data_parallel_shard_degree` | 256 / 128 / 64 |
| `parallelism.num_pp_microbatches` | 60 / 120 / 240 |
| `parallelism.expert_parallel_degree` | 64 |
| `parallelism.data_parallel_replicate_degree` | 1 |
| `parallelism.tensor_parallel_degree`, `parallelism.context_parallel_degree` | 1 |
| `parallelism.enable_sequence_parallel` | `False` |
| `parallelism.pipeline_parallel_schedule` | `Interleaved1F1B` for PP2/PP4 |
| `parallelism.pipeline_parallel_module_fqns_per_model_part` | No split for SPMD; the 16-stage split above for PP2/PP4 |
| `parallelism.fsdp_reshard_after_forward` | `"never"` |
| `parallelism.fsdp_defer_gradient_reduction` | Eager: `True`; GT: `False` |
| `training.num_tokens_per_microbatch_per_dp_rank` | 4096 |
| `training.num_tokens_per_train_step` | 62,914,560 (15,360 sequences) |
| `training.steps` | 60 by default |
| `training.disable_cuda_graphs` | `False` |
| `activation_remat` (`activation_checkpoint` in the pinned revision) | `None` |
| `dist_moe_runtime.scratch_capacity_factor` | Natural: `8.0`; balanced: `1.0` |
| `dist_moe_runtime.activation_slot_capacity_factor` | `1.0` |
| `dist_moe_runtime.vmm_capacity_factor` | `None` |
| `debug.seed`, `debug.deterministic` | `42`, `False` in benchmark mode |

Balanced factories also replace each MoE router's expert selection with
round-robin routing. GT factories convert the trainer/model configuration to
GraphTrainer and enable weight-gradient accumulation fusion; Eager factories
use the standard Trainer and in-place expert weight-gradient accumulation.

## 3. Preflight and launch

First, construct all 12 configurations and read one microbatch for each
topology on every node. This checks imports, CLI options, tokenizer and data
without allocating model weights or initializing distributed training:

```bash
python - <<'PY'
from torchtitan.config import ConfigLoader
from torchtitan.components.tokenizer import HuggingFaceTokenizer

module = "scripts.graph_trainer_runs.perf_configs"
for topology in ("spmd", "pp2", "pp4"):
    for routing in ("natural", "force_balanced"):
        for engine in ("graph_trainer", "eager"):
            name = f"{topology}_256gpu_{engine}_{routing}"
            config = ConfigLoader().load(["--module", module, "--config", name])
            print(name, "OK")
    tokenizer = HuggingFaceTokenizer.Config().build(tokenizer_path=config.hf_assets_path)
    loader = config.dataloader.build(
        dp_world_size=config.parallelism.data_parallel_shard_degree,
        dp_rank=0, tokenizer=tokenizer, max_context_length=4096,
        num_tokens_per_microbatch=4096,
    )
    try:
        batch = next(iter(loader))
        print(topology, batch.input.shape, "valid tokens", int((~batch.padding_mask).sum()))
    finally:
        loader.close()
PY
```

Run from the same checkout on each allocated node. Assign `GT_RUN_NODE_RANK`
uniquely from 0..63 in the fabric order described in section 1. Static
rendezvous preserves that ordering: `global_rank = 4 * node_rank + local_rank`.
Use a reachable host/port on node 0 and identical settings on every node:

```bash
source .venv/bin/activate
export GT_RUN_TOKENIZER_PATH=/shared/assets/DeepSeek-V3.1-Base
export GT_RUN_CONFIG=pp2_256gpu_graph_trainer_natural
export GT_RUN_NODE_RANK=0
export GT_RUN_RDZV_ENDPOINT=node0.example.org:29500
export GT_RUN_ID=pp2-gt-natural-repeat1
export GT_RUN_STEPS=60
export GT_RUN_MODE=benchmark
export GT_RUN_PROFILE=0
export GT_RUN_MEMORY_PROFILE=0
export LOG_RANK=0,128,192
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True

torchrun --nnodes=64 --nproc-per-node=4 \
  --node-rank="$GT_RUN_NODE_RANK" \
  --rdzv-backend=static --rdzv-endpoint="$GT_RUN_RDZV_ENDPOINT" \
  --rdzv-id="$GT_RUN_ID" \
  -m torchtitan.train \
  --module scripts.graph_trainer_runs.perf_configs \
  --config "$GT_RUN_CONFIG" \
  --output-dir "/shared/outputs/$GT_RUN_ID"
```

Repeat with each factory above, using a fresh run ID/output directory. Launch
GT first, then the Eager counterpart with identical settings. A scheduler should
allocate the nodes and invoke this command once per node. For eight-GPU nodes,
use 32 nodes and eight processes per node, with each eight-node block inside a
supported NVLink domain; the total must remain 256. `--node-rank` is ignored
with `c10d` rendezvous, so keep `static` for this placement scheme.

Before constructing the full model, check NCCL and symmetric peer access using
the same allocation and static launcher. Save this as
`/shared/gt_ep64_preflight.py` and replace `-m torchtitan.train` and its trainer
arguments with that filename in the launch command:

```python
import os
import socket
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.distributed import _symmetric_memory as symm_mem

device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=5))
rank = dist.get_rank()
assert dist.get_world_size() == 256
print(rank, socket.gethostname(), device, flush=True)
total = torch.tensor([rank], dtype=torch.int64, device=device)
dist.all_reduce(total)
assert total.item() == sum(range(256))
for start in range(0, 256, 64):
    group = dist.new_group(list(range(start, start + 64)))
    if start <= rank < start + 64:
        ep_group, ep_start = group, start
buffer = symm_mem.empty(1, dtype=torch.int64, device=device)
buffer.fill_(rank)
handle = symm_mem.rendezvous(buffer, group=ep_group)
torch.cuda.synchronize()
dist.barrier(group=ep_group)
for peer in range(64):
    remote = handle.get_buffer(peer, (1,), torch.int64)
    assert remote.clone().item() == ep_start + peer
dist.barrier()
print(rank, "NCCL and EP64 peer access OK", flush=True)
dist.destroy_process_group()
```

Every rank must print success. Check the hostname/rank map against the intended
fabric domains. This validates placement and communication; the upstream
DistMoE runtime tests separately validate JIT kernels on the installed toolkit.

## 4. Measurements

Open the saved events with `tensorboard --logdir /shared/outputs`. Each fresh
run writes `<output-dir>/tb/<YYYYMMDD-HHMM>/` with these scalar tags:

| Measurement | TensorBoard tag |
|---|---|
| Configured token slots/s/GPU | `throughput(tps)` |
| Global average loss | `loss_metrics/global_avg_loss` |
| Gradient norm | `grad_norm` |
| Peak active / reserved GiB | `memory/max_active(GiB)` / `memory/max_reserved(GiB)` |

For the default 60-step benchmark, use updates **11..60 excluding 35..40** for
both engines: 44 updates. This removes startup and the optional profiling
window. Run at least three fresh repetitions of each configuration and report
each result plus the arithmetic mean of the repetition means. Keep the source,
environment, allocation, data, steps and profiling toggles identical per pair.

TPS/GPU is the arithmetic mean over the selected updates from the retained
metrics rank; multiply by 256 for global configured-slot TPS. This is distinct
from total slots divided by total elapsed time. Compute `GT TPS / Eager TPS`
per topology/routing and `natural TPS / balanced TPS` per topology/engine.
Useful-token TPS requires a separate count of non-padding tokens.

The retained metrics rank is **0 / 128 / 192 for SPMD / PP2 / PP4**. Its memory
tags are interval peaks on that GPU; take the maximum over the declared window.
They do not measure the fleet maximum or every PP stage. To measure fleet
peaks, set `config.metrics.save_for_all_ranks = True` in the selected Python
factories for separate matched memory runs, then aggregate the maximum over all
`tb/<timestamp>/rank_<global-rank>/` event directories and the same step window.

For a separate deterministic numerical comparison, set `GT_RUN_MODE=verify`;
this disables profiling and enables deterministic execution with seed 42.
Keep topology and routing fixed between engines: PP uses a topology-dependent
seed policy. Require matching complete step sets and exact equality of stored
loss and gradient norm at every update. Never enable
`debug.deterministic_warn_only`. TensorBoard stores these scalars as FP32;
equality here is a regression check at that precision. A claim of bitwise
equality at higher precision needs the original scalar values or tensors.

Use this reader for either a benchmark pair or a separate verify pair. Select
the actual event directory for each run and choose the mode explicitly:

```bash
export GT_RUN_GT_EVENTS=/shared/outputs/pp2-gt-natural-repeat1/tb/YYYYMMDD-HHMM
export GT_RUN_EAGER_EVENTS=/shared/outputs/pp2-eager-natural-repeat1/tb/YYYYMMDD-HHMM
python - "$GT_RUN_GT_EVENTS" "$GT_RUN_EAGER_EVENTS" benchmark <<'PY'
import math
import statistics
import struct
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

expected = set(range(1, 61))
window = expected - set(range(1, 11)) - set(range(35, 41))
events = [EventAccumulator(path, size_guidance={"scalars": 0}).Reload() for path in sys.argv[1:3]]

def read(event, tag):
    rows = event.Scalars(tag)
    values = {row.step: row.value for row in rows}
    if len(rows) != len(values) or set(values) != expected:
        raise ValueError(f"{tag}: expected one value for each update 1..60")
    for step, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{tag}, update {step}: non-finite value {value}")
    return values

if sys.argv[3] == "verify":
    for tag in ("loss_metrics/global_avg_loss", "grad_norm"):
        gt, eager = [read(event, tag) for event in events]
        for step in sorted(expected):
            if struct.pack("!f", gt[step]) != struct.pack("!f", eager[step]):
                raise ValueError(f"{tag}, update {step}: GT={gt[step]:.17g}, Eager={eager[step]:.17g}")
        print(tag, "identical stored FP32 values at all 60 updates")
elif sys.argv[3] == "benchmark":
    means = []
    for engine, event in zip(("GT", "Eager"), events):
        tps = read(event, "throughput(tps)")
        mean = statistics.mean(tps[step] for step in window)
        means.append(mean)
        active = read(event, "memory/max_active(GiB)")
        reserved = read(event, "memory/max_reserved(GiB)")
        print(f"{engine}: {mean:.2f} slots/s/GPU, {256 * mean:.2f} slots/s globally; "
              f"selected-rank peak Active={max(active[s] for s in window):.2f} GiB, "
              f"Reserved={max(reserved[s] for s in window):.2f} GiB")
    print(f"GT/Eager TPS ratio: {means[0] / means[1]:.4f}")
else:
    raise ValueError("mode must be benchmark or verify")
PY
```

Use `verify` as the final argument for deterministic runs. The reader requires
fresh, complete 60-step runs; choose and document a new window for other run
lengths. Collect profiles in additional paired runs to keep throughput runs
free of profiler overhead.

## 5. Profiles

Set `GT_RUN_PROFILE=1` for traces or `GT_RUN_MEMORY_PROFILE=1` for allocator
snapshots on every node. Only global ranks 0 and 128 record them:

| Topology | Global rank 0 | Global rank 128 | Metrics rank |
|---|---|---|---:|
| SPMD | DP rank 0 | DP rank 128, another host | 0 |
| PP2 | pp-rank0, DP rank 0 | pp-rank1, DP rank 0 | 128 |
| PP4 | pp-rank0, DP rank 0 | pp-rank2, DP rank 0 | 192 |

PP4 profiling covers two of four physical PP ranks; pp-rank1 and pp-rank3,
including its metrics rank, are unprofiled. At the default 60 steps, profiler
warmup covers updates 35..37, traces record 38..39 and snapshots record 39:

```text
<output-dir>/profiling/traces/iteration_39/rank0_trace.json.gz
<output-dir>/profiling/traces/iteration_39/rank128_trace.json.gz
<output-dir>/profiling/memory_snapshot/step_000000000039/000000_step_39.pickle
<output-dir>/profiling/memory_snapshot/step_000000000039/000128_step_39.pickle
```

For shorter runs, the profile step is `min(39, GT_RUN_STEPS - 1)` and the
warmup/active window moves with it. Open raw traces in
[Perfetto](https://ui.perfetto.dev/) and snapshots in
[PyTorch memory_viz](https://pytorch.org/memory_viz). Newly captured raw traces
have not undergone the compaction used for the published examples below.

Download a trace below and open it with **Open trace file** in
[Perfetto](https://ui.perfetto.dev/). These illustrative profiles come from
earlier 256-GPU benchmarks; the public recipes above use C4 and seed-initialized
weights. SPMD uses global rank 0. Each PP2 pair includes **global rank 0
(pp-rank0)** and **global rank 128 (pp-rank1)** on different hosts.
These ranks form the recorded pipeline process group `[0, 128]`.
Each PP4 pair includes **global ranks 0/64/128/192 (pp-rank0/1/2/3)**,
the recorded pipeline group `[0, 64, 128, 192]`. The PP4 GT examples come from
completed runs with the parameter-preparation fix. The default profile capture
above selects only ranks 0/128; to capture all four PP4 stages in a fresh run,
change the `selected_rank` membership set from `{0, 128}` to
`{0, 64, 128, 192}` in `_build` in
`scripts/graph_trainer_runs/perf_configs.py` for both engines.
Each file contains a one-second GPU timeline excerpt, starting ten seconds
after that rank's first profiled training CUDA graph launch.
Every overlapping kernel from that training replay is retained with its
original name, timestamp and duration; copies, memsets and PP annotations are
included where available.
CUDA graph streams are grouped into compute and communication lanes. CPU
events, machine/job metadata and nonessential event arguments are omitted.
Use these excerpts to inspect scheduling and the measurement procedure above
for throughput comparisons.

| Topology | Routing | GT traces | Eager traces |
|---|---|---|---|
| SPMD | natural | [rank 0](profiles/graph_trainer_256gpu/spmd_graph_trainer_natural_rank0.json.gz) | [rank 0](profiles/graph_trainer_256gpu/spmd_eager_natural_rank0.json.gz) |
| SPMD | balanced | [rank 0](profiles/graph_trainer_256gpu/spmd_graph_trainer_balanced_rank0.json.gz) | [rank 0](profiles/graph_trainer_256gpu/spmd_eager_balanced_rank0.json.gz) |
| PP2 | natural | [rank 0 (pp-rank0)](profiles/graph_trainer_256gpu/pp2_graph_trainer_natural_rank0.json.gz) / [rank 128 (pp-rank1)](profiles/graph_trainer_256gpu/pp2_graph_trainer_natural_rank128.json.gz) | [rank 0 (pp-rank0)](profiles/graph_trainer_256gpu/pp2_eager_natural_rank0.json.gz) / [rank 128 (pp-rank1)](profiles/graph_trainer_256gpu/pp2_eager_natural_rank128.json.gz) |
| PP2 | balanced | [rank 0 (pp-rank0)](profiles/graph_trainer_256gpu/pp2_graph_trainer_balanced_rank0.json.gz) / [rank 128 (pp-rank1)](profiles/graph_trainer_256gpu/pp2_graph_trainer_balanced_rank128.json.gz) | [rank 0 (pp-rank0)](profiles/graph_trainer_256gpu/pp2_eager_balanced_rank0.json.gz) / [rank 128 (pp-rank1)](profiles/graph_trainer_256gpu/pp2_eager_balanced_rank128.json.gz) |
| PP4 | natural | [rank 0 (pp-rank0)](profiles/graph_trainer_256gpu/pp4_graph_trainer_natural_rank0.json.gz) / [rank 64 (pp-rank1)](profiles/graph_trainer_256gpu/pp4_graph_trainer_natural_rank64.json.gz) / [rank 128 (pp-rank2)](profiles/graph_trainer_256gpu/pp4_graph_trainer_natural_rank128.json.gz) / [rank 192 (pp-rank3)](profiles/graph_trainer_256gpu/pp4_graph_trainer_natural_rank192.json.gz) | [rank 0 (pp-rank0)](profiles/graph_trainer_256gpu/pp4_eager_natural_rank0.json.gz) / [rank 64 (pp-rank1)](profiles/graph_trainer_256gpu/pp4_eager_natural_rank64.json.gz) / [rank 128 (pp-rank2)](profiles/graph_trainer_256gpu/pp4_eager_natural_rank128.json.gz) / [rank 192 (pp-rank3)](profiles/graph_trainer_256gpu/pp4_eager_natural_rank192.json.gz) |
| PP4 | balanced | [rank 0 (pp-rank0)](profiles/graph_trainer_256gpu/pp4_graph_trainer_balanced_rank0.json.gz) / [rank 64 (pp-rank1)](profiles/graph_trainer_256gpu/pp4_graph_trainer_balanced_rank64.json.gz) / [rank 128 (pp-rank2)](profiles/graph_trainer_256gpu/pp4_graph_trainer_balanced_rank128.json.gz) / [rank 192 (pp-rank3)](profiles/graph_trainer_256gpu/pp4_graph_trainer_balanced_rank192.json.gz) | [rank 0 (pp-rank0)](profiles/graph_trainer_256gpu/pp4_eager_balanced_rank0.json.gz) / [rank 64 (pp-rank1)](profiles/graph_trainer_256gpu/pp4_eager_balanced_rank64.json.gz) / [rank 128 (pp-rank2)](profiles/graph_trainer_256gpu/pp4_eager_balanced_rank128.json.gz) / [rank 192 (pp-rank3)](profiles/graph_trainer_256gpu/pp4_eager_balanced_rank192.json.gz) |

The [profile manifest](profiles/graph_trainer_256gpu/manifest.json) records
each excerpt's configuration, time window, event counts and SHA256 checksums.
