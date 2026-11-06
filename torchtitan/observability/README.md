# Profiling

## Enable Memory Profiling

Configure profiling in the selected recipe:

```python
from torchtitan.observability.profiler import MemorySnapshotProfiler
from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel


def my_debug_recipe() -> Trainer.Config:
    config = llama3_debugmodel()
    config.profiler.memory_snapshot = MemorySnapshotProfiler.Config(
        save_folder="memory_snapshot", freq=3
    )
    return config
```

Launch it with `MODULE=my_project.recipes CONFIG=my_debug_recipe
./run_train.sh`. In case of an OOM, the snapshot is written under
`step_{step:012d}_exit`; regular snapshots use `step_{step:012d}`.

* `save_folder`: configures the output folder.
* `freq`: controls how often regular snapshots are taken (default 10).
* `max_entries`: caps the recorded alloc/free history.

You can find the saved pickle files in your output folder.
To visualize a snapshot file, you can drag and drop it to <https://pytorch.org/memory_viz>. To learn more details on memory profiling, please visit this [tutorial](https://pytorch.org/blog/understanding-gpu-memory-1/).

## Profile CUDA with Cuspy

Setting `profiler.trace` to `CuspyTraceProfiler` instead of `KinetoTraceProfiler` collects CUDA activity with Cuspy, PyTorch's in-process CUPTI collector, instead of Kineto. On top of what Kineto records, it can add:

* GPU performance metrics sampled with CUPTI PM sampling, e.g. SM activity, DRAM and NVLink bandwidth.
* GPU power, clock and thermal counters (`enable_environment_counters`).
* CUDA graph dependency edges and event-node ids.
* Perfetto protobuf output (`trace_format="pftrace"`), which is smaller than JSON.

See [`torch.profiler.CuspyConfig`](https://docs.pytorch.org/docs/main/profiler.html#torch.profiler.CuspyConfig) for the underlying options.

Requirements:

* A PyTorch nightly from 2026-09-21 or later (or PyTorch 2.15 or newer, once it's released).
* An NVIDIA driver for CUDA 13 (R580 or newer).
* libcupti 13.3 or newer (`pip install cupti-python`; CUDA 13.4 PyTorch wheels already include it).

Profiler startup raises a `ValueError` if torch or libcupti lacks Cuspy support.

Enable it in a Python recipe:

```python
from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel
from torchtitan.observability.profiler import CuspyTraceProfiler


def llama3_debugmodel_cuspy():
    config = llama3_debugmodel()
    config.profiler.trace = CuspyTraceProfiler.Config(
        performance_metrics=["sm__cycles_active.avg"]
    )
    return config
```

With CUDA graphs on, the forward-backward, optimizer, and validation graphs are captured with CUPTI-based annotations, so replayed kernels keep their `mark_kernels` metadata. This starts Cuspy's CUPTI subscription at the first capture and holds it for the rest of the run, so other CUPTI tools (Nsight Systems, on-demand Kineto traces via dynolog) can't attach to that process.

Set `record_py_stacks=True` to also record each CUDA graph kernel's Python launch stack. The stacks are written beside each trace as `rank{rank}_cuda_graph_py_stacks.json.gz`, keyed by `(graph id << 32) | graph node id` of the trace's kernel events. Kernels launched outside CUDA graphs get no stack.

### PM sampling configuration

`performance_metrics` takes CUPTI metric names, such as `sm__cycles_active.avg` or `dram__throughput.avg.pct_of_peak_sustained_elapsed`, and each one becomes a counter track in the trace. The whole set must be collectable in a single hardware pass.

To see what the current GPU supports, list its base metric names. A full metric name is a base name plus a suffix such as `.avg` or `.avg.pct_of_peak_sustained_elapsed`:

```python
from torch.profiler._cuspy.pm_sampling import supported_metrics

print(sorted(supported_metrics()))  # e.g. "sm__cycles_active" -> "sm__cycles_active.avg"
```

* `pm_sampling_interval_ms`: time between samples. Defaults to 1 ms.
* `pm_lookback_window_ms`: how much sample history the buffer keeps. Defaults to 10 s.

The trace gets the PM samples still in the buffer when profiling ends. If the warmup and active steps together take longer than the lookback window, the earliest samples are dropped.

The sample buffer is held in process memory while profiling, not written to the trace. CUPTI stores raw hardware counters for every sample (several KB each, depending on the metrics), so it grows with lookback / interval: e.g. about 1.1 GiB at 100 s / 2 ms with 7 metrics.

## Always-on kernel spans

torchtitan supports always-on profiling through the Cuspy backend, typically
adding under 1% overhead in real-scale training. Every step, each rank records
every CUDA kernel's GPU start/end time and keeps the most recent steps on disk,
so a slow step can be inspected after the fact. It stays cheap because:

* each kernel is recorded as a few integers (graph node id, start, end, stream);
* the host collects records without a device sync, while it waits on the GPU;
* kernel names come from the CUDA graph node table, recorded once per graph
  and joined offline, not per kernel during training;
* steps are written in batches of `steps_per_file`, and old files are deleted
  so each rank keeps the last `num_steps_retained` steps.

```python
from torchtitan.observability.kernel_spans import KernelSpanRecorder

config.profiler.kernel_spans = KernelSpanRecorder.Config(num_steps_retained=1000)
```

```
<dump_folder>/profiling/kernel_spans/
  rank0/
    steps_000000000901_000000001000.npz   # steps_per_file (100) steps each, oldest deleted first
    steps_000000001001_000000001100.npz
  rank1/
    ...
```

To look at steps, convert them into a trace for <https://ui.perfetto.dev>,
passing the file that holds them from each rank you want:

```bash
python scripts/kernel_spans_to_trace.py \
    outputs/profiling/kernel_spans/rank{0..3}/steps_000000001001_000000001100.npz \
    --steps 1050-1052 --output trace.json.gz
```

Caveats:

* A step holds the kernels that started between the previous step's end on
  the host and its own. When the host runs ahead of the GPU, a step can include
  the tail of earlier steps.
* Kernels launched outside CUDA graphs are recorded without names.
* Steps not yet written to a file are lost if the process is killed.
* It can run alongside `CuspyTraceProfiler`, but not `KinetoTraceProfiler`.

The file format is described in [`kernel_spans.py`](kernel_spans.py).

Requirements:

* Accelerator graphs (the default), since kernels are named from graph nodes.
* The Cuspy requirements above (libcupti 13.3 or newer).
* CUDA graph node ids: a CUDA 13.1 or newer driver, or cuda-compat 13.1 or
  newer on `LD_LIBRARY_PATH` (see `torch.cuda.graph_annotations.is_available`).
  CUDA 13.5 is not required.
