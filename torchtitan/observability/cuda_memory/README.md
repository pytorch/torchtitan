# CUDA allocation requests

This optional collector records successful CUDA allocation API calls, their byte counts and native stacks. It uses `Cuspy.register_callback_handler` and the existing `torch.utils.get_cpp_backtrace`. Stack module/symbol evidence identifies NCCL, cuBLAS/cuBLASLt, cuDNN and recognizable PyTorch allocator callers; other callers remain unknown with their raw stacks retained. This attribution identifies a requester, not exclusive memory ownership.

## Relationship to Cuspy profiling

Allocation capture uses Cuspy's shared callback infrastructure. Setting `config.profiler.cuda_memory` starts the required callbacks automatically, independently of scheduled CUDA profiling.

| Scheduled profiling (`enable_profiling`) | `profiler.cuspy` | Allocation capture behavior |
| --- | --- | --- |
| `False` | Unset or configured | Starts the required Cuspy callbacks automatically. |
| `True` | `CuspyProfilerConfig()` | Shares Cuspy's subscription with scheduled CUDA traces. |
| `True` | `None` (Kineto) | Raises an error before trainer construction. Select Cuspy or disable scheduled profiling. |

These rows assume `profiler.cuda_memory` is configured. Missing Cuspy callback APIs, missing NVIDIA binding packages and incompatible versions raise startup errors instead of silently disabling capture. See [Capture library allocation requests](../README.md#capture-library-allocation-requests) for examples of both modes.

## Enable in a Titan recipe

```python
from torchtitan.observability.profiler import CudaMemoryProfilerConfig

config.profiler.enable_profiling = False  # Allocation capture only.
config.profiler.cuda_memory = CudaMemoryProfilerConfig(
    capture_steps=3,
    stacks=True,
    max_records=262144,
)
```

Use the normal `torchtitan.train` entry point and a fresh output directory. Capture starts after local CUDA device selection and before trainer construction, including communicator initialization. It stops after the configured number of completed training iterations, counted from this invocation rather than a resumed checkpoint step. This startup window is independent of scheduled profiler warmup/active periods and excludes allocations after its cutoff. Shorter successful runs close the window at scope exit; failed initialization/training releases capture without exporting the unfinished window. An already exported window does not establish that subsequent training succeeded.

Each rank writes `memory.json.gz` and `memory-trace.json.gz` under `<dump_folder>/<save_traces_folder>/cuda_memory/rankN/`. Existing rank output is never overwritten. The JSON report contains `coverage`, individual `requests`, and `summary` rows grouped by requesting library, API, device and memory location. Unknown device/location values are `-1`. Request summaries from different APIs are separate: a pool request and physical backing creation can describe different views of the same storage. No live-memory or combined physical-memory total is computed. Failed allocation calls are excluded.

The feature is disabled when `config.profiler.cuda_memory` is `None`. Disabled runs import no collector and perform no extra CUDA work. PyTorch allocator snapshots remain separate.

Custom training entry points must wrap trainer construction and training with `record_cuda_memory(config.profiler, base_folder=config.dump_folder)` from `torchtitan.observability.profiler` and advance the matching normal `Profiler.step()` once per iteration. The standard training entry point manages this automatically. Without the early scope, a configured profiler raises an error because it would have missed communicator initialization.

## Coverage and compatibility

The supported API families are `cuMemCreate`, `cuMemAlloc`, `cuMemAllocPitch`, `cuMemAllocManaged`, `cuMemAllocAsync` and `cuMemAllocFromPoolAsync`. Their legacy and per-thread variants make ten callback IDs. `cuMemCreate` reports backing-allocation requests; async/pool APIs report requests, not GPU completion or newly committed backing. Pitched allocation sizes use returned pitch times height. Allocations before the window, pinned-host allocation APIs, arrays, imports, graphics resources and graph residency are outside this feature. Frees, mappings and releases are not tracked.

`callbacks.py` uses NVIDIA's `cupti.CallbackData`, named `DriverApiTraceCbid` members and typed CUDA driver parameter wrappers. `cuda.bindings.driver.CUmemAllocationProp_st` supplies nested VMM location fields. The only raw reads are primitive result codes, returned device pointers/handles and returned pitch values; legacy pointer/pitch outputs remain 32-bit. All fields become ordinary Python values before callback return. Titan has no copied parameter structs, layout offsets, generator or runtime compiler, and needs no custom PyTorch allocation decoder.

This draft requires a CUDA 13.5 PyTorch build exposing the shared private Cuspy callback registry, loaded CUPTI 13.5, and both NVIDIA Python binding packages from the 13.5 release family. A build without `register_callback_handler`, `unregister_callback_handler`, `arm_callback` and `disarm_callback` needs the upstream callback-registry change first. Install the optional bindings in the capture environment:

```bash
pip install 'cupti-python~=13.5.0' 'cuda-bindings~=13.5.0'
```

Bindings are imported only when capture starts. Startup checks package versions and required wrapper/enum types before registering callbacks, then checks PyTorch's CUDA version and the shared subscriber's loaded CUPTI version. The scalar contract is 64-bit little-endian Linux; x86_64 and aarch64 require runtime qualification. Other release families are outside this draft. Extending compatibility means checking NVIDIA's parameter wrapper and enum coverage and qualifying each supported runtime pair, without maintaining a local struct layout table.

This draft intentionally excludes the earlier PyTorch cleanup and dropped-record fixes, including transactional callback arming. The inspected base can retain an incorrect shared arm count if native enable fails while another consumer remains registered but unarmed. That failure path remains unresolved; the corresponding mock regression is marked expected-failure. This source is for design review and is not runtime-qualified.

Callbacks execute synchronously on the CUDA-calling thread and take the GIL. Arguments are copied before callback return. Native stacks are optional, collected only for successful allocations, bounded to 128 frames and 65,536 retained characters per record, and marked when unavailable/truncated. The record buffer is bounded; overflow, decoding or stack-collection errors fail capture instead of exporting a successful partial report. Start/stop perform no explicit CUDA synchronization. Stop closes callback admission and waits for admitted callbacks; no GPU-completion claim is made. Registration/lifecycle changes must occur outside callbacks and must not race other components' shared-subscriber lifecycle changes.

## Reuse and tests

```python
from torchtitan.observability.cuda_memory import MemoryCapture

with MemoryCapture(stacks=True) as capture:
    workload()
capture.export("memory.json.gz")
capture.export_chrome_trace("memory-trace.json.gz")
```

`MemoryCapture` is independent of Titan trainer classes. Use `stacks=False` for byte counts without library attribution. Each object represents one window and cannot be restarted. `callbacks.py` decodes with NVIDIA bindings; `capture.py` manages registration and records; `stack.py` parses native backtraces; `attribution.py` labels requesters and groups requests; `session.py` owns Titan's startup window.

```bash
python -m pytest tests/unit_tests/cpu/observability/cuda_memory
python -m pytest tests/unit_tests/gpu/observability/cuda_memory
```

Test sources cover scalar output widths, failed-call filtering, async/pool requests, returned pitch, VMM properties, the Python collector, stack parsing, session lifecycle and CUDA/NCCL integration. The decoder unit sources mock vendor wrappers and exercise real scalar buffers; the GPU sources must qualify the actual wrappers. These sources have been adapted but not run for this design. No build, lint, type check or test was run for this revision, as requested for design review. Earlier test results and NCCL measurements apply to prior implementations. Native teardown, CUDA 13.5 runtime behavior, numerical equivalence and overhead remain unqualified.
