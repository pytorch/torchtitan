# CUDA graphs

CUDA graph capture is optional. A CUDA graph records GPU work once and replays it
with fixed tensor addresses. Tensor input values can change between replays.
Input shapes and structure must stay fixed.

Capture runs the Python function and records its GPU work. It does not execute
that GPU work. The wrapper immediately replays the graph once. A call that
captures therefore counts as one real run. Replay does not call the Python
function again.

Before capture, TorchTitan runs the work without a graph to initialize its state.
These are warmup calls. Each warmup call is a real run. Its results and state
changes are kept. This document uses "eager" for execution without a CUDA graph
and "eval" for validation. A training step is one optimizer update.

## Configuration

A microbatch group is one local forward/backward call without pipeline parallelism
(PP), or one PP schedule call with its microbatches. All groups in a training step
contribute to the same optimizer update.

Training capture requires `training.disable_cuda_graphs=False`.

| Setting | Captured work |
| --- | --- |
| `training.cuda_graph_per_microbatch_group=False` | Forward and backward for all microbatch groups in one optimizer update |
| `training.cuda_graph_per_microbatch_group=True` | Forward and backward for one microbatch group, replayed for each group |
| `optim.enable_cuda_graph=True` | Gradient clipping, finite checks, and optimizer update |
| `training.disable_cuda_graphs=False` | One validation microbatch group, if validation is configured |

An eval microbatch group contains one microbatch without PP. With PP, it contains
the microbatches for one eval schedule call. A full validation pass runs the
configured number of groups and reports one validation loss.

With `cuda_graph_per_microbatch_group=False`, one graph captures all microbatch
groups. Each replay runs all groups for one update. The number of groups and their
input structure must stay fixed across updates.

With `cuda_graph_per_microbatch_group=True`, one graph captures one microbatch
group. Each group replays that graph. Groups must have the same input structure,
tensor shapes, dtypes, devices, and set of parameters receiving gradients. The
number of groups can change between updates. Each group reduces gradients
separately. With hybrid sharded data parallelism (HSDP), this can change results
because eager accumulation reduces replicas once per update.

Training and validation use the same capture flag. Disabling training capture
also disables validation capture.
Optimizer capture requires training capture to be enabled. Learning rate
scheduling and exponential moving average (EMA) updates run outside the optimizer
graph. Optimizer capture uses tensor learning rates so each replay reads the
current learning rate.

Capture requires NVIDIA CUDA. During capture, the CPU cannot wait for GPU results,
even at fixed points. This includes `.item()` and `torch.cuda.synchronize()`.
CPU waits must run outside capture. GPU stream waits are allowed under CUDA capture
rules. DistMoE supports expert parallelism (EP) capture. PP requires a schedule that
supports the requested training or eval work. FluxValidator, GraphTrainer, and
TorchFT keep validation eager and warn when training capture is enabled.

## Shared pool and capture order

`AcceleratorGraphManager` owns one memory pool, one capture stream, and the graph wrappers.
The module exposes one instance, `accelerator_graph_manager`. Training, optimizer, and
validation graphs share its pool. They run one at a time.

During capture, the allocator can reuse storage after its tensor is released.
This includes temporary tensors from another graph in the pool. Storage still
held by a live tensor is not available for a capture allocation.

`AcceleratorGraphWrapper` handles warmup, capture, and replay. Training and optimizer
wrappers each use two eager warmup calls. The first initializes state, such as
optimizer tensors. The second runs with that state in place. This is TorchTitan's
warmup policy, not a requirement that CUDA graphs always need two warmup calls.

The eval wrapper uses one eager warmup call. Eval has a separate forward path:
models use eval mode, autograd is disabled, and PP runs without backward.
Training warmup does not warm this path. The full first validation pass stays
eager, even if it contains several groups.

When eval capture is enabled, validation blocks first capture for all wrappers.
Their warmup calls still run on the shared stream. Validation runs after each
training update until eval captures. Each call runs one full validation pass.

The first pass warms eval. A later pass allows capture once all registered
wrappers have finished warmup. Its first group captures and replays eval.
The remaining groups replay that graph. Training and optimizer can capture on
their next calls. Later validation follows its configured frequency.

With one training wrapper call per update and optimizer capture enabled:

```text
Step 1: Training warmup 1 -> optimizer warmup 1 -> eager eval
Step 2: Training warmup 2 -> optimizer warmup 2 -> eval capture/replay
Step 3: Training capture/replay -> optimizer capture/replay -> eval if due
```

Warmup counts wrapper calls, not training steps. With capture per microbatch
group, several training warmup calls can occur in one update. The capture gate
keeps later groups eager until eval captures. Wrappers register on their first
call. `all_wrappers_warmed_up` checks those registered wrappers.
`capture_allowed` controls capture of wrappers that do not have a graph.
It does not block replay of existing graphs.

The first process-local validation pass stays eager on resumed runs too.
A run that ends before capture still reports its validation results. With
training capture disabled, validation follows its configured schedule.

Each validation pass builds its own loader. No warmup result is discarded.
RNG state and model buffers are not restored. Startup validation can change later
training results if eval uses RNG or changes model buffers. Numerical comparisons
must use the same validation schedule.

Before returning, eval releases model outputs and tensors saved by the PP schedule.
It also finishes fully sharded data parallel (FSDP) cleanup. The wrapper keeps only
the loss result.

PP schedules rebuild metadata when switching between eval and training.
`prepare_pipeline_schedule_for_cuda_graph()` prepares metadata for the requested
mode. It can exchange metadata with other ranks and cannot run inside capture.
A wrapper's `prepare_for_capture` callback runs this setup once on the shared
stream, outside capture. Warmup uses the eager schedule's own initialization.

With training capture enabled, capturing eval before training warmup ends can
raise peak memory. The graph pool stays reserved while eager training allocates
from the normal pool. Waiting until warmup finishes avoids this overlap during
training warmup. The shared stream also avoids separate eager caches for
training and eval.

Before its first capture, each wrapper calls `torch.cuda.empty_cache()`. This
releases unused normal cache. It cannot release storage held by live tensors or
live graph pools. Replay does not call it.

## Memory use with and without `_CUDAGraphGradientState`

The figures show possible reuse of one GPU address range in the shared pool.
Each row shows when its tensor values are needed. They do not show total memory
use or promise that every temporary tensor shares gradient storage.

"Temporary" means a tensor that is released before the next use of that address
range. Activations needed by backward must stay live until backward consumes them.
Parameters and optimizer state also stay live. The optimizer updates their values,
but temporary tensors cannot use their storage.

The figures use one forward/backward call. With several microbatch groups or
interleaved PP calls, gradients from earlier backward calls must stay intact during
later forward calls.

### Without `_CUDAGraphGradientState`

`zero_grad(set_to_none=False)` keeps the gradient buffers allocated during eager
warmup. These buffers are outside the graph pool. They stay allocated through
forward, backward, optimizer, and eval. The graph allocator cannot reuse their
storage. Eval and training can still share temporary storage inside the graph pool:

```text
Capture phase       eval            forward         backward        optimizer
Eager grad storage  [======================================================] ...
Eval temporary      [==========]
Forward temporary                   [==========]
Backward temporary                                  [==========]
Shared range        [eval temp]     [forward temp]  [backward temp]
```

Replay keeps the eager gradient buffer separate from temporary storage:

```text
Replay phase        forward         backward        optimizer       eval
Eager grad storage  [======================================================] ...
Gradient values                     [========================]
Forward temporary   [==========]
Backward temporary                  [==========]
Eval temporary                                                      [==========]
Shared range        [forward temp]  [backward temp]                  [eval temp]
```

Capture per microbatch group uses this approach. It clears gradients with
`set_to_none=False` once per update and accumulates them across group replays.

Using `set_to_none=True` without saving and restoring gradient references fails.
Replay does not repeat autograd's assignment to `param.grad`. After that reference
is cleared, the optimizer sees `param.grad is None` and skips the parameter update.

### With `_CUDAGraphGradientState`

Capturing all microbatch groups together uses `_CUDAGraphGradientState` to save and
restore gradient references. Before training capture,
`zero_grad(set_to_none=True)` releases eager gradient buffers. Parameter gradients
are absent when the first forward starts. Autograd creates them during backward
inside the graph pool. It can use storage released by forward tensors.

Eval capture runs first and releases its temporary tensors before training capture.
This lets the same address range serve eval, forward, and gradients:

```text
Capture phase       eval            forward         backward        optimizer
Eval temporary      [==========]
Forward temporary                   [==========]
Gradient values                                     [======================]
Shared range        [eval temp]     [forward temp]  [gradient               ]
```

After capture, the state keeps references to the gradient tensors. Before each
training replay, `zero_grad(set_to_none=True)` clears `param.grad`. The saved
references keep the storage alive. Replay writes fresh gradients at the captured
addresses. The state then restores `param.grad` so the optimizer can read them.

Capture chose addresses for forward temporary tensors before allocating gradients.
Replay uses those addresses without asking the allocator for new storage. Forward
kernels can write temporary data to saved gradient storage when gradient values
are not needed. Eval can also write to it after the optimizer reads the gradients
because eval was captured first:

```text
Replay phase        forward         backward        optimizer       eval           next forward
Gradient values                     [========================]
Forward temporary   [==========]                                                   [==========]
Eval temporary                                                      [==========]
Shared range        [forward temp]  [gradient               ]       [eval temp]    [forward temp]
```

The gradient tensors still exist outside the "Gradient values" interval.
The next backward writes fresh gradients. Eval must not run between backward
and the optimizer update. It must not overlap training or optimizer GPU work.

Forward and eval can share temporary storage. They use separate graphs.
The training graph contains backward work as well as forward work. Eval uses
eval mode and disables autograd. PP and FSDP can also use different communication.

### Why eval captures first

CUDA allows eval capture after training and optimizer capture. That order can use
more memory with `_CUDAGraphGradientState`. The state holds gradient tensors until
teardown. Clearing `param.grad` or finishing the optimizer update does not release
the saved references. A later eval capture cannot allocate tensors in that live
storage. Other released training storage remains available.

When eval captures first, it releases its temporary tensors before training capture
allocates gradients. The allocator can assign the same storage to those gradients.
During replay, eval and forward use their recorded addresses. They can write to
that storage when gradient values are not needed.

## Inputs, outputs, and cleanup

`post_validate_fsdp_for_cuda_graph()` finishes pending grouped forwards and
reshards full parameters so later capture can reuse their storage. With chunked
loss, the model skips `lm_head`, then the loss calls it outside the root forward.
The FSDP group containing `norm` and `lm_head` can still have `norm` pending.
Validation has no backward callback to clear that state. Eager and captured
validation use the same cleanup.

Before replay, the wrapper copies tensor input values into the buffers used for
capture. Inputs marked as static are not copied. Their addresses must stay fixed.
Non-tensor input values and tensor shapes, dtypes, and devices must also stay fixed.
`AcceleratorGraphInputSpec` exposes tensors stored in `BlockMask` so replay can update them.

Graph outputs point into storage owned by the graph. Read or clone results before
a later replay can overwrite them. Another graph sharing the pool may also
overwrite that storage.

`accelerator_graph_teardown()` destroys all registered graphs and clears saved inputs,
outputs, and gradient references. `TrainingEngine.close()` finishes pending work
before teardown. Gradient storage remains live until those references are released.

For memory checks, compare peak allocated memory, peak reserved memory, and
shared-pool size. Reserved memory includes unused cache. Check warmup as well as
replay. CUDA and communication libraries also use device memory outside the PyTorch
allocator.
