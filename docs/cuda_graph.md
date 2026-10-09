# CUDA graphs

CUDA graph capture is optional. CUDA graphs record GPU work once and replay it
with fixed tensor addresses.
Tensor input values can change between replays. Input shapes and structure must
stay fixed.

Before capture, TorchTitan runs the work without a graph to initialize its state.
These are warmup calls. Each warmup call is a real run. Its results and state
changes are kept. This document uses "eager" for execution without a CUDA graph
and "eval" for validation.

## Configuration

An accumulation group is one local forward/backward call without pipeline
parallelism (PP), or one PP schedule call with its microbatches. All groups in an
update contribute to the same optimizer update.

Training capture requires `training.disable_cuda_graphs=False`.

| Setting | Captured work |
| --- | --- |
| `training.cuda_graph_per_accumulation_group=False` | Forward and backward for all accumulation groups in one optimizer update |
| `training.cuda_graph_per_accumulation_group=True` | Forward and backward for one accumulation group, replayed for each group |
| `optim.enable_cuda_graph=True` | Gradient clipping, finite checks, and optimizer update |
| `validator.enable_cuda_graphs=True` | One validation batch, or one group of PP microbatches |

With `cuda_graph_per_accumulation_group=False`, one graph captures all accumulation
groups. Each replay runs all groups for one update. The number of groups and their
input structure must stay fixed across updates.

With `cuda_graph_per_accumulation_group=True`, one graph captures one accumulation
group. Each group replays that graph. Groups must have the same input structure,
tensor shapes, dtypes, devices, and set of parameters receiving gradients. The
number of groups can change between updates. Each group reduces gradients
separately. With hybrid sharded data parallelism (HSDP), this can change results
because eager accumulation reduces replicas once per update.

Validation capture can be enabled without training capture. Optimizer capture
requires training CUDA graphs to be enabled. Learning rate scheduling and
exponential moving average (EMA) updates run outside the optimizer graph.
Optimizer capture uses tensor learning rates so each replay reads the current
learning rate.

Capture requires NVIDIA CUDA. During capture, the CPU cannot wait for GPU results,
even at fixed points. This includes `.item()` and `torch.cuda.synchronize()`.
CPU waits must run outside capture. GPU stream waits are allowed under CUDA capture
rules. DistMoE supports expert parallelism (EP) capture. PP requires a schedule that
supports the requested training or eval work. FluxValidator, GraphTrainer, and
TorchFT do not support validation capture.

## Shared pool and capture order

`_CUDAGraphManager` owns one memory pool, one capture stream, and the graph wrappers.
Training, optimizer, and validation graphs share the pool. They run one at a time.
During capture, the allocator can reuse a memory block after its tensor is released.
This includes temporary tensors from another graph in the pool. A block still held
by a live tensor is not available for a capture allocation.

`CUDAGraphWrapper` handles warmup, capture, and replay for training, optimizer, and
eval. Training and optimizer each use two warmup calls. These calls contribute to
real training updates. The eval wrapper uses one warmup call. Each call handles one
batch, or one group of PP microbatches.

When eval capture is enabled, the trainer blocks capture until training and
optimizer warmup finish. Scheduled validation still runs eagerly during this time.
After warmup finishes, the trainer runs validation twice at the current training
step. Both runs use the configured validation length and report validation loss.

The first of these two runs stays eager for all batches. It warms up eval and keeps
random number generator (RNG) and model-buffer changes. If scheduled eval has
already warmed up the wrapper, this run still prepares PP for eval after training.
PP metadata setup cannot run inside capture. The trainer then allows capture.
The second run captures and replays its first batch. Its remaining batches use
replay. This also works when validation has only one batch. If validation is due
at this step, these two runs replace that scheduled run. Otherwise, both runs are
extra. Later validation follows its configured frequency.

Only enabled graph wrappers need warmup. With eval-only capture, both startup
validation runs occur after the first training update. A run that ends before
warmup finishes still reports scheduled validation results. Without eval capture,
validation follows its configured schedule.

With one training wrapper call per update and optimizer capture enabled:

```text
Step 1: Training warmup 1 -> optimizer warmup 1 -> real eager eval
Step 2: Training warmup 2 -> optimizer warmup 2 -> real eager eval -> real captured eval
Step 3: Training capture/replay -> optimizer capture/replay -> eval if due
```

Warmup uses wrapper call counts, not training step numbers. In per-group mode,
each accumulation group counts as one training wrapper call.

Capture records GPU work without executing it. The wrapper then immediately
replays that work once. The first capture call therefore counts as one real run,
not two. Each validation run builds its own loader. No warmup result is discarded.
RNG state and model buffers are not restored. Startup validation can change later
training results if eval uses RNG or changes model buffers. Numerical comparisons
must use the same validation schedule.

Before returning, eval releases model outputs and tensors saved by the PP schedule.
It also finishes fully sharded data parallel (FSDP) cleanup. The wrapper keeps only
the loss result.

PP schedules rebuild metadata when switching between eval and training. The engine
prepares training metadata before capture. This setup can exchange metadata with
other ranks and cannot run inside capture.

Capturing eval before training warmup ends can raise peak memory. The graph pool
stays reserved while eager training allocates from the normal pool. Waiting until
warmup finishes avoids this overlap. The shared stream also avoids separate eager
caches for training and eval.

Before its first capture, each wrapper calls `torch.cuda.empty_cache()`. This
releases unused normal cache. It cannot release storage held by live tensors or
live graph pools. Replay does not call it.

## Memory use with and without `_CUDAGraphGradientState`

The figures show possible reuse of one memory block. "Temporary" means a tensor
that is released before the next use of that block. Activations needed by backward
must stay live until backward consumes them. Parameters and optimizer state also
stay live. The optimizer updates their values, but temporary tensors cannot use
their storage.

The figures use one forward/backward call. With several accumulation groups or
interleaved PP calls, gradients from earlier backward calls must stay intact during
later forward calls.

### Without `_CUDAGraphGradientState`

`zero_grad(set_to_none=False)` keeps the gradient buffers allocated during eager
warmup. These buffers are outside the graph pool. They stay allocated through
forward, backward, optimizer, and eval. The graph allocator cannot reuse their
storage. Eval and training can still share temporary storage inside the graph pool:

```text
Capture time        eval            forward         backward        optimizer
Eager grad buffer   [======================================================] ...
Eval temporary      [==========]
Forward temporary                   [==========]
Backward temporary                                  [==========]
One pool block      [eval temp]     [forward temp]  [backward temp]
```

Replay keeps the eager gradient buffer separate from temporary storage:

```text
Replay time         forward         backward        optimizer       eval
Eager grad buffer   [======================================================] ...
Gradient use                        [========================]
One pool block      [forward temp]  [backward temp]                  [eval temp]
```

Per-group capture uses this approach. It clears gradients with
`set_to_none=False` once per update and accumulates them across group replays.

Using `set_to_none=True` without saving and restoring gradient references fails.
Replay runs GPU work. It does not repeat autograd's Python assignment to
`param.grad`. After that reference is cleared, the optimizer sees
`param.grad is None` and skips the parameter update.

### With `_CUDAGraphGradientState`

Capturing all accumulation groups together uses `_CUDAGraphGradientState` to save
and restore gradient references. Before training capture,
`zero_grad(set_to_none=True)` releases eager gradient buffers. Parameter gradients
are absent when the first forward starts. Autograd creates them during backward
inside the graph pool. It can use blocks released by forward tensors.

Eval capture runs first and releases its temporary tensors before training capture.
This lets the same pool block serve eval, forward, and gradients:

```text
Capture time        eval            forward         backward        optimizer
Eval temporary      [==========]
Forward temporary                   [==========]
Gradient use                                        [======================]
One pool block      [eval temp]     [forward temp]  [gradient               ]
```

After capture, the state keeps references to the gradient tensors. Before each
training replay, `zero_grad(set_to_none=True)` clears `param.grad`. The saved
references keep the storage alive. Replay writes fresh gradients at the captured
addresses. The state then restores `param.grad` so the optimizer can read them.

Capture chose the addresses for forward temporary tensors before allocating the
gradients. Replay uses those same addresses without asking the allocator for new
blocks. Forward kernels can therefore write temporary data to a saved gradient
block when its gradient values are not needed. Eval can also write to it after the
optimizer finishes reading the gradients because eval was captured first:

```text
Replay time         forward         backward        optimizer       eval           next forward
Gradient use                        [========================]
Eval temporary                                                      [==========]
One pool block      [forward temp]  [gradient               ]       [eval temp]    [forward temp]
```

The "Gradient use" row shows when gradient values must be kept. The saved gradient
tensors still exist outside that interval. The next backward writes fresh gradients.
Eval must not run between backward and the optimizer update. It must not overlap
training or optimizer GPU work.

### Why eval captures first

After training capture, the state holds gradient tensors until teardown. Clearing
`param.grad` or finishing the optimizer update does not release the saved
references. If eval captures after training, the allocator cannot assign those
blocks to eval tensors. Other released training blocks remain available.

When eval captures first, it releases its temporary tensors before training capture
allocates gradients. The allocator can assign the same blocks to those gradients.
During replay, eval and forward use their recorded addresses. They can write to
those blocks when gradient values are not needed.

## Inputs, outputs, and cleanup

Replay executes recorded GPU work. It does not call the captured function again.

Before replay, the wrapper copies tensor input values into the buffers used for
capture. Inputs marked as static are not copied. Their addresses must stay fixed.
Non-tensor input values and tensor shapes, dtypes, and devices must also stay fixed.
`CUDAGraphInputSpec` exposes tensors stored in `BlockMask` so replay can update them.

Graph outputs point into storage owned by the graph. Read or clone results before
a later replay can overwrite them. Another graph sharing the pool may also
overwrite that storage.

`cuda_graph_teardown()` destroys all registered graphs and clears saved inputs,
outputs, and gradient references. `TrainingEngine.close()` finishes pending work
before teardown. Gradient storage remains live until those references are released.

For memory checks, compare peak allocated memory, peak reserved memory, and
shared-pool size. Reserved memory includes unused cache. Check warmup as well as
replay. CUDA and communication libraries also use device memory outside the PyTorch
allocator.
