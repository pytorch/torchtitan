# Activation offloading

Activation offloading lowers peak memory by copying activations from GPU HBM
to CPU DRAM during forward and back during backward. This doc covers how
TorchTitan implements it (`activation_offload.py`), how it composes with remat
([ACTIVATION_REMAT.md](ACTIVATION_REMAT.md)), FSDP2, pipeline parallelism and
CUDA graphs, and what it costs.

## What is offloaded

The offloader works on **saved tensors**, the tensors autograd keeps from
forward for backward, through saved-tensor hooks: `pack` runs when a tensor is
saved, `unpack` when backward reads it.

Offloading always runs under remat. The offloader installs
`torch_remat.saved_tensors_hooks` around each transformer block's remat
checkpoint, so it sees only what remat keeps (anything remat recomputes is
never packed), and for each tensor why it is kept (`SavedTensorInfo.kind`) and
by which save region (`SavedTensorInfo.region`):

- `CHECKPOINT_INPUT`: the block input, where the backward replay starts;
- `SAVE_OUTPUT`: a save region's output, reused by the replay;
- `BACKWARD`: tensors a save region's ops keep for their own gradient.

**Offloading is part of the remat policy**: each retained region is kept on
GPU or sent to CPU, in `ActivationRematConfig`:

| Field | Default | Selects |
|---|---|---|
| `offload_block_input` | `False` | each block's input hidden state (`CHECKPOINT_INPUT` that requires grad); under full remat, the whole stash |
| `offload_regions` | `[]` | glob patterns over retained save-region names, e.g. `["attention.*"]`; `["*"]` is everything save regions retain |
| `offload_last_block` | `False` | also the last block of each model part (see [The last block](#the-last-block)) |

Both can differ per block through `block_overrides`
([ACTIVATION_REMAT.md](ACTIVATION_REMAT.md#per-block-policies)), e.g. to
offload only some blocks.
Offloading in place of recomputing is `save_regions=["*"]` plus offload: only
the cheap code outside regions is recomputed. Tensors recomputation needs but
no region retains (`recompute_needs_tensor`) stay on GPU; the first-forward
summary lists them as "outside any region", and a large amount there points
at a missing region.

When the policy offloads anything, `ActivationRematConfig.apply` checkpoints
the blocks, installs an `ActivationOffloader` on the model part, and returns
it. Encoder blocks of a
multimodal model cannot be offloaded: `apply` raises if the policy offloads
one. Offloading also requires CUDA: `apply` raises on other accelerators.

Never offloaded, whatever the config:

- parameters, including the inner tensors of subclass parameters (the qdata
  and scales of FSDP2's unsharded MXFP8 weights, which MXFP8 Dist-MoE saves):
  the module holds them, so nothing is freed, and under FSDP2 the copy would
  race with resharding;
- storages shared with the model part's input: the caller (e.g. a pipeline
  stage) holds them;
- tensors that are not plain CUDA tensors;
- storages under 1 MiB (routing metadata, loss scalars): no measurable saving.

A storage that something other than autograd frees before the free point
(e.g. FSDP2 data the checks above miss) raises there, since the copy may have
read reused memory.

**In-place updates.** Autograd records the version of each tensor saved for
backward (`save_for_backward`) and checks it when backward reads it
(`saved_tensors`), so a tensor modified in place after it was saved raises
instead of producing wrong gradients. Autograd skips this check when
saved-tensor hooks pack the tensor, so the offloader records the version at
`pack` and runs the same check at `unpack`.

## How it works

Queuing a copy saves no memory; dropping the GPU copy does, and that is only
safe once the copy has finished. In backward, a tensor must be back before
it is read. So everything turns on two points, and the trade-off is how much
compute each copy gets to overlap with:

- the **free point**, where the compute stream waits for a group's D2H copies
  and drops the GPU copies. Later means more copy time but higher peak
  memory; a copy still running there stalls compute.
- the **reload trigger**, which starts a group's H2D copies ahead of use. A
  copy still running when `unpack` needs it stalls compute.

All waits are GPU-side event waits between streams; the CPU never blocks.

![Activation offload timeline](../../assets/images/activation_offload_timeline.svg)

| Term | Code | Meaning |
|---|---|---|
| offload group | `_OffloadGroup` | the storages packed by one call of one transformer block; freed and reloaded together |
| copy-done event | `group.copy_done_event` | recorded on the offload stream when the block's forward returns; completes after all of the group's D2H copies |
| reload event | `_OffloadedStorage.reload_event` | recorded on the reload stream after one storage's H2D copy |

**Forward, block `i`:**

1. Block `i`'s pre-forward hook opens group `i` and enters the saved-tensor
   hooks.
2. `pack` queues each selected storage's D2H copy on the offload stream, which
   first waits on the compute stream so the copy starts after the tensor is
   produced. A storage is copied once even if several saves view it (e.g. q,
   k and v slices of a fused QKV output); each save is rebuilt as a view of
   the reloaded storage.
3. When block `i` returns, `copy_done_event` is recorded.
4. When block `i + 1` returns, group `i`'s **free point** runs: the compute
   stream waits on its `copy_done_event`, then its GPU copies are dropped.
   Group `i`'s copies overlap block `i + 1`'s forward, and at most two groups
   are on the GPU at once. The last group is freed when the model part's
   forward returns.

**Backward:**

1. Block `i + 1`'s output gradient is ready: its **reload trigger** queues
   group `i`'s H2D copies on the reload stream, each into a GPU buffer
   allocated on the compute stream (so it is freed in compute-stream order),
   and records each storage's `reload_event`. The model part's output
   triggers the last group.
2. Group `i`'s reloads overlap block `i + 1`'s backward, roughly 2x its
   forward (more with the replay), so a volume that does not stall the
   forward rarely stalls the backward.
3. Block `i`'s `unpack` makes the compute stream wait on the storage's
   `reload_event`. Each reloaded copy is freed with autograd's saved tensor.
   A storage no trigger reloaded (its block's output got no gradient) raises
   at `unpack`.

Reloads are queued in pack order, which suits the replay: it reads the block
input and the `SAVE_OUTPUT`s first, the `BACKWARD` saves only in the gradient
pass. Queuing `BACKWARD` saves last would help only when reloads barely
finish in time (TODO).

## Design choices

**Offload around remat, at remat's granularity.** The nesting that works is
`offload(remat(block))`: the block's checkpoint runs inside the offloader's
saved-tensor hooks, so remat first decides what the block keeps (its inputs,
and what its save regions retain) and `pack` sees exactly that. Offloading
attention and MLP apart under block remat reverses it to
`remat(offload(sub-block))`, and the offloader no longer sees what remat
keeps. For example, remat packs the block input when the checkpoint is
entered, before any sub-block's hooks are active, so it would never be
offloaded.

So offload and remat share a granularity, offload outside. We align with
today's remat boundary: one transformer block per offload group. Finer groups
need finer remat checkpoints (Megatron does both per submodule).

### The last block

Only the loss runs between the last block's forward and its backward, so
offloading it would pay two transfers to free memory for almost no time, and
both the free point and the reload would likely stall. By default
(`offload_last_block=False`) the last block of each model part is not hooked
and keeps everything on GPU. This overrides the base offload fields; a block
override that offloads that block raises, since under PP which block is last
depends on the stage split. Offloading it pays off only
when the peak falls between its forward and backward, e.g. a large-vocabulary
loss: its activations are freed before the logits are allocated.

Under PP every stage has this problem, at every stage boundary: the forward
leaves the stage right after its last block, and that block's backward starts
as soon as the gradient arrives. `offload_last_block=False` keeps one block
per in-flight microbatch on GPU on every stage.

## Bandwidth

Measured on GB300 with nvbandwidth on an idle host (GB/s; two GPUs per Grace
CPU, 900 GB/s of NVLink-C2C per Grace, so 225 per GPU per direction on paper):

| Traffic | Per GPU, each direction | Grace total |
|---|---|---|
| one GPU, one direction | 212 host-to-device, 194 device-to-host | |
| one GPU, both directions | ~170 | ~340 |
| two GPUs, one direction | ~175 host-to-device, ~183 device-to-host | 350-365 |
| two GPUs, both directions | ~85 | ~340 |

Each GPU's link carries about 200 GB/s each way, but all host traffic of a
Grace, in either direction and from either GPU, shares about 350 GB/s. CPU
memory traffic on that Grace (a CPU optimizer step, data loading, FSDP2 CPU
parameter offload) draws on the same budget; a CPU memory copy roughly halves
it. So:

- Without PP, offloads (in forward) and reloads (in backward) get about
  180 GB/s per GPU.
- With PP, one microbatch's offloads and another's reloads can overlap; while
  they do, each gets about 85 GB/s per GPU.
- A block's offloads are hidden when its offloaded bytes at that rate take no
  longer than the next block's forward; its reloads, when they take no longer
  than the following block's backward.

## FSDP2

The design borrows FSDP2's structure: per-block hooks, one stream per transfer
direction, event waits between streams, and a backward prefetch one block
ahead, triggered by a hook on the block's output gradient.

**Why not inside FSDP2.** Offload must understand torch_remat (what is saved
vs recomputed, save-region names, `is_recomputing()`); torch_remat is a
separate package, and remat concepts do not belong in pytorch/pytorch's
FSDP2. (It should also work without FSDP, a weaker reason: degree 1 FSDP
would do.)

**Hook order.** `parallelize` applies remat, and with it offloading, before
`fully_shard`. FSDP2 registers its pre-forward hook with `prepend=True`, so it
still runs first; its post-forward runs after ours:

| When | First | Then |
|---|---|---|
| block pre-forward | FSDP2: unshard parameters | offload: open the group, enter the saved-tensor hooks |
| block post-forward | offload: close the group, free the previous group, register the reload trigger | FSDP2: reshard; register pre-backward on the output |
| block output gradient ready | FSDP2: unshard this block, prefetch the previous block | offload: reload the previous block's group |

The post-forward hooks do not depend on each other: offload never touches
parameters, and the reload trigger is not ordered by registration (below).

In backward the order holds however they were applied: FSDP2's pre-backward
is a tensor hook on the output, the reload trigger a pre-hook on the output's
autograd node, and autograd runs a node's tensor hooks first. It matters:
unsharding runs the all-gather copy-out on the compute stream, which starts
with a small H2D copy of `split_with_sizes_copy`'s metadata. The reload
stream waits on the compute stream, so the reload queues behind that copy;
issued before it, the small copy could wait in the copy engine for the whole
reload, delaying a block that does not read it (seen on H100 in
pytorch/torchtitan#467; not reproduced on GB300).

**FSDP2 CPU parameter offload** (`CPUOffloadPolicy`) is the one FSDP2 traffic
on the CPU-GPU link: parameter shards host-to-device in each unshard's
copy-in, gradient shards device-to-host after each reduce-scatter. In
forward, its copies run opposite to the offloads but share the Grace's
bandwidth (see [Bandwidth](#bandwidth)). In block `i`'s backward,
block `i - 1`'s parameter shards (prefetch, `all_gather_copy_in_stream`) and
activations (reload, `_reload_stream`) go the same direction; both become
ready when block `i`'s copy-out finishes and share the bandwidth in no
defined order. The two compose (a unit test checks gradients bitwise on two
ranks), but when the copies do not fit in block `i`'s backward, parameters
first is better: they still need the all-gather after their copy, the
activations need nothing more. Shared, both finish after `(P + A) / BW` and
the all-gather starts only then; parameters first saves up to
`min(all-gather, A / BW)` (P, A: parameter and activation bytes; BW: link
bandwidth). Under `set_unshard_in_current_stream` (TorchTitan's
`fsdp_unshard_in_current_stream`) or a single rank, FSDP2 copies parameters
on the compute stream, so reloads already follow them.

TODO: make the reload stream also wait on `all_gather_copy_in_stream`, where
FSDP2 queues block `i - 1`'s parameter copies just before the trigger runs (a
no-op wait in the modes above). It needs FSDP2 to expose that stream; today
it is only reachable through FSDP2's private comm context.

## Pipeline parallelism

Basic PP needs no changes: each stage is a model part with its own offloader,
and groups are per block call, kept alive by each forward's autograd graph,
so several microbatches can be in flight per stage. The cost is each stage's
last block (see [The last block](#the-last-block)).

To offload it too, the schedule drives the stage boundary, as it drives
FSDP's UNSHARD/RESHARD. It names microbatches by the index its actions
already carry (`_Action(stage_index, computation_type, microbatch_index)`), so
the offloader assumes no particular schedule or backward order:

| Call | When | Effect |
|---|---|---|
| `defer_to_schedule()` | once, at setup | the schedule, not the model part's hooks, frees and reloads each microbatch's last group |
| `set_microbatch(i)` | before the stage's forward of microbatch `i` | tags the forward; an untagged forward (e.g. metadata inference) runs as without a schedule |
| `wait_offload(i)` | after an action following that forward | frees microbatch `i`'s last group, whose copies had that action's compute to finish |
| `reload_backward(i, num_groups=1)` | ahead of microbatch `i`'s backward | reloads its last `num_groups` groups (all blocks pays off only for stages of one or two blocks) |

Inside the stage, gradient hooks still prefetch one block ahead, and the
model part's output keeps its reload trigger as a fallback. With
`ActivationRematConfig.pp_schedule_driven`, `TrainingEngine` calls
`defer_to_schedule()` on each stage's offloader and registers it with the
schedule (`register_activation_offloader`), which then makes the other calls.

## Pinned host memory and CUDA graphs

**Why pinned.** Mainly so the copies do not block the CPU. A `non_blocking`
copy to or from pageable memory returns only after the data has moved, so the
CPU thread that also queues the forward and backward kernels would stall on
every transfer. Pinned memory also works under CUDA graph capture, which
rejects copies from or to pageable memory.

**Why a pool.** Allocating a pinned buffer per offload and dropping it after
the reload would let PyTorch's caching host allocator recycle it, except
under CUDA graphs, which TorchTitan uses for the whole step by default: the
allocator never reuses a block allocated during capture, so every captured
microbatch would hold fresh pinned memory for the life of the process. Each
offloader keeps its own pool (`_PinnedBufferPool`):

1. An offload takes a free buffer of the storage's size rounded up to a power
   of two, or allocates one. The caching host allocator rounds to that size
   anyway, and rounding lets storages whose size changes every step (the
   tokens routed to an EP rank) share buffers.
2. When a reload's H2D copy is queued, its buffer returns with the copy's
   `reload_event`.
3. The next offload into it makes the offload stream wait on that event, so
   reuse is ordered on the GPU.

**Size.** The pool grows to the most bytes offloaded and not yet reloaded at
once: one microbatch's worth per rank with gradient accumulation, times the
microbatches in flight under PP, times the rounding (none for power-of-two
sizes, e.g. a bf16 32K x 8192 block input is 512 MiB; under 2x in general).
Buffers live as long as the offloader.

**Capture.** The hooks run once, during capture, and the copies, free points
and reload triggers replay unchanged (static shapes). Every cross-stream
dependency is an event wait and every GPU copy is allocated on the compute
stream, so all of it is capturable, and the pool keeps the host addresses
baked into the graph valid. A unit test checks a captured two-microbatch step
is bitwise equal to eager and shares buffers with eager steps.

**Graph boundary.** Reuse inside a graph is correct only if the graph holds
both the offload and the reload of every transfer: a replay reuses the host
addresses recorded at capture, and nothing picks a new buffer. Both
TorchTitan graph modes do:

| Mode | One graph holds | Pinned memory |
| --- | --- | --- |
| default | the whole `forward_backward`: every accumulation microbatch, with PP the whole schedule | the eager peak |
| `accelerator_graph_per_microbatch_group` | one microbatch group (with PP, one schedule step), replayed per group | one group's peak; replays run one after another |

Separate forward and backward graphs with several microbatches in flight
(e.g. PP 1F1B) would break it: replaying the forward for microbatch 2
overwrites buffers microbatch 1's backward has not reloaded. That needs a
buffer set per in-flight microbatch. Eager mode is unaffected, so a failure
only with CUDA graphs points here. Not enforced yet (TODO in
`_PinnedBufferPool`).

## Other compositions

- **Remat.** Offload hooks are forward hooks on the block (`__call__`), so
  they enclose `remat.checkpoint`; the backward replay re-runs the block's
  forward and is skipped (`remat.is_recomputing()`).
- **Dist-MoE.** Expert intermediates live in Dist-MoE's own buffers, not in
  autograd saves, so offload cannot reach them (28.95 GiB per rank at Muse
  Spark 816B). The region's output and routing metadata are ordinary saves
  and are offloaded.
- **GraphTrainer** rejects a remat policy that offloads: the eager hooks do
  not run in a traced step, which has its own CPU-offload graph pass.
