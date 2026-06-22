# GroupedOwned Reshard-After-Forward Investigation

## Context

This note records why `reshard_after_forward` (RAF) does not reduce peak
memory for FlexShard `GroupedOwned` expert buckets in the DSV3 1D setup.

The investigated workload was:

- TorchTitan FlexShard DSV3 16B
- DP8 / EP1, so expert buckets use the FSDP mesh
- `seq_len=32`, local batch size 1
- activation checkpointing off
- compile off
- memory snapshots enabled

Snapshot artifacts:

- RAF always:
  `/data/users/weif/code-review/agent_space/torchtitan_flex_dsv3_1d_raf_always_warm_mem_20260620_011713/memory_snapshot`
- RAF never:
  `/data/users/weif/code-review/agent_space/torchtitan_flex_dsv3_1d_raf_never_mem_20260620_011248/memory_snapshot`
- Parsed summary:
  `/data/users/weif/code-review/agent_space/flex_dsv3_1d_raf_memory_summary.json`
- Fixed RAF alias-aware run:
  `/data/users/weif/code-review/agent_space/torchtitan_flex_dsv3_1d_raf_alias_lru2_mem_20260621_170142/memory_snapshot`
- Fixed parsed summary:
  `/data/users/weif/code-review/agent_space/flex_dsv3_1d_raf_alias_lru2_memory_summary.json`

## What The Snapshots Show

RAF is active for `GroupedOwned`: backward recomputes the expert all-gather.

Per rank, per training step:

| Mode | GroupedOwned AG recv alloc count | GroupedOwned AG recv total alloc |
|---|---:|---:|
| `reshard_after_forward=never` | 26 | 26.81 GiB |
| `reshard_after_forward=always` | 52 | 53.62 GiB |

The doubled allocation count and bytes mean the RAF backward recompute path is
running.

However, peak live GroupedOwned gathered memory does not improve:

| Mode | Peak live GroupedOwned gathered memory |
|---|---:|
| `reshard_after_forward=never` | 26.81 GiB |
| `reshard_after_forward=always` | 27.84 GiB |

The end-of-step live bytes for the GroupedOwned gathered buffers are `0.00 GiB`
in both runs, so this is not a step-boundary leak. The problem is forward-window
retention: autograd still saves views that keep the original forward gathered
expert buffers alive until backward.

## Why Shard Works

The `Shard` placement creates standalone full-parameter tensors during unshard.
For the DSV3 dense/non-expert buckets, the relevant path is:

- `Shard.prepare_unshard_bucket()` allocates the all-gather result.
- `Shard.finish_prepared_unshard()` copy-outs into full parameter tensors.
- The exposed parameter owns its storage.
- If forward later saves a transpose/view of that parameter, the saved tensor's
  `_base` is the registered full parameter.

The RAF saved-tensor hook can then replace the saved view with a
`_RafSavedTensorView`, which recomputes the full parameter in backward.

The snapshots match this:

- `Shard` AG recv allocation count roughly doubles under RAF.
- Peak Shard AG live memory stays bounded.
- Only one small prefetched Shard unshard remains live at the snapshot boundary.

## Why GroupedOwned Does Not Work

`GroupedOwned` deliberately uses view-out for DSV3 expert weights to preserve
the packed grouped-mm layout:

- `GroupedOwned.prepare_unshard_bucket()` allocates `gathered`:
  `world_size * layout.padded_rank_numel`.
- `GroupedOwned.finish_prepared_unshard()` calls
  `_views_from_padded_gathered()`.
- For DSV3 expert tensors, `_views_from_padded_gathered()` returns `w1`, `w2`,
  and `w3` as direct `as_strided` views into the large `gathered` buffer.

DSV3 grouped-mm then uses:

```python
torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets)
torch._grouped_mm(x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets)
torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets)
```

Since `w1`, `w2`, and `w3` are themselves views, the transpose view's `_base`
is the root gathered buffer, not the exposed parameter view.

The current RAF saved-tensor hook registers only the exposed parameter tensor:

- `UnshardedParamSlot.get_unsharded_param()` registers `current`.
- `_RafSavedTensorContext.pack()` checks:
  - exact tensor id
  - `tensor._base` id

For a saved grouped-mm transpose:

- exact tensor id is the transpose, not registered
- `tensor._base` is `gathered`, not registered
- the registered `w1`/`w2`/`w3` view is skipped

So the hook returns the raw tensor. Autograd saves that tensor and pins the
entire gathered buffer through backward.

## Minimal Alias Repro

The key difference is the `_base` chain.

```python
import torch
from torchtitan.experiments.flex_shard.flex_shard.unsharded_param_getters import (
    _RafSavedTensorContext,
)

class Handle:
    def __init__(self, tensor):
        self.tensor = tensor

    def unpack_raf_saved_tensor(self):
        return self.tensor

def pack_result_for_grouped_mm_weight(weight):
    ctx = _RafSavedTensorContext()
    ctx.register(weight, Handle(weight))

    saved = weight.bfloat16().transpose(-2, -1)
    packed = ctx.pack(saved)
    return saved._base is weight, saved._base is weight._base, type(packed).__name__

# Shard-like: full parameter owns storage.
shard_weight = torch.empty(2, 16, 16, device="cuda", dtype=torch.bfloat16)
print(pack_result_for_grouped_mm_weight(shard_weight))
# (True, False, "_RafSavedTensorView")

# GroupedOwned-like: full parameter is a view into a larger gathered buffer.
gathered = torch.empty(4, 16, 16, device="cuda", dtype=torch.bfloat16)
grouped_owned_weight = gathered.as_strided(
    (2, 16, 16),
    (256, 16, 1),
    storage_offset=0,
)
print(pack_result_for_grouped_mm_weight(grouped_owned_weight))
# (False, True, "Tensor")
```

The Shard-like case is packed into a RAF view handle. The GroupedOwned-like case
is saved as a raw tensor and therefore retains `gathered`.

## Root Cause

`GroupedOwned` RAF is defeated by root-storage aliasing:

```text
gathered buffer
  -> w1/w2/w3 expert parameter views
      -> transposed grouped-mm weight views saved by autograd
```

The RAF hook knows about `w1/w2/w3`, but the saved transpose points directly at
`gathered`. Since `gathered` has no RAF handle, the hook cannot replace the saved
tensor with a recompute handle.

## Fix Options

### Option 1: Disable GroupedOwned view-out under RAF

The conservative fix is to avoid returning views into `gathered` when
`reshard_after_forward=True`.

Instead:

1. Copy out `w1`, `w2`, and `w3` into standalone full-parameter tensors.
2. Register those standalone tensors with the existing RAF saved-tensor hook.
3. Let saved transpose views be packed as `_RafSavedTensorView`.

This should restore expected RAF behavior quickly, but it adds copy-out cost and
loses the zero-copy view-out path.

### Option 2: Make RAF alias-aware for gathered bases

The better long-term fix is to teach RAF about root gathered buffers.

Possible design:

1. When a placement returns view-out tensors, expose optional RAF alias metadata:
   - root base tensor
   - per-param view size, stride, and storage offset
   - bucket/param handle needed to recompute the bucket
2. Register a bucket-level RAF handle for the gathered base, not only per-param
   handles for `w1/w2/w3`.
3. In `_RafSavedTensorContext.pack()`, when a saved tensor's `_base` is the
   gathered base, pack it as a view relative to a recomputed gathered base.
4. In unpack, recompute the GroupedOwned bucket once and reconstruct the saved
   tensor with `torch.as_strided()`.

This preserves grouped-mm view-out and should restore RAF memory behavior, but
it requires placement/runtime API changes. It is not enough to register the
param view's storage, because saved transpose offsets are relative to the root
gathered base.

## Implemented Fix

The implemented fix uses Option 2, with one extra detail discovered during
iteration.

### 1. Register root gathered bases

`_RafSavedTensorContext.register()` now also checks whether the registered tensor
is a view. If it is, and the tensor's handle can provide a base handle, the
context registers `tensor._base` as well.

For GroupedOwned this means:

```text
registered w1/w2/w3 view -> also register gathered base
```

Then when grouped-mm saves `w1.transpose(-2, -1)`, the saved tensor's `_base`
matches the registered gathered base and packs as `_RafSavedTensorView` instead
of being saved as a raw tensor.

`RafSavedUnshardedParam` provides the base handle via
`base_handle_for_raf_saved_tensor()`. The base handle recomputes the bucket,
finds the recomputed full param, and returns that param's root `_base`.

### 2. Bound the RAF recompute cache

Alias-aware packing alone was not enough.

After the first implementation, the original forward gathered buffers were no
longer saved by grouped-mm, but `BucketCommContext.raf_saved_unshard_cache`
kept every recomputed bucket's full params until the end-of-backward callback.
For GroupedOwned view-out, those recomputed full params again alias large
gathered bases, so memory still accumulated across all 26 expert layers.

The cache is now a tiny LRU with limit 2:

- it keeps repeated saved tensor unpacks within the same local backward window
  from recomputing unnecessarily;
- it prevents all layer expert gathered buffers from accumulating;
- it still clears at the end-of-backward callback.

The limit of 2 matters. A limit of 1 fixed memory but caused extra GroupedOwned
recomputes because same-layer dense/expert saved-tensor unpack can evict each
other. Limit 2 restored the expected one backward recompute per layer while
keeping memory bounded.

## Confirmed Results

Final DSV3 16B DP8/EP1 RAF run:

```text
/data/users/weif/code-review/agent_space/torchtitan_flex_dsv3_1d_raf_alias_lru2_mem_20260621_170142
```

Per rank, per step:

| Metric | Before fix, RAF always | Alias only | Alias + LRU2 |
|---|---:|---:|---:|
| GroupedOwned AG recv alloc count | 52 | 52 | 52 |
| GroupedOwned AG recv total alloc | 53.62 GiB | 53.62 GiB | 53.62 GiB |
| Peak live GroupedOwned gathered memory | 27.84 GiB | 26.81 GiB | 2.06 GiB |
| Peak live GroupedOwned gathered buffers | 27 | 26 | 2 |
| Current live GroupedOwned gathered at snapshot | 0.00 GiB | 0.00 GiB | 0.00 GiB |

Step 2 in the final run:

| Metric | Value |
|---|---:|
| GroupedOwned AG recv alloc count | 104 |
| GroupedOwned AG recv total alloc | 107.25 GiB |
| Peak live GroupedOwned gathered memory | 2.06 GiB |
| Peak combined unshard memory | 2.44 GiB |
| Current live GroupedOwned gathered at snapshot | 0.00 GiB |

The job log also dropped from roughly 60 GiB step memory in the broken RAF run
to roughly 34 GiB in the fixed RAF run.

## Follow-Ups

The current fix preserves GroupedOwned view-out and restores the expected RAF
memory behavior. Follow-up work should focus on whether LRU size 2 is the right
policy for all models:

- make the cache limit configurable if another model has a wider saved-tensor
  interleave window;
- profile whether the LRU ever causes extra recomputes at larger batch/sequence
  sizes;
- consider per-bucket saved-tensor reference counting if a future workload needs
  exact lifetime instead of a small bounded cache.

The older fallback recommendation remains useful only if alias-aware RAF is
disabled or regresses:

- gate GroupedOwned view-out off when RAF is enabled;
- rerun the DSV3 16B memory snapshot comparison;
- verify GroupedOwned peak gathered memory drops from roughly 27 GiB to a small
  bounded prefetch amount, similar to Shard.
