# Selective-gather attention

Attention-gym owns sliding-window block routing, K/V exchange, assembly,
gradient return, and transport resources. TorchTitan maps attention inputs
to compact K/V and sets checkpoint policy.

## Supported attention

Use `SelectiveGatherKVSlidingWindowVarlenInnerAttention`.
Copy engines require one NVLink domain, which can span hosts.
PyTorch symmetric-memory setup checks peer access.

Sliding-window Varlen supports contiguous sharding and head-tail load balancing.
Decoder applies a configured balancer only when the model has full attention.
Each CP chunk must contain whole `gather_block_size` blocks. Attention-gym checks
the token ranges before building the exchange.
The default `gather_block_size=256` is a tested starting point. It rounds a
4096-token window's 4095-token lookback to 4096 tokens, adding one unused token
per halo. Use a smaller block when CP chunks are shorter than 256 tokens.
Its plan is a causal-window superset of packed-document demand.
TorchTitan passes each rank's `cp_query_ranges` to attention-gym's public
`build_sliding_window_plan`. Returned global block IDs locate K/V slices in
the compact buffer.
Document boundaries may cause safe overfetch. It keeps standard
`VarlenAttentionMetadata`; its
document offsets can change without changing the exchange plan.

Use `AllGatherKVVarlenInnerAttention` for full causal Varlen layers. A dense
head-tail plan can need almost all remote K/V, so worst-case selective-gather
buffers cost almost as much memory as all-gather.

```python
from torchtitan.config.transform import ContextParallelTransform
from torchtitan.models.common.attention import (
    SlidingWindowVarlenInnerAttention,
    VarlenInnerAttention,
)
from torchtitan.models.common.attention.cp_attention import (
    AllGatherKVVarlenInnerAttention,
    SelectiveGatherKVSlidingWindowVarlenInnerAttention,
)

transform = ContextParallelTransform(
    inner_attention_map={
        VarlenInnerAttention: AllGatherKVVarlenInnerAttention,
        SlidingWindowVarlenInnerAttention: (
            SelectiveGatherKVSlidingWindowVarlenInnerAttention
        ),
    }
)
```

## K/V memory

Sliding-window Varlen packs local K/V and its halos in original token
order. Overlapping ranges share the same blocks. The last rank's adjacent
head and tail chunks do not duplicate their shared keys. Output size is fixed
at the largest selected block count across CP ranks, with padding at the end.
Padding is uninitialized. Attention views exclude it.

Varlen uses one view for contiguous sharding and two for head-tail.
It does not reorder or copy the assembled K/V again. Document packing changes the
mask or offsets, not the transport plan or buffer shape.

Contiguous sharding needs at most one remote halo; head-tail needs two.
Each halo rounds the window's
lookback up to a transport block. With 128K local tokens, a 4096-token window,
and 256-token blocks, the assembled buffer holds at most 136K tokens:

| CP8 K/V buffer | Upper bound relative to local K/V |
| --- | --- |
| All-gather output | 8x |
| Compact output, without checkpointing | 1.0625x |
| Saved halo, with checkpointed attention | 0.0625x |

These are buffer-size bounds, not measurements of model peak memory.
They exclude other model activations and shared transport buffers.
Full attention remains on all-gather in the sliding-window recipes.

The saved activations depend on the remat policy:

| Sliding-layer policy | All-gather | Selective gather |
| --- | --- | --- |
| Save inner attention | Retain gathered K/V for backward. | Retain compact K/V for backward. |
| Full remat | Release gathered K/V and fetch it again in backward. | Retain remote halos and rebuild compact K/V locally in backward. |

With full remat, all-gather also releases its gathered K/V. Selective gather
retains halos to avoid fetching them again. Measure model peak memory for the
chosen policy; full attention layers and CUDA Graph pools also contribute.

## Benchmark

52-layer Muse Glimmer, CP16 on 16 GB300s in one NVLink domain, head-tail sharding,
256K packed tokens, and documents capped at 16K. The 39 sliding layers use a
2048-token window; all 13 full layers use all-gather in both variants.
Activations are BF16 with FA4 and CUDA Graphs.

| Remat | Sliding backend | Step (ms) | Peak allocated (GiB) | Peak reserved (GiB) |
| --- | --- | ---: | ---: | ---: |
| Full | All-gather | 2866.2 | 43.393 | 57.793 |
| Full | Copy engine | 2701.7 | 43.545 | 57.867 |
| Inner attention | All-gather | 2404.6 | 141.706 | 159.648 |
| Inner attention | Copy engine | 2227.7 | 141.857 | 159.139 |

Copy engine reduces step time by 5.74% under full remat and 7.36% under inner
remat. Peak allocated memory is 0.151 GiB higher in both pairs. Loss and grad_norm
match bit-for-bit across all 20 steps, both backends, and both remat policies.

Inner remat uses `save_regions=["*"]` and
`recompute_regions=["attention.inner_attention*"]`.

Step time averages the slowest rank over 17 graph replays on one allocation.
Allocator peaks include build, warmup, capture, and training. Symmetric-memory
buffers can be outside the PyTorch allocator.

## Checkpointing

See [Selective-gather attention in REMAT.md](REMAT.md#selective-gather-attention)
for halo retention, assembly recomputation, and save patterns.

## Setup and teardown

Install the attention-gym checkout that provides `CopyEngineBlockExchange`.
The base TorchTitan dependency does not provide this API.
The checkout must provide `build_sliding_window_plan(local_ranges, ...)`.
Use attention-gym `4bc108cc689632cef6a1ab3b90740496d3247272`.

<!-- TODO(fegin): Pin an attention-gym release that includes block exchange. -->

```bash
pip install -e /path/to/attention-gym
```

Use FA3 on Hopper or FA4 on Blackwell for Varlen.

Warm up every exchange before CUDA Graph capture. Static exchange objects
are built on the first eager forward. Layers with matching CP groups, plans,
K/V shapes, dtypes, and devices share landing zones. Changed
token counts, shapes, dtypes, or CP groups are rejected.

Attention metadata is a PyTorch pytree. TorchTitan's graph wrapper copies its
tensors into stable graph inputs.

Order eager exchange after a graph replay on another stream with `wait_stream`.
Do not replay graphs that share an exchange at the same time.
Release unused eager graphs outside capture. Symmetric-memory cleanup can
synchronize the device.

Attention modules keep their exchanges alive. The shared cache holds only weak
references and does not extend their lifetime. Tear down graphs before releasing
the model. Do not free exchanges during capture or replay.

## GPU tests

The H100 CI lane runs the four-GPU suite. It compares two layers sharing one
exchange against all-gather, including backward, unused-buffer poisoning,
changed documents, CUDA Graph replay, and a one-token window with no remote K/V.
The generic A10G lane does not run this suite. It requires peer access and an
attention-gym build that provides block exchange.

The H100 integration recipe `fsdp+cp+varlen_selective_gather` uses Muse Glimmer
with FSDP2, CP2, head-tail balancing, full remat, and selective sliding layers.
Its full attention layers use all-gather.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python -m pytest -q tests/unit_tests/gpu/models/common/test_varlen_selective_gather.py
```
