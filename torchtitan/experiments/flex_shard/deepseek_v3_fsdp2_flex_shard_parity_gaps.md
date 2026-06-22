# DeepSeek V3 FSDP2 vs FlexShard Parity Gaps

## Goal

Run an authentic apple-to-apple comparison between the original TorchTitan
DeepSeek V3 recipe under FSDP2 and the same recipe under FlexShard.

The clean target is:

```text
CONFIG=deepseek_v3_16b
```

with the recipe unchanged, and only the data-parallel backend switched between:

- FSDP2
- FlexShard

The current experimental FlexShard DSV3 entry point is useful for isolating
FlexShard mechanics, but it still uses a separate module/config path, so it is
not yet a strict backend-only comparison.

## Current Divergence

`flex_shard_deepseek_v3_16b_dp8()` starts from `deepseek_v3_16b()` and now
preserves the original DP/EP/AC/loss-compile recipe knobs.

The original `deepseek_v3_16b()` recipe uses:

- `local_batch_size = 4`
- `seq_len = 4096`
- `expert_parallel_degree = 8`
- selective activation checkpointing
- loss compile enabled
- `data_parallel_shard_degree = -1`
- `pipeline_parallel_schedule = "Interleaved1F1B"`

So the current FlexShard DSV3 path is closer to the original recipe, but still
is not a strict backend-only switch because it uses a separate module/config
path.

## Remaining Gaps

### Recipe Knob Status

The FlexShard 16B config now inherits the original DSV3 recipe knobs, including
loss-only compile. The FlexShard parallelize path still explicitly rejects
unsupported main-path features:

- PP
- TP
- CP
- HSDP / DP replicate
- CPU offload
- model compile

Loss compile remains allowed because the original DSV3 16B recipe uses
`CompileConfig(enable=True, components=["loss"])`.

### 1. Reuse the main DSV3 parallelization flow

Today `torchtitan/experiments/flex_shard/deepseek_v3/parallelize.py` carries a
simplified copy of the main DSV3 parallelization path.

For parity, the better shape is to keep the main DSV3 flow shared and inject
only the final data-parallel wrapper:

```text
apply_fsdp(...)       # FSDP2 baseline
apply_flex_shard(...) # FlexShard comparison
```

This avoids hidden differences in CP, TP, EP, AC, compile ordering, and mesh
construction.

### EP Parity Status

The original 16B recipe has:

```python
expert_parallel_degree = 8
```

The FlexShard 16B config now inherits that value from `deepseek_v3_16b()`.
It also preserves the same `apply_moe_ep_tp(...)` behavior and the same
`ep` / `efsdp` mesh structure.

On a single 8-GPU node, EP8 may mean routed experts are mostly EP-local and no
longer stress the same GroupedOwned FSDP-over-expert path. That is acceptable
for recipe parity: the point is to compare the original DSV3 recipe, not to
force the workload to exercise GroupedOwned.

### 2. Validate AC plus RAF together

The RAF alias-aware fix restores expected memory behavior for GroupedOwned
view-out, but the authentic recipe uses selective activation checkpointing.

The parity run should validate FlexShard with:

```python
activation_checkpoint.mode = "selective"
```

which the FlexShard 16B config now inherits from `deepseek_v3_16b()`.

### Loss Compile Status

Original DSV3 16B enables compile for the loss only:

```python
CompileConfig(enable=True, components=["loss"])
```

FlexShard rejects model compile, but the original recipe does not compile the
model. Loss compile is allowed, so this recipe knob now matches FSDP2.

### 3. Match prefetch and overlap behavior

FSDP2 adds explicit prefetching when EP is enabled. FlexShard needs equivalent
prefetch behavior, or the comparison must deliberately constrain both backends
to the same prefetch/overlap policy.

Otherwise, profiler differences may reflect prefetch policy rather than the
placement backend itself.

### 4. Align bucket policy with FSDP2 wrapping boundaries

FlexShard currently hand-authors DSV3 buckets per layer and parameter class.

For a cleaner comparison, bucket boundaries should intentionally mirror the
FSDP2 wrapping structure:

- token embeddings
- one bucket or group per transformer block
- routed expert handling matching the EP/FSDP mesh split
- final norm and lm head

The bucket policy does not need to be byte-for-byte identical to FSDP2 internal
flattening, but it should not change the high-level unit of communication unless
that is the feature being measured.

## EP4 Comparison Run

I ran a single-node EP4 comparison with the closest currently supported recipe
parity:

```text
NGPU=8
data_parallel_shard_degree=8
expert_parallel_degree=4
seq_len=4096
local_batch_size=4
activation_checkpoint.mode=selective
compile.enable=True, components=["loss"]
mixed_precision_param=bf16
mixed_precision_reduce=fp32
dataloader.dataset=c4_test
hf_assets_path=./tests/assets/tokenizer
```

The local `c4_test` dataset/tokenizer override avoids external data dependency;
both backends used the same override.

### Required FlexShard Fix

The first FlexShard attempt failed before profiling:

```text
NotImplementedError: There was no rule registered for HigherOrderOperator
inductor_compiled_code and mode _MarkRecomputeTorchDispatchMode
```

Root cause: selective AC recompute runs compiled FlexAttention under
FlexShard's RAF recompute dispatch mode. That mode wrapped the SAC recompute
mode but did not declare higher-order-operator support, so PyTorch rejected the
compiled FlexAttention higher-order op.

The narrow fix was:

```python
class _MarkRecomputeTorchDispatchMode(TorchDispatchMode):
    supports_higher_order_operators = True
```

After this, the focused RAF plus activation-checkpointing test passed and the
EP4 FlexShard run completed.

### Artifacts

Memory snapshot runs:

- FSDP2:
  `/data/users/weif/code-review/agent_space/torchtitan_dsv3_16b_ep4_fsdp2_20260621_172310`
- FlexShard:
  `/data/users/weif/code-review/agent_space/torchtitan_dsv3_16b_ep4_flex_shard_20260621_172728`

Clean profile-only runs:

- FSDP2:
  `/data/users/weif/code-review/agent_space/torchtitan_dsv3_16b_ep4_fsdp2_profile_20260621_172950`
- FlexShard:
  `/data/users/weif/code-review/agent_space/torchtitan_dsv3_16b_ep4_flex_shard_profile_20260621_173105`

Rank0 Perfetto links:

- FSDP2: `https://fburl.com/8xatelmu`
- FlexShard: `https://fburl.com/x7wzr5rs`

### Trace Summary

Clean rank0 profile window:

| Metric | FSDP2 | FlexShard |
|---|---:|---:|
| `ProfilerStep#4` duration | 1.959 s | 2.053 s |
| NCCL all-gather kernels | 112 | 117 |
| NCCL all-gather kernel time | 141.7 ms | 237.6 ms |
| NCCL reduce-scatter kernels | 60 | 63 |
| NCCL reduce-scatter kernel time | 307.2 ms | 98.3 ms |
| NCCL send/recv kernels | 130 | 130 |
| NCCL send/recv kernel time | 352.6 ms | 355.5 ms |

The high-level result is that FlexShard runs, but is about 5% slower in this
rank0 trace window. It also shifts where communication cost appears:

- FSDP2 has higher reduce-scatter kernel time.
- FlexShard has higher all-gather kernel time.
- Token-dispatch send/recv work is effectively identical.

FlexShard-specific trace costs:

| FlexShard range | Count | Inclusive time |
|---|---:|---:|
| `FlexShard::grouped_owned_reduce_scatter_copy_in` | 52 | 916.6 ms |
| `FlexShard::copy_in` | 234 | 82.6 ms |
| `FlexShard::copy_out` | 130 | 46.6 ms |
| `FlexShard::post_backward_reduce_grad_wait` | 9 | 72.1 ms |

The biggest remaining profiler gap is the GroupedOwned gradient pack/copy-in
path. Even with the Triton pack path available, the EP4 trace still shows large
CPU/inclusive time in the per-layer GroupedOwned reduce-scatter copy-in ranges.

### Memory Snapshot Summary

Step-4 memory snapshots:

| Metric | FSDP2 | FlexShard |
|---|---:|---:|
| Current allocated, mean per rank | 30.90 GiB | 30.98 GiB |
| Current allocated, max per rank | 30.90 GiB | 30.98 GiB |
| Reserved, mean per rank | 76.61 GiB | 78.86 GiB |
| Reserved, max per rank | 77.42 GiB | 84.50 GiB |
| Reconstructed trace-window peak, mean per rank | 67.24 GiB | 68.76 GiB |
| Reconstructed trace-window peak, max per rank | 68.22 GiB | 74.79 GiB |

Current live allocation at the snapshot boundary is nearly identical, so there
is no obvious end-of-step leak. The remaining memory gap is peak/reserved:
FlexShard is roughly 1.5 GiB higher on mean peak and roughly 6.6 GiB higher on
the worst ranks in this run.

The worst FlexShard ranks were ranks 3 and 7. The reconstructed peak occurred
inside `ProfilerStep#1`, with large live allocations from model compute plus
FlexShard runtime buffers. This needs a cleaner allocator-lifetime breakdown,
but it is already enough to say current FlexShard EP4 does not yet match FSDP2
peak memory.

### Correctness/Scaling Gap

Resolved for the current experimental DSV3 path.

FlexShard now has bucket-level `gradient_reduce_op` metadata and a model-level
`disable_flex_shard_gradient_division(model)` API mirroring
`disable_fsdp_gradient_division(model)`. The DSV3 FlexShard parallelization path
calls it immediately after `flex_shard(model, buckets=...)`, so Shard,
Owned, GroupedOwned, RaggedShard, and GroupedRaggedShard reduce gradients with
SUM/no-division semantics when this API is enabled.

Focused distributed test:

```text
python -m pytest -q torchtitan/experiments/flex_shard/tests/test_flex_shard_training.py \
  -k "disable_gradient_division_uses_sum_reduce"
```

Result:

```text
1 passed, 4 deselected
```

EP4 DSV3 16B rerun:

| Backend | Step-1 grad norm |
|---|---:|
| FSDP2 | 8.5942 |
| FlexShard | 8.3248 |

Run folders:

```text
/data/users/weif/code-review/agent_space/torchtitan_dsv3_16b_ep4_fsdp2_sumapi_20260621_180110
/data/users/weif/code-review/agent_space/torchtitan_dsv3_16b_ep4_flex_shard_sumapi_20260621_180201
```

The old DP8-sized mismatch is gone: FlexShard is now in the same grad-norm
range as FSDP2 instead of being smaller by roughly 8x. The remaining difference
is not evidence of the previous gradient-division bug; it should be treated as
normal backend/run drift until the remaining recipe and implementation gaps are
closed.

### EP4 Remaining Gaps

After this run, the concrete EP4 parity gaps are:

1. GroupedOwned reduce-scatter pack/copy-in is still a visible CPU/profiler
   cost.
2. FlexShard all-gather kernel time is higher than FSDP2 in this EP4 run.
3. FlexShard peak/reserved memory is higher on the worst ranks despite similar
   current allocation at the snapshot boundary.

## Target End State

The desired end state is a backend switch, not a separate recipe:

```text
MODULE=deepseek_v3 CONFIG=deepseek_v3_16b DP_BACKEND=fsdp2
MODULE=deepseek_v3 CONFIG=deepseek_v3_16b DP_BACKEND=flex_shard
```

or an equivalent TorchTitan config field.

At that point, the comparison can be interpreted as:

```text
same model
same data
same seq_len and batch size
same EP/TP/CP/PP settings
same AC policy
same loss compile setting
same mixed precision policy
different DP backend
```

Until then, the experimental
`torchtitan/experiments/flex_shard/deepseek_v3` path should be treated as a
FlexShard development harness rather than an authentic FSDP2 parity baseline.
