# Tensor logging

Tensor logging records compact statistics about model tensors without retaining the tensors themselves.

```text
tensor [0, 1, -2, 3]
       |
       +-> count=4, zero_count=1, abs_sum=6, square_sum=14, abs_max=3
       |
       +-> fixed buffer slot -> one packed drain -> TensorBoard/W&B
```

Use it to find exploding activations and dead gradients across a distributed training job.

## Mental model

```text
1. Register metric names while modules are constructed.
2. `init()` freezes one globally ordered buffer slot for every name.
3. `log_stats()` writes sufficient statistics into those fixed slots.
4. On a selected step, `collect()` reduces the packed buffers and derives scalars.
5. TorchTitan publishes the scalars through its existing loggers.
```

The ordinary path does not inspect TP, CP, DP, EP, or PP meshes. It reports statistics over the tensor occurrences emitted by participating ranks. A metric that needs a particular semantic population reconstructs that value first, then uses the same `log_stats()` call.

## Enable it

Enable tensor logging and a metrics sink on an existing recipe:

```bash
NGPU=8 MODULE=qwen3 CONFIG=qwen3_debugmodel ./run_train.sh \
  --metrics.enable-wandb \
  --metrics.tensor-logging.enabled \
  --metrics.tensor-logging.freq 5
```

See the [observability README](../README.md) for TensorBoard and Weights & Biases setup.

Tensor work runs only when both its requested cadence and ordinary `metrics.log_freq` select the step. For tensor cadence 15 and scalar cadence 10, tensor metrics publish at steps 30, 60, and so on. Step 1 publishes tensor metrics only when their cadence is 1.

The default tensor cadence is 5. The default publication filter keeps `numel`, `nonfinite_count`, `abs_mean`, `square_mean`, and `abs_max`.

## Record a tensor

Register names in module construction, then record their current values at the callsite:

```python
from torchtitan.observability import tensor_logging


class Attention(nn.Module):
    def __init__(self, ...):
        super().__init__()
        tensor_logging.register(self, ["scores"])

    def forward(self, x):
        scores = self.compute_scores(x)
        tensor_logging.log_stats(self, scores=scores)
        return self.apply_scores(scores)
```

`register()` declares a short name on one module. The trainer calls `init()` after model parallelization and optimizer construction, joins the module path and short name, and assigns that full name a fixed buffer row. Emitting an unregistered name is an error.

### Record forward and backward values

```python
class Attention(nn.Module):
    def __init__(self, ...):
        super().__init__()
        tensor_logging.register_fwd_bwd(self, ["xq"])

    def forward(self, x):
        xq = self.query(x)
        tensor_logging.log_fwd_bwd_stats(self, xq=xq)
        return self.attend(xq)
```

This publishes:

```text
<module>.xq.x.<statistic>   forward tensor
<module>.xq.dx.<statistic>  gradient arriving at xq during backward
```

The call returns `None`; continue using the original `xq` tensor. The backward recorder is attached directly to it.

### Boundaries and internal tensors use the same API

```python
# Residual-branch boundary
tensor_logging.log_fwd_bwd_stats(
    transformer_block,
    attn_stream=residual,
    attn_out=attention_output,
)

# Internal attention projection
tensor_logging.log_fwd_bwd_stats(attention, xq=xq)
```

“Boundary” describes where the tensor sits in the model; it is not a different logging operation. TorchTitan does not infer a reduction mesh from either call.

## From slots to metrics

Each observation contributes mergeable sufficient statistics:

```text
sum_statistics = [numel, nonfinite_count, zero_count, observation_count,
                  abs_sum, square_sum, fourth_moment_sum]
maxima         = [abs_max]
```

For finite values, these slots derive `zero_frac`, `abs_mean`, `square_mean`, RMS, `abs_max`, and kurtosis about zero. For `[0, 1, -2, 3]` recorded once:

```text
numel=4
zero_frac=1/4
abs_mean=(0+1+2+3)/4=1.5
square_mean=(0+1+4+9)/4=3.5
abs_max=3
kurtosis=(0+1+16+81)/4/3.5^2 - 3 = -1
```

Adding another ordinary metric does not add another collective. On a selected step, all slots share two packed buffers and run SUM then MAX sequentially:

```text
float32 sum_statistics --SUM--> counts (at FP32 integer precision) and moments
float32 maxima         --MAX--> absolute maxima
```

Every rank allocates the same slot order. Under PP, ranks that do not own a metric contribute identity values for its slot.

Ordinary slots count physical observations. A replicated tensor contributes once per rank holding it, so absolute counts include replica multiplicity. Ratios such as `abs_mean`, `zero_frac`, and kurtosis are unchanged by uniform replication.

The LM-head recorder can run more than once per model forward when chunked loss is enabled. With eight loss chunks, `lm_head.output.observation_count` is eight times the count of a once-per-forward boundary; this is per-call accounting, not duplicate recording.

## Publication filter

`metrics.tensor_logging.publish_filter_regex` is an allowlist over dotted metric names:

```text
layers.0.attention.xq.x.abs_max
```

For example, `r"layers\.0\..*\.abs_max$"` accepts `layers.0.attention.xq.x.abs_max`, rejects the same metric under `layers.1`, and `r".*"` publishes everything. Filtering controls what gets logged, not what gets computed, so a narrow filter reduces sink volume but not GPU statistic collection.

## Compatibility

`✓` supported and tested
`✗` rejected during setup

```text
Configuration                                                   Status
DP / TP / CP / EP                                               ✓
spmd_types backend                                              ✓
torch.compile(fullgraph=True)                                   ✓
Full activation checkpointing                                   ✓
CUDA graphs                                                     ✓
Gradient accumulation                                           ✓
Graph Trainer, in-process and PP disabled                       ✓
FaultTolerantTrainer / TorchFT                                  ✗

DP × CP × TP / EP + FullAC + spmd_types                         ✓
DP × PP × TP / EP + compile + FullAC + spmd_types               ✓

Pipeline parallel, 1F1B                                         ✓
Pipeline parallel, Interleaved1F1B                              ✓
Other pipeline schedules                                        ✗
Graph Trainer + pipeline parallel (GraphPP)                     ✗
Graph Trainer precompiled artifacts                             ✗
```

The first compound row is a previously validated eight-rank eager CP topology: DP-shard 2 × CP 2 × TP 2, with EP 2 carved. The second is the Stage-A eight-rank compiled PP topology: DP-shard 2 × PP 2 × TP 2, with EP 2 carved. Unsupported modes fail during setup rather than publishing incomplete metrics.

The Triton fast path handles tensors whose PyTorch device type is `cuda`, including ROCm's CUDA-compatible PyTorch API. Other device types use the ordinary PyTorch fallback path.

## Execution details

### What happens on a non-logging step?

In eager mode, the logging helper returns before it scans the tensor.

In a compiled or CUDA-captured forward/backward, the logging operation stays in the graph so the same graph can be reused on every step. It reads a device value named `enabled`. When `enabled` is 0, the Triton kernel returns before scanning the tensor or changing the statistic buffers.

Code that builds an optional value before `log_stats()` can still run if that code was included in the compiled graph. For example, `scores.mean(dim=0)` can run even though `log_stats(router, scores=...)` does not update a buffer.

For a CUDA-captured tensor whose layout needs `.contiguous()`, that copy is also part of the captured graph and still runs. The `enabled=0` check prevents the statistics scan and buffer update, not earlier work used to prepare the value.

The packed SUM/MAX all-reduces, CPU derivation, and publication do not run on a non-logging step. They happen in `collect()` after forward/backward and the optimizer step, outside the compiled or captured function.

### Why does changing cadence not create a new graph?

The logging operations stay in the compiled or captured graph. Cadence changes only the scalar tensor `StatisticBuffers.enabled`:

```text
enabled = 0 -> logging kernel returns without changing buffers
enabled = 1 -> logging kernel scans the tensor and adds statistics
```

The graph contains the same operations in both cases, so it can be reused. Cadence does not remove calculations from the graph: code that builds a value for `log_stats()` still runs on every replay.

### Are reductions inside the graph?

No. The compiled or captured function contains forward, backward, and the device-side statistic updates. After that function returns on a logging step, trainer Python calls `collect()`.

`collect()` runs one packed SUM all-reduce, one packed MAX all-reduce, copies the two reduced tensors to CPU, derives scalar metrics, clears the buffers, and returns the metric dictionary. Trainer then passes that dictionary to the normal metrics logger. None of this runs on non-logging steps.

### How are backward statistics recorded?

`log_fwd_bwd_stats()` records `<name>.x` immediately and attaches an autograd hook. During backward, the hook receives the incoming cotangent and an ordered custom operation records `<name>.dx` in that metric's buffer row.

If the tensor does not require gradients, the forward `.x` statistics are still recorded but no `.dx` metric is observed.

### Why does activation checkpointing not double-count statistics?

Eager FullAC records the original forward. Its recompute context sets a flag that makes `log_stats()` and `log_fwd_bwd_stats()` return before recording the second forward. Under compiled FullAC, the checkpoint policy preserves the mutation operation, so recomputation does not execute the update again.

### Does PP divide metrics by pipeline degree?

No. The stage that owns a metric writes its row. Other stages contribute zero to additive fields and negative infinity to the maximum. WORLD SUM/MAX therefore reproduces the owning stage's values without dividing by PP degree.

Only `1F1B` and `Interleaved1F1B` are supported. Tensor logging rejects every other regular pipeline schedule during setup until a focused test proves complete forward and backward metric coverage.

### Which Graph Trainer modes are supported?

In-process, non-PP Graph Trainer tracing is supported: tensor-logging buffers exist before the lazy trace, and replay uses the same buffers and device cadence flag.

GraphPP is rejected during setup. Its copied FX forward graphs do not yet point at the live statistic buffers, so allowing the run would publish backward rows while silently omitting forward rows.

Separately generated or loaded precompiled artifacts are also rejected: their saved graph has no portable binding to the later process's metric names and live buffers.

### What performance optimizations are used?

- Buffer rows are allocated once during `init()` and reused.
- On a non-logging compiled or captured step, the device kernel returns before scanning the tensor.
- The Triton kernel scans contiguous storage. Common transposes and permutations become no-copy views; `.contiguous()` is used only when the layout still has gaps.
- Every additive field shares one SUM all-reduce and every maximum shares one MAX all-reduce, regardless of how many metrics are registered.
- The all-reduces and scalar derivation run only on logging steps and outside compiled or captured forward/backward.

## Current limitations

- The publication filter does not skip GPU collection.
- Publication is synchronous with the training step.
- Unsupported PP schedules, GraphPP, and separately precompiled Graph Trainer artifacts fail during setup.
- FaultTolerantTrainer rejects tensor logging during setup.
- Optional visualizations, asynchronous publication, and additional built-in metrics are follow-up work.
