# Fused DeepSeek-V3 overrides

The router override preserves native checkpoint parameters and distributed
ownership while specializing routing pointwise work and its backward. The
gate stays an ordinary BF16 linear; sequence auxiliary loss remains external.

## Authorship

Original implementation and tests: Yidi Wu (@ydwu4),
[#5258](https://github.com/pytorch/torchtitan/pull/5258), source revision
`a5d8990c5e9b46e3a87007b33de8e03c09fb97b7`. Copyright/license are preserved.
This integration adapts BF16 storage and per-branch gradient rounding, padding,
explicit round-robin benchmark routing, current model/metric APIs and required
dispatch checks. General-shape kernels and native reduction-order handling
are additional work in this PR. It is not an unmodified kernel or a wrapper-only port.

## Shape and launch contract

The fused API accepts nonempty CUDA BF16/FP32 logits `[T,E]`, valid top-k and
group counts, with optional bias and padding. Token counts, expert counts and
top-k are compile-time launch parameters, not a public benchmark-shape allowlist.
FP32 score/group arithmetic, normalization and native unsorted expert order
are preserved. Gradient reductions retain ATen's row/column-major accumulator
and warp order, including unaligned vector tails.

```text
native gate -> logits[T,E]
                    |
             fused operator
             /            \
 tuned 671B geometry       other legal geometries
 preserved CuTe kernel    parameterized Triton kernel
             \            /
      weights[T,K], IDs[T,K], map[T,E], counts[E], scores[T,E]
                    |
       two score-gradient branches -> native gate backward
```

The general forward zeros fresh integer counts and fuses sigmoid, group/top-k
selection, routing weights, the map and exact integer usage accumulation.
Its saved state is scores, selected values/IDs and one denominator per row.
Calls at different shapes do not share a shape-dependent arrival-counter budget.
The tuned CuTe path retains its existing stream-ordered counter and layouts.
Hardware-specific launch variants do not replace general fusion with eager code.

## Precision and execution

```text
BF16 hidden + BF16 gathered gate weight
                    |
             native BF16 GEMM
                    |
               BF16 logits
                    |
       fused FP32 sigmoid / group top-k
                    |
          +---------+------------+
          |                      |
 BF16 selected weights    FP32 unbiased scores
          |                      |
   routed expert path      external aux loss
          |                      |
 FP32 route derivative    FP32 aux derivative
          |                      |
     round BF16             round BF16
          +----------+-----------+
                     |
           BF16 gradient accumulation
                     |
          native gate DGRAD / WGRAD
```

The 671B learned recipe uses256 experts, top8, eight groups/four selected groups,
scale2.5 and sigmoid scores. Each backward branch follows NVIDIA's sigmoid
arithmetic and rounds at the BF16 logit boundary before accumulation. FP32
input behavior remains available for the source-compatible path. Padding does
not contribute usage counts; validation does not accumulate training usage or
auxiliary metrics. Forced round-robin is a declared benchmark deviation, not
a substitute for learned routing.

The PP2/PP4 recipes select learned or forced router factories with
`require_fusion=True`. Unsupported precision/mesh/device contracts fail explicitly;
other supported shapes remain fused. Standalone overrides allow native fallback
for unsupported contracts, such as CPU execution or non-sigmoid score functions.
GraphTrainer integration is out of scope. Kernel/unit checks do not establish
full-model convergence or distributed throughput.

## Unbiased sequence-wise balancing

Original loss implementation/tests: Yidi Wu (@ydwu4),
[#5259](https://github.com/pytorch/torchtitan/pull/5259), source revision
`25dafa9a6db1e7d5327377a0e28913485e704a1f`. The adaptation retains native
objective injection, coefficient/token denominator and detached metric state,
while computing unbiased assignments inside the fused raw-loss kernel.
General-shape CUDA kernels and native-order column/CTA reductions are additional
integration work; the tuned CuTe implementation remains credited and preserved.

```text
FP32 scores -----------------------+     dispatch map
       |                          |          |
 normalize across all experts     |     not used for
       |                          |   unbiased objective
 plain top-k -> valid frequencies  |          X
       |                          |
       +-------- dot -------- valid normalized-score sums
                    |
                 raw loss
                    |
        existing coefficient / token denominator
                    |
       existing gradient injection and metric state
```

Unbiased mode consumes nonnegative sigmoid scores and uses
`sum(scores)+1e-20`, independently of group limits, bias and forced dispatch.
Padding contributes neither counts nor score sums/gradients. Generic
provided-map/L1 normalization remains a distinct mode. All supported positive
T/E shapes and configured top-k values stay fused, including unaligned score
storage. Unsupported precision/mesh/device contracts use native behavior only
when fusion is not required. The production router's aligned scores retain
the tuned specialization.

The general implementation derives native reduction partitions from T/E and
device resources; no shape-dependent module counter controls those launches:

```text
scores[T,E] + provided map / unbiased top-k
                    |
        fused row norm, selection, padding
                    |
      temporary normalized scores / selected map
                    |
       fused column partial sums[partitions,E]
                    |
    fused native-order finish -> frequencies[E], raw loss

saved original scores + frequencies + upstream scalar
                    |
           one fused row backward
```

Normalized scores/maps and column scratch are temporary forward buffers,
not autograd saved state. Backward retains the original scores, frequencies
and optional padding mask. Row norms account for the actual storage offset;
native scalar clamp-min's strict boundary derivative is preserved. The tuned
path retains its existing single-forward/single-backward launch and counter.

Training's DP objective stays local; distributed collection of detached
metrics is separate. Router and loss counters have independent ownership.
The composed factory reuses the existing router and loss factories within
one override claim, avoiding nested-claim conflicts without merging their
implementations. MTP cross-entropy remains native at this boundary.
