# GraphTrainer CODA fusion results

This file records the graph evidence and isolated performance result for each
CODA fusion pass. Full tlparse dumps stay under `outputs/` because they are
generated artifacts; the grounded samples and exact artifact paths are listed
here.

## Standalone 12-pattern tuning suite

The standalone suite is implemented in
`benchmarks/coda_fusion_microbench.py`. Every case has an explicit plain
PyTorch eager function and an explicit FlexGEMM function at a real shape from
the DSV3-671B joint graph. The primary baseline is the same eager function
compiled with Inductor. Source eager is also reported to keep the epilogue
implementation visible and independently measurable.

The primary timing contract uses fixed-pointer CUDA graph replay, 25 warmups,
10 alternating candidate-order rounds, and 200 replays per round. F2-Q and
F2-KV used 100 replays per round because of their larger live input sets.
Compilation and tuning are outside the timed region. Every generated source is
checked for the expected number of `flex_gemm_epilogue` calls, and the selected
QuACK configurations are extracted from the generated code. All outputs pass
elementwise tolerance checks plus max-absolute, mean-absolute, and relative-L2
reporting.

Environment: one NVIDIA GB300 (SM 10.3), PyTorch
`2.14.0.dev20260811+cu130` at `50e2fa0ee83b`, CUTLASS DSL `4.6.2`. Results are
milliseconds; speedup is compiled eager divided by FlexGEMM.

| Pattern | Source eager | Compiled eager | FlexGEMM | Speedup | Selected config(s) |
| --- | ---: | ---: | ---: | ---: | --- |
| B1 | 13.443 | 13.636 | 14.226 | 0.959x | `256x256 c2x1 dynamic` |
| B2 | 6.140 | 5.743 | 5.797 | 0.991x | `128x256 c2x1 dynamic`; `256x256 c2x2 dynamic` |
| B4 | 7.205 | 6.799 | 0.587 | 11.582x | `256x256 c2x2 dynamic` |
| B5 | 4.408 | 4.522 | 4.841 | 0.934x | `256x256 c2x1 dynamic` |
| B6 | 1.713 | 1.748 | 1.772 | 0.987x | `256x256 c2x1 dynamic` |
| B7 | 2.302 | 2.086 | 2.058 | 1.013x | `256x256 c2x1 dynamic` |
| F2-KV | 3.052 | 3.048 | 3.135 | 0.972x | `256x192 c2x1 dynamic`; `256x256 c2x1 dynamic` |
| F2-Q | 5.946 | 6.078 | 6.062 | 1.003x | `256x512 c2x1 static`; `256x256 c2x1 dynamic` |
| F3-A | 14.190 | 14.385 | 14.932 | 0.963x | `256x512 c2x1 static` |
| F3-B | 3.042 | 2.558 | 2.720 | 0.940x | `256x512 c2x1 static` |
| F4 | 3.725 | 3.730 | 3.598 | 1.037x | two `256x256 c2x1 dynamic` |
| F6 | 5.454 | 5.404 | 0.573 | 9.428x | `256x256 c2x1 dynamic` |

Every FlexGEMM uses `tuned=True`. The F4 SiLU FlexGEMM and F6 sigmoid
FlexGEMM also use `fast_math=True`; the non-SiLU F4 FlexGEMM does not. B4's
BF16 result has `max_abs=4.883e-4` and `relative_l2=1.707e-3` versus source
eager. F6's two FP32 results have `max_abs=1.392e-5` and relative L2 below
`5e-6`. The large B4 and F6 speedups therefore satisfy the pattern-specific
tolerances but are not bitwise-equivalence claims.

### Automatic tuning behavior

FlexGEMM can tune without an explicit configuration:

```python
kernel_options={"backend": "QUACK", "tuned": True}
```

With this form, the lowering obtains all device-compatible QuACK candidates,
filters them for the epilogue and local-reduction geometry, passes the resulting
template choices to Inductor's `autotune_select_algorithm()`, and caches the
winner. An explicit partial `config` constrains that candidate set while still
using the tuned path.

The results above use `benchmarks/coda_fusion_autotune.py`, which supplies one
fully constrained configuration to each fresh child process and performs the
search outside Inductor. This is a reliability workaround, not a FlexGEMM API
requirement. On this build, unconstrained in-process tuning can execute an SM100
candidate that raises `cudaErrorNoKernelImageForDevice` on SM103 or hangs in
the kernel. Inductor subprocess tuning cannot currently transport the resulting
TVM-FFI exception. Process isolation lets the suite time out or reject one bad
candidate without losing the entire search.

The tuner searches 12 measured-priority configurations on every pattern,
remeasures per-GPU finalists sequentially on GPU 0, and performs a final full
timing run in another fresh process. Multi-FlexGEMM patterns use coordinate
descent. A full 74-configuration non-TMA, non-transposed SM100 search was also
run for B4 and F6. It retained the priority winners shown above, so the broader
space did not improve either decisive win.

Artifacts:

```text
outputs/coda_fusion_microbench/20260811_tournament_autotune
outputs/coda_fusion_microbench/20260811_full_autotune
```

Representative commands:

```bash
python -m torchtitan.experiments.graph_trainer.benchmarks.coda_fusion_autotune \
  --case f4_shared_expert_swiglu --search priority --devices 0,1,2,3

python -m torchtitan.experiments.graph_trainer.benchmarks.coda_fusion_autotune \
  --case f6_router_sigmoid_bias --search full --devices 0,1,2,3
```

## B1 LM-head input-gradient cast

Pattern name: `b1_lm_head_input_grad_cast`

The full B1 region contains eight chunked LM-head input-gradient GEMMs, FP32
chunk writes and accumulation, a final BF16 conversion, and RMSNorm backward.
The supported FlexGEMM boundary is intentionally smaller: each
`BF16 mm -> reshape -> alias -> FP32 cast` chain is fused only when it is an
LM-head backward GEMM and the cast is the source of the corresponding chunk
`copy_`. The chunk writes, their cross-chunk accumulation, final conversion,
and RMSNorm backward remain unchanged.

Real before sample:

```python
mm_548 = torch.ops.aten.mm.default(view_4384, _unsafe_view_2737)
view_4385 = torch.ops.aten.reshape.default(mm_548, [24, 512, 7168])
alias_436 = torch.ops.aten.alias.default(view_4385)
_to_copy_1737 = torch.ops.aten._to_copy.default(
    alias_436, dtype=torch.float32
)
copy_ = torch.ops.aten.copy_.default(slice_798, _to_copy_1737)
```

Real after sample:

```python
_coda_b1_lm_head_input_grad_body_0 = (
    self._coda_b1_lm_head_input_grad_body_0
)
flex_gemm = torch.ops.higher_order.flex_gemm(
    torch.ops.aten.mm.default,
    _coda_b1_lm_head_input_grad_body_0,
    (view_4384, _unsafe_view_2737),
    {},
    {"backend": "QUACK"},
)
getitem_25338 = flex_gemm[0]
reshape_default = torch.ops.aten.reshape.default(
    getitem_25338, [24, 512, 7168]
)
copy_ = torch.ops.aten.copy_.default(slice_798, reshape_default)
```

The FlexGEMM body explicitly models accumulator FP32 -> BF16 -> FP32
conversion, preserving the original BF16 GEMM store rounding while producing
the FP32 chunk value.

### Graph proof

Configuration: DSV3-671B, fake communication backend, FSDP 256, EP 64, local
batch 24, sequence length 4096, full activation checkpointing, `c4_test`.

Artifact root:

```text
outputs/profiling/dsv3_fake/graph/coda-b1-proof-20260810/tlparse/-_-_-_-
```

Before and after dumps:

```text
before_fuse_b1_lm_head_input_grad_cast_pass_274.txt
after_fuse_b1_lm_head_input_grad_cast_pass_275.txt
```

Pass diff:

| Operation | Before | After | Delta |
| --- | ---: | ---: | ---: |
| root `mm.default` | 2,148 | 2,140 | -8 |
| root `_to_copy.default` | 2,486 | 2,478 | -8 |
| root `alias.default` | 2,825 | 2,817 | -8 |
| root `flex_gemm` | 0 | 8 | +8 |
| root `get_attr` | 427 | 435 | +8 |
| root `getitem` | 24,620 | 24,628 | +8 |

All eight grounded chunk chains fused. Root reshape and `copy_` counts are
unchanged because each removed pre-cast reshape is replaced by a post-FlexGEMM
reshape. The proof run disabled regional Inductor and the CUDA graph pass so
tlparse could record the rewrite in isolation; execution subsequently OOMed in
unfused FlexAttention after all before/after artifacts had been written.

### GB300 microbenchmark

The exact fused boundary was benchmarked at the real chunk shape:
`(12,288, 129,280) @ (129,280, 7,168)`, BF16 inputs and FP32 output. Five
warmups and 20 CUDA-event iterations were run on one GB300.

| Implementation | Median | Min | Max |
| --- | ---: | ---: | ---: |
| eager `mm` + cast | 13.265 ms | 12.952 ms | 13.502 ms |
| QUACK FlexGEMM | 15.874 ms | 15.226 ms | 17.246 ms |

Speedup: `0.84x` (FlexGEMM is about 20% slower). This is retained because the
project accepts FlexGEMM fusions without a speedup. The result was bitwise
identical to eager (`max_abs_error=0`). The benchmark excludes the unchanged
chunk `copy_`.

## B6 BF16 weight-gradient cast

Pattern name: `b6_bf16_weight_grad_cast`

The matcher was derived from the DSV3-671B post-bucketing graph. It requires an
`aten.mm.default` with BF16 output, exactly one user, and an
`aten._to_copy.default(dtype=torch.float32)` user. Multi-use GEMMs and the FP32
router round trip are not matched.

Real before sample:

```python
t_702 = torch.ops.aten.t.default(view_4444)
mm_570 = torch.ops.aten.mm.default(t_702, view_4377_recomputed)
_to_copy_1775 = torch.ops.aten._to_copy.default(
    mm_570, dtype=torch.float32
)
```

Real after sample:

```python
_coda_b6_body_8 = self._coda_b6_body_8
flex_gemm_8 = torch.ops.higher_order.flex_gemm(
    torch.ops.aten.mm.default,
    _coda_b6_body_8,
    (t_702, view_4377_recomputed),
    {},
    {"backend": "QUACK"},
)
getitem_25346 = flex_gemm_8[0]
```

The body explicitly models accumulator FP32 -> BF16 -> FP32 conversion. This
preserves the original BF16 GEMM store rounding while returning the FP32 value
expected by FSDP reduce-scatter bucketing.

### Graph proof

Configuration: DSV3-671B, fake communication backend, FSDP 256, EP 64, local
batch 24, sequence length 4096, full activation checkpointing, `c4_test`.

Artifact root:

```text
outputs/profiling/dsv3_fake/graph/coda-b6-bf16-proof-20260810-142424/tlparse/-_-_-_-
```

Before and after dumps:

```text
before_fuse_b6_bf16_weight_grad_cast_pass_279.txt
after_fuse_b6_bf16_weight_grad_cast_pass_280.txt
```

Pass diff:

| Operation | Before | After | Delta |
| --- | ---: | ---: | ---: |
| root `mm.default` | 2,148 | 1,652 | -496 |
| root `_to_copy.default` | 2,486 | 1,990 | -496 |
| root `flex_gemm` | 0 | 496 | +496 |
| root `get_attr` | 427 | 923 | +496 |
| root `getitem` | 24,620 | 25,116 | +496 |

All 496 grounded BF16 chains fused. The 58 FP32 router chains remain in the
after graph. The proof run disabled regional Inductor so tlparse could record
the rewrite in isolation; execution subsequently OOMed in unfused
FlexAttention after all before/after artifacts had been written.

### GB300 microbenchmark

Representative shape: `(2048, 98304) @ (98304, 7168)`, BF16 inputs, FP32
output. This is the most frequent B6 shape, with 116 occurrences in the graph.
Five warmups and 20 CUDA-event iterations were run on one GB300.

| Implementation | Median | Min | Max |
| --- | ---: | ---: | ---: |
| eager `mm` + cast | 1.534 ms | 1.511 ms | 1.553 ms |
| QUACK FlexGEMM | 2.094 ms | 2.062 ms | 2.390 ms |

Speedup: `0.73x` (FlexGEMM is 37% slower). This is retained because the project
accepts FlexGEMM fusions without a speedup. The output was bitwise identical to
eager (`max_abs_error=0`) and every FP32 output value was BF16-representable.

### Dependencies and exclusions

The pass requires PyTorch commit `bd2911838e0`, which preserves marked
FlexGEMM accumulator conversions through joint-graph cleanup and makes explicit
CuTeDSL casts use physical dtypes. The active environment also has the PyTorch
CI-pinned `nvidia-cutlass-dsl==4.5.2` and `apache-tvm-ffi==0.1.11` packages.

The 58 FP32 router `mm -> BF16 -> FP32` chains are intentionally excluded.
QUACK compilation for the real `(256, 98304) @ (98304, 7168)` shape failed on
GB300 with `cudaErrorNoKernelImageForDevice`. No custom router kernel is added
without evidence that it is faster than the original implementation.

## F6 router sigmoid and expert bias

Pattern name: `f6_router_sigmoid_bias`

The matcher follows the real FP32 router chain through its canonical reshape:
`mm -> reshape -> sigmoid`. The GEMM and reshape must each have one user, the
last reshape dimension must equal the GEMM output width, and an expert bias is
captured only when it is the exact one-dimensional output-width tensor. Other
sigmoid users are preserved.

That last point is required by the training graph. The original forward score
is consumed both by the biased top-k path and by forced-load-balance routing;
the recomputed score is also consumed by sigmoid backward. FlexGEMM therefore
returns the raw sigmoid as its main output and, for the 58 forward cases, the
biased score as a same-shape auxiliary output.

Real before sample:

```python
mm_29 = torch.ops.aten.mm.default(view_109, t_29)
_unsafe_view_29 = torch.ops.aten.reshape.default(mm_29, [24, 4096, 256])
sigmoid = torch.ops.aten.sigmoid.default(_unsafe_view_29)
add_7 = torch.ops.aten.add.Tensor(sigmoid, arg1938_1)
```

Real after sample:

```python
reshape_default = torch.ops.aten.reshape.default(arg1938_1, [1, 256])
_coda_f6_body_0 = self._coda_f6_body_0
flex_gemm = torch.ops.higher_order.flex_gemm(
    torch.ops.aten.mm.default,
    _coda_f6_body_0,
    (view_109, t_29, reshape_default),
    {},
    {"backend": "QUACK"},
)
raw_scores = torch.ops.aten.reshape.default(flex_gemm[0], [24, 4096, 256])
biased_scores = torch.ops.aten.reshape.default(flex_gemm[1], [24, 4096, 256])
```

### Graph proof

Configuration: DSV3-671B, fake communication backend, FSDP 256, EP 64, local
batch 24, sequence length 4096, full activation checkpointing, `c4_test`.

Artifact root:

```text
outputs/profiling/dsv3_fake/graph/coda-f6-proof-20260810-144205/tlparse/-_-_-_-
```

Before and after dumps:

```text
before_fuse_f6_router_sigmoid_bias_pass_274.txt
after_fuse_f6_router_sigmoid_bias_pass_275.txt
```

Pass diff for the root graph:

| Operation | Before | After | Delta |
| --- | ---: | ---: | ---: |
| root `mm.default` | 2,148 | 2,032 | -116 |
| root `sigmoid.default` | 116 | 0 | -116 |
| root `add.Tensor` | 1,017 | 959 | -58 |
| root `flex_gemm` | 0 | 116 | +116 |
| root `get_attr` | 427 | 543 | +116 |
| root `getitem` | 24,620 | 24,794 | +174 |
| root `reshape.default` | 6,822 | 6,938 | +116 |

All 58 original and 58 recomputed router sigmoid chains fused. The 58 original
expert-bias adds became auxiliary epilogue outputs. The proof run disabled
regional Inductor to retain the rewrite as FX text, then OOMed in unfused
FlexAttention after all pass artifacts had been written.

### GB300 microbenchmark

Representative shape: `(98304, 7168) @ (7168, 256)`, FP32 inputs and outputs,
with both raw and biased router scores returned. Five warmups and 20 CUDA-event
iterations were run on one GB300.

| Implementation | Median | Min | Max |
| --- | ---: | ---: | ---: |
| eager `mm` + sigmoid + bias | 5.534 ms | 5.513 ms | 5.611 ms |
| QUACK FlexGEMM | 0.983 ms | 0.966 ms | 1.032 ms |

Speedup: `5.63x`. For random FP32 inputs scaled by `0.02`, both raw and biased
outputs had `max_abs_error=1.395e-5` and `mean_abs_error=1.984e-6` versus eager.

## F4 dense and shared-expert SwiGLU

Pattern name: `f4_dense_swiglu`

The matcher requires the grounded BF16 two-GEMM layout:

```text
mm(W1) -> reshape -> silu --+
                               mul
mm(W3) -> reshape ----------+
```

Both GEMMs and their reshapes must be shape-compatible, and each GEMM must feed
only its reshape. Routed `_grouped_mm` SwiGLU is excluded and remains assigned
to distMoE.

The first FlexGEMM emits SiLU. The second captures that same-shape tile and
emits both the BF16-rounded W3 result and the product. Recomputed W1
preactivations have an additional `silu_backward` consumer, so those first
FlexGEMMs also return the rounded preactivation as an auxiliary output. The 61
original forward cases do not create that unused auxiliary output. Explicit
FP32 -> BF16 conversions inside both bodies preserve the original GEMM store
rounding.

Real before sample:

```python
mm_5 = torch.ops.aten.mm.default(view_25, t_5)
_unsafe_view_5 = torch.ops.aten.reshape.default(mm_5, [24, 4096, 18432])
silu = torch.ops.aten.silu.default(_unsafe_view_5)
mm_6 = torch.ops.aten.mm.default(view_27, t_6)
_unsafe_view_6 = torch.ops.aten.reshape.default(mm_6, [24, 4096, 18432])
mul_2 = torch.ops.aten.mul.Tensor(silu, _unsafe_view_6)
```

Real after sample:

```python
flex_gemm = torch.ops.higher_order.flex_gemm(
    torch.ops.aten.mm.default,
    _coda_f4_silu_body_0,
    (view_25, t_5),
    {},
    {"backend": "QUACK"},
)
silu_2d = flex_gemm[0]
flex_gemm_1 = torch.ops.higher_order.flex_gemm(
    torch.ops.aten.mm.default,
    _coda_f4_mul_body_0,
    (view_27, t_6, silu_2d),
    {},
    {"backend": "QUACK"},
)
gate_2d = flex_gemm_1[0]
product_2d = flex_gemm_1[1]
```

The first recomputed case has two outputs instead: `flex_gemm_122[0]` is SiLU
and `flex_gemm_122[1]` is the saved W1 preactivation.

### Graph proof

Configuration: DSV3-671B, fake communication backend, FSDP 256, EP 64, local
batch 24, sequence length 4096, full activation checkpointing, `c4_test`.

Artifact root:

```text
outputs/profiling/dsv3_fake/graph/coda-f4-full-proof-20260810-150132/tlparse/-_-_-_-
```

Before and after dumps:

```text
before_fuse_f4_dense_swiglu_pass_274.txt
after_fuse_f4_dense_swiglu_pass_275.txt
```

Pass diff for the root graph:

| Operation | Before | After | Delta |
| --- | ---: | ---: | ---: |
| root `mm.default` | 2,148 | 1,904 | -244 |
| root `silu.default` | 238 | 116 | -122 |
| root `mul.Tensor` | 1,249 | 1,127 | -122 |
| root `flex_gemm` | 0 | 244 | +244 |
| root `get_attr` | 427 | 671 | +244 |
| root `getitem` | 24,620 | 25,047 | +427 |
| root `reshape.default` | 6,822 | 7,005 | +183 |

All 6 dense and 116 shared-expert original/recomputed chains fused. The 116
routed grouped-GEMM chains remain unchanged. The proof run disabled regional
Inductor to retain the rewrite as FX text, then OOMed in unfused FlexAttention
after all pass artifacts had been written.

### GB300 microbenchmark

Representative shared-expert shape: two `(98304, 7168) @ (7168, 2048)` BF16
GEMMs, returning SiLU, the raw W3 gate, and their product. Five warmups and 20
CUDA-event iterations were run on one GB300.

| Implementation | Median | Min | Max |
| --- | ---: | ---: | ---: |
| eager two GEMMs + SiLU + multiply | 3.613 ms | 3.332 ms | 3.739 ms |
| two QUACK FlexGEMMs | 3.711 ms | 3.619 ms | 4.069 ms |

Speedup: `0.97x` (FlexGEMM is 2.7% slower). This is retained under the project
policy that accepts FlexGEMM fusions without a speedup. SiLU, raw gate, and
product were bitwise identical to eager on the full representative shape.

## B2 dense and shared-expert SwiGLU backward

Pattern name: `b2_dense_swiglu_backward`

The matcher covers the two grounded BF16 GEMM epilogues in a dense or shared
SwiGLU backward block. The W2 input-gradient GEMM feeds two derivatives:

```text
mm(W2 input gradient) -> reshape --+-> mul(saved SiLU)
                                   +-> mul(saved W3) -> silu_backward(saved W1)
```

The first FlexGEMM captures the three saved activations and returns both
derivatives. A second FlexGEMM captures the already-computed W3 input-gradient
GEMM and folds its add into the W1 input-gradient GEMM. The second match also
requires sibling `w3` and `w1` module FQNs from the same feed-forward block,
with the captured W3 GEMM preceding W1 in the graph. Routed grouped GEMMs are
excluded and remain assigned to distMoE.

Real before sample from layer 60:

```python
mm_571 = torch.ops.aten.mm.default(view_4444, _unsafe_view_2734)
view_4445 = torch.ops.aten.reshape.default(mm_571, [24, 4096, 2048])
mul_357 = torch.ops.aten.mul.Tensor(view_4445, silu_118_recomputed)
mul_358 = torch.ops.aten.mul.Tensor(view_4445, _unsafe_view_660_recomputed)
silu_backward = torch.ops.aten.silu_backward.default(
    mul_358, _unsafe_view_659_recomputed
)

mm_573 = torch.ops.aten.mm.default(view_4447, _unsafe_view_2735)
view_4448 = torch.ops.aten.reshape.default(mm_573, [24, 4096, 7168])
mm_575 = torch.ops.aten.mm.default(view_4450, _unsafe_view_2733)
view_4451 = torch.ops.aten.reshape.default(mm_575, [24, 4096, 7168])
add_362 = torch.ops.aten.add.Tensor(view_4448, view_4451)
```

Real after sample:

```python
flex_gemm = torch.ops.higher_order.flex_gemm(
    torch.ops.aten.mm.default,
    _coda_b2_branch_body_0,
    (
        view_4444,
        _unsafe_view_2734,
        reshape_default,
        reshape_default_1,
        reshape_default_2,
    ),
    {},
    {"backend": "QUACK"},
)
gate_grad = flex_gemm[0]
silu_grad = flex_gemm[1]

flex_gemm_61 = torch.ops.higher_order.flex_gemm(
    torch.ops.aten.mm.default,
    _coda_b2_input_add_body_0,
    (view_4450, _unsafe_view_2733, mm_573),
    {},
    {"backend": "QUACK"},
)
input_grad = flex_gemm_61[0]
```

Both bodies explicitly model the original BF16 GEMM store rounding with an
FP32 -> BF16 conversion before evaluating their epilogues.

### Graph proof

Configuration: DSV3-671B, fake communication backend, FSDP 256, EP 64, local
batch 24, sequence length 4096, full activation checkpointing, `c4_test`.

Artifact root:

```text
outputs/profiling/dsv3_fake/graph/coda-b2-proof-20260810-continued/tlparse/-_-_-_-
```

Before and after dumps:

```text
before_fuse_b2_dense_swiglu_backward_pass_274.txt
after_fuse_b2_dense_swiglu_backward_pass_275.txt
```

Pass diff for the root graph:

| Operation | Before | After | Delta |
| --- | ---: | ---: | ---: |
| root `mm.default` | 2,148 | 2,026 | -122 |
| root `mul.Tensor` | 1,249 | 1,127 | -122 |
| root `silu_backward.default` | 119 | 58 | -61 |
| root `add.Tensor` | 1,017 | 956 | -61 |
| root `flex_gemm` | 0 | 122 | +122 |
| root `get_attr` | 427 | 549 | +122 |
| root `getitem` | 24,620 | 24,803 | +183 |
| root `reshape.default` | 6,822 | 7,005 | +183 |

All 61 dense/shared branch-derivative chains and all 61 matching
input-gradient adds fused. The proof run disabled regional Inductor to retain
the rewrite as FX text, then OOMed on the known 192 GiB unfused FlexAttention
allocation after all pass artifacts had been written.

### GB300 microbenchmark

The full representative shared-expert backward block used BF16 tensors with
`M=98,304`, model width `7,168`, and shared-expert width `2,048`. It includes
the W2 input-gradient GEMM, both derivative branches, both W3/W1 input-gradient
GEMMs, and their final add. Five warmups and 20 CUDA-event iterations were run
on one GB300.

| Implementation | Median | Min | Max |
| --- | ---: | ---: | ---: |
| eager three GEMMs + derivative epilogues | 6.046 ms | 5.663 ms | 6.277 ms |
| two QUACK FlexGEMMs + one eager GEMM | 6.955 ms | 6.769 ms | 7.899 ms |

Speedup: `0.87x` (FlexGEMM is 15% slower). This is retained under the project
policy that accepts FlexGEMM fusions without a speedup. The gate derivative was
bitwise identical to eager. The SiLU derivative and final input gradient had
maximum absolute errors of `3.052e-5` and `1.526e-5`, respectively, with random
BF16 inputs scaled by `0.02`.

## F2 MLA Q projection and RMSNorm

Pattern name: `f2_q_rmsnorm`

This is an opt-in, numerics-changing CODA reparameterization of the grounded
MLA Q chain:

```text
mm(wq_a) -> RMSNorm(q_norm) -> mm(wq_b)
```

The first FlexGEMM applies the 1,536-element norm weight without the row scale
and emits three 512-column partial mean-square values per token. Root-graph
pointwise ops finalize one inverse-RMS value per token. The second FlexGEMM
captures that row value and applies it after the `wq_b` accumulation.

The pass rewrites both the 61 original and 61 rematerialized Q chains. The
rematerialized first FlexGEMM additionally returns raw BF16 Q. The pass
reconstructs normalized Q and reshapes `rstd` for the existing `wq_b` weight
gradient and RMSNorm backward consumers, so activation checkpointing does not
mix CODA forward values with an unrelated recomputation path.

Real before sample:

```python
mm = torch.ops.aten.mm.default(view_4, t)
_unsafe_view = torch.ops.aten.reshape.default(mm, [24, 4096, 1536])
_fused_rms_norm_1 = torch.ops.aten._fused_rms_norm.default(
    _unsafe_view, [1536], _unsafe_view_1273, 1e-05
)
getitem_2 = _fused_rms_norm_1[0]
view_7 = torch.ops.aten.reshape.default(getitem_2, [98304, 1536])
mm_1 = torch.ops.aten.mm.default(view_7, t_1)
```

Real original-forward after sample:

```python
flex_gemm = torch.ops.higher_order.flex_gemm(
    torch.ops.aten.mm.default,
    _coda_f2_q_first_body_0,
    (view_4, t, reshape_default),
    {},
    {"backend": "QUACK"},
)
weighted_q = flex_gemm[0]
partial_mean_square = flex_gemm[1]
mean_square = torch.ops.aten.mean.dim(partial_mean_square, [-1], True)
rstd = torch.ops.aten.rsqrt.default(
    torch.ops.aten.add.Scalar(mean_square, 1e-05)
)
flex_gemm_1 = torch.ops.higher_order.flex_gemm(
    torch.ops.aten.mm.default,
    _coda_f2_q_second_body_0,
    (weighted_q, t_1, rstd),
    {},
    {"backend": "QUACK"},
)
q = flex_gemm_1[0]
```

The rematerialized first HOP has a third output: raw BF16 Q is output 1 and
the partial statistics move to output 2. Generated-kernel validation confirmed
that QUACK supports this combination of a same-shape auxiliary output and a
compressed physical reduction.

### Graph proof

Configuration: DSV3-671B, fake communication backend, FSDP 256, EP 64, local
batch 24, sequence length 4096, full activation checkpointing, `c4_test`.

Artifact root:

```text
outputs/profiling/dsv3_fake/graph/coda-f2-q-full-proof-20260810-2/tlparse/-_-_-_-
```

Before and after dumps:

```text
before_fuse_f2_q_rmsnorm_pass_274.txt
after_fuse_f2_q_rmsnorm_pass_275.txt
```

Pass diff for the root graph:

| Operation | Before | After | Delta |
| --- | ---: | ---: | ---: |
| root `mm.default` | 2,148 | 1,904 | -244 |
| root `_fused_rms_norm.default` | 490 | 368 | -122 |
| root `flex_gemm` | 0 | 244 | +244 |
| root `get_attr` | 427 | 671 | +244 |
| root `getitem` | 24,620 | 24,864 | +244 |
| root `mean.dim` | 0 | 122 | +122 |
| root `add.Scalar` | 0 | 122 | +122 |
| root `rsqrt.default` | 0 | 122 | +122 |
| root `_to_copy.default` | 2,486 | 2,608 | +122 |
| root `mul.Tensor` | 1,249 | 1,371 | +122 |

All 61 original and 61 rematerialized Q chains fused. The remaining 368 norms
include KV, residual/FFN, final, and their rematerialized copies. The proof run
disabled regional Inductor to retain the rewrite as FX text, then OOMed on the
known 192 GiB unfused FlexAttention allocation after all pass artifacts had
been written.

### GB300 microbenchmarks

The representative Q chain uses BF16 tensors with `M=98,304`, input width
`7,168`, Q low-rank width `1,536`, and projected width `24,576`. Five warmups
and 20 CUDA-event iterations were run on one GB300.

| Variant | Eager | CODA FlexGEMM | Speedup |
| --- | ---: | ---: | ---: |
| original forward | 5.816 ms | 6.020 ms | 0.97x |
| rematerialized with saved values | 5.919 ms | 6.294 ms | 0.94x |

Both variants are retained under the project policy that accepts FlexGEMM
fusions without a speedup. On the full original-forward shape, projected Q had
`max_abs_error=0.03125` and `mean_abs_error=0.001034`. In the rematerialized
variant, raw Q was bitwise exact, saved `rstd` had `max_abs_error=5.722e-6`,
and reconstructed normalized Q had `max_abs_error=0.015625` and
`mean_abs_error=3.656e-8`.

This F2-Q pass requires convergence validation because moving the row scale
across BF16 GEMMs changes rounding. The separate F2-KV pass below handles the
512-of-576 KV chain without moving RoPE into the GEMM epilogue.

## F2 MLA KV projection and segmented RMSNorm

Pattern name: `f2_kv_rmsnorm`

This pass rewrites the grounded segmented MLA KV chain:

```text
mm(wkv_a) -> split [512, 64] -> RMSNorm(kv_norm) -> mm(wkv_b)
                            -> 64-column RoPE tail
```

The first FlexGEMM retains its physical 576-column output. The 512-element norm
weight is padded with 64 ones, so the epilogue emits a gamma-weighted full-width
value, a raw full-width auxiliary, and nine 64-column mean-square partials. The
root graph uses the first eight partials to form `rstd`; the ninth corresponds
to the unnormalized RoPE tail. The second FlexGEMM consumes the first 512
weighted columns and applies `rstd` after `wkv_b` accumulation. The raw
auxiliary continues through the original split, preserving the RoPE tail and
the rematerialized RMSNorm-backward input.

Real before sample:

```python
mm_2 = torch.ops.aten.mm.default(view_10, t_2)
kv = torch.ops.aten.reshape.default(mm_2, [24, 4096, 576])
kv, k_pe = torch.ops.aten.split_with_sizes.default(kv, [512, 64], -1)
norm = torch.ops.aten._fused_rms_norm.default(
    kv, [512], kv_norm_weight, 1e-05
)
mm_3 = torch.ops.aten.mm.default(norm[0].reshape(98304, 512), t_3)
```

Real after sample:

```python
gamma = torch.ops.aten.reshape.default(kv_norm_weight, [1, 512])
gamma_full = torch.ops.aten.constant_pad_nd.default(gamma, [0, 64], 1.0)
first = torch.ops.higher_order.flex_gemm(
    torch.ops.aten.mm.default,
    _coda_f2_kv_first_body_0,
    (view_10, t_2, gamma_full),
    {},
    {"backend": "QUACK"},
)
weighted_full, raw_full, partials = first
weighted = torch.ops.aten.slice.Tensor(weighted_full, 1, 0, 512)
active_partials = torch.ops.aten.slice.Tensor(partials, 1, 0, 8)
rstd = torch.ops.aten.rsqrt.default(
    torch.ops.aten.add.Scalar(
        torch.ops.aten.mean.dim(active_partials, [-1], True), 1e-05
    )
)
second = torch.ops.higher_order.flex_gemm(
    torch.ops.aten.mm.default,
    _coda_f2_kv_second_body_0,
    (weighted, t_3, rstd),
    {},
    {"backend": "QUACK"},
)
```

### Graph proof

Configuration: DSV3-671B, fake communication backend, FSDP 256, EP 64, local
batch 24, sequence length 4096, full activation checkpointing, `c4_test`.

Artifact root:

```text
outputs/profiling/dsv3_fake/graph/coda-f2-kv-proof-20260810/tlparse/-_-_-_-
```

Before and after dumps:

```text
before_fuse_f2_kv_rmsnorm_pass_274.txt
after_fuse_f2_kv_rmsnorm_pass_275.txt
```

Pass diff for the root graph:

| Operation | Before | After | Delta |
| --- | ---: | ---: | ---: |
| root `mm.default` | 2,148 | 1,904 | -244 |
| root `_fused_rms_norm.default` | 490 | 368 | -122 |
| root `flex_gemm` | 0 | 244 | +244 |
| root `get_attr` | 427 | 671 | +244 |
| root `getitem` | 24,620 | 24,864 | +244 |
| root `mean.dim` | 0 | 122 | +122 |
| root `add.Scalar` | 0 | 122 | +122 |
| root `rsqrt.default` | 0 | 122 | +122 |
| root `_to_copy.default` | 2,486 | 2,608 | +122 |
| root `constant_pad_nd.default` | 3,904 | 4,026 | +122 |
| root `mul.Tensor` | 1,249 | 1,371 | +122 |
| root `reshape.default` | 6,822 | 6,883 | +61 |
| root `slice.Tensor` | 615 | 920 | +305 |

All 61 original and 61 rematerialized KV chains fused. The raw 576-column
auxiliary preserves all 122 existing split/RoPE paths. The proof run disabled
regional Inductor to retain the rewrite as FX text, then OOMed on the known
192 GiB unfused FlexAttention allocation after all artifacts were written.

### GB300 microbenchmark

The representative chain uses BF16 tensors with `M=98,304`, input width
`7,168`, WKV-A width `576`, active RMSNorm width `512`, and WKV-B output width
`32,768`. Five warmups and 20 CUDA-event iterations were run on one GB300.

| Implementation | Median | Min | Max |
| --- | ---: | ---: | ---: |
| eager WKV-A + segmented RMSNorm + WKV-B | 2.816 ms | 2.654 ms | 3.602 ms |
| two QUACK FlexGEMMs + partial reduction | 3.516 ms | 3.435 ms | 4.559 ms |

Speedup: `0.80x`. This is retained under the project policy that accepts
FlexGEMM fusions without a speedup. The raw RoPE tail was bitwise exact. The
WKV-B output had `max_abs_error=0.015625` and `mean_abs_error=0.000596412`.
Moving `rstd` across WKV-B requires convergence validation. This result does
not justify a custom kernel.

## B4 router input-gradient cast and add

Pattern name: `b4_router_input_grad_add`

This pattern fuses the router input-gradient GEMM's FP32 -> BF16 store rounding
and the addition of the expert-path input gradient. The matcher requires the
GEMM's module FQN to end in `.moe.router.gate`, an FP32 two-dimensional GEMM,
a sole reshape and BF16 cast chain, and a BF16 residual with the same final
shape. These constraints exclude unrelated FP32 linears and multi-use values.

Real before sample from layer 60:

```python
view_4467 = torch.ops.aten.reshape.default(sigmoid_backward, [98304, 256])
mm_577 = torch.ops.aten.mm.default(view_4467, _to_copy_1718_recomputed)
view_4468 = torch.ops.aten.reshape.default(mm_577, [24, 4096, 7168])
_to_copy_1784 = torch.ops.aten._to_copy.default(
    view_4468,
    dtype=torch.bfloat16,
    layout=torch.strided,
    device=torch.device("cuda:0"),
)
add_366 = torch.ops.aten.add.Tensor(add_364, _to_copy_1784)
```

Real after sample:

```python
reshape_default = torch.ops.aten.reshape.default(add_364, [98304, 7168])
flex_gemm = torch.ops.higher_order.flex_gemm(
    torch.ops.aten.mm.default,
    _coda_b4_router_body_0,
    (view_4467, _to_copy_1718_recomputed, reshape_default),
    {},
    {"backend": "QUACK"},
)
router_input_grad = flex_gemm[0]
reshape_default_1 = torch.ops.aten.reshape.default(
    router_input_grad, [24, 4096, 7168]
)
```

The FlexGEMM body performs the original BF16 conversion before the add, so the
expert-path gradient is not added to an unrounded FP32 accumulator.
`sigmoid_backward` feeds the GEMM and is therefore a prologue, not an epilogue;
the following `_fused_rms_norm_backward` consumes the fused result but is not
part of this fusion.

### Graph proof

Configuration: DSV3-671B, fake communication backend, FSDP 256, EP 64, local
batch 24, sequence length 4096, full activation checkpointing, `c4_test`.

Artifact root:

```text
outputs/profiling/dsv3_fake/graph/coda-b4-proof-20260810/tlparse/-_-_-_-
```

Before and after dumps:

```text
before_fuse_b4_router_input_grad_add_pass_274.txt
after_fuse_b4_router_input_grad_add_pass_275.txt
```

Pass diff for the root graph:

| Operation | Before | After | Delta |
| --- | ---: | ---: | ---: |
| root `mm.default` | 2,148 | 2,090 | -58 |
| root `_to_copy.default` | 2,486 | 2,428 | -58 |
| root `add.Tensor` | 1,017 | 959 | -58 |
| root `flex_gemm` | 0 | 58 | +58 |
| root `get_attr` | 427 | 485 | +58 |
| root `getitem` | 24,620 | 24,678 | +58 |
| root `reshape.default` | 6,822 | 6,880 | +58 |

All 58 router input-gradient chains fused. The proof run disabled regional
Inductor to retain the rewrite as FX text, then OOMed on the known 192 GiB
unfused FlexAttention allocation after all pass artifacts had been written.

### GB300 microbenchmark

The representative full shape is FP32 `(98,304, 256) @ (256, 7,168)` followed
by a BF16 conversion and addition to a BF16 `(98,304, 7,168)` residual. Inputs
were scaled by `0.02`; five warmups and 20 CUDA-event iterations were run on one
GB300.

| Implementation | Median | Min | Max |
| --- | ---: | ---: | ---: |
| eager GEMM + BF16 cast + add | 7.276 ms | 7.266 ms | 7.300 ms |
| QUACK FlexGEMM | 0.784 ms | 0.761 ms | 0.865 ms |

Speedup: `9.29x`. The fused output had `max_abs_error=0.00048828125` and
`mean_abs_error=1.494e-6` relative to eager.

## B7 Q/KV attention input-gradient merge

Pattern name: `b7_attention_grad_merge`

The attention backward graph computes separate BF16 input gradients for
`wkv_a` and `wq_a`, reshapes both to the residual-stream shape, and adds them
before attention RMSNorm backward. The matcher requires both branches to be
backward `mm` nodes from the same transformer layer, with exact
`attention.wkv_a` and `attention.wq_a` module annotations, one reshape user per
GEMM, and one common BF16 add. This prevents unrelated same-shape gradient
adds from matching.

The KV GEMM remains independent. The Q FlexGEMM captures its 2D BF16 output,
rounds the Q accumulator to BF16 exactly where the original GEMM stored it,
and performs the add in the original operand order. Its root-graph output is
2D and the original 3D reshape remains after the HOP. The following
`_fused_rms_norm_backward` is not part of this fusion.

Real before sample:

```python
mm_583 = torch.ops.aten.mm.default(view_4484, slice_787_recomputed)
view_4485 = torch.ops.aten.reshape.default(mm_583, [24, 4096, 7168])
mm_587 = torch.ops.aten.mm.default(view_4492, _unsafe_view_2724)
view_4493 = torch.ops.aten.reshape.default(mm_587, [24, 4096, 7168])
add_368 = torch.ops.aten.add.Tensor(view_4485, view_4493)
```

Real after sample:

```python
flex_gemm = torch.ops.higher_order.flex_gemm(
    torch.ops.aten.mm.default,
    _coda_b7_attention_grad_merge_body_0,
    (view_4492, _unsafe_view_2724, mm_583),
    {},
    {"backend": "QUACK"},
)
getitem_25338 = flex_gemm[0]  # bf16[98304, 7168]
reshape_default = torch.ops.aten.reshape.default(
    getitem_25338, [24, 4096, 7168]
)
```

### Graph proof

Configuration: DSV3-671B, fake communication backend, FSDP 256, EP 64, local
batch 24, sequence length 4096, full activation checkpointing, `c4_test`.

Artifact root:

```text
outputs/profiling/dsv3_fake/graph/coda-b7-proof-final-20260810/tlparse/-_-_-_-
```

Before and after dumps:

```text
before_fuse_b7_attention_grad_merge_pass_274.txt
after_fuse_b7_attention_grad_merge_pass_275.txt
```

Pass diff for the root graph:

| Operation | Before | After | Delta |
| --- | ---: | ---: | ---: |
| root `mm.default` | 2,148 | 2,087 | -61 |
| root `add.Tensor` | 1,017 | 956 | -61 |
| root `reshape.default` | 6,822 | 6,761 | -61 |
| root `flex_gemm` | 0 | 61 | +61 |
| root `get_attr` | 427 | 488 | +61 |
| root `getitem` | 24,620 | 24,681 | +61 |

All 61 transformer-layer Q/KV input-gradient merges fused. The proof run
disabled regional Inductor to retain the rewrite as FX text, then OOMed on the
known 192 GiB unfused FlexAttention allocation after all pass artifacts had
been written.

### GB300 microbenchmark

The representative full shape uses a KV GEMM `(98,304, 576) @ (576, 7,168)`
and a Q GEMM `(98,304, 1,536) @ (1,536, 7,168)`, followed by their BF16 add.
Inputs were scaled by `0.02`; five warmups and 20 CUDA-event iterations were
run on one GB300.

| Implementation | Median | Min | Max |
| --- | ---: | ---: | ---: |
| eager two GEMMs + add | 2.195 ms | 2.149 ms | 2.285 ms |
| KV GEMM + Q FlexGEMM add epilogue | 2.115 ms | 1.999 ms | 2.156 ms |

Speedup: `1.04x`. The fused output was bitwise identical to eager
(`max_abs_error=0`).

## F3 residual RMSNorm between GEMMs

Pattern name: `f3_residual_rmsnorm`

### Current terminal implementation

The current pass implements two GEMM-residual-RMSNorm epilogues and stops at
the normalized activation:

```text
F3-A: layers.L.attention.wo + residual -> layers.L.ffn_norm

F3-B: layers.L.feed_forward.w2 + residual
      -> layers.L+1.attention_norm

F3-B: layers.L.moe.shared_experts.w2 + routed output + residual
      -> layers.L+1.attention_norm
```

Each match creates one FlexGEMM. Its body preserves the source graph's BF16
GEMM store and addition order, then returns the raw residual sum and
512-column FP32 mean-square partials. The root graph reduces the partials,
forms `rstd`, applies `rstd` and gamma to the raw sum in FP32, and stores the
normalized BF16 activation. The raw sum and optional saved `rstd` remain
available for checkpointed backward.

This terminal form does not inspect or rewrite consumers of the norm. The
following Q/KV, dense SwiGLU, router, shared-expert, and routed grouped GEMMs
remain available to F2, F4, F6, and distMoE. F3 and F4 consequently have no
pass-order dependency.

The focused CPU tests cover both boundary families, the shared-expert two-add
form, saved forward values, arbitrary downstream consumers, F4 composition,
and invalid module-role pairs. The full CODA pass test file has 37 passing
tests. A refreshed DSV3-671B fake-backend graph proof and GB300 benchmark are
still required; the counts and timings below predate the terminal rewrite.

### Superseded cross-GEMM implementation

The following evidence records the earlier implementation of the paper's
central GEMM-residual-RMSNorm-GEMM
reparameterization. It follows PyTorch's end-to-end FlexGEMM coverage in
`test_mm_coda_rmsnorm_rewrite_e2e`: the first FlexGEMM emits a gamma-weighted
activation and 512-column mean-square partials, a small root-graph reduction
forms the row-wise `rstd`, and each downstream FlexGEMM applies `rstd` in its
epilogue. The TorchTitan body additionally emits the raw residual sum because
the model reuses that value as the residual stream and checkpointed backward
requires it.

The matcher covers three grounded boundaries:

```text
layers.L.attention.wo -> residual -> layers.L.ffn_norm
    -> layers.L.feed_forward.{w1,w3}

layers.L.feed_forward.w2 -> residual -> layers.L+1.attention_norm
    -> layers.L+1.attention.{wq_a,wkv_a}

layers.L.moe.shared_experts.w2 + routed output -> residual
    -> layers.L+1.attention_norm
    -> layers.L+1.attention.{wq_a,wkv_a}
```

The first form is restricted to the three dense FFN layers. MoE normalized
activations also feed routing and grouped GEMMs, so that boundary remains
assigned to distMoE. The third form starts at the shared expert's ordinary W2
GEMM after the routed expert collective. It captures both the routed result and
the residual in their original BF16 addition order; it does not rewrite either
grouped expert GEMM.

For the first form, F3 also recognizes the exact dense SwiGLU epilogues handled
by F4. Its W1 FlexGEMM applies `rstd` and SiLU, then the W3 FlexGEMM applies
`rstd` and multiplies by the captured W1 activation. This keeps both fusions on
the six overlapping original/recomputed boundaries. When both patterns are
enabled, `f3_residual_rmsnorm` must therefore precede `f4_dense_swiglu`; config
validation rejects the reverse order instead of silently losing F3 coverage.

Real before sample from layer 0:

```python
mm_4 = torch.ops.aten.mm.default(view_22, t_4)
attention_out = torch.ops.aten.reshape.default(mm_4, [24, 4096, 7168])
residual = torch.ops.aten.add.Tensor(embedding, attention_out)
norm = torch.ops.aten._fused_rms_norm.default(
    residual, [7168], ffn_norm_weight, 1e-05
)
normalized = norm[0]
w1 = torch.ops.aten.mm.default(normalized.reshape(98304, 7168), t_5)
w3 = torch.ops.aten.mm.default(normalized.reshape(98304, 7168), t_6)
```

Real after sample:

```python
residual_2d = torch.ops.aten.reshape.default(embedding, [98304, 7168])
gamma_2d = torch.ops.aten.reshape.default(ffn_norm_weight, [1, 7168])
first = torch.ops.higher_order.flex_gemm(
    torch.ops.aten.mm.default,
    _coda_f3_residual_first_body_0,
    (view_22, t_4, residual_2d, gamma_2d),
    {},
    {"backend": "QUACK"},
)
weighted = first[0]
residual = first[1]
partial_mean_square = first[2]
rstd = torch.ops.aten.rsqrt.default(
    torch.ops.aten.add.Scalar(
        torch.ops.aten.mean.dim(partial_mean_square, [-1], True), 1e-05
    )
)
activated = torch.ops.higher_order.flex_gemm(
    torch.ops.aten.mm.default,
    _coda_f3_f4_silu_body_0,
    (weighted, t_5, rstd),
    {},
    {"backend": "QUACK"},
)[0]
gate, product = torch.ops.higher_order.flex_gemm(
    torch.ops.aten.mm.default,
    _coda_f3_f4_mul_body_0,
    (weighted, t_6, rstd, activated),
    {},
    {"backend": "QUACK"},
)
```

Real MoE-output before and after samples from layer 3:

```python
# Before
shared = torch.ops.aten.mm.default(view_159, t_34)
shared = torch.ops.aten.reshape.default(shared, [24, 4096, 7168])
moe_output = torch.ops.aten.add.Tensor(routed_output, shared)
residual = torch.ops.aten.add.Tensor(previous_residual, moe_output)
norm = torch.ops.aten._fused_rms_norm.default(
    residual, [7168], attention_norm_weight, 1e-05
)

# After
routed_2d = torch.ops.aten.reshape.default(routed_output, [98304, 7168])
residual_2d = torch.ops.aten.reshape.default(previous_residual, [98304, 7168])
gamma_2d = torch.ops.aten.reshape.default(attention_norm_weight, [1, 7168])
first = torch.ops.higher_order.flex_gemm(
    torch.ops.aten.mm.default,
    _coda_f3_residual_first_body_6,
    (view_159, t_34, routed_2d, residual_2d, gamma_2d),
    {},
    {"backend": "QUACK"},
)
```

FSDP bucketing can define the next norm's unsharded gamma after the producing
GEMM. The new first HOP is therefore placed immediately before the norm, where
the GEMM inputs, residual, and gamma all dominate it. A late-gamma unit test
covers this real graph ordering.

### Graph proof

Configuration: DSV3-671B, fake communication backend, FSDP 256, EP 64, local
batch 24, sequence length 4096, full activation checkpointing, `c4_test`.

Artifact root:

```text
outputs/profiling/dsv3_fake/graph/coda-all-composed-f3-f4-20260810/tlparse/-_-_-_-
```

Before and after dumps:

```text
before_fuse_f3_residual_rmsnorm_pass_278.txt
after_fuse_f3_residual_rmsnorm_pass_279.txt
```

Pass diff for the root graph:

| Operation | Before | After | Delta |
| --- | ---: | ---: | ---: |
| root `mm.default` | 2,024 | 1,826 | -198 |
| root `_fused_rms_norm.default` | 490 | 424 | -66 |
| root `add.Tensor` | 959 | 836 | -123 |
| root `flex_gemm` | 124 | 322 | +198 |
| root `get_attr` | 551 | 749 | +198 |
| root `getitem` | 24,802 | 25,072 | +270 |
| root `mean.dim` | 0 | 66 | +66 |
| root `add.Scalar` | 0 | 66 | +66 |
| root `rsqrt.default` | 0 | 66 | +66 |
| root `_to_copy.default` | 2,478 | 2,481 | +3 |
| root `reshape.default` | 6,938 | 7,007 | +69 |
| root `silu.default` | 238 | 232 | -6 |

All six original dense boundaries, 57 original MoE-output boundaries, and three
within-layer recomputed dense boundaries fused, covering 132 downstream
projections and six dense SwiGLU epilogues. The following F4 pass fused all 116
remaining, non-overlapping SwiGLU chains. Cross-layer residual outputs are
`MUST_SAVE` under full activation checkpointing, so their backward-side norms
have no recomputed producer GEMM to fuse. The proof run then OOMed on the known
192 GiB unfused FlexAttention allocation after all pass artifacts were written.

### GB300 microbenchmarks

The representative attention-to-dense-FFN boundary uses BF16 tensors with
`M=98,304`, attention width `16,384`, model width `7,168`, and two FFN
projections of width `18,432`. It includes all three GEMMs, the residual add,
RMSNorm, partial-statistics reduction, row-scale epilogues, SiLU, and the final
gate multiply. Inputs were scaled by `0.02`; five warmups and 20 CUDA-event
iterations were run on one GB300.

| Implementation | Median | Min | Max |
| --- | ---: | ---: | ---: |
| eager full boundary | 48.657 ms | 46.389 ms | 51.643 ms |
| three composed QUACK FlexGEMMs | 48.837 ms | 46.274 ms | 51.238 ms |

Speedup: `1.00x` (`0.996x`; FlexGEMM is 0.4% slower). This remains under the
project policy that accepts FlexGEMM fusions without a speedup. The residual
output was bitwise exact. The final SwiGLU output had `max_abs_error=0.5` and
`mean_abs_error=0.00283090`. Moving `rstd` across the downstream GEMMs changes
rounding, so this pattern requires convergence validation.

The representative MoE-output boundary uses a shared-expert W2 GEMM with
`M=98,304`, input width `2,048`, and model width `7,168`. Its first epilogue
captures both model-width routed and residual tensors, and the two downstream
attention projections have widths `1,536` and `576`. The benchmark otherwise
uses the same input scaling, warmups, and iteration count.

| Implementation | Median | Min | Max |
| --- | ---: | ---: | ---: |
| eager shared W2 + two adds + RMSNorm + projections | 4.464 ms | 4.284 ms | 5.673 ms |
| three QUACK FlexGEMMs + two captures + partial reduction | 5.120 ms | 4.999 ms | 6.504 ms |

Speedup: `0.87x`. The residual output was bitwise exact. Both projected outputs
had `max_abs_error=0.0625`; their mean absolute errors were `0.00223414` and
`0.00223534`. This does not justify a custom kernel under the project policy.

## B5 MLA projection and RMSNorm backward

Pattern name: `b5_mla_rmsnorm_backward`.

The pass matches only backward BF16 input-gradient GEMMs from
`layers.*.attention.wkv_b` and `layers.*.attention.wq_b` whose sole reshape
feeds `_fused_rms_norm_backward` with both `dx` and `dweight` requested. The
FlexGEMM preserves the original BF16 GEMM store and emits 128-column partials
of `x_hat * grad_x_hat`. Regional Inductor then completes the row dot product,
`dx`, and the independent token-axis `dweight` reduction. The latter cannot be
put in the same FlexGEMM body because the two reductions use different grouped
layouts.

Real before sample from the layer-60 KV path:

```python
mm_581 = torch.ops.aten.mm.default(view_4476, _unsafe_view_2729)
view_4477 = torch.ops.aten.reshape.default(mm_581, [24, 4096, 512])
_fused_rms_norm_backward_2 = (
    torch.ops.aten._fused_rms_norm_backward.default(
        view_4477,
        getitem_8778_recomputed,
        [512],
        alias_473,
        _unsafe_view_2728,
        [True, True],
    )
)
```

Real after sample:

```python
flex_gemm = torch.ops.higher_order.flex_gemm(
    torch.ops.aten.mm.default,
    _coda_b5_mla_rmsnorm_body_0,
    (
        view_4476,
        _unsafe_view_2729,
        reshape_default,
        reshape_default_1,
        reshape_default_2,
    ),
    {},
    {"backend": "QUACK"},
)
rounded = flex_gemm[0]
partial_row_dot = flex_gemm[1]  # f32[98304, 4]
row_dot = torch.ops.aten.sum.dim_IntList(partial_row_dot, [-1], True)
grad_input = torch.ops.aten.sub.Tensor(grad_x_hat, correction)
grad_weight = torch.ops.aten.sum.dim_IntList(grad_weight_terms, [0])
```

Proof artifacts:

- Before:
  `outputs/profiling/dsv3_fake/graph/coda-b5-proof-20260810/tlparse/-_-_-_-/before_fuse_b5_mla_rmsnorm_backward_pass_274.txt`
- After:
  `outputs/profiling/dsv3_fake/graph/coda-b5-proof-20260810/tlparse/-_-_-_-/after_fuse_b5_mla_rmsnorm_backward_pass_275.txt`

The DSV3-671B fake-backend proof used local batch 24, sequence 4,096, EP64,
FSDP256, full activation checkpointing, and `c4_test`. It fused all 122 grounded
chains, two per transformer layer. Root `mm` and
`_fused_rms_norm_backward` counts each fell by 122; 122 FlexGEMMs were added.
The remaining 123 RMSNorm backward nodes are outside the MLA pattern. As in the
other proof runs, execution later OOMed on the known 192 GiB unfused
FlexAttention allocation after all graph artifacts were written.

### GB300 microbenchmarks

Both real shapes used `M=98,304`, BF16 inputs scaled by `0.02`, `rstd` sampled
from `[0.5, 1.5]`, five warmups, and 20 CUDA-event samples on one GB300. The KV
case is `(98,304, 32,768) @ (32,768, 512)`; the Q case is
`(98,304, 24,576) @ (24,576, 1,536)`.

| Shape | Implementation | Median | Min | Max |
| --- | --- | ---: | ---: | ---: |
| KV, `N=512` | eager GEMM + fused RMSNorm backward | 1.975 ms | 1.966 ms | 2.100 ms |
| KV, `N=512` | FlexGEMM row partial + compiled completion | 2.558 ms | 2.523 ms | 2.915 ms |
| Q, `N=1,536` | eager GEMM + fused RMSNorm backward | 4.218 ms | 3.960 ms | 4.410 ms |
| Q, `N=1,536` | FlexGEMM row partial + compiled completion | 5.454 ms | 5.194 ms | 5.733 ms |

The FlexGEMM path is `0.772x` eager for KV and `0.773x` for Q, about 29% slower
in both cases. It remains accepted under the project policy that permits a
FlexGEMM pattern without a speedup. KV `dweight` was bitwise exact; KV `dx` had
`max_abs_error=7.451e-9` and `mean_abs_error=1.573e-16`. Q `dx` had
`max_abs_error=5.821e-11` and `mean_abs_error=5.951e-19`; Q `dweight` had
`max_abs_error=4.768e-7` and `mean_abs_error=3.104e-10`. These differences come
from the 128-column partial reduction order and require convergence validation.

## Historical composed graph proof

This proof predates the terminal F3 implementation above. Its F3 row, total
FlexGEMM count, and composed F3/F4 conclusions must not be used to validate the
current pass. All 11 FlexGEMM patterns were enabled together after
`joint_transformer_block_bucketing_reordering_pass`. The DSV3-671B fake-backend
run used FSDP256, EP64, local batch 24, sequence length 4,096, full activation
checkpointing, force-balanced MoE routing, deterministic seed 42, and
`c4_test`.

Artifact root:

```text
outputs/profiling/dsv3_fake/graph/coda-all-proof-20260810/tlparse/-_-_-_-
```

| Pass | Grounded matches | FlexGEMMs added |
| --- | ---: | ---: |
| B1 LM-head input-gradient cast | 8 | 8 |
| F6 router sigmoid and bias | 116 | 116 |
| F3 residual RMSNorm | 66 boundaries across 132 projections | 198 |
| F4 dense/shared-expert SwiGLU | 116 chains | 232 |
| B2 dense/shared-expert SwiGLU backward | 61 derivative + 61 add chains | 122 |
| F2 MLA Q RMSNorm | 62 | 124 |
| F2 MLA KV RMSNorm | 62 | 124 |
| B4 router input-gradient add | 58 | 58 |
| B5 MLA RMSNorm backward | 122 | 122 |
| B6 BF16 weight-gradient cast | 496 | 496 |
| B7 attention input-gradient merge | 61 | 61 |

The final root graph has 1,661 FlexGEMMs. Relative to the post-bucketing graph,
root `mm.default` fell from 2,148 to 487, `sigmoid.default` from 116 to zero,
`_fused_rms_norm.default` from 490 to 300, and
`_fused_rms_norm_backward.default` from 245 to 123. F3 ran before F4, so its
six dense SwiGLU epilogues were retained inside the composed F3 boundary while
F4 handled the remaining 116 shared-expert chains. Every isolated pattern
retained its expected match count.

The run disabled regional Inductor and CUDA graphs to preserve every pass dump.
After all 19 configured graph passes and artifacts completed, execution hit the
known 192 GiB allocation in unfused FlexAttention. This failure is downstream
of the rewrite proof and is not a CODA pass failure.

## CODA-kernels comparison

CODA-kernels was measured from `~/local/coda-kernels` on branch
`gb300-perf-v061`, commit `b5afe0d`. The branch starts from CODA commit
`c9c4447`, the last commit with a coherent Quack 0.6.1 API, and adds SM100/110
GEMM dispatch. CODA `main` at `8c7c4d5` starts a Quack 0.6.2 migration but mixes
the old batch-last ABI with new epilogue imports and does not run as checked
out.

The two implementations require dependency-isolated runs:

- PyTorch FlexGEMM: CUTLASS DSL 4.5.2 and PyTorch's vendored Quack.
- CODA-kernels: Quack 0.6.1 and CUTLASS DSL 4.6.1 with the CUDA 13 library
  package, needed to import `GemmSm100` on this CUDA 13.0 host.

Both runs used BF16 tensors scaled by `0.02`, five warmups, 20 CUDA-event
samples, and one GB300. The tables therefore compare steady-state GPU time but
not a single shared Python process. This matters for small differences: the F4
eager median was `3.482 ms` in the CODA process and `3.613 ms` in the earlier
FlexGEMM process.

### F1 LM-head and cross entropy

The real per-chunk shape is `(12,288, 7,168) @ (7,168, 129,280)`. The eager
boundary is the exact graph sequence: BF16 GEMM, FP32 log-softmax, summed NLL,
FP32 log-softmax backward, BF16 `dlogits`, and the two activation/weight
gradient GEMMs. The loss is divided by the full 98,304-token local batch.

FlexGEMM has no valid implementation for this row. Its local-reduction API
cannot combine online max/sum state across the full 129,280-column vocabulary
or select the target column. CODA instead uses `gemm_lse`, overwrites the BF16
logits buffer with `dlogits` using `cross_entropy_fwd_bwd_`, and applies the loss
scale in the two gradient GEMM epilogues. This avoids the 6.35 GB FP32
log-softmax tensor while retaining the backward data.

| Implementation | Median | Min | Max |
| --- | ---: | ---: | ---: |
| eager full forward/backward boundary | 46.735 ms | 45.926 ms | 51.256 ms |
| PyTorch QUACK FlexGEMM | unsupported | unsupported | unsupported |
| CODA fused-LSE boundary | 42.794 ms | 41.780 ms | 43.763 ms |

CODA is `1.092x` faster than eager. This is a fixed-config result, not an
autotuned result: `tile_m=256`, `tile_n=256`, `cluster_m=2`, `cluster_n=1`,
dynamic persistent scheduling, and no ping-pong. CODA's generic LSE autotuner
currently includes SM100 layouts whose epilogue has more than one N warp and
fails its `warps_in_N == 1` invariant, so the known-valid configuration was
selected directly.

Against eager, CODA had `max_abs_error=1.192e-7` in loss and
`max_abs_error=7.451e-9` in both gradients. Mean absolute errors were
`3.164e-10` for the activation gradient and `2.963e-11` for the weight
gradient. CODA computes LSE from the FP32 GEMM accumulator and stores unscaled
BF16 `dlogits`, then moves the loss scale into the gradient GEMM epilogues; the
source graph rounds BF16 logits before FP32 log-softmax and rounds scaled
`dlogits` before the GEMMs. The faster kernel is therefore a performance
candidate, not a numerically identical replacement.

This kernel was not wired into the GraphTrainer pass pipeline. CODA requires
Quack 0.6.1 and CUTLASS DSL 4.6.1, while the PyTorch build that provides the
landed FlexGEMM passes uses its vendored Quack and CUTLASS DSL 4.5.2. Loading
CODA directly would make the two implementations share an incompatible
`cutlass` Python package. The benchmark used the isolated environment
`~/local/coda-kernels/.venv-coda061`.

### F4 shared-expert SwiGLU

Real shape: `M=98,304`, `K=7,168`, `P=2,048`. CODA interleaves W1/W3 columns
and executes one `gemm_swiglu` with a `K x 2P` weight. This performs the same
GEMM FLOPs as the two eager/FlexGEMM projections and returns the combined raw
preactivation plus the final `P`-wide product.

| Implementation | Median | Min | Max |
| --- | ---: | ---: | ---: |
| eager two GEMMs + SiLU + multiply | 3.482 ms | 3.359 ms | 3.743 ms |
| two PyTorch QUACK FlexGEMMs | 3.711 ms | 3.619 ms | 4.069 ms |
| CODA `gemm_swiglu` (autotuned) | 3.583 ms | 3.529 ms | 3.853 ms |

CODA is `1.036x` faster than FlexGEMM but `0.972x` versus its paired eager
baseline. That difference is within the observed cross-process eager drift, so
there is no defensible F4 winner. CODA's preactivation was bitwise exact versus
the paired eager GEMMs. Its final product had `max_abs_error=6.104e-5` and
`mean_abs_error=8.497e-7`; CODA applies SwiGLU to the FP32 accumulator before
the BF16 output store, unlike the graph's explicit BF16 intermediate.

### Historical composed F3 and F4 dense boundary

Real shape: `(98,304, 16,384) @ (16,384, 7,168)`, followed by two
`(98,304, 7,168) @ (7,168, 18,432)` projections. CODA uses
`gemm_residual_partial_rmsnorm` followed by one interleaved
`gemm_rmsnorm_swiglu`, reducing three GEMMs to two without changing the total
FLOPs.

| Implementation | Median | Min | Max |
| --- | ---: | ---: | ---: |
| eager full graph boundary | 48.657 ms | 46.389 ms | 51.643 ms |
| three composed PyTorch QUACK FlexGEMMs | 48.837 ms | 46.274 ms | 51.238 ms |
| two CODA GEMMs, SM100 default config | 46.715 ms | 45.555 ms | 47.510 ms |

The CODA row is `1.045x` faster than FlexGEMM and `1.042x` faster than the
recorded eager median. It is not an autotuned result: CODA's full search was
stopped after more than 20 minutes while tuning the second GEMM. The measured
config was its SM100 default, `tile_m=256`, `tile_n=256`, `cluster_m=2`, and
`cluster_n=1`.

This CODA path is not currently a drop-in numerical replacement. It adds the
residual in the GEMM accumulator, while the source graph stores BF16 before the
residual add. Against an `addmm`-ordered reference, the residual had
`max_abs_error=0.001953` and `mean_abs_error=6.117e-5`; the final product had
`max_abs_error=0.0001221` and `mean_abs_error=2.161e-6`. The result is a useful
performance bound, but landing it requires an explicit numerical policy and
convergence validation.

### Pattern coverage

| GraphTrainer pattern | Existing CODA-kernels analogue | Comparison status |
| --- | --- | --- |
| F1 LM-head plus cross entropy | `gemm_lse` + `cross_entropy_fwd_bwd_` + scaled GEMMs | measured above; faster external candidate, FlexGEMM unsupported |
| `b1_lm_head_input_grad_cast` | none with the required FP32 post-store cast | FlexGEMM/eager only |
| `b6_bf16_weight_grad_cast` | none with the required FP32 post-store cast | FlexGEMM/eager only |
| `f6_router_sigmoid_bias` | no public sigmoid-plus-bias GEMM | FlexGEMM/eager only |
| `f4_dense_swiglu` | `gemm_swiglu` | measured above |
| `b2_dense_swiglu_backward` | `gemm_swiglu_bwd_zdz` | ABI mismatch: CODA emits packed interleaved `dZ` and a full-row `ZdZ`; the graph consumes separate W1/W3 branches |
| `f2_q_rmsnorm` | QKV square-sum and RMS-scaled GEMM primitives | no exact 1,536-wide segmented boundary wrapper |
| `f2_kv_rmsnorm` | RMS-scaled GEMM primitives | no exact 512-plus-64 tail/RoPE wrapper |
| `b4_router_input_grad_add` | no public residual-add-only GEMM | FlexGEMM/eager only |
| `b5_mla_rmsnorm_backward` | `gemm_residual_partial_rmsnorm_bwd` has a different residual/`ZdZ` contract | FlexGEMM/eager measured above; no exact CODA row |
| `b7_attention_grad_merge` | no public captured-add-only GEMM | FlexGEMM/eager only |
| `f3_residual_rmsnorm` | residual partial RMSNorm plus RMS-scaled GEMM | measured with F4 above |

The routed grouped-GEMM opportunities remain assigned to distMoE. They were
not included in this comparison.
