# GraphTrainer CODA fusion results

This file records the graph evidence and isolated performance result for each
CODA fusion pass. Full tlparse dumps stay under `outputs/` because they are
generated artifacts; the grounded samples and exact artifact paths are listed
here.

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
