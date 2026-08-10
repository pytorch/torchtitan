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
