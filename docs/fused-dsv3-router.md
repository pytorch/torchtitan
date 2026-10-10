# Fused DeepSeek V3 learned router

This experimental override replaces the operations after the gate projection
with `FusedDSv3RouterFunction`, a `torch.autograd.Function` with one CUDA forward
launch and one CUDA backward launch. Its operator wrappers and CUDA launchers
are separate from the model override.

```bash
--override torchtitan_recipes.overrides.fused_dsv3_router.fused_dsv3_router
```

The forward fuses sigmoid, biased group selection (top two scores in each of
eight groups, choose four groups), top-eight expert selection, unbiased weight
normalization and scaling, the boolean routing map, expert counts, and the raw
microbatch-wise auxiliary loss. The backward combines routing-weight and
auxiliary-loss gradients and applies the sigmoid derivative. Expert IDs, the
routing map, counts, and selection bias have no gradients.

The gate projection and its backward stay native. `AuxLoss.inject` still owns
the dynamic denominator, coefficient, metric update, and retained remat region.
The router updates its native counter once per forward. The state-dict keys and
expert-parallel settings are unchanged.

The fast path requires training, GB300 with at least 128 SMs, 4096 local tokens,
256 experts, top-k 8, 8 groups/4 selected groups, sigmoid, route normalization
with epsilon `1e-20`, scale `2.5`, the native HiMidLo gate and microbatch-wise
loss, no padding mask or autocast, and CP=TP=1. Other configurations call the
native router. No communication operation is replaced or simulated.

This is a joint routing/loss specialization. Use it as an alternative to the
loss-only override, not as an additional loss injection. Both factories target
the same DeepSeek V3 router config, so the override framework rejects using
both on the same router.

CUDA sources are included in the wheel and JIT-built with PyTorch's
`load_inline`; a matching CUDA toolkit and C++ compiler are required. Warm up
both directions before capture, or call `kernels.prepare()` from the override
module before `torch.compile`/CUDA graph capture. A Triton backend can use the
same operator and autograd interfaces; the current backend is CUDA C++.

## Validation and component timings

On PyTorch `2.16.0.dev20261007+cu130`, GB300, the tests check bit patterns for
normal, tied, near-tied, saturated, and unbiased selection, all gradient paths,
strided upstream gradients, native gate gradients at `[4096,7168]` BF16, counter
and metric state under remat, full-graph Function compilation, and CUDA replay.
CPU tests cover config replacement, fallback, and checkpoint key compatibility.

```bash
python -m pytest --import-mode=importlib tests/unit_tests/{cpu,gpu}/test_fused_dsv3_router.py
python -m benchmarks.fused_dsv3_router --buffers 96 --samples 30
```

Median microseconds per post-gate component call, with 96 independent FP32
`[4096,256]` inputs (384 MiB), interleaved CUDA-graph samples:

| Phase | Native | Fused |
| --- | ---: | ---: |
| Forward | 199.17 | 15.25 |
| Backward | 58.49 | 5.23 |
| Forward + backward | 251.97 | 19.38 |

Backward timing retains the forward autograd graph. The benchmark includes the
raw auxiliary objective; it excludes the gate GEMMs and native metric/counter
bookkeeping. These are component results, not full-model TPS. Independent
roofline acceptance and scaling validation remain open; this is an opt-in
experimental implementation.
