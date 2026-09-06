# CODA investigation handoff

Last updated: 2026-08-21

## Resume point

The work is preserved on the local branch `coda-flex-gemm-passes`. Its base is
TorchTitan commit `9228564523aa63f78c5e3e038068a886572e90de`. Resume with:

```bash
cd /home/bahuang/local/torchtitan
git switch coda-flex-gemm-passes
git log --oneline --decorate -20
```

The branch is local and has no configured upstream. Do not delete it when
starting unrelated work from `main`.

The main implementation is in:

```text
/home/bahuang/local/torchtitan/torchtitan/experiments/graph_trainer/coda_passes.py
/home/bahuang/local/torchtitan/torchtitan/experiments/graph_trainer/configs.py
/home/bahuang/local/torchtitan/torchtitan/experiments/graph_trainer/passes.py
/home/bahuang/local/torchtitan/torchtitan/experiments/graph_trainer/tests/test_coda_passes.py
```

The graph evidence and pattern-by-pattern status are in:

```text
/home/bahuang/local/torchtitan/torchtitan/experiments/graph_trainer/CODA_FUSION_RESULTS.md
```

## Branch history

The CODA branch contains these commits after its TorchTitan base:

```text
73408dc18 Fuse B6 BF16 weight-gradient casts with FlexGEMM
d2cf35a47 Fuse F6 router sigmoid and bias with FlexGEMM
32ede641b Fuse F4 dense SwiGLU with FlexGEMM
538cf666c Fuse B2 dense SwiGLU backward with FlexGEMM
ccd9d2e6f Fuse F2 MLA Q RMSNorm with FlexGEMM
dff1bbe10 Fuse B4 router input gradient with FlexGEMM
f2efa2187 Fuse F3 residual RMSNorm with FlexGEMM
421675a7b Fuse F2 MLA KV RMSNorm with FlexGEMM
f11d8c709 Fuse B7 attention gradient merge with FlexGEMM
6eb058388 Fuse B1 LM-head input gradient casts with FlexGEMM
841ad78d6 Compose F3 RMSNorm and F4 SwiGLU FlexGEMMs
8aeed44d6 Document CODA kernel GB300 comparisons
08b7cc582 Fuse B5 MLA RMSNorm backward with FlexGEMM
dab40eeb2 Document CODA linear cross-entropy benchmark
38f0ad020 Document composed CODA graph proof
09c7c538f Implement terminal F3 residual RMSNorm fusions
388cdae19 Add CODA FlexGEMM autotuning benchmarks
100b595ef Add reusable DSV3 profiling workflow
```

The handoff document itself is in the next branch commit. Use `git log` rather
than relying on this list if more commits are added after resuming.

## Implemented pass scope

The branch implements GraphTrainer rewrites for:

- B1: LM-head input-gradient BF16 store rounding plus FP32 cast.
- B2: dense/shared-expert SwiGLU backward branch derivatives and input add.
- B4: router input-gradient GEMM, BF16 store rounding, and expert-gradient add.
- B5: MLA projection plus RMSNorm backward.
- B6: BF16 weight-gradient store rounding plus FP32 cast.
- B7: Q and KV attention input-gradient merge.
- F2-Q: Q low-rank projection, RMSNorm, and Q-B projection.
- F2-KV: segmented KV low-rank projection, RMSNorm, and KV-B projection.
- F3-A: attention WO projection, residual add, and following RMSNorm.
- F3-B: FFN/shared-expert W2 projection, residual path, and following RMSNorm.
- F4: dense/shared-expert W1/W3 SwiGLU forward.
- F6: router GEMM, sigmoid, and optional expert bias.

F3 and F4 composition is implemented so the terminal F3 producer can stop at
RMSNorm without consuming the next layer's projection. The pass runs after
`joint_transformer_block_bucketing_reordering_pass` because the patterns and
FSDP bucketing boundaries must be final before matching.

Routed `aten._grouped_mm` chains are intentionally excluded. They belong to
DistMoE, especially under MinimalAsyncEP. Do not add dense FlexGEMM rewrites
around those routed grouped GEMMs without coordinating with the DistMoE
implementation.

F1 linear cross-entropy is documented but is not implemented as a FlexGEMM
pass. CODA kernels can reduce the full vocabulary materialization and remain a
better external candidate for that pattern.

## FX graph evidence

The original DSV3-671B traced graph is:

```text
/home/bahuang/local/torchtitan/outputs/profiling/dsv3_fake/graph/20260805-110153/tlparse/-_-_-_-/make_fx_graph_traced_257.txt
```

The post-bucketing graph used for the CODA analysis is:

```text
/home/bahuang/local/torchtitan/outputs/profiling/dsv3_fake/graph/20260805-110153/tlparse/-_-_-_-/after_joint_transformer_block_bucketing_reordering_pass_273.txt
```

The most useful composed F3/F4 proof graph is:

```text
/home/bahuang/local/torchtitan/outputs/profiling/dsv3_fake/graph/coda-all-composed-f3-f4-20260810/tlparse/-_-_-_-/after_joint_transformer_block_bucketing_reordering_pass_273.txt
```

The unfused DSV3-16B post-bucketing graph used to extract the smaller-model
benchmark shapes is:

```text
/home/bahuang/local/torchtitan/outputs/profiling/dsv3_fake/graph/dsv3-16b-unfused-shapes-20260812/tlparse/-_-_-_-/after_joint_transformer_block_bucketing_reordering_pass_142.txt
```

Pattern-specific proof directories are listed under each `Graph proof`
section of `CODA_FUSION_RESULTS.md`.

## Benchmark code

The standalone source-eager, compiled-eager, and FlexGEMM suite is here:

```text
/home/bahuang/local/torchtitan/torchtitan/experiments/graph_trainer/benchmarks/coda_fusion_microbench.py
/home/bahuang/local/torchtitan/torchtitan/experiments/graph_trainer/benchmarks/coda_fusion_microbench_16b.py
/home/bahuang/local/torchtitan/torchtitan/experiments/graph_trainer/benchmarks/coda_fusion_autotune.py
/home/bahuang/local/torchtitan/tests/unit_tests/test_coda_fusion_microbench.py
```

Every FlexGEMM uses `tuned: true`. Epilogues containing SiLU or sigmoid use
fast math. The external tuner isolates every explicit QUACK configuration in
a fresh process because some SM100 configurations fail or hang on SM103.
Multi-GEMM cases use coordinate descent; a full search evaluates all 74
configurations per FlexGEMM, not the Cartesian product of configurations.

Run a 16B case with the known CUDA paths pinned:

```bash
cd /home/bahuang/local/torchtitan
CUDA_ROOT=/home/bahuang/local/venvs/torch-cu132-nightly/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib"
export TORCH_NATIVE_SKIP_VERSION_CHECK=1

python -m torchtitan.experiments.graph_trainer.benchmarks.coda_fusion_autotune \
  --suite 16b \
  --case f4_shared_expert_swiglu \
  --devices 0,1,2,3 \
  --search full \
  --passes 1
```

Before rerunning, verify that this environment really reports CUDA 13.2. As of
2026-08-21, the venv at that path has drifted and reports PyTorch
`2.15.0a0+git1b5baff`, CUDA 13.0, and git revision
`1b5baff2649da3d5c57ad14d72cc2dbc24dbdf72`. It is not the environment that
produced the recorded CUDA 13.2 results.

The recorded 16B exhaustive run used:

```text
PyTorch: 2.15.0.dev20260812+cu132
PyTorch git: 3eb0e5d0968d26971fdd6684ab5c9b605bfec4a6
CUDA: 13.2
CUTLASS DSL: 4.6.2
Triton: 3.8.0+git675c5987
GPU: NVIDIA GB300, compute capability 10.3
```

## Performance conclusions

The DSV3-671B 12-pattern results are recorded in
`CODA_FUSION_RESULTS.md`. The strongest isolated results were B4 at `11.582x`
and F6 at `9.428x` versus compiled eager. F4 was `1.037x`; most large-GEMM and
RMSNorm patterns were neutral or slower.

The DSV3-16B exhaustive run covered 13 cases, 18 FlexGEMMs, and 1,332
candidate evaluations. The primary long-run results versus compiled eager
were:

| Pattern | Speedup | Conclusion |
| --- | ---: | --- |
| B4 router input-gradient add | `4.531x` | Strong, stable win |
| F4 shared-expert SwiGLU | `1.191x` | Useful win |
| B7 attention input-gradient merge | `1.039x` | Small win |
| F4 dense SwiGLU | `1.015x` | Marginal; verify end to end |
| B1 LM-head cast | `1.014x` | Effectively neutral |
| B6 shared weight-gradient cast | `1.063x` final, `0.979x` verification | Noisy/inconclusive |
| F2-KV RMSNorm | `0.981x` | Slight regression |
| B5 KV RMSNorm backward | `0.891x` | Regression |
| B2 shared/dense backward | `0.944x` / `0.953x` | Regression |
| F3 attention/MoE/dense | `0.849x` / `0.779x` / `0.959x` | Regression |

The complete 16B report and raw results are:

```text
/home/bahuang/local/torchtitan/outputs/coda_fusion_microbench/dsv3_16b_exhaustive_autotune/FULL_REPORT.md
/home/bahuang/local/torchtitan/outputs/coda_fusion_microbench/dsv3_16b_exhaustive_autotune/16b
```

The portable 671B autotuning handoff is:

```text
/home/bahuang/local/torchtitan/outputs/coda_fusion_autotune_handoff
/home/bahuang/local/torchtitan/outputs/coda_fusion_autotune_handoff.zip
```

## Known issues

1. Reduction configuration coverage is the largest FlexGEMM limitation. F3
   accepts only 2 of the 74 generic configurations because the local RMSNorm
   reduction requires group 512. B5 and F2-KV have similar restrictions.
2. Some configuration-specific QUACK kernels raise
   `cudaErrorNoKernelImageForDevice` on SM103. This does not prevent other
   FlexGEMM configurations from running on GB300.
3. Native unconstrained `tuned: true` can lose the whole search when a bad
   SM103 candidate fails. The process-isolated tuner is a workaround; the
   backend should reject unsupported candidates before execution.
4. Small gains below roughly 3% are sensitive to GPU load and clocks. Require
   repeated fixed-GPU and end-to-end evidence before enabling those patterns.
5. Moving RMSNorm state such as `rstd` across a projection changes scheduling
   and requires convergence validation, even when isolated outputs pass the
   microbenchmark tolerance.
6. The branch is based on an older TorchTitan `main` and must be rebased or
   replayed deliberately before upstreaming. Audit the pass pipeline and all
   config callsites after rebasing.

## Recommended next work

1. Rebase the branch onto a current TorchTitan `main`, keeping each logical
   pattern commit separate so regressions can be bisected.
2. Run GraphTrainer end-to-end A/B tests with only B4 enabled, then only F6,
   then both. Use `c4_test`, at least 10 steps, seed 42, deterministic mode,
   identical parallelism, and compare full-precision loss and grad norm.
3. Add a minimized PyTorch FlexGEMM test for SM103 candidates that currently
   produce `cudaErrorNoKernelImageForDevice`, then fix candidate filtering.
4. Extend local-reduction layout support for group 512 and retune F3. More
   generic search is not useful until additional layouts are legal.
5. Compare specialized CODA kernels against FlexGEMM for F1 and the
   reduction-heavy F2/F3/B5 cases using the exact FX shapes.
6. Keep routed grouped GEMMs under DistMoE ownership. Test any dense epilogue
   changes with standard EP and MinimalAsyncEP separately.
7. Performance-gate broad graph rewrites. The evidence supports prioritizing
   B4 and F6; the remaining patterns need stronger end-to-end justification.

The local CODA kernel checkout used for earlier comparisons is:

```text
/home/bahuang/local/coda-kernels
branch: gb300-perf-v061
commit: b5afe0d9572b66845445efbc0399a4151e51235a
```

## Figures

```text
/home/bahuang/local/torchtitan/outputs/coda_fusion_patterns/coda_fusion_patterns.png
/home/bahuang/local/torchtitan/outputs/dsv3_one_layer_graph/dsv3_layer60_gemms.png
/home/bahuang/local/torchtitan/outputs/dsv3_one_layer_graph/dsv3_layer60_coda_forward_patterns.png
/home/bahuang/local/torchtitan/outputs/dsv3_one_layer_graph/dsv3_cross_layer_f3.png
/home/bahuang/local/torchtitan/outputs/dsv3_one_layer_graph/dsv3_layer60_coda_backward_patterns.png
```

## Validation state

Before the final handoff commit:

- `tests/unit_tests/test_coda_fusion_microbench.py`: 8 passed.
- Benchmark Python files: `flake8`, `ufmt`, `pydoclint`, `codespell`, and
  targeted Pyrefly passed.
- Profiling shell script: `bash -n` and ShellCheck passed.
- Trace uploader: Python compilation, `flake8`, `ufmt`, `pydoclint`,
  `codespell`, and targeted Pyrefly passed.
- Repository `pre-commit` could not bootstrap because fetching
  `pre-commit-hooks` from GitHub returned proxy HTTP 403.
- Repository-wide Pyrefly reaches unrelated optional-dependency errors for
  `transformers` and `rich`; targeted checks for the added Python files pass.

Run the core pass tests again after rebasing because the pass manager and FX
graph details may have changed on current `main`:

```bash
cd /home/bahuang/local/torchtitan
python -m pytest \
  torchtitan/experiments/graph_trainer/tests/test_coda_passes.py \
  torchtitan/experiments/graph_trainer/tests/test_passes.py -q
```
