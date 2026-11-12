# Glue roofline: how much of the fusable time the agentic compiler captured

2026-11-09, one GB300, fake PG, torch 3c53c6f. One profiled step (step 10, CUDA graphs off) per run, in
each loop's measurement setup, without kernels (`*_base`) and with the loop's kernels (`*_pack`). Base and
pack of a model ran back to back on the same GPU (DSv3 GPU 0, Qwen3 30B GPU 1, GPT-OSS GPU 2, Qwen3 235B
GPU 3). `runs.sh <model> <gpu>` reproduces them.

Method (`roofline.py`): every kernel is GEMM (incl. Dist-MoE grouped GEMMs with fused dispatch/combine),
attention (FA4), communication (launched under a collective op; fake-PG emulation, excluded from the step),
or glue (everything else). Roofline = glue share of the step: the step reduction if all glue took no time.
Captured = (base - pack) / base glue, in ms, so it does not depend on whether communication is counted.
`glue_groups.py` breaks the glue down by kind.

| Model | Step ms, base -> pack | GEMM + attention | Glue (roofline) | Achieved | Captured | Without optimizer step: glue / captured |
|---|---|---:|---:|---:|---:|---|
| DSv3 671B | 583.9 -> 502.1 | 42.7% | 57.3% | 14.0% | 24% | 49.5% / 28% |
| Qwen3 30B-A3B | 967.6 -> 690.1 | 48.9% | 51.1% | 28.7% | 56% | 45.0% / 64% |
| GPT-OSS 120B | 2002.7 -> 1676.6 | 16.1% | 83.9% | 16.3% | 19% | 82.7% / 20% |
| Qwen3 235B-A22B | 1031.0 -> 734.0 | 54.2% | 45.8% | 28.8% | 63% | 44.9% / 64% |

GPT-OSS without the deterministic index ops (sort + `indexing_backward`, 1093 ms, which the pack leaves in
place) and the optimizer step: glue 28.2%, captured 56%.

Results: `results.json` (per-category ms for every run).

Glue by kind, without -> with the loop's kernels: `groups_<model>.txt`, from
`python glue_groups.py <model>=<model>_base,<model>_pack` (2026-11-11, same runs). The run directories
(traces) stay in the local workspace `outputs/agentic_compiler/glue_roofline/`.
