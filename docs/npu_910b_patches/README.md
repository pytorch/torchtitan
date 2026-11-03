# Ascend 910B flex patches (reusable)

These patches are **not** TorchTitan source changes. They patch installed
`torch_npu` so Muse Spark debug recipes can finish on **Ascend 910B4**
(non-950). Ascend 950 keeps the stock mask-out / decoding paths.

`make_reduction` / `NPUCombinedScheduling` / remat null-storage belong to
`torchtitan-ascend` (`torchtitan_ascend/patches/` and `fixes/torch_npu/storage.patch`).
Do not reintroduce those here.

## Target versions

| Package | Version |
| --- | --- |
| `torch` | `2.15.0+cpu` (PrivateUse1 / NPU via torch_npu) |
| `torch_npu` | `2.15.0.dev20261005+git14dc54e` |

Expected MD5 after apply:

| File under site-packages | MD5 |
| --- | --- |
| `torch_npu/_inductor/kernel/flex_attention.py` | `659bf1dbc6c9652d8a5fdf9dad9bedeb` |
| `torch_npu/_inductor/runtime/triton_heuristics.py` | `e0fcd810ba31c602be44fb7c5e49b058` |

## What each patch does

1. `flex_attention.py.patch`
   - Non-950: pointwise `aten.index` for mask subgraphs when indirect-memory
     codegen is off; disable mask-out; disable flex decoding; default
     backward tiles to 16.
2. `triton_heuristics.py.patch`
   - Non-950: do not emit `simt_only` for `simd_simt_mix`.
   - Serialize multi-rank `flex_attention` autotune with
     `/tmp/npu_flex_attention_autotune.lock`.

Checklist of all flex non-950 adaptation points (including env compile
threads and recipe timeouts): see
`docs/npu_pta_fixes.md` section **flex attention 非 Ascend 950 适配点**.

## Apply

From a checkout that has these files, with the training venv active:

```bash
# optional: point at another prefix
# export NPU_PATCH_SITE_PACKAGES=/path/to/site-packages
bash docs/npu_910b_patches/apply_npu_910b_patches.sh
```

Also source the compile-thread limits before training:

```bash
source docs/npu_910b_patches/env_npu_910b.sh
```

## TorchTitan commits still required

Apply these patches **and** use the NPU Titan branch (for example
`npu-muse-spark-debug-r04`): RoPE complex64 indexing, SwiGLU fallback,
vision `contiguous()`, loss compile-time assert skip, recipe timeouts, etc.
See `docs/npu_pta_fixes.md`. Remat storage and Inductor API gaps come from
`torchtitan-ascend`, not this directory.

## Dry-run / restore

```bash
bash docs/npu_910b_patches/apply_npu_910b_patches.sh --dry-run
bash docs/npu_910b_patches/apply_npu_910b_patches.sh --restore
```
