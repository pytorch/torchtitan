# Hardware-specific integration test skips

This directory contains one TOML file per hardware backend (`cuda.toml`,
`rocm.toml`, `xpu.toml`). Each file lists integration tests that must not run on
that backend; a backend with no skips keeps an empty list.
`tests/integration_tests/run_tests.py` reads the files from `_filter_tests`: a
matching test is dropped before anything is launched, and the skip is logged.

## File format

The file name is for readability; the top-level key inside is the backend the
skips apply to:

```toml
# rocm.toml
rocm = [
  "sdc_replay_accelerator_graph",
  "fsdp_pp_deferred_gradient_reduction",
  "deepseek_v3_fused_mla_swiglu_fsdp+ep",
  "dist_gemm",   # symmetric memory is CUDA-only
]
```

- A key aligns with `utils.device_type` (for example `xpu`). `cuda` and `rocm`
  are special cases: because ROCm also reports `device_type == "cuda"`, they are
  distinguished with `torch.version.hip`.
- The value lists `test_name`s of `IntegrationTestDefinition` entries.
- Reasons are trailing comments on the entry; an entry without a comment is
  fine.
- Skips match by test name, the same way `disabled` does.
- Every `*.toml` file in this directory is loaded; additional backends can also
  register a file through the entry point below.

## Registering skips from an out-of-tree device repo

A device repo ships its own file(s) with the same schema and registers them.
`register_hardware_skips` accepts either a single file or a directory; a
directory loads every `*.toml` in it. Two ways, in order of preference:

### 1. Entry point (automatic)

```toml
# pyproject.toml of the device repo
[project.entry-points."torchtitan.hardware_skips"]
my_device = "my_device.tests:register_skips"
```

```python
# my_device/tests.py
from pathlib import Path

from tests.integration_tests.hardware_skips import register_hardware_skips


def register_skips() -> None:
    register_hardware_skips(Path(__file__).parent / "hardware_skips")
```

### 2. Explicit call

Call `register_hardware_skips(path_or_dir)` from the device repo before its
runner calls `run_tests()`, i.e. before the test list is filtered.

Both ways use the same parser and the same schema; no runner change is needed.

## Where a skip belongs

| Mechanism | Use for | Lives in |
|---|---|---|
| `hardware_skips/<backend>.toml` | permanent, backend-wide skips | this directory |
| `disabled=True` | skips not caused by hardware coupling, and temporary one-off skips | the definition |
| `--exclude a,b,c` | skipping a few cases for one invocation | the command line |
