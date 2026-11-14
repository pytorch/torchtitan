# Hardware-specific integration test skips

This directory contains one TOML file per hardware backend (`cuda.toml`,
`rocm.toml`, `xpu.toml`), listing the integration tests that must not run on
that backend. `tests/integration_tests/run_tests.py` reads these files: a
matching test is dropped before anything is launched, and the skip is logged.

## File format

The file name is for readability; the top-level key inside is what takes
effect:

```toml
# rocm.toml
rocm = [
  "sdc_replay_accelerator_graph",
  "fsdp_pp_deferred_gradient_reduction",
  "deepseek_v3_fused_mla_swiglu_fsdp+ep",
  "dist_gemm",   # symmetric memory is CUDA-only
]
```

- A key aligns with `utils.device_type` (for example `xpu`).
- `cuda` and `rocm` are special cases: ROCm also reports
  `device_type == "cuda"`, so the two are distinguished with
  `torch.version.hip`.
- The value is the list of `test_name`s of `IntegrationTestDefinition` entries.
- Every `*.toml` file in this directory is loaded; out-of-tree hardware
  registers its own file as described below.

## How an out-of-tree device repo registers skips

Out-of-tree hardware registers its skips through `register_hardware_skips`. A
device repo is another repository that provides an additional device backend to
torchtitan. `register_hardware_skips` accepts a single file or a directory
(a directory loads every `*.toml` in it). There are two ways, in order of
preference:

### 1. Entry point (automatic registration)

A device repo adds the following to its `pyproject.toml`:

```toml
[project.entry-points."torchtitan.hardware_skips"]
example_hardware = "example_hardware.tests:register_skips"
```

It also needs to add a Python file that calls the registration function with the TOML
file or directory it maintains:

```python
# example_hardware/tests.py
from tests.integration_tests.hardware_skips import register_hardware_skips


def register_skips() -> None:
    register_hardware_skips(toml_file_path_or_dir)
```

### 2. Explicit call

Call `register_hardware_skips` explicitly with the TOML file or directory you
maintain. An in-repo example is
`torchtitan/experiments/graph_trainer/tests/integration_tests.py`:

```python
# torchtitan/experiments/graph_trainer/tests/integration_tests.py
register_hardware_skips(Path(__file__).parent / "hardware_skips")
run_tests(args, test_list)
```

## Where a skip belongs

| Mechanism | Use for | Lives in |
|---|---|---|
| `hardware_skips/<backend>.toml` | permanent, backend-wide skips | this directory |
| `disabled=True` | skips not caused by hardware coupling, and temporary one-off skips | the definition |
| `--exclude a,b,c` | skipping a few cases for one invocation | the command line |

`--exclude` is already used in-repo when one CI job drops a few cases:
[scripts/ci/pytorch_ci_test_runner.sh:37](../../../scripts/ci/pytorch_ci_test_runner.sh#L37),
[.github/scripts/validate_release_gpu.sh:382](../../../.github/scripts/validate_release_gpu.sh#L382),
[.github/scripts/run_8xgpu_integration_tests.sh:81](../../../.github/scripts/run_8xgpu_integration_tests.sh#L81).
