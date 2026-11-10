# Integration test skips

`skip_cases.toml` in this directory lists integration tests that must not run on
a given backend. `tests/integration_tests/run_tests.py` reads it from
`_filter_tests`: a matching test is dropped before anything is launched, and the
skip is logged with the section that matched.

## File format

One section per backend, one key per suite, one array of test names:

```toml
[rocm]
features = [
  "sdc_replay_accelerator_graph",
  "fsdp_pp_deferred_gradient_reduction",
]
models = [
  "deepseek_v3_fused_mla_swiglu_fsdp+ep",
  "qwen3_5_moe_fsdp+tp+ep",
]
h100 = [
  "dist_gemm",   # symmetric memory is CUDA-only
]
```

- A section name aligns with `utils.device_type` (for example `xpu`). `all`,
  `cuda`, and `rocm` are special cases: `all` applies to every backend, and
  because ROCm also reports `device_type == "cuda"`, `cuda` and `rocm` are
  distinguished with `torch.version.hip`.
- Keys are suite names. By default they are the `_TEST_SUITES` values of
  `tests/integration_tests/run_tests.py`; a registered file can extend them with
  `suites=`.
- Values are `test_name`s of that suite's `IntegrationTestDefinition` entries.
- Reasons are trailing comments on the entry; an entry without a comment is
  fine.
- Suite names are checked against the suites the file may name, and test names
  against the suite being run; either mismatch raises `ValueError` before any
  test starts.

## Registering skips from an out-of-tree device repo

A device repo ships its own file with the same schema and registers it. Two
ways, in order of preference:

### 1. Entry point (automatic)

```toml
# pyproject.toml of the device repo
[project.entry-points."torchtitan.skip_cases"]
my_device = "my_device.tests:register_skips"
```

```python
# my_device/tests.py
from tests.integration_tests.skips import register_skip_cases


def register_skips() -> None:
    register_skip_cases("path/to/your_toml")
```

Set `TORCHTITAN_SKIP_CASES_AUTOLOAD=0` to disable automatic registration.

### 2. Explicit call

Call `register_skip_cases(path)` from the device repo before its runner calls
`run_tests()`, i.e. before the test list is filtered.

In both cases `suites=` can be omitted; the file is then held to titan's own
suites. Pass it only when the file introduces new suites, for example when an
experiment registers suites that are not in `_TEST_SUITES`:

```python
register_skip_cases(path, suites={"my_suite"})
```

Both ways use the same parser and the same section/suite schema, so a device
file behaves exactly like the in-tree one.

## Where a skip belongs

| Mechanism | Use for | Lives in |
|---|---|---|
| `skip_cases.toml` | permanent, backend-wide skips | this directory |
| `disabled=True` | skips not caused by hardware coupling, and temporary one-off skips | the definition |
| `--exclude a,b,c` | skipping a few cases for one invocation | the command line |
