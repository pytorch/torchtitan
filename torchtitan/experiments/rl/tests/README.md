# RL tests

## Routing

| Test type | Location | Marker |
|---|---|---|
| CPU unit test | `tests/test_*.py` | none |
| CUDA unit test | `tests/test_*.py` | `@gpu_test(num_gpus=N)` |
| H100 unit test | `tests/test_*.py` | `@h100_test(num_gpus=N)` |
| Distributed check that needs `torchrun` | wrapper test in `tests/test_*.py` that launches a script under torchrun; see `test_bitwise_parity.py` and `scripts/bitwise_parity.py` | hardware marker on the wrapper |
| E2E GRPO config | append to `build_rl_test_list()` or `build_rl_h100_test_list()` in `tests/test_integration.py`; the parametrized test applies `gpu_test` / `h100_test` with `case.ngpu` | n/a — applied from `case.ngpu` |

CI lanes pick up tests by marker:

- CPU CI: `pytest tests -m "not gpu"`
- 4-GPU CI: `pytest tests -m "gpu and not h100"`
- H100 CI: `pytest tests -m h100`
