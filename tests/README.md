# Tests

This directory contains tests for the torchtitan project, including unit tests and integration tests.

## Test Structure

- `unit_tests/`: Contains unit tests for individual components
- `integration_tests/`: Contains integration tests that test multiple components together
  - `features.py`: Tests for torchtitan features and composability, based on Llama3
  - `flux.py`: Tests for the FLUX model
  - `h100.py`: Tests cases for H100 GPUs
  - `models.py`: Tests for specific model architectures and configurations, other than Llama3 and FLUX
- `assets/`: Contains test assets and fixtures used by the tests
  - `tokenizer/`: Tokenizer configuration and vocabulary files for testing
  - `custom_schedule.csv`: Custom PP schedule for testing

## Running Tests

### Prerequisites

For most users, install stable PyTorch and the rest of the dependencies
from PyPI:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

PyTorch comes in as a transitive dependency of `torchdata` (stable from
PyPI). This is the fast path — it works for unit tests and most local
development, but it pins you to whatever stable torch happens to be
current, which may lag behind what CI tests against.

#### Developer path: nightly torch matching CI

To match the exact torch / torchao / torchdata / torchvision versions
that CI runs against (and to pick up features only in nightlies), install
them explicitly from the nightly index for your hardware **before**
`pip install -e .`. PyTorch publishes one nightly index per hardware
target:

| Hardware | Index URL |
|---|---|
| CPU only | `https://download.pytorch.org/whl/nightly/cpu` |
| CUDA 13.0 | `https://download.pytorch.org/whl/nightly/cu130` |
| CUDA 12.8 | `https://download.pytorch.org/whl/nightly/cu128` |
| ROCm 6.4 | `https://download.pytorch.org/whl/nightly/rocm6.4` |

(See <https://pytorch.org/get-started/locally/> for the current list of
channels.) The install pattern below mirrors what
`.github/workflows/integration_test_8gpu_h100.yaml` runs in CI, with the
index URL substituted for your hardware:

```bash
# Pick the URL for your hardware from the table above
INDEX_URL=https://download.pytorch.org/whl/nightly/cu130

# 1. Pre-install torch's pure-Python deps (faster than letting pip re-resolve).
pip install filelock typing-extensions "setuptools<82" sympy networkx jinja2 fsspec numpy

# 2. torch + torchvision — clear PIP_EXTRA_INDEX_URL so a default extra
#    index can't slip in a +cpu wheel when you wanted CUDA.
PIP_EXTRA_INDEX_URL= pip install --force-reinstall --pre torch torchvision \
  --index-url ${INDEX_URL}

# 3. torchao — USE_CPP=0 matches CI: skips C++/CUDA kernels for a faster
#    install. Drop it if you need the accelerated kernels.
USE_CPP=0 PIP_EXTRA_INDEX_URL= pip install --pre torchao \
  --index-url ${INDEX_URL}

# 4. torchdata
pip install --pre torchdata \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# 5. Finally install torchtitan + dev tooling. The earlier nightly installs
#    satisfy torch/torchao/torchdata so the resolver won't replace them.
pip install -e .
pip install -r requirements-dev.txt
```

### Running Integration Tests

To run the integration tests:

```bash
python -m tests.integration_tests.run_tests <output_dir> [--module MODULE] [--config CONFIG] [--test_suite TEST_SUITE] [--test_name TEST_NAME] [--ngpu NGPU]
```

Arguments:
- `output_dir`: (Required) Directory where test outputs will be stored
- `--module`: (Optional) Model module to use for training (default: "llama3"). Passed as `MODULE` env var to `run_train.sh`.
- `--config`: (Optional) Config function to use for training (default: "llama3_debugmodel"). Passed as `CONFIG` env var to `run_train.sh`.
- `--test_suite`: (Optional) Specific test suite to run by name (default: "features")
- `--test_name`: (Optional) Specific test to run by name (default: "all")
- `--ngpu`: (Optional) Number of GPUs to use for testing (default: 8)

Examples:
```bash
# Run all feature integration tests with default module/config (llama3/llama3_debugmodel)
python -m tests.integration_tests.run_tests test_output

# Run feature tests with a specific module and config
python -m tests.integration_tests.run_tests test_output --module llama3 --config llama3_8b

# Run only core functionality tests for features
python -m tests.integration_tests.run_tests test_output --test_suite features

# Run a specific test with 2 GPUs
python -m tests.integration_tests.run_tests test_output --test_suite features --test_name gradient_accumulation --ngpu 2
```

### Running Unit Tests

To run only the unit tests:

```bash
pytest -s tests/unit_tests/
```

### Running Specific Unit Test Files

To run a specific test file:

```bash
pytest -s tests/unit_tests/test_config_manager.py
```

### Running Specific Test Functions in Unit Tests

To run a specific test function:

```bash
pytest -s tests/unit_tests/test_config_manager.py::TestConfigManager::test_cli_overrides
```
