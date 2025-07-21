# Tests

This directory contains tests for the TorchTitan project, including unit tests and integration tests.

## Test Structure

- `unit_tests/`: Contains unit tests for individual components
- `integration_tests/`: Contains integration tests that test multiple components together
  - `integration_tests.py`: Main integration tests for various model configurations
  - `integration_tests_h100.py`: Tests specifically designed for H100 GPUs, utilizing symmetric memory and float8
  - `base_config.toml`: Base configuration file for integration tests
- `assets/`: Contains test assets and fixtures used by the tests

## Running Tests

### Prerequisites

Ensure you have all development dependencies installed:

```bash
pip install -r dev-requirements.txt
pip install -r requirements.txt
```

### Running Integration Tests

To run the integration tests:

```bash
python -m tests.integration_tests.integration_tests <output_dir> [--config_path CONFIG_PATH] [--test_name TEST_NAME] [--test_suite TEST_SUITE] [--model MODEL] [--ngpu NGPU]
```

Arguments:
- `output_dir`: (Required) Directory where test outputs will be stored
- `--config_path`: (Optional) Path to the base config file (default: "./tests/integration_tests/base_config.toml")
- `--test_name`: (Optional) Specific test to run by name (default: "all")
- `--test_suite`: (Optional) Test suite to run: 'core', 'parallelism', or 'all' (default: "all")
- `--model`: (Optional) Specify the model to run tests on (default: "all")
- `--ngpu`: (Optional) Number of GPUs to use for testing (default: 8)

Examples:
```bash
# Run all integration tests with 8 GPUs
python -m tests.integration_tests.integration_tests ./test_output

# Run a specific test with 4 GPUs
python -m tests.integration_tests.integration_tests ./test_output --test_name tp_only --ngpu 4

# Run only core functionality tests
python -m tests.integration_tests.integration_tests ./test_output --test_suite core
```

### Running Unit Tests

To run only the unit tests:

```bash
pytest -s tests/unit_tests/
```

### Running Specific Unit Test Files

To run a specific test file:

```bash
pytest -s tests/unit_tests/test_job_config.py
```

### Running Specific Test Functions in Unit Tests

To run a specific test function:

```bash
pytest -s tests/unit_tests/test_job_config.py::TestJobConfig::test_command_line_args
```
