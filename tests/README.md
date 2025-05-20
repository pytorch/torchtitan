# Tests

This directory contains tests for the TorchTitan project, including unit tests and integration tests.

## Test Structure

- `unit_tests/`: Contains unit tests for individual components
- `integration_tests.py`: Contains integration tests that test multiple components together
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
python ./tests/integration_tests.py <output_dir> [--config_dir CONFIG_DIR] [--test TEST] [--ngpu NGPU]
```

Arguments:
- `output_dir`: (Required) Directory where test outputs will be stored
- `--config_dir`: (Optional) Directory containing configuration files (default: "./torchtitan/models/llama3/train_configs")
- `--test`: (Optional) Specific test to run, use test names from the `build_test_list()` function (default: "all")
- `--ngpu`: (Optional) Number of GPUs to use for testing (default: 8)

Examples:
```bash
# Run all integration tests with 8 GPUs
python ./tests/integration_tests.py ./test_output

# Run a specific test with 4 GPUs
python ./tests/integration_tests.py ./test_output --test default --ngpu 4

# Run all tests with a custom config directory
python ./tests/integration_tests.py ./test_output --config_dir ./my_configs
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
