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

Ensure you have all development dependencies installed:

```bash
pip install -r requirements-dev.txt
pip install -r requirements.txt
```

### Running Integration Tests

To run the integration tests:

```bash
python -m tests.integration_tests.run_tests <output_dir> [--test_suite TEST_SUITE] [--test_name TEST_NAME] [--ngpu NGPU]
```

Arguments:
- `output_dir`: (Required) Directory where test outputs will be stored
- `--test_suite`: (Optional) Specific test suite to run by name (default: "features")
- `--test_name`: (Optional) Specific test to run by name (default: "all")
- `--ngpu`: (Optional) Number of GPUs to use for testing (default: 8)

Each test names the full configurations it runs, one per run, and the runner passes
them to `run_train.sh` as the `MODULE` and `CONFIG` env vars. The configurations
live in [torchtitan_recipes/tests](../torchtitan_recipes/tests/), one module per
test suite. To run something else, add a configuration there and a test entry
that names it.

Examples:
```bash
# Run all feature integration tests (features is the default suite)
python -m tests.integration_tests.run_tests test_output

# Run the model test suite
python -m tests.integration_tests.run_tests test_output --test_suite models

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
