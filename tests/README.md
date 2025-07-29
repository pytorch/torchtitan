# Tests

This directory contains tests for the TorchTitan project, including unit tests and integration tests.

## Test Structure

- `unit_tests/`: Contains unit tests for individual components
- `integration_tests/`: Contains integration tests that test multiple components together
  - `base_config.toml`: Base configuration file for integration tests
  - `features.py`: Tests for specific TorchTitan features
  - `ft.py`: Fine-tuning integration tests
  - `h100.py`: Tests specifically designed for H100 GPUs, utilizing symmetric memory and float8
  - `models.py`: Tests for specific model architectures and configurations
- `assets/`: Contains test assets and fixtures used by the tests
  - `tokenizer/`: Tokenizer configuration and vocabulary files for testing
  - `custom_schedule.csv`: Custom learning rate schedule for testing

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
python -m tests.integration_tests.<test_module> <output_dir> [--config_path CONFIG_PATH] [--test_name TEST_NAME] [--ngpu NGPU]
```

Where `<test_module>` can be one of:
- `features`: For feature-specific tests
- `ft`: For fine-tuning tests
- `h100`: For H100 GPU-specific tests
- `models`: For model-specific tests

Arguments:
- `output_dir`: (Required) Directory where test outputs will be stored
- `--config_path`: (Optional) Path to the base config file (default: "./tests/integration_tests/base_config.toml")
- `--test_name`: (Optional) Specific test to run by name (default: "all")
- `--ngpu`: (Optional) Number of GPUs to use for testing (default: 8)

Examples:
```bash
# Run all model integration tests with 8 GPUs
python -m tests.integration_tests.models test_output

# Run only core functionality tests for features
python -m tests.integration_tests.features test_output

# Run a specific test with 2 GPUs
python -m tests.integration_tests.features test_output --test_name gradient_accumulation --ngpu 2

# Run H100-specific tests
python -m tests.integration_tests.h100 test_output
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

## Test Categories

### Feature Tests (`features.py`)
Tests specific TorchTitan features like attention mechanisms, optimizers, and other components.

### Fine-tuning Tests (`ft.py`)
Tests for fine-tuning capabilities including LoRA, QLoRA, and other parameter-efficient fine-tuning methods.

### H100 Tests (`h100.py`)
Tests specifically designed for H100 GPUs, focusing on features like symmetric memory and float8 precision.

### Model Tests (`models.py`)
Tests for specific model architectures and configurations, ensuring they train and generate correctly.
