# Tests

This directory contains tests for the torchtitan project, including unit tests and integration tests.

## Test Structure

- `unit_tests/`: Contains unit tests for individual components
- `integration_tests/`: Contains integration tests that test multiple components together
  - `base_config.toml`: Base configuration file for integration tests
  - `features.py`: Tests for torchtitan features and composability
  - `ft.py`: Fault-tolerance integration tests
  - `h100.py`: Tests cases for H100 GPUs
  - `models.py`: Tests for specific model architectures and configurations
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
python -m tests.integration_tests.run_tests <output_dir> [--config_path CONFIG_PATH] [--test_suite TEST_SUITE] [--test_name TEST_NAME] [--ngpu NGPU]
```

Arguments:
- `output_dir`: (Required) Directory where test outputs will be stored
- `--test_suite`: (Optional) Specific test suite to run by name (default: "features")
- `--config_path`: (Optional) Path to the base config file (default: "./tests/integration_tests/base_config.toml")
- `--test_name`: (Optional) Specific test to run by name (default: "all")
- `--ngpu`: (Optional) Number of GPUs to use for testing (default: 8)

Examples:
```bash
# Run all model integration tests with 8 GPUs
python -m tests.integration_tests.run_tests test_output

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
pytest -s tests/unit_tests/test_job_config.py
```

### Running Specific Test Functions in Unit Tests

To run a specific test function:

```bash
pytest -s tests/unit_tests/test_job_config.py::TestJobConfig::test_command_line_args
```

## Parity Testing (HF Baseline Comparison)

To verify that torchtitan model implementations produce numerically identical outputs to their HuggingFace counterparts, use the parity test script in `scripts/checkpoint_conversion/`:

```bash
python scripts/checkpoint_conversion/numerical_tests_example.py
```

**What it does:**
1. Loads the same weights into both a torchtitan model and HF `AutoModelForCausalLM`
2. Runs forward passes on identical random inputs (100 prompts by default)
3. Computes KL divergence between output logit distributions
4. Reports average KL divergence per test configuration

**Expected results:** With correct state dict conversion (including any necessary weight permutations), the KL divergence should be on the order of **1e-13** (essentially zero). For example, Llama 3 8B:
```
Average loss for test from_hf is -1.45365707318601e-13
```

**Prerequisites:**
- A HuggingFace checkpoint for the model you want to test
- A converted DCP checkpoint (use `scripts/checkpoint_conversion/convert_from_hf.py`)
- GPU with enough memory to load the model

**Running for a specific model:**
1. Convert the HF checkpoint to DCP format:
   ```bash
   python scripts/checkpoint_conversion/convert_from_hf.py <hf_dir> <dcp_dir> --model_name <model> --model_flavor <flavor>
   ```
2. Update the paths in `numerical_tests_example.py` (or pass them as arguments) to point to your model and checkpoint
3. Run the script and verify KL divergence is near zero

For the full methodology and details, see [`scripts/checkpoint_conversion/README.md`](../scripts/checkpoint_conversion/README.md).

**Supported models with StateDictAdapters:** Llama 3, Llama 4, DeepSeek-V3, Qwen3, Flux, GPT-OSS. See each model's README in `torchtitan/models/<model>/` for model-specific parity notes.
