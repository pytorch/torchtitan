# Tests

This directory contains tests for the torchtitan project, including unit tests and integration tests.

## Test Structure

- `unit_tests/cpu/`: Unit tests that run without a GPU
- `unit_tests/gpu/`: Tests that require GPUs; multi-GPU tests use the
  `multi_gpu` pytest marker
- `integration_tests/`: Contains integration tests that test multiple components together
  - `features.py`: Tests for torchtitan features and composability
  - `flux.py`: Tests for the FLUX model
  - `h100.py`: Tests cases for H100 GPUs
  - `b200.py`: Test cases that require SM100 or SM103 GPUs
  - `models.py`: Tests for model architectures
- `assets/`: Contains test assets and fixtures used by the tests
  - `losses/`: Golden loss and gradient norm curves for the numerics guards
  - `tokenizer/`: Tokenizer configuration and vocabulary files for testing
  - `custom_schedule.csv`: Custom PP schedule for testing

## TorchTitan CI Design

### Integration tests (Goal: E2E composability)

#### Principle

Use Fake PG as much as possible on pull requests for fast, broad functional
coverage. Every enabled test runs before landing: tests compatible with Fake PG
use one physical GPU, while tests marked `use_real_pg=True` use eight physical
GPUs. Scheduled and post-merge runs execute the complete suite with Real PG.

#### Cadence

- 1 GPU Fake PG cadence: pull requests on open, update, reopen, or
  ready-for-review. Reusable workflow callers run Fake PG by default.
- 8 GPU Real PG cadence: every pull request event above runs tests marked
  `use_real_pg=True` in separate `required subset - features` and
  `required subset - models` jobs. Adding the `ciflow/8gpu` pull request label
  creates a `ciflow/8gpu/*` tag and runs separate `full suite - features` and
  `full suite - models` jobs with Real PG. Pushes and merges to `main`,
  six-hour schedules, and manual dispatches also run both full-suite jobs with
  Real PG. Reusable workflow callers can explicitly request
  `execution_mode: real_pg`, as the ROCm workflow does.
- 8 GPU H100 cadence: opt-in pull requests carrying the `ciflow/h100.8` label.
  The lane always uses Real PG; updates and reopened events rerun it while the
  label remains attached.
- B200 cadence: opt-in pull requests carrying the `ciflow/b200` label and
  pushes affecting Kimi K3 on `main`. The lane uses Real PG and currently runs
  the Kimi K3 multimodal FSDP test.

Feature tests provide depth of infrastructure composability. Fake-PG runs check
that feature combinations configure, transform, and complete training, while
Real-PG runs additionally cover real collectives and distributed state. Model
tests provide width across supported implementations. Their definitions remain
separate for clarity. Each 8 GPU Real-PG suite runs as its own CI job with an
independent timeout; the model job also runs the FLUX integration tests.

### Numerics tests (Goal: deterministic regression coverage)

Selected model integration tests also check numerics by setting
`golden_numerics_path`. The runner derives metrics and step count from that file,
creates a seed checkpoint for A10G Real-PG execution, and runs the integration
case through `loss_compare.py`. Fake-PG execution skips seed-checkpoint creation
and uses its fixed initialization path.

- A10G cases run on one physical GPU with Fake PG for pull requests and eight
  physical A10Gs with Real PG after merge, on schedule, or when triggered by
  the `ciflow/8gpu` label. Their golden paths can use `{execution_mode}` to
  select the `fake_pg/` or `real_pg/` directory.
  Shared numerical cases use the same configuration in both modes.
- Fake-PG goldens guard PyTorch FakeProcessGroup's deterministic synthetic
  numerical contract. They do not validate remote-rank values or EP load
  balance.
- Large Fake-PG gradient norms may still be deterministic under that synthetic
  contract. Non-finite gradients are not accepted: training stops before the
  optimizer update. A golden catches changes to finite synthetic values; it
  does not establish that the simulated gradients are numerically
  representative.
- Golden directories identify the PG mode, filenames identify the model and
  hardware tier, and the exact parallelism plan is recorded in the header.
- Qwen3.5 MoE FSDP 4 x TP 2, EP 4 does not have a numerical golden. Its A10G
  Real-PG results are not bitwise deterministic, so the case provides
  end-to-end coverage only.

| A10G model | Topology (Fake-PG and Real-PG) |
| --- | --- |
| Llama 3 | FSDP 2 x TP 2 x CP 2 |
| Llama 3 SFT | FSDP 2 |
| DeepSeek V3 | FSDP 8, EP 8 |
| GPT-OSS | FSDP 4 x TP 2, EP 4 |
| Qwen3 | FSDP 2 x TP 2 x CP 2, EP 8 |
| Muse Glimmer text | FSDP 8 |
| Qwen3.5 MoE multimodal | FSDP 4 x TP 2, EP 4 |

Kimi K2.5 continues to run as an FSDP 8, EP 8 integration case. Its multimodal
backward uses bicubic upsampling, whose CUDA backward has no deterministic
implementation, so the case does not carry a numerical golden. For manual
comparisons, `loss_compare.py` can create the model-only seed checkpoint with a
model-equivalent AdamW config while the measured run continues to use DistMuon.

Additional A10G Real-PG-only cases exercise CP and pipeline communication:

| A10G model | Pipeline-parallel topology |
| --- | --- |
| DeepSeek V3 | FSDP 2 x CP 2 x PP 2, EP 4, Interleaved1F1B |
| Llama 3 | FSDP 2 x TP 2 x PP 2, 1F1B |
| GPT-OSS | FSDP 2 x CP 2 x PP 2, EP 4, Interleaved1F1B |

On pull requests, the Fake-PG lane and the `real_pg_required` Real-PG scope
partition the enabled A10G tests without overlap. Post-merge and scheduled
Real-PG lanes run the complete selected suite. A selected configuration fails
validation if it uses checkpointing, pipeline parallelism, an explicit non-Fake
communication backend, or another known incompatibility without
`use_real_pg=True`. Hardware is encoded by suite: `features` and `models` run in
A10G lanes, while `h100` and `b200` run only with Real PG in their
hardware-specific workflows.

### Unit tests (Goal: module functionality)

- CPU versus GPU requirements are encoded by the `unit_tests/cpu/` and
  `unit_tests/gpu/` directories.
- GPU tests that require multiple physical devices use the `multi_gpu` pytest
  marker. The 1-GPU lane selects `not multi_gpu`, while the multi-GPU lane
  selects `multi_gpu` from the same GPU directory.

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
python -m tests.integration_tests.run_tests <output_dir> [--test_suite TEST_SUITE[,TEST_SUITE...]] [--execution_mode {fake_pg,real_pg}] [--test_scope {all,real_pg_required}] [--test_name TEST_NAME] [--ngpu NGPU]
```

Arguments:
- `output_dir`: (Required) Directory where test outputs will be stored
- `--test_suite`: (Optional) Comma-separated test suites to run (default: "features")
- `--execution_mode`: (Optional) Run with Fake PG or Real PG (default: `real_pg`)
- `--test_scope`: (Optional) Run all selected tests or only tests marked
  `use_real_pg=True` (default: `all`)
- `--test_name`: (Optional) Specific test to run by name (default: "all")
- `--ngpu`: (Optional) Number of GPUs to use for testing (default: 8)
- `--export-numerics`: (Optional) Export results for numerical tests instead of
  comparing against their `golden_numerics_path` files

Each test names the full configurations it runs, one per run, and the runner passes
them to `run_train.sh` as the `MODULE` and `CONFIG` env vars. The configurations
live in [torchtitan_recipes/tests](../torchtitan_recipes/tests/), one module per
test suite. To run something else, add a configuration there and a test entry
that names it.

Examples:
```bash
# Run all feature integration tests (features is the default suite)
python -m tests.integration_tests.run_tests test_output

# Run the complete A10G matrix with Fake PG on one physical GPU
python -m tests.integration_tests.run_tests test_output --test_suite features,models --execution_mode fake_pg --ngpu 1

# Run the complete A10G matrix with real process groups
python -m tests.integration_tests.run_tests test_output --test_suite features,models --execution_mode real_pg --ngpu 8

# Run only cases that explicitly require real process groups
python -m tests.integration_tests.run_tests test_output --test_suite features,models --execution_mode real_pg --test_scope real_pg_required --ngpu 8

# Run H100-only cases with real process groups
python -m tests.integration_tests.run_tests test_output --test_suite h100 --execution_mode real_pg --ngpu 8

# Run B200-only cases with real process groups
python -m tests.integration_tests.run_tests test_output --test_suite b200 --execution_mode real_pg --ngpu 8
```

### Running Unit Tests

To run only the unit tests:

```bash
pytest -s tests/unit_tests/cpu/

# Single-GPU tests
pytest -s tests/unit_tests/gpu/ -m "not multi_gpu"

# Multi-GPU tests
pytest -s tests/unit_tests/gpu/ -m multi_gpu
```

### Running Specific Unit Test Files

To run a specific test file:

```bash
pytest -s tests/unit_tests/cpu/test_config_manager.py
```

### Running Specific Test Functions in Unit Tests

To run a specific test function:

```bash
pytest -s tests/unit_tests/cpu/test_config_manager.py::TestConfigManager::test_cli_overrides
```
