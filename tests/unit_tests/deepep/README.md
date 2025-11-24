# DeepEP Tests

Tests for DeepEP-based components in TorchTitan.

## DeepEPTokenDispatcher Tests

Comprehensive test suite with unit tests (mocked) and distributed tests (multi-GPU).

### Quick Start

```bash
# Unit tests (no GPU required)
python -m unittest tests.unit_tests.deepep.test_primus_turbo_flex_token_dispatcher.TestUnit -v

# Distributed tests (requires 2+ GPUs)
torchrun --nproc_per_node=2 -m tests.unit_tests.deepep.test_primus_turbo_flex_token_dispatcher

# Run all tests with script
./tests/unit_tests/deepep/run_tests.sh
```

### Test Coverage

**Unit Tests (10 tests)**
- Initialization and configuration
- Dispatch preprocessing/postprocessing
- Token dispatch/combine operations
- Edge cases (zero capacity)

**Distributed Tests (5 tests)**
- Multi-GPU initialization
- Cross-rank communication
- Expert distribution
- Shape consistency

### Requirements

**Unit Tests:**
- torch
- No GPU required
- No distributed environment

**Distributed Tests:**
- 2+ CUDA GPUs
- torch.distributed with NCCL
- Run with torchrun

### Script Usage

```bash
# Run all tests (auto-detects environment)
./tests/unit_tests/deepep/run_tests.sh

# Run with specific GPU count
./tests/unit_tests/deepep/run_tests.sh --ngpu 4

# Skip specific test types
./tests/unit_tests/deepep/run_tests.sh --skip-unit
./tests/unit_tests/deepep/run_tests.sh --skip-distributed
```

### CI/CD Integration

```yaml
- name: Unit tests
  run: python -m unittest tests.unit_tests.deepep.test_primus_turbo_flex_token_dispatcher.TestUnit

- name: Distributed tests
  if: ${{ env.GPU_COUNT >= 2 }}
  run: torchrun --nproc_per_node=2 -m tests.unit_tests.deepep.test_primus_turbo_flex_token_dispatcher
```
