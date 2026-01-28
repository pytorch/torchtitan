# CLAUDE.md

This file provides guidance for AI assistants working with the torchtitan codebase.

## Project Overview

TorchTitan is a PyTorch-native platform for large-scale generative AI model training. It provides composable, multi-dimensional parallelism (FSDP, Tensor Parallel, Pipeline Parallel, Context Parallel, Expert Parallel) with minimal model code changes. Current version: 0.2.1.

## Repository Structure

```
torchtitan/                    # Main package
├── train.py                   # Core training loop / Trainer class
├── config/                    # JobConfig dataclass-based configuration
├── components/                # Training components (checkpoint, dataloader, loss, lr_scheduler, metrics, tokenizer, quantization)
├── distributed/               # Parallelism infrastructure (FSDP, TP, PP, CP, activation checkpointing)
├── models/                    # Core model implementations (llama3, llama4, deepseek_v3, qwen3, gpt_oss, flux)
├── experiments/               # Experimental features (simple_fsdp, vlm, autoparallel, compiler_toolkit, rl, etc.)
├── protocols/                 # Abstract interfaces (TrainSpec, Model, ModelArgs, StateDictAdapter)
├── hf_datasets/               # HuggingFace dataset utilities
└── tools/                     # Logging, profiling, device utilities
tests/
├── unit_tests/                # CPU-based unit tests
├── integration_tests/         # GPU-based integration tests (8 GPUs)
└── assets/                    # Test fixtures
scripts/                       # Checkpoint conversion, estimation, generation, downloads
docs/                          # Technical documentation
```

### Model code layout convention

Each model lives in `torchtitan/models/<name>/` with:
- `model/model.py` — single-device model implementation
- `model/args.py` — model args dataclass with flavor dictionary
- `infra/parallelize.py` — parallelization function
- `infra/pipeline.py` — pipeline parallel splitting (optional)
- `train_configs/` — TOML config files per flavor

## Build & Install

```bash
pip install -r requirements.txt   # from source
pip install torchtitan             # from PyPI
```

Requires Python ≥ 3.10. Key deps: torch, torchdata, datasets, tokenizers, tyro, tensorboard.

## Common Commands

### Training
```bash
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh
```

### Tests
```bash
# Unit tests (CPU)
pytest -s tests/unit_tests/

# Integration tests (8 GPUs)
python -m tests.integration_tests.run_tests <output_dir> [--test_suite features]
```

### Linting & Formatting
```bash
pre-commit run --all-files
```

This runs: flake8 (with bugbear, pep8-naming, torchfix), ufmt (black + usort), pydoclint, codespell, pyrefly, and license header checks.

## Code Style & Conventions

- **Max line length**: 120 characters (enforced by flake8 B950, not E501)
- **Formatter**: black (via ufmt)
- **Import sorting**: usort (via ufmt)
- **Type checking**: pyrefly
- **Docstrings**: validated by pydoclint
- Lambda assignments (E731) and uppercase variable names for Triton kernels (N803/N806) are allowed
- `import torch.nn.functional as F` and DDP-style acronym imports (N812/N817) are allowed

## Architecture Key Concepts

**TrainSpec** — The central registration protocol. Each model registers a TrainSpec containing its model class, args, parallelization function, and builder functions for optimizers, LR schedulers, dataloaders, tokenizers, and loss.

**JobConfig** — Dataclass-based TOML configuration with sections: Job, Model, Optimizer, LRScheduler, Training, Parallelism, Checkpoint, Logging, Profiling, Metrics, Compile, Experimental. Extensible via `--job.custom_config_module`.

**ParallelDims** — Manages device mesh with dimensions: dp_replicate, dp_shard, cp, tp, pp, ep, etp.

**Trainer** — Core class in `train.py` that orchestrates the training loop, extending `torch.distributed.checkpoint.stateful.Stateful`.

## Testing Expectations

- New features need CPU unit tests and GPU integration tests
- Integration tests verify loss convergence against baseline files
- Loss baselines are architecture-specific (CUDA/ROCm)
- `pytest -s tests/unit_tests/` must pass before submitting

## Adding New Models

Follow the pattern in `torchtitan/models/README.md`:
1. Create `torchtitan/models/<name>/` with model/args/parallelize/pipeline modules
2. Register a TrainSpec in `__init__.py`
3. Add TOML train configs
4. Implement StateDictAdapter for checkpoint interop with HuggingFace

## Adding Experiments

Place in `torchtitan/experiments/<name>/`. Experiments can have their own dependencies but must reuse core components via TrainSpec. Include a README with clear ownership.

## CI

GitHub Actions workflows run on push/PR:
- `unit_test_cpu.yaml` — CPU unit tests
- `lint.yaml` — pre-commit on changed files
- `integration_test_8gpu_*.yaml` — various GPU test suites
- All use PyTorch nightly builds
