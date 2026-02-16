# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

torchtitan is a PyTorch-native platform for pretraining generative AI models (LLMs, MoE, vision). It implements multi-dimensional parallelism (FSDP2, TP, PP, CP, EP) with minimal model code changes. The codebase prioritizes clean, minimal, readable code over heavy abstraction.

## Common Commands

### Training
```bash
# 8-GPU local training (default: Llama 3 debug model)
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh

# Custom GPU count and config
NGPU=4 CONFIG_FILE="path/to/config.toml" ./run_train.sh

# Dry-run for config validation (no GPU execution)
NGPU=32 COMM_MODE="fake_backend" ./run_train.sh

# Single-GPU debugging with simulated multi-GPU
NGPU=32 COMM_MODE="local_tensor" ./run_train.sh
```

### Testing
```bash
# Unit tests
pytest -s tests/unit_tests/

# Single unit test
pytest -s tests/unit_tests/test_job_config.py::TestJobConfig::test_command_line_args

# Integration tests (requires GPUs)
python -m tests.integration_tests.run_tests <output_dir> --test_suite features --ngpu 8

# Specific integration test
python -m tests.integration_tests.run_tests <output_dir> --test_suite features --test_name gradient_accumulation --ngpu 2
```

### Linting
```bash
pre-commit run --all-files
```

### Install (from source)
```bash
pip install -r requirements.txt
# Dev dependencies:
pip install -e ".[dev]"
```

## Architecture

### Core Training Loop
- `torchtitan/train.py` — `Trainer` class: main training loop, initialization, forward/backward step. Entry point: `main(Trainer)`.

### Key Abstractions

**TrainSpec** (`torchtitan/protocols/train_spec.py`): Central registration mechanism that bundles model class, model args, parallelization functions, and builder functions for all training components (optimizer, LR scheduler, dataloader, tokenizer, loss). Models register a `TrainSpec` via `get_train_spec()` in their `__init__.py`. External users can call `register_train_spec()`.

**ModelProtocol** (`torchtitan/protocols/model.py`): Interface models must implement (`__init__`, `init_weights`).

**ModelConverter** (`torchtitan/protocols/model_converter.py`): Plugin interface for model transformations (e.g., Float8 quantization). Has `convert()` (pre-parallelization) and `post_optimizer_hook()`.

### Package Structure
- `torchtitan/models/` — Supported model implementations (llama3, llama4, deepseek_v3, qwen3, flux, gpt_oss). Each model folder contains:
  - `model/` — Model definition and args
  - `infra/` — Parallelization (`parallelize.py`) and pipeline (`pipeline.py`) logic
  - `train_configs/` — TOML configuration files
  - `__init__.py` — `get_train_spec()` function and model arg variants
- `torchtitan/components/` — Reusable training components: checkpoint, dataloader, loss, lr_scheduler, metrics, optimizer, tokenizer, validate, quantization (float8, mx)
- `torchtitan/distributed/` — Parallelism implementations: parallel_dims, tensor_parallel, pipeline_parallel, context_parallel, expert_parallel, activation_checkpoint
- `torchtitan/config/` — `JobConfig` (dataclass-based) and `ConfigManager` (TOML + CLI arg parsing via tyro)
- `torchtitan/protocols/` — Interfaces/protocols (TrainSpec, ModelProtocol, ModelConverter, BaseStateDictAdapter)
- `torchtitan/tools/` — Logging, profiling, device utilities
- `torchtitan/experiments/` — Experimental features (lower contribution bar, separate guidelines in `experiments/README.md`)

### Configuration
All training is configured via TOML files (see `train_configs/` in each model folder). CLI args override TOML values. `JobConfig` can be extended with `--job.custom_config_module` for model-specific configs.

### Parallelism Application Order
When parallelizing a model in `parallelize.py`, techniques are applied in this order:
1. TP (and EP for MoE)
2. Activation checkpointing
3. `torch.compile`
4. FSDP / HSDP

### Adding a New Model
Follow the structure in `torchtitan/models/llama3/` — see `torchtitan/models/README.md` for full instructions. New models should start in `torchtitan/experiments/` first.
