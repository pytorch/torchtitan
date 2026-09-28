# Relevant Files

## Core training path

### `torchtitan/trainer.py`

Relevant class and methods:

- `Trainer`
- `Trainer.train_step()`
- `Trainer.microbatch_generator()`

`train_step()` gathers microbatches, accumulates ordinary valid-token counts, performs the data-parallel reduction, calls `TrainingEngine.prepare_step()`, and passes the denominator into forward/backward execution.

### `torchtitan/training_engine.py`

Relevant class and methods:

- `TrainingEngine`
- `TrainingEngine.prepare_step()`
- `TrainingEngine.forward_backward_microbatch()`
- `TrainingEngine._non_pp_forward_backward_body()`
- `TrainingEngine._pp_forward_backward_body()`

This file owns the step-level denominator setup and dispatches model outputs and loss arguments into ordinary or pipeline execution.

## MTP implementation

### `torchtitan/models/deepseek_v3/mtp.py`

Relevant functions and classes:

- `roll_mtp_sequence()`
- `MTPTransformerBlock`
- `MTPDecoder.preprocess_inputs()`
- `MTPDecoder.forward()`
- `MTPLoss`

This is the primary implementation. It constructs shifted MTP inputs and labels, tracks validity through masks and `IGNORE_INDEX`, produces one output per depth, and combines the main and MTP cross-entropy sums.

### `torchtitan/models/deepseek_v4/mtp.py`

DeepSeek V4 imports and reuses `MTPLoss` and `roll_mtp_sequence()` from the DeepSeek V3 MTP implementation. It matters for compatibility and configuration coverage if the denominator behavior is changed.

### `torchtitan/models/deepseek_v3/MTP.md`

Documents the intended DeepSeek MTP alignment, shifted labels, packed-document handling, and current objective description. It is useful for comparing the documented objective with the actual denominator behavior.

## Loss implementation

### `torchtitan/components/loss.py`

Relevant functions and classes:

- `IGNORE_INDEX`
- `cross_entropy_loss()`
- `BaseLoss`
- `CrossEntropyLoss`
- `ChunkedLossWrapper`

Cross entropy uses sum reduction and ignores `IGNORE_INDEX`. `CrossEntropyLoss` normally applies `global_valid_tokens`; `MTPLoss` instead combines raw sums and applies the shared denominator after adding the MTP terms. `ChunkedLossWrapper` matters because MTP configurations may use chunked loss execution.

## Data and valid-token construction

### `torchtitan/components/data/collators.py`

The token collator creates `TokenizedTrainingMicrobatch` and computes `num_valid_tokens` from ordinary labels not equal to `IGNORE_INDEX`. This is the source of the current main-loss denominator.

### `torchtitan/components/data/types.py`

Relevant class:

- `TokenizedTrainingMicrobatch`

Defines the model-facing tensors, including labels, positions, padding masks, and `num_valid_tokens`.

## Shared auxiliary-loss infrastructure

### `torchtitan/models/common/aux_loss.py`

Relevant class and functions:

- `AuxLoss`
- `AuxLoss.set_step_denominator()`
- `AuxLoss.inject()`
- `collect_aux_loss_metrics()`

This is primarily used for MoE auxiliary losses, not the DeepSeek MTP `MTPLoss`. It is relevant because it documents the existing step-level denominator convention and because the MTP TODO is located near the same denominator setup. It should not be generalized unnecessarily for this project.

## Existing tests

### `tests/unit_tests/cpu/test_loss.py`

Relevant tests include:

- `test_roll_mtp_sequence_respects_packed_document_boundaries()`
- `test_mtp_preprocess_aligns_tokens_and_labels()`
- MTP loss tests around `MTPLoss`
- `test_chunked_mtp_matches_full_objective()`

These tests already validate shifted labels, packed boundaries, validity masks, and chunked/full objective agreement. They are the natural location for denominator-specific hand-computed tests.

### `tests/unit_tests/cpu/test_aux_loss.py`

Tests the shared `AuxLoss` denominator and distributed metric behavior. It is useful for understanding existing denominator conventions, but MTP normalization tests should primarily live with MTP/loss tests unless the implementation intentionally changes shared auxiliary-loss code.

### `tests/unit_tests/cpu/test_deepseek_v3_moe.py`

Contains MTP-related sharding tests, including validity-mask placement under sequence parallelism. These tests matter if per-depth counts are transported alongside existing MTP metadata or must respect the same sharding boundaries.

### `tests/unit_tests/cpu/test_deepseek_v4_mtp.py`

Validates DeepSeek V4 MTP configuration and sharding behavior. It should be reviewed for regression coverage if V4 uses the changed loss path.

### `tests/unit_tests/cpu/test_trainer.py`

Tests trainer-to-engine loss argument propagation and global valid-token handling. It is relevant for verifying that any new MTP-specific denominator metadata survives the existing forward/backward dispatch.

## Distributed and numerical validation references

### `torchtitan/distributed/parallelism_context.py`

Defines the mesh context used to determine enabled data, context, tensor, expert, and pipeline parallel dimensions.

### `torchtitan/distributed/utils.py`

Contains distributed reduction helpers such as the data-parallel sum used for global valid-token calculation.

### `tests/integration_tests/`

Contains model and feature integration definitions, including DeepSeek configurations and real/Fake Process Group execution.

### `scripts/loss_compare.py`

Provides the repository's existing numerical comparison workflow for training loss and gradient norms. It is relevant to the later experiment, but must not be run during this documentation phase.
