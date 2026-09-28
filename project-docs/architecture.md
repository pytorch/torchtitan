# Current and Proposed MTP Loss Architecture

## Original execution path

### 1. `Trainer.train_step()`

`Trainer.train_step()` in `torchtitan/trainer.py` gathers the microbatches for one optimizer step, including gradient-accumulation groups and pipeline microbatches.

For each microbatch it adds `microbatch.num_valid_tokens`. The collator computes that field from the ordinary labels:

```python
int((labels != IGNORE_INDEX).sum())
```

The count is accumulated across all microbatches contributing to the optimizer step.

### 2. Global valid-token calculation

The trainer creates a device tensor from the local count. If data parallelism is enabled, it reduces that tensor over the data-parallel mesh with `dist_sum_tensor`. The result is the global ordinary valid-token count for the optimizer step.

This count includes gradient accumulation, but it does not include separate MTP-specific target counts.

### 3. `TrainingEngine.prepare_step()`

`TrainingEngine.prepare_step()` in `torchtitan/training_engine.py`:

- records the accumulation plan;
- clears optimizer gradients;
- stores the global ordinary valid-token tensor through `AuxLoss.set_step_denominator()`;
- returns the same tensor to the trainer.

The MTP-related TODO is next to this denominator assignment. The current API establishes one shared step denominator.

### 4. Forward/backward dispatch

`Trainer.train_step()` passes `global_valid_tokens` into `TrainingEngine.forward_backward_microbatch()`.

For non-pipeline execution, `_non_pp_forward_backward_body()` in `training_engine.py`:

1. calls the model;
2. calls `self.loss_fn(pred, labels, **loss_kwargs)`;
3. calls `loss.backward()`.

For pipeline execution, the same loss arguments are routed through the pipeline schedule and consumed by the loss stage. The denominator remains a single shared tensor.

### 5. DeepSeek MTP preprocessing

`MTPDecoder.preprocess_inputs()` in `torchtitan/models/deepseek_v3/mtp.py` creates one shifted input/label pair for every MTP depth.

For depth `k`, it calls `roll_mtp_sequence()` with `shift=k`:

- shifted input tokens use fill value `0`;
- shifted labels use fill value `IGNORE_INDEX`;
- a validity mask is created for the shifted input path;
- positions and padding are used to prevent invalid cross-document or padded targets.

The method returns tuples containing:

```text
(main_inputs, mtp_inputs_depth_1, mtp_inputs_depth_2, ...)
(main_labels, mtp_labels_depth_1, mtp_labels_depth_2, ...)
```

DeepSeek V4 reuses the MTP loss and sequence-shifting utilities from the DeepSeek V3 implementation.

### 6. `roll_mtp_sequence()`

`roll_mtp_sequence()` shifts a sequence along its token dimension. For each destination position `i`, depth `k` reads source position `i + k`.

The shifted position is invalid when:

- `i + k` is outside the row;
- the shifted source is padding;
- position IDs show that the shift crosses a packed-document boundary.

Invalid shifted labels are set to `IGNORE_INDEX`. Therefore the valid count for depth `k` is obtainable from:

```python
(mtp_labels_k != IGNORE_INDEX).sum()
```

The current implementation constructs these labels but does not carry their counts into the denominator path.

### 7. MTP model forward

`MTPDecoder.forward()` computes the ordinary decoder hidden state, then runs the MTP layers sequentially. Each MTP layer receives the shifted token embedding, the preceding depth's hidden state, and the shifted-input validity mask.

The output is a tuple containing the main output and one output per MTP depth. With the language-model head enabled, each output is converted to logits.

### 8. `MTPLoss`

`MTPLoss` in `torchtitan/models/deepseek_v3/mtp.py` receives aligned prediction and label tuples.

It calls the ordinary `CrossEntropyLoss` implementation separately for the main prediction and each MTP prediction. Each call uses sum reduction and ignores `IGNORE_INDEX` labels.

The MTP sums are combined with:

```python
mtp_weight = self.mtp_scale / num_mtp_layers
```

The current code then divides the complete combined loss by the single `global_valid_tokens` argument.

### 9. Cross-entropy reduction

`cross_entropy_loss()` in `torchtitan/components/loss.py` uses `reduction="sum"` by default. `CrossEntropyLoss.__call__()` then divides the result by `global_valid_tokens` when that argument is present.

`MTPLoss` bypasses per-call normalization by invoking its parent loss without a denominator, combines the raw sums, and applies the shared denominator once at the end.

## Original denominator flow

```text
ordinary labels
  -> microbatch.num_valid_tokens
  -> sum over accumulation microbatches
  -> data-parallel reduction
  -> Trainer.train_step(): global_valid_tokens
  -> TrainingEngine.prepare_step()
  -> forward_backward_microbatch()
  -> MTPLoss(global_valid_tokens=...)
  -> main sum + weighted MTP sums
  -> divide entire objective by global_valid_tokens
  -> backward()
```

The original path had no separate `global_mtp_valid_tokens` tensor. The current implementation adds one through a model-level MTP count hook, optimizer-step accumulation, and DP reduction.

## Implemented denominator flow

```text
raw microbatch labels, positions, padding
  -> MTP model count hook
  -> local per-depth counts
  -> accumulation across optimizer-step microbatches
  -> DP reduction
  -> global_mtp_valid_tokens
  -> MTPLoss loss kwargs
  -> per-depth normalization
```

## Proposed minimal architecture

### 1. Per-depth valid-token counts

For every shifted MTP label tensor, derive one valid-target count:

```text
N_k_local = (mtp_labels_k != IGNORE_INDEX).sum()
```

The counts must include all microbatches contributing to the optimizer step.

### 2. Global reduction

Reduce each depth's accumulated count over the distributed dimensions required by the loss semantics. At minimum, data-parallel ranks must contribute to a global count. Context-parallel and pipeline configurations must preserve the same global-token meaning as the existing loss path.

The result is:

```text
N_k_global = sum of local depth-k valid-target counts
```

The denominator must not be rank-local.

### 3. Passing denominators to `MTPLoss`

Pass the tuple of global per-depth denominators through the existing loss-input path, for example as a dedicated MTP-specific loss keyword. The ordinary `global_valid_tokens` argument remains available for the main loss.

The change should be limited to the MTP path and should not require a generalized loss-weighting abstraction.

### 4. Per-depth normalization

`MTPLoss` should compute:

```text
main_loss = L_main_sum / N_main_global

mtp_loss = mtp_scale / D * sum_k(L_mtp_k_sum / N_k_global)

loss = main_loss + mtp_loss
```

The configured `mtp_scale / num_mtp_layers` weighting remains unchanged.

Zero-valid-token handling must be defined before implementation, especially for very short or heavily packed examples.

### 5. Preserved behavior

The proposed architecture must preserve:

- current main-loss normalization;
- current behavior for models without MTP;
- current gradient-accumulation semantics;
- current MTP weighting;
- existing distributed loss semantics;
- existing pipeline and chunked-loss compatibility.

This is a denominator-flow change, not a redesign of MTP blocks, model architectures, or auxiliary-loss infrastructure.
