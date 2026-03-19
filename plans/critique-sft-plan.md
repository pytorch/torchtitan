# Critique: SFT Implementation (Updated)

## What's solid

- **Minimal diff to core torchtitan**: Only touches `experiments/__init__.py` in existing code. Everything else is new files under `experiments/sft/`.
- **RoPE positions fix (#2559)**: Packed sequences now yield per-document positions that reset at boundaries. This flows through `extra_inputs` -> `Decoder.forward(positions=...)` -> `apply_rotary_emb`. Correct by construction.
- **Config-level validation**: `SFTTrainerConfig.__post_init__` catches three common misconfigurations before training starts: wrong attention backend, missing `block_causal`, and `validator.steps=-1` with packing.
- **17 unit tests** covering masking, packing, positions, checkpointing, shuffling, and validation rejection.

## Remaining issues

### 1. `_iter_unpacked` doesn't yield positions (Low-Medium)

`_iter_unpacked` yields `{"input": tensor}` with no `positions` key. This is fine when each sequence is a single document (positions naturally start at 0). But the inconsistency means the model sees `positions=None` for unpacked sequences and explicit positions for packed ones. If someone adds padding-aware position tracking later, or if the model's behavior differs between `positions=None` (inferred) and `positions=tensor` (explicit), this could surprise.

Not a bug today, but worth documenting.

### 2. Padding positions are sequential from 0, not matching their document (Low)

In `_iter_greedy_packed`, padding positions are `list(range(pad_len))` -- they start at 0 and count up. Since padding tokens are masked by both attention (`block_causal` isolates them) and loss (`IGNORE_INDEX`), the position values don't affect training. But they're semantically wrong (they don't correspond to any document). This is harmless but could confuse someone reading the code.

### 3. No integration test for packing + flex attention + positions (Medium)

The integration test uses `sft_debugmodel` which has `pack_sequences=False`. There's no integration test that exercises the packed path with flex attention and position resets. The unit tests verify positions reset correctly at the dataset level, but don't verify the full forward pass with RoPE consuming the positions tensor.

### 4. Checkpointing with `num_workers > 0` silently breaks (Pre-existing, Medium)

`SFTDataset` tracks `_sample_idx` in the main process, but with `num_workers > 0`, iteration happens in worker subprocesses. The state captured by `state_dict()` wouldn't reflect worker-side progress. The default is `num_workers=0` so this doesn't trigger, but there's no guard preventing someone from setting it.

### 5. No `process_sample` error handling for malformed GSM8K (Low)

`answer.rsplit("####", 1)` in `sft_qwen3_8b_math`'s `process_sample` crashes on samples without `####`. All GSM8K samples have it, so this is theoretical. A guard would be more defensive.

### 6. `sft_debugmodel` uses `pack_sequences=False` but no `attn_mask_type` override (Low)

`sft_debugmodel` uses `llama3_registry("debugmodel")` which has `attn_mask_type="causal"` by default. This is fine since packing is off, but if someone copies the config and enables packing, the `__post_init__` validation will catch it. No action needed.

## Summary

The implementation is clean and addresses #2559 correctly. The main gap is test coverage for the full packed + flex + positions path in an integration test. Everything else is minor or pre-existing.
