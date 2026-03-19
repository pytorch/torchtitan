# SFT Implementation Plan for TorchTitan

## Context

TorchTitan currently supports pretraining workflows where raw text is tokenized, concatenated, and chunked into fixed-length sequences. We want to add supervised fine-tuning (SFT) support under `torchtitan/experiments/sft/`. The key differences from pretraining are:

1. **Data format**: Chat-structured data (single-turn user/assistant pairs) instead of raw text
2. **Label masking**: Loss is only computed on assistant response tokens (prompt tokens masked with `IGNORE_INDEX=-100`)
3. **Sequence packing**: Pack multiple complete examples into a single sequence for efficiency (no mid-example truncation since this is post-training)
4. **Model loading**: Load pretrained weights (via existing `checkpoint.initial_load_path`) rather than training from scratch
5. **Chat templates**: Apply the model's chat template to format messages before tokenization
6. **Per-document RoPE positions**: When packing, positions reset to 0 at each document boundary so RoPE embeddings match what the model saw during pretraining (fixes [#2559](https://github.com/pytorch/torchtitan/issues/2559))

The existing loss function (`cross_entropy_loss` in `torchtitan/components/loss.py`) already uses `ignore_index=IGNORE_INDEX`, and the training loop's token counting already handles `IGNORE_INDEX` correctly, so the trainer infrastructure works as-is.

## Dataset

- **Training**: GSM8K (`openai/gsm8k`, `main` split) -- math word problems with step-by-step answers
- **Validation**: GSM8K `test` split
- **Test fixture**: Synthetic JSON file in `tests/assets/sft_test/` with 10 chat-formatted samples for unit/integration tests

### GSM8K data characteristics

- ~7,473 training samples, ~1,319 test samples
- All samples contain the `####` separator
- Average tokenized sample length (with Qwen3 chat template): ~220 tokens
- With `seq_len=2048` and greedy packing: ~9-10 samples per packed sequence
- With 8 GPUs and `local_batch_size=1`: dataset re-loops every ~100 steps
- `infinite=True` is needed for most training runs

### GSM8K reasoning traces

GSM8K answers have the format `step-by-step reasoning #### final_answer`. The `process_sample` function in `sft_qwen3_8b_math` splits on `####`:
- Reasoning (before `####`) goes in `reasoning_content` on the assistant message
- Final answer (after `####`) goes in `content`

For Qwen3's chat template, `reasoning_content` is automatically wrapped in `<think>...</think>` tags. The model trains on both the reasoning trace and the final answer (only the user prompt is masked).

## Files

### 1. `torchtitan/experiments/sft/__init__.py`
Exports `SFTDataLoader`.

### 2. `torchtitan/experiments/sft/dataset.py`
Core SFT dataset and dataloader implementation.

**`SFTDataset(IterableDataset, Stateful)`**:
- Constructor takes: `dataset`, `tokenizer`, `sample_processor`, `seq_len`, `dp_rank`, `dp_world_size`, `infinite`, `pack_sequences`
- Splits data by DP rank using `split_dataset_by_node`
- `_validate_messages()`: static method that enforces single-turn `[user, assistant]` format, raises `ValueError` on multi-turn or wrong roles
- `_tokenize_sample()`: tokenizes a sample using incremental prefix re-tokenization to find the prompt/response boundary (same approach as [torchtune's HFTokenizer.tokenize_messages](https://github.com/pytorch/torchtune/blob/main/torchtune/modules/transforms/tokenizers/_hf_tokenizer.py)). Masks prompt tokens with `IGNORE_INDEX`. Drops over-length examples. Returns `(input_ids, label_ids)` or `None`.
- **Greedy packing** (`_iter_greedy_packed`): packs examples sequentially into `seq_len`-length sequences with per-document position tracking. Yields `{"input": tensor, "positions": tensor}, labels`. Positions reset to 0 at each document boundary for correct RoPE embeddings.
- **Unpacked** (`_iter_unpacked`): yields each example independently, padded to `seq_len`. Yields `{"input": tensor}, labels` (no positions needed since each sequence is a single document).
- `_flush_pack_buffer()`: helper that converts pack buffers (input, label, positions) to tensors and clears them
- **Epoch shuffling**: `Dataset.shuffle(seed=42 + epoch)` for deterministic but varied ordering between epochs
- `state_dict()` / `load_state_dict()`: checkpoints `sample_idx`, `epoch`, and all pack buffer state (input, label, positions)
- Logs the first sample's full text for debugging

**`SFTDataLoader(ParallelAwareDataloader)`**:
- Config has `dataset_path`, `load_dataset_kwargs`, `sample_processor`, `infinite`, `pack_sequences`
- `sample_processor` is `Annotated[Callable, tyro.conf.Suppress]` (set in config functions, not CLI)
- Constructs `SFTDataset` and passes to parent

### 3. `torchtitan/experiments/sft/configs.py`
`SFTTrainerConfig(Trainer.Config)` -- overrides the `dataloader` field type. `__post_init__` validation:
- Errors if `pack_sequences=True` with `attn_backend` not in `("flex", "varlen")`
- Errors if `pack_sequences=True` with `attn_mask_type != "block_causal"`
- Errors if `validator.steps == -1` with `pack_sequences=True` (hang risk)

### 4. `torchtitan/experiments/sft/config_registry.py`
Two config functions:
- `sft_debugmodel()`: LLaMA3 debugmodel, test dataset, no packing, 10 steps
- `sft_qwen3_8b_math()`: Qwen3-8B SFT on GSM8K with flex attention + `block_causal` mask, greedy packing, validation, WandB logging. Each config defines its own `process_sample` inline.

### 5. `tests/assets/sft_test/data.json`
10 synthetic samples with `question` and `answer` fields, simple math problems with `####` separator.

### 6. `tests/unit_tests/test_sft_dataset.py`
17 unit tests:
- Label masking: prompt tokens have `IGNORE_INDEX`, response tokens have correct IDs
- Shifted tokens match the tokenized full text
- Sequence packing: multiple short examples correctly packed, padding uses EOS/IGNORE_INDEX
- **Positions reset per document**: positions start at 0 for each packed document, sequential between resets
- Drop on overflow: over-length examples are dropped
- No packing: each example yields independently
- Checkpointing: `state_dict()` / `load_state_dict()` round-trips correctly
- Shuffling: data order changes between epochs, epoch counter persists in state_dict
- Chat template: correct format, prompt-only rendering with `add_generation_prompt`
- Single-turn validation: rejects multi-turn, wrong first role, wrong second role

### 7. `torchtitan/experiments/sft/tests/integration_tests.py`
Integration test: SFT training with debugmodel on 2 GPUs for 10 steps.

### 8. `torchtitan/experiments/__init__.py`
Added `"sft"` to `_supported_experiments`.

## Key Design Decisions

1. **Reuse the existing `Trainer`** -- no custom trainer subclass needed. The standard training loop, loss function, optimizer, checkpoint, and validation infrastructure all work for SFT. We just need a different dataloader.

2. **Incremental prefix re-tokenization for label masking**: For single-turn, tokenize the user message with `add_generation_prompt=True` to get the prompt prefix. The delta gives the exact prompt/response boundary. This is correct by construction -- no BPE merge errors. Same approach as torchtune.

3. **Single-turn only**: Each sample is a single `[user, assistant]` turn. Enforced by `_validate_messages()`. Multi-turn support is a follow-up.

4. **Per-document RoPE positions**: When packing multiple documents into one sequence, positions reset to 0 at each document boundary. The `positions` tensor is yielded alongside `input` and flows through the trainer's `extra_inputs` to `Decoder.forward(positions=...)`, which indexes into the RoPE cache correctly. This fixes [#2559](https://github.com/pytorch/torchtitan/issues/2559).

5. **`block_causal` attention required for packing**: `SFTTrainerConfig.__post_init__` validates both `attn_backend` (flex/varlen) and `attn_mask_type` (block_causal). Without `block_causal`, packed sequences have cross-document attention leakage.

6. **Greedy packing**: Whole-sequence greedy packing -- examples are added sequentially until one doesn't fit, then pad and yield. Simple and fast but may leave padding gaps.

7. **Model loading**: Leverage existing `checkpoint.initial_load_path` + `checkpoint.initial_load_in_hf=True`. No new infrastructure needed.

8. **Self-contained config functions**: Each config function defines its own `process_sample` inline. No shared module-level processors or string-key registries.

9. **Validation hang prevention**: `__post_init__` errors if `validator.steps == -1` with packed sequences. Different DP ranks can produce different numbers of packed sequences, causing all-reduce deadlocks.

## Lessons Learned

### Validation with `steps=-1` hangs
Different ranks get different numbers of packed sequences due to uneven sample lengths across shards. When one rank finishes before another, the all-reduce for loss hangs. Always use a fixed number of validation steps. Enforced by `SFTTrainerConfig.__post_init__`.

### Qwen3 `enable_thinking` and `reasoning_content`
- `enable_thinking` is a kwarg to `apply_chat_template`, only affects the `add_generation_prompt=True` path
- `reasoning_content` is a field on assistant message dicts -- the template wraps it in `<think>...</think>` tags
- Using `reasoning_content` with actual reasoning data is the correct approach for SFT

### BPE merge boundaries
`tokenize(A+B) != tokenize(A) + tokenize(B)` at merge boundaries. We use the torchtune-style incremental prefix approach which is correct by construction. See `plans/bpe-boundary-research.md`.

### Qwen3 flex attention + TP is incompatible
`parallelize_qwen3` uses `ColwiseParallel(use_local_output=False)` on wq/wk/wv but `flex_attention` has no DTensor dispatch rule. For SFT with packing on Qwen3, use DP-only (no TP).

### `slots=True` dataclasses and `super()`
Python's zero-argument `super()` doesn't work in `__post_init__` of `slots=True` dataclasses. Use `ParentClass.__post_init__(self)` instead.

### The trainer forwards all non-"input" keys as model kwargs
`post_dataloading_process` extracts `input_dict["input"]` and passes all other keys (e.g., `positions`) as `**extra_inputs` to `model.forward()`. Only include keys the model accepts.

### RoPE positions must reset per document when packing
Without explicit positions, the model uses sequential `[0, 1, ..., seq_len-1]` across the entire packed sequence. The second document in a pack gets positions continuing from where the first ended, causing wrong RoPE embeddings. The fix is to yield a `positions` tensor with per-document resets. See [#2559](https://github.com/pytorch/torchtitan/issues/2559).

### `attn_mask_type` defaults to `"causal"`, not `"block_causal"`
Qwen3's model registry doesn't set `attn_mask_type`. The default is `"causal"`, which means flex attention uses a plain causal mask with no document boundary awareness. Must explicitly set `attn_mask_type="block_causal"` when packing.

## Verification

1. **Unit tests**: `conda run -n joe-titan-v2 python -m pytest tests/unit_tests/test_sft_dataset.py -v` -- 17 tests
2. **Integration test**: `python torchtitan/experiments/sft/tests/integration_tests.py /tmp/sft_test_output --ngpu 2`
3. **Manual training**: `./run_train.sh --module sft --config sft_qwen3_8b_math --checkpoint.initial_load_path ./assets/hf/Qwen3-8B`

## Next Steps

1. **Multi-turn conversations**: Support multiple user/assistant turns per sample, masking all user turns. The prefix re-tokenization approach generalizes naturally.

2. **Alternative packing strategies**: Sorted/bin packing for better efficiency. Multi-bin packing with multiple buffers.

3. **Packing efficiency logging**: Track content_tokens / total_tokens ratio per step.

4. **Additional dataset formats**: ShareGPT, Alpaca, etc. Each config provides its own `process_sample`.

5. **Multi-dataset mixing**: Accept `(dataset, processor, weight)` tuples for proportional sampling.

6. **Qwen3 flex attention + TP**: Update `parallelize_qwen3` to convert Q/K/V to local tensors after norms when `attn_backend="flex"`.

7. **EOS tokens inside response text**: Current EOS-based document boundary detection could be confused. A future fix would use explicit document IDs.
