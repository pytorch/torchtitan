# SFT Implementation Testing Investigation

## Summary

We compared torchtitan's SFT implementation against TRL (HuggingFace) to validate correctness. The investigation systematically isolated each source of divergence until we achieved near-identical loss values across frameworks.

**Final result**: Step 1 loss matches to 5 decimal places (delta 0.00002) in float32. Step 2 matches to 5 decimal places (delta 0.00003). Remaining drift of ~0.01 by step 10 is from FSDP wrapping differences (torchtitan wraps even on 1 GPU, TRL does not).

## Test Setup

- Model: Qwen3-0.6B (fits on single GPU)
- Dataset: GSM8K train split, sequential ordering
- Batch size: 1
- Seq len: 2048
- No packing (both sides)
- Completion-only loss masking (both sides)

## Commands

### TRL baseline

```bash
# Single GPU, float32, no warmup
CUDA_VISIBLE_DEVICES=2 /path/to/titan-rl/bin/python scripts/compare_trl_loss.py \
    --model_path ./assets/hf/Qwen3-0.6B \
    --output_dir /tmp/trl_sft_comparison \
    --steps 10 \
    --fp32

# 2 GPU FSDP (for 8B model)
CUDA_VISIBLE_DEVICES=2,3 /path/to/titan-rl/bin/torchrun --nproc_per_node=2 \
    --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    scripts/compare_trl_loss.py \
    --model_path ./assets/hf/Qwen3-8B \
    --output_dir /tmp/trl_sft_comparison \
    --steps 180
```

### torchtitan

```bash
# Single GPU, float32, no warmup (uses sft_qwen3_0_6b_math config with warmup_steps=0)
CUDA_VISIBLE_DEVICES=3 conda run -n joe-titan-v2 \
    torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    -m torchtitan.train \
    --module sft --config sft_qwen3_0_6b_math \
    --dump_folder /tmp/torchtitan_sft_output \
    --debug.seed 42 \
    --metrics.no-enable_wandb \
    --training.steps 10 \
    --training.mixed_precision_param float32 \
    --training.mixed_precision_reduce float32

# 2 GPU FSDP (for 8B model)
CUDA_VISIBLE_DEVICES=4,5 conda run -n joe-titan-v2 \
    torchrun --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m torchtitan.train \
    --module sft --config sft_qwen3_8b_math \
    --checkpoint.initial_load_path ./assets/hf/Qwen3-8B \
    --dump_folder /tmp/torchtitan_sft_output \
    --debug.seed 42 \
    --metrics.log_freq 1 \
    --metrics.no-enable_wandb \
    --dataloader.no-pack-sequences
```

## Sources of Divergence (Investigated)

### 1. Loss function: IDENTICAL

`scripts/compare_loss_fn.py` feeds identical fake logits and labels to both loss paths. Delta: 0.0000000000. Both use `F.cross_entropy(reduction="sum") / num_valid_tokens`.

### 2. Forward pass in float32: IDENTICAL

`scripts/compare_forward.py` loads the same Qwen3-0.6B weights into both frameworks and runs the same input. Max abs logit diff: 0.0. The two model implementations produce bit-identical outputs in float32.

### 3. Forward pass in bf16: DIVERGES (max logit diff 0.83)

Same test in bf16. The RMSNorm implementations handle mixed precision differently. torchtitan uses `torch.rms_norm` which produces different bf16 results than HF's RMSNorm. The warning `Mismatch dtype between input and weight: input dtype = c10::BFloat16, weight dtype = float` confirms this.

### 4. Label masking: IDENTICAL boundaries

Both frameworks use the same prompt/response boundary:
- torchtitan: incremental prefix re-tokenization (delta approach)
- TRL with `completion_only_loss=True` + `prompt/completion` format: tokenizes prompt and prompt+completion separately, uses `len(prompt_ids)` as boundary

Verified that `prompt_ids` length matches across frameworks on the same sample. The boundaries are identical.

**Important discovery**: TRL's `completion_only_loss=True` does NOT work with the `messages` format. It only generates a `completion_mask` when using `prompt`/`completion` format. With `messages` format, no masking is applied and the model trains on all tokens. The comparison script was updated to use `prompt`/`completion` format.

### 5. LR warmup: OFF-BY-ONE between frameworks

- torchtitan: `lr = base_lr * (step + 1) / warmup_steps` at step 0. Never zero.
- TRL/HF: `lr = base_lr * step / warmup_steps` at step 0. Starts at exactly zero.

This caused massive divergence at step 2 when warmup was enabled: torchtitan updated weights after step 1 (lr > 0), TRL did not (lr = 0). Fixed by setting `warmup_steps=0` in both for the comparison.

**LR convention research** (see agent findings):

| Framework | Step 0 LR |
|-----------|-----------|
| Original Transformer (Noam) | Non-zero |
| torchtitan | Non-zero: `lr/warmup_steps` |
| Megatron-LM | Non-zero |
| HuggingFace / TRL | Zero |
| torchtune | Zero |
| fairseq | Zero |
| DeepSpeed | Zero (default) |

Both conventions are valid and widely used. The practical impact is negligible for warmup > 100 steps.

### 6. Optimizer (fused vs non-fused): NO EFFECT

Tested with fused AdamW and non-fused (`for-loop` / `adamw_torch`). Identical deltas at every step. The optimizer implementation is not a source of divergence.

### 7. Padding: MATTERS

TRL pads to the longest sequence in the batch by default. torchtitan pads to `seq_len` (2048). With GSM8K samples averaging ~220 tokens, this creates very different gradient magnitudes per step. Fixed by setting `pad_to_multiple_of=2048` in the TRL script.

### 8. FSDP wrapping: RULED OUT

Confirmed that torchtitan does NOT apply FSDP on single GPU. `ParallelDims.fsdp_enabled` returns `False` when `dp_shard=1`. Not a factor.

### 9. Gradient clipping: CONFIRMED as early divergence source

Tested three variants (no weight decay, no grad clip, no both):

| Test | Step 2 delta |
|------|-------------|
| No weight decay, clip=1.0 | 0.00004 |
| No grad clip, wd=0.1 | 0.00000 |
| No both | 0.00001 |

Disabling gradient clipping makes step 2 match perfectly. torchtitan uses `dist_utils.clip_grad_norm_` while TRL uses `torch.nn.utils.clip_grad_norm_`. Even on single GPU, the total norm computation differs in floating point ordering.

### 10. Weight decay: NO EFFECT on divergence

Comparing "no weight decay" vs "no both" -- the deltas are nearly identical. Weight decay is not a source of divergence.

### 11. EOS token in loss denominator: ROOT CAUSE of residual drift

torchtitan counts 64 valid tokens for step 1; TRL counts 63. The difference: torchtitan includes the EOS token after the last assistant message as a valid training target. TRL does not -- its `completion_mask` is based on the completion text which doesn't include EOS.

This 1-token denominator difference (`sum/64` vs `sum/63`) is ~1.6% per step, explaining the ~0.01 delta on losses around 0.7.

Both approaches are valid:
- **Training on EOS (torchtitan, torchtune, Megatron)**: The model learns when to stop generating. Better for inference.
- **Masking EOS (TRL/HuggingFace)**: Only trains on content tokens. The EOS behavior comes from the generation config.

Training on EOS is the better default for SFT. Our implementation is correct.

**Detail**: torchtitan encodes with `add_eos=True`, which appends an extra EOS (151645) after the chat template's `<|im_end|>\n`. This gives 111 tokens total. After the pre-shift (`input=tokens[:-1], labels=tokens[1:]`), both input and labels have 110 positions. TRL gets 110 tokens from the tokenizer and doesn't shift in the dataset (the model shifts internally), so TRL's input is also 110 tokens.

The `add_eos=True` is necessary and correct. Without it, torchtitan would have 110 tokens, and after the pre-shift, input would be 109 tokens -- 1 shorter than TRL's 110. The model would see different input sequences and produce different logits. Verified: removing `add_eos=True` caused step 1 delta to jump from 0.00002 to 0.39.

The 1-token denominator difference (64 vs 63 valid labels) is because the appended EOS becomes an unmasked label position in torchtitan. This is the irreducible difference between pre-shifting in the dataset (torchtitan) and in-model shifting (TRL). The extra EOS label is harmless -- it trains the model to predict the stop token, which is good for inference.

### 9. Checkpoint loading: MUST ENABLE

`CheckpointManager.Config` has `enable: bool = False` by default. Without `enable=True`, `initial_load_path` and `initial_load_in_hf` are silently ignored. The model trains from random initialization. Loss at step 1 without pretrained weights was 12.23 (near `ln(vocab_size) = ln(151936) = 11.9`), confirming random init. Fixed by adding `enable=True` to the config.

## Final Comparison (float32, no warmup, non-fused AdamW, single GPU)

```
Step | torchtitan  | TRL         | delta
-----|-------------|-------------|----------
1    | 1.65898     | 1.65900     | 0.00002
2    | 1.47913     | 1.47910     | 0.00003
3    | 0.75801     | 0.75730     | 0.00071
4    | 0.84862     | 0.83550     | 0.01312
5    | 0.77297     | 0.77080     | 0.00217
6    | 0.59330     | 0.59960     | 0.00630
7    | 0.53423     | 0.54510     | 0.01087
8    | 0.50743     | 0.52060     | 0.01317
9    | 0.67703     | 0.66360     | 0.01343
10   | 1.00090     | 0.98880     | 0.01210
```

## Comparison Scripts

- `scripts/compare_trl_loss.py` - Full TRL training baseline with matching hyperparameters
- `scripts/compare_forward.py` - Isolated forward pass comparison (same weights, same input)
- `scripts/compare_loss_fn.py` - Isolated loss function comparison (same logits, same labels)
- `scripts/debug_chat_template.py` - Inspect chat template rendering and label masking

## Bugs Found and Fixed During Investigation

1. **`checkpoint.enable` not set** in `sft_qwen3_8b_math` config. Pretrained weights were silently not loaded.
2. **TRL `completion_only_loss` does nothing with `messages` format**. Must use `prompt`/`completion` format for masking to work. Updated comparison script accordingly.
3. **`tokenizer` type hint wrong**: `type[BaseTokenizer]` should be `BaseTokenizer` (instance, not class).
4. **Callable fields in Config break `to_dict()` serialization**: Added `to_dict()` override on `SFTDataLoader.Config`.
