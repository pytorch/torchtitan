"""Compare SFT loss curves between torchtitan and TRL.

Runs TRL's SFTTrainer on the same data (GSM8K) and model (Qwen3-8B) with
matching hyperparameters, logging per-step loss to a JSON file. Run this
script, then compare against torchtitan's WandB/log output.

Uses completion_only_loss=True to match torchtitan's approach of only
computing loss on assistant response tokens.

Usage (2 GPUs with FSDP, matching torchtitan):
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
        scripts/compare_trl_loss.py \
        --model_path ./assets/hf/Qwen3-8B \
        --output_dir /tmp/trl_sft_comparison \
        --steps 180

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=2 python scripts/compare_trl_loss.py \
        --model_path ./assets/hf/Qwen3-8B \
        --output_dir /tmp/trl_sft_comparison \
        --steps 180
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from torch.utils.data import SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer


class SequentialSFTTrainer(SFTTrainer):
    """SFTTrainer with sequential (non-shuffled) data ordering."""

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        return SequentialSampler(train_dataset)


def format_dataset(dataset, tokenizer):
    """Convert each sample to prompt/completion format for TRL.

    Uses prompt/completion format (not messages) so that TRL's
    completion_only_loss actually generates a completion_mask.
    The prompt is rendered via apply_chat_template with add_generation_prompt=True,
    and the completion is the full conversation minus the prompt.
    This mirrors torchtitan's incremental prefix re-tokenization approach.
    """

    def _format(sample):
        answer = sample["answer"]
        reasoning, final_answer = answer.rsplit("####", 1)
        messages = [
            {"role": "user", "content": sample["question"]},
            {
                "role": "assistant",
                "reasoning_content": reasoning.strip(),
                "content": final_answer.strip(),
            },
        ]
        prompt_messages = [m for m in messages if m["role"] != "assistant"]

        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        completion_text = full_text[len(prompt_text):]

        return {"prompt": prompt_text, "completion": completion_text}

    return dataset.map(_format, remove_columns=dataset.column_names)


class LossLogger(TrainerCallback):
    """Log per-step training loss."""

    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(
                {"step": state.global_step, "loss": logs["loss"]}
            )


def main():
    parser = argparse.ArgumentParser(description="TRL SFT baseline for loss comparison")
    parser.add_argument("--model_path", type=str, default="./assets/hf/Qwen3-8B")
    parser.add_argument("--output_dir", type=str, default="/tmp/trl_sft_comparison")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--packing", action="store_true", default=False)
    parser.add_argument("--fp32", action="store_true", default=False,
                        help="Run in float32 instead of bf16")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = torch.float32 if args.fp32 else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=model_dtype,
        attn_implementation="sdpa",
    )

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    formatted = format_dataset(dataset, tokenizer)

    # Match torchtitan sft_qwen3_8b_math hyperparameters
    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        max_length=args.seq_len,
        packing=args.packing,
        # completion_only_loss masks prompt tokens, matching torchtitan's approach
        completion_only_loss=True,
        # Pad to seq_len to match torchtitan's fixed-length padding
        pad_to_multiple_of=args.seq_len,
        learning_rate=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.1,
        optim="adamw_torch",
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": args.lr * 0.1},
        warmup_steps=args.warmup_steps,
        max_grad_norm=1e10,  # effectively disabled for comparison
        bf16=not args.fp32,
        fp16=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=42,
        data_seed=42,
        dataloader_drop_last=True,
        # FSDP for multi-GPU (only enabled when distributed)
        **({
            "fsdp": "full_shard auto_wrap",
            "fsdp_transformer_layer_cls_to_wrap": "Qwen3DecoderLayer",
        } if torch.distributed.is_initialized() else {}),
    )

    loss_logger = LossLogger()

    trainer = SequentialSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted,
        processing_class=tokenizer,
        callbacks=[loss_logger],
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    loss_path = os.path.join(args.output_dir, "loss_log.json")
    with open(loss_path, "w") as f:
        json.dump(loss_logger.losses, f, indent=2)

    print(f"Loss log saved to {loss_path} ({len(loss_logger.losses)} entries)")
    print(f"Config: packing={args.packing}, completion_only_loss=True")


if __name__ == "__main__":
    main()
