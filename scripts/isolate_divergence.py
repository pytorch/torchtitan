"""Isolate source of ~0.01 divergence between torchtitan and TRL.

Tests weight_decay=0 and max_grad_norm=inf to see which one
causes the divergence.

Usage:
    # Test 1: no weight decay
    CUDA_VISIBLE_DEVICES=4 python scripts/isolate_divergence.py --test no_weight_decay
    # Test 2: no grad clipping
    CUDA_VISIBLE_DEVICES=5 python scripts/isolate_divergence.py --test no_grad_clip
    # Test 3: neither (baseline)
    CUDA_VISIBLE_DEVICES=6 python scripts/isolate_divergence.py --test no_both
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

MODEL_PATH = "./assets/hf/Qwen3-0.6B"
STEPS = 10


class SequentialSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        return SequentialSampler(train_dataset)


class LossLogger(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append({"step": state.global_step, "loss": logs["loss"]})


def format_dataset(dataset, tokenizer):
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
        return {"prompt": prompt_text, "completion": full_text[len(prompt_text):]}

    return dataset.map(_format, remove_columns=dataset.column_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["no_weight_decay", "no_grad_clip", "no_both"],
                        required=True)
    args = parser.parse_args()

    weight_decay = 0.0 if args.test in ("no_weight_decay", "no_both") else 0.1
    max_grad_norm = 1e10 if args.test in ("no_grad_clip", "no_both") else 1.0

    print(f"Test: {args.test}, weight_decay={weight_decay}, max_grad_norm={max_grad_norm}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, attn_implementation="sdpa"
    )

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    formatted = format_dataset(dataset, tokenizer)

    output_dir = f"/tmp/trl_isolate_{args.test}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=output_dir,
        max_steps=STEPS,
        per_device_train_batch_size=1,
        max_length=2048,
        packing=False,
        completion_only_loss=True,
        pad_to_multiple_of=2048,
        learning_rate=2e-5,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=weight_decay,
        optim="adamw_torch",
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 2e-6},
        warmup_steps=0,
        max_grad_norm=max_grad_norm,
        bf16=False,
        fp16=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=42,
        data_seed=42,
        dataloader_drop_last=True,
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

    loss_path = os.path.join(output_dir, "loss_log.json")
    with open(loss_path, "w") as f:
        json.dump(loss_logger.losses, f, indent=2)
    print(f"Saved to {loss_path}")
    for d in loss_logger.losses:
        print(f"  step {d['step']}: {d['loss']:.8f}")


if __name__ == "__main__":
    main()
