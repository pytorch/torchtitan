# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generator actor for RL training.

Generates trajectories for the digit sum task.
Weight sync uses load_state_dict from the trainer.
"""

import re

import torch
from monarch.actor import Actor, current_rank, endpoint

from sum_digits import SumDigitsSpec
from task import Task, extract_answer
from reverse_digits import ReverseDigitsSpec
from trainer import Trajectory
from transformers import AutoModelForCausalLM, AutoTokenizer


class Generator(Actor):
    """Individual generator worker.

    Uses setup() for heavy initialization (model loading).
    Weight sync uses load_state_dict from the trainer.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        task: str = "sum_digits",
    ):
        # Lightweight init - just store config
        self.model_name = model_name
        self.device_config = device
        self.task = task
        self.rank = current_rank().rank
        self._ready = False
        print(f"[Generator:{self.rank}] Spawned, waiting for setup()...")

    @endpoint
    def setup(self) -> dict:
        """Heavy initialization: load model."""
        import os

        if self._ready:
            return {"status": "already_ready"}

        # Generators use GPU 1 + rank (trainer uses GPU 0)
        gpu_id = 1 + self.rank
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_version = 0
        self.generations = 0

        print(
            f"[Generator:{self.rank}] Loading model {self.model_name} on GPU {gpu_id}..."
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.task == "reverse_digits":
            self.spec = ReverseDigitsSpec(seed=42 + self.rank)
        else:
            self.spec = SumDigitsSpec(seed=42 + self.rank)

        self._ready = True
        print(f"[Generator:{self.rank}] Ready on GPU {gpu_id}!")

        return {"status": "ready", "rank": self.rank, "gpu": gpu_id}

    @endpoint
    def get_version(self) -> int:
        return self.policy_version

    @endpoint
    def sync_weights(self, state_dict: dict, version: int) -> bool:
        """Sync weights from trainer via load_state_dict."""
        if version <= self.policy_version:
            return False
        self.model.load_state_dict(state_dict)
        self.policy_version = version
        return True

    @endpoint
    def generate_trajectory(self) -> Trajectory:
        """Generate a trajectory using a self-generated task.

        Each generator has its own seeded spec, so broadcasting this endpoint
        to all generators produces diverse trajectories from different tasks.
        """
        self.model.eval()
        task = self.spec.generate_task()
        return self._run_generation(task)

    def _run_generation(self, task: Task) -> Trajectory:
        """Generate a response, score it, and return a Trajectory."""
        messages = [
            {"role": "system", "content": self.spec.get_system_prompt()},
            {"role": "user", "content": task.question},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        prompt_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
            )

        response_text = self.tokenizer.decode(
            outputs[0][prompt_length:], skip_special_tokens=True
        )

        self.generations += 1

        extracted = extract_answer(response_text)
        is_correct = extracted == task.correct_answer
        has_answer_tag = bool(re.search(r"\[ANSWER\]", response_text))

        reward = 1.0 if is_correct and has_answer_tag else -1.0
        

        return Trajectory(
            task_question=task.question,
            response_text=response_text,
            reward=reward,
            is_correct=is_correct,
            has_answer_tag=has_answer_tag,
            input_ids=outputs[0].tolist(),
            prompt_length=prompt_length,
        )
