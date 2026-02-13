"""
Generator actor for RL training.

Generates trajectories with tool-augmented multi-turn inference.
Weight sync uses load_state_dict from the trainer.
"""

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from zorplex import ZorplexSpec, Task, generate_with_tools

from trainer import Trajectory

from monarch.actor import Actor, endpoint, current_rank


class Generator(Actor):
    """Individual generator worker.

    Uses setup() for heavy initialization (model loading).
    Weight sync uses load_state_dict from the trainer.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
    ):
        # Lightweight init - just store config
        self.model_name = model_name
        self.device_config = device
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

        print(f"[Generator:{self.rank}] Loading model {self.model_name} on GPU {gpu_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.spec = ZorplexSpec(seed=42 + self.rank)

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
    def generate(self, question: str, answer: int, max_turns: int = 4) -> Trajectory:
        """Generate a trajectory for a given task (used for examples/debugging)."""
        self.model.eval()
        task = Task(question=question, correct_answer=answer, metadata={})
        return self._run_generation(task, max_turns)

    @endpoint
    def generate_trajectory(self, max_turns: int = 4) -> Trajectory:
        """Generate a trajectory using a self-generated task.

        Each generator has its own seeded spec, so broadcasting this endpoint
        to all generators produces diverse trajectories from different tasks.
        """
        self.model.eval()
        task = self.spec.generate_task()
        return self._run_generation(task, max_turns)

    def _run_generation(self, task: Task, max_turns: int) -> Trajectory:
        """Shared generation logic: run inference, compute tokens, return Trajectory."""
        import re as _re

        result = generate_with_tools(
            self.model, self.tokenizer, self.spec, task, self.device,
            max_turns=max_turns, max_tokens_per_turn=150,
        )

        self.generations += 1

        # Build model-only text (generated tokens without injected tool results)
        model_only_text = "".join(t.generated_text for t in result.turns)

        # Detect [ANSWER] tag and classify failure mode
        has_answer_tag = bool(_re.search(r'\[ANSWER\]', result.final_text))
        if result.is_correct:
            failure_mode = "success"
        elif not has_answer_tag:
            failure_mode = "wrong_format"
        elif result.total_tool_calls > 3:
            failure_mode = "tool_spam"
        else:
            failure_mode = "wrong_answer"

        # Pre-tokenize for the trainer: prompt + model_only_text
        messages = [
            {"role": "system", "content": self.spec.get_system_prompt()},
            {"role": "user", "content": task.question},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        prompt_length = prompt_ids.shape[1]

        full_ids = self.tokenizer(
            prompt_text + model_only_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=1024,
        )["input_ids"]

        # Reward shaping:
        #   +1.0 for correct answer
        #   +0.2 for format compliance ([ANSWER] tag), only when correct
        reward = 0.0
        if result.is_correct:
            reward += 1.0
            if has_answer_tag:
                reward += 0.2

        return Trajectory(
            task_question=task.question,
            task_answer=task.correct_answer,
            response_text=result.final_text,
            reward=reward,
            is_correct=result.is_correct,
            num_turns=len(result.turns),
            num_tool_calls=result.total_tool_calls,
            generator_id=self.rank,
            policy_version=self.policy_version,
            model_only_text=model_only_text,
            has_answer_tag=has_answer_tag,
            failure_mode=failure_mode,
            input_ids=full_ids[0].tolist(),
            prompt_length=prompt_length,
        )
