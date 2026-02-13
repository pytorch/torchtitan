"""
Trainer actor for RL training.

Trains the model on trajectory batches using REINFORCE.
Exposes get_state_dict for weight distribution to generators.
"""

import torch
import torch.nn.functional as F

from dataclasses import dataclass, field

from transformers import AutoModelForCausalLM, AutoTokenizer

from zorplex import ZorplexSpec, generate_with_tools

from monarch.actor import Actor, endpoint, current_rank


@dataclass
class Trajectory:
    """A single trajectory from generation."""
    task_question: str
    response_text: str
    reward: float
    is_correct: bool
    has_answer_tag: bool = False  # Whether model emitted [ANSWER]
    # Pre-tokenized sequence and prompt boundary for training.
    # The generator populates these so the trainer never needs to re-tokenize.
    input_ids: list[int] = field(default_factory=list)
    prompt_length: int = 0  # Number of prompt tokens (response starts here)


@dataclass
class TrainMetrics:
    """Metrics from a training step."""
    step: int
    loss: float
    batch_size: int
    avg_reward: float
    policy_version: int
    correct_rate: float = 0.0  # Fraction with correct answer
    format_rate: float = 0.0  # Fraction that emitted [ANSWER]


class Trainer(Actor):
    """Trains the model on trajectories.

    Uses setup() for heavy initialization (model loading).
    Exposes get_state_dict for weight distribution to generators.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        lr: float = 5e-6,
        device: str = "cuda",
    ):
        # Lightweight init - just store config
        self.model_name = model_name
        self.lr = lr
        self.device_config = device
        self.rank = current_rank().rank
        self._ready = False
        print(f"[Trainer:{self.rank}] Spawned, waiting for setup()...")

    @endpoint
    def setup(self) -> dict:
        """Heavy initialization: load model, create optimizer."""
        import os

        if self._ready:
            return {"status": "already_ready"}

        # Trainer always uses GPU 0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_version = 0
        self.train_steps = 0

        print(f"[Trainer:{self.rank}] Loading model {self.model_name} on GPU 0...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        self.model.gradient_checkpointing_enable()

        self._ready = True
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[Trainer:{self.rank}] Ready! {param_count:,} params")

        return {
            "status": "ready",
            "params": param_count,
        }

    @endpoint
    def get_state_dict(self) -> tuple[dict, int]:
        """Get model state dict and current policy version."""
        return self.model.state_dict(), self.policy_version

    @endpoint
    def get_version(self) -> int:
        return self.policy_version

    @endpoint
    def train_step(self, trajectories: list[Trajectory], baseline: float) -> TrainMetrics:
        """Train on a batch of trajectories using REINFORCE.

        Each trajectory carries pre-tokenized input_ids and a prompt_length
        boundary from the generator, so we just slice and compute log-probs.
        """
        if len(trajectories) == 0:
            return TrainMetrics(
                step=self.train_steps, loss=0.0, batch_size=0,
                avg_reward=0.0, policy_version=self.policy_version,
            )

        self.model.train()
        self.optimizer.zero_grad()

        losses = []
        valid_count = 0

        for traj in trajectories:
            if not traj.input_ids or traj.prompt_length == 0:
                continue

            # Load pre-tokenized sequence from the generator
            full_ids = torch.tensor(traj.input_ids, device=self.device).unsqueeze(0)
            prompt_len = traj.prompt_length

            if full_ids.shape[1] <= prompt_len + 1:
                continue

            # Forward pass, then slice at prompt_length for response-only log-probs
            with torch.amp.autocast('cuda', enabled=self.device == "cuda"):
                logits = self.model(full_ids).logits

            # logits[i] predicts token[i+1], so start at prompt_len - 1
            shift_logits = logits[:, prompt_len - 1:-1, :]
            shift_labels = full_ids[:, prompt_len:]
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

            # REINFORCE loss = -log_prob * advantage
            advantage = traj.reward - baseline
            losses.append(-token_log_probs.sum() * advantage)
            valid_count += 1

        # Optimizer step
        if valid_count > 0:
            avg_loss = torch.stack(losses).sum() / valid_count
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        else:
            avg_loss = torch.tensor(0.0)

        self.policy_version += 1
        self.train_steps += 1

        avg_reward = sum(t.reward for t in trajectories) / len(trajectories)
        correct_rate = sum(1 for t in trajectories if t.is_correct) / len(trajectories)
        format_rate = sum(1 for t in trajectories if t.has_answer_tag) / len(trajectories)

        return TrainMetrics(
            step=self.train_steps,
            loss=avg_loss.item() if torch.is_tensor(avg_loss) else avg_loss,
            batch_size=len(trajectories),
            avg_reward=avg_reward,
            policy_version=self.policy_version,
            correct_rate=correct_rate,
            format_rate=format_rate,
        )

    @endpoint
    def evaluate_zorplex(self, num_samples: int = 10, seed: int = 42, max_turns: int = 4) -> dict:
        """Evaluate current model on compositional Zorplex tasks."""
        import re as _re
        self.model.eval()
        torch.manual_seed(seed)  # Deterministic evaluation
        spec = ZorplexSpec(seed=seed)
        correct = 0
        total_turns = 0
        total_tools = 0
        format_ok = 0
        failure_modes = {"success": 0, "wrong_format": 0, "tool_spam": 0, "wrong_answer": 0}
        for _ in range(num_samples):
            task = spec.generate_task()
            result = generate_with_tools(
                self.model, self.tokenizer, spec, task,
                self.device, max_turns=max_turns,
                temperature=0.0, do_sample=False,
            )
            correct += int(result.is_correct)
            total_turns += len(result.turns)
            total_tools += result.total_tool_calls
            has_tag = bool(_re.search(r'\[ANSWER\]', result.final_text))
            format_ok += int(has_tag)
            if result.is_correct:
                failure_modes["success"] += 1
            elif not has_tag:
                failure_modes["wrong_format"] += 1
            elif result.total_tool_calls > 3:
                failure_modes["tool_spam"] += 1
            else:
                failure_modes["wrong_answer"] += 1
        return {
            "accuracy": correct / num_samples,
            "correct": correct,
            "total": num_samples,
            "avg_turns": total_turns / num_samples,
            "avg_tools": total_tools / num_samples,
            "format_rate": format_ok / num_samples,
            "failure_modes": failure_modes,
        }
