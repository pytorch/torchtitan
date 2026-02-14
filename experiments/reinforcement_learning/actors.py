"""
Actor definitions for async RL training.

Provides the 4 actor classes used in the RL training pipeline:
- ZorplexWorker: Tool execution environment (managed by a Service)
- ReplayBuffer: Stores trajectories for async training
- TrainerActor: Trains the model on trajectory batches (REINFORCE)
- GeneratorWorker: Generates trajectories with tool-augmented inference
"""

from collections import deque
import random

import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

from zorplex_rl import get_spec, Task
from zorplex_rl.evaluate import generate_with_tools

from rl_primitives import Trajectory, TrainMetrics

from monarch.actor import Actor, endpoint, current_rank

# RDMA imports (with fallback)
try:
    from monarch.rdma import RDMABuffer, is_rdma_available
    _rdma_available = is_rdma_available()
except Exception:
    RDMABuffer = None
    _rdma_available = False


def rdma_available():
    return _rdma_available


class ZorplexWorker(Actor):
    """Worker actor that handles Zorplex tool execution.

    Managed by a Service for load balancing across replicas.
    """

    def __init__(self, difficulty: str = "easy", seed: int = 42):
        self.rank = current_rank().rank
        self.spec = get_spec("compositional", difficulty=difficulty, seed=seed + self.rank)
        self.calls_served = 0
        print(f"[ZorplexWorker:{self.rank}] Initialized with difficulty={difficulty}")

    @endpoint
    def ping(self) -> bool:
        return True

    @endpoint
    def generate_task(self) -> tuple[str, int]:
        """Generate a new task. Returns (question, correct_answer)."""
        task = self.spec.generate_task()
        return task.question, task.correct_answer

    @endpoint
    def execute_tool(self, tool_name: str, argument: str) -> str:
        """Execute a tool call."""
        from zorplex_rl.task_specs import ToolCall
        tc = ToolCall(tool_name, argument)
        result = self.spec.execute_tool(tc)
        self.calls_served += 1
        return str(result)

    @endpoint
    def get_system_prompt(self) -> str:
        """Get the system prompt with tool hints."""
        return self.spec.get_system_prompt(with_hint=True)

    @endpoint
    def check_answer(self, model_output: str, correct_answer: int) -> tuple[bool, int | None]:
        """Check if model output contains the correct answer."""
        extracted = self.spec.extract_answer(model_output, [])
        is_correct = extracted == correct_answer
        return is_correct, extracted

    @endpoint
    def stats(self) -> dict:
        return {"rank": self.rank, "calls_served": self.calls_served}


class ReplayBuffer(Actor):
    """Stores trajectories for async RL training."""

    def __init__(self, max_size: int = 1000):
        self.buffer: deque[Trajectory] = deque(maxlen=max_size)
        self.total_added = 0
        print(f"[ReplayBuffer] Initialized with max_size={max_size}")

    @endpoint
    def add(self, trajectory: Trajectory) -> None:
        """Add a trajectory to the buffer."""
        self.buffer.append(trajectory)
        self.total_added += 1

    @endpoint
    def sample(self, batch_size: int) -> list[Trajectory]:
        """Sample a batch of trajectories."""
        if len(self.buffer) == 0:
            return []
        n = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), n)

    @endpoint
    def size(self) -> int:
        return len(self.buffer)

    @endpoint
    def clear(self) -> int:
        """Clear the buffer. Returns number of items removed."""
        count = len(self.buffer)
        self.buffer.clear()
        return count

    @endpoint
    def stats(self) -> dict:
        if len(self.buffer) == 0:
            return {"size": 0, "total_added": self.total_added, "avg_reward": 0.0}
        rewards = [t.reward for t in self.buffer]
        failure_modes = {}
        for t in self.buffer:
            fm = t.failure_mode or "unknown"
            failure_modes[fm] = failure_modes.get(fm, 0) + 1
        return {
            "size": len(self.buffer),
            "total_added": self.total_added,
            "avg_reward": sum(rewards) / len(rewards),
            "correct_rate": sum(1 for t in self.buffer if t.is_correct) / len(self.buffer),
            "format_rate": sum(1 for t in self.buffer if t.has_answer_tag) / len(self.buffer),
            "failure_modes": failure_modes,
        }


class TrainerActor(Actor):
    """Trains the model on trajectories.

    Uses setup() for heavy initialization (model loading, RDMA registration).
    Implements circular buffer with CPU staging for weight distribution.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        lr: float = 1e-5,
        device: str = "cuda",
        n_buffer_slots: int = 3,
    ):
        # Lightweight init - just store config
        self.model_name = model_name
        self.lr = lr
        self.device_config = device
        self.n_buffer_slots = n_buffer_slots
        self.rank = current_rank().rank
        self._ready = False
        print(f"[Trainer:{self.rank}] Spawned, waiting for setup()...")

    @endpoint
    def setup(self) -> dict:
        """Heavy initialization: load model, create optimizer, set up circular buffer."""
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

        # Circular buffer with CPU staging: each slot is a single contiguous
        # CPU buffer holding ALL model parameters packed end-to-end.
        total_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

        self._slots = [
            torch.empty(total_bytes, dtype=torch.uint8)
            for _ in range(self.n_buffer_slots)
        ]

        self._slot_handles = []
        if rdma_available() and RDMABuffer is not None:
            try:
                for slot in self._slots:
                    self._slot_handles.append(RDMABuffer(slot))
                print(f"[Trainer:{self.rank}] RDMA handles registered for {self.n_buffer_slots} circular buffer slots")
            except Exception as e:
                print(f"[Trainer:{self.rank}] RDMA registration failed: {e}")
                self._slot_handles = []

        self._param_meta = {}
        offset = 0
        for name, p in self.model.named_parameters():
            self._param_meta[name] = (offset, tuple(p.shape), p.dtype)
            offset += p.numel() * p.element_size()

        self._publish_weights()

        self._ready = True
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[Trainer:{self.rank}] Ready! {param_count:,} params, "
              f"RDMA={len(self._slot_handles) > 0}, "
              f"buffer_slots={self.n_buffer_slots}")

        return {
            "status": "ready",
            "params": param_count,
            "rdma": len(self._slot_handles) > 0,
            "buffer_slots": self.n_buffer_slots,
        }

    def _publish_weights(self):
        """Copy GPU params to the current circular buffer slot (D2H)."""
        slot_idx = self.policy_version % self.n_buffer_slots
        slot = self._slots[slot_idx]
        for name, p in self.model.named_parameters():
            off, shape, dtype = self._param_meta[name]
            nbytes = p.numel() * p.element_size()
            slot[off:off + nbytes].copy_(
                p.data.view(-1).view(torch.uint8).cpu(), non_blocking=True
            )
        torch.cuda.synchronize()  # Ensure D2H complete before RDMA reads

    @endpoint
    def get_weight_handle(self) -> tuple:
        """Get RDMA handle for the latest weight slot.

        Returns (handle_or_None, param_meta, version, total_bytes).
        If RDMA unavailable, handle is None and caller should use get_state_dict().
        """
        total_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        if self._slot_handles:
            slot_idx = self.policy_version % self.n_buffer_slots
            return self._slot_handles[slot_idx], self._param_meta, self.policy_version, total_bytes
        return None, self._param_meta, self.policy_version, total_bytes

    @endpoint
    def get_state_dict(self) -> tuple[dict, int]:
        """Fallback: get state dict directly (when RDMA not available)."""
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

        # Optimizer step, then publish weights to circular buffer
        if valid_count > 0:
            avg_loss = torch.stack(losses).sum() / valid_count
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        else:
            avg_loss = torch.tensor(0.0)

        # Bump version, then publish weights to the new slot.
        self.policy_version += 1
        self._publish_weights()
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
    def evaluate_zorplex(self, num_samples: int = 10, seed: int = 42) -> dict:
        """Evaluate current model on compositional Zorplex tasks."""
        import re as _re
        self.model.eval()
        torch.manual_seed(seed)  # Deterministic evaluation
        spec = get_spec("compositional", seed=seed)
        correct = 0
        total_turns = 0
        total_tools = 0
        format_ok = 0
        failure_modes = {"success": 0, "wrong_format": 0, "tool_spam": 0, "wrong_answer": 0}
        for _ in range(num_samples):
            task = spec.generate_task()
            result = generate_with_tools(
                self.model, self.tokenizer, spec, task,
                self.device, max_turns=5,
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


class GeneratorWorker(Actor):
    """Individual generator worker.

    Uses setup() for heavy initialization (model loading).
    Weight sync uses CPU staging buffer for explicit RDMA -> H2D flow.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        difficulty: str = "easy",
        device: str = "cuda",
    ):
        # Lightweight init - just store config
        self.model_name = model_name
        self.difficulty = difficulty
        self.device_config = device
        self.rank = current_rank().rank
        self._ready = False
        print(f"[GeneratorWorker:{self.rank}] Spawned, waiting for setup()...")

    @endpoint
    def setup(self) -> dict:
        """Heavy initialization: load model, create weight buffer."""
        import os

        if self._ready:
            return {"status": "already_ready"}

        # Generators use GPU 1 + rank (trainer uses GPU 0)
        gpu_id = 1 + self.rank
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_version = 0
        self.generations = 0

        print(f"[GeneratorWorker:{self.rank}] Loading model {self.model_name} on GPU {gpu_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.spec = get_spec("compositional", difficulty=self.difficulty, seed=42 + self.rank)

        self._sync_buf = None  # CPU staging buffer for RDMA weight sync

        self._ready = True
        print(f"[GeneratorWorker:{self.rank}] Ready on GPU {gpu_id}!")

        return {"status": "ready", "rank": self.rank, "gpu": gpu_id}

    @endpoint
    def get_version(self) -> int:
        return self.policy_version

    @endpoint
    def sync_weights_from_buffer(self, handle, param_meta: dict, version: int, total_bytes: int) -> bool:
        """Sync weights via RDMA from trainer's circular buffer.

        Flow: Trainer CPU slot --RDMA--> Generator CPU staging --H2D--> Generator GPU params
        """
        if version <= self.policy_version:
            return False

        # Allocate CPU staging buffer on first sync (reuse thereafter).
        if self._sync_buf is None or self._sync_buf.numel() < total_bytes:
            self._sync_buf = torch.empty(total_bytes, dtype=torch.uint8)

        # RDMA read: trainer CPU slot -> generator CPU staging buffer
        byte_view = self._sync_buf[:total_bytes].flatten()
        handle.read_into(byte_view).get()

        # Scatter from CPU staging into GPU model params (H2D copy per parameter)
        for name, p in self.model.named_parameters():
            off, shape, dtype = param_meta[name]
            nbytes = p.numel() * p.element_size()
            src = self._sync_buf[off:off + nbytes].view(dtype).view(shape)
            p.data.copy_(src)
        self.policy_version = version
        return True

    @endpoint
    def sync_weights(self, state_dict: dict, version: int) -> bool:
        """Sync weights directly (fallback when RDMA unavailable)."""
        if version <= self.policy_version:
            return False
        self.model.load_state_dict(state_dict)
        self.policy_version = version
        return True

    @endpoint
    def generate(self, question: str, answer: int, max_turns: int = 5) -> Trajectory:
        """Generate a trajectory for a given task (used for examples/debugging)."""
        self.model.eval()
        task = Task(question=question, correct_answer=answer, metadata={})
        return self._run_generation(task, max_turns)

    @endpoint
    def generate_trajectory(self, max_turns: int = 5) -> Trajectory:
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
            {"role": "system", "content": self.spec.get_system_prompt(with_hint=True)},
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
        #   +0.2 for format compliance ([ANSWER] tag)
        #   -0.1 per tool call beyond 2 (discourages tool spam)
        reward = 0.0
        if result.is_correct:
            reward += 1.0
        if has_answer_tag:
            reward += 0.2
        excess_tools = max(0, result.total_tool_calls - 2)
        reward -= 0.1 * excess_tools

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
