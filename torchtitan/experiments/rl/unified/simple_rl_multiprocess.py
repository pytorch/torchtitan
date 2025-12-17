# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Multiprocess RL training loop using Monarch Actors.

This demonstrates:
1. Distributed actor architecture with Generator (vLLM) and Trainer (TorchTitan) components
2. Asynchronous communication via queues
3. File based weight synchronization between trainer and generator #TODO: enable RDMA-based weight transfer
4. Event-driven architecture for efficient RL training

The architecture mirrors grpo_actor.py but adapted for vLLM rollouts + TorchTitan training.
"""
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

import asyncio
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
from monarch.actor import Actor, endpoint, this_host
from monarch.rdma import RDMABuffer
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.batch_invariant import init_batch_invariance


from torchtitan.experiments.rl.vllm_compat.simple_rl import (
    compute_grpo_advantages,
    compute_grpo_advantages_stable,
    compute_policy_gradient_loss_vllm,
    download_and_convert_model,
    load_gsm8k_dataset,
    math_reward_function,
    trivial_reward_function,
)
from torchtitan.experiments.rl.vllm_compat.weights.converter import (
    torchtitan_to_vllm,
)

from torchtitan.experiments.rl.unified.models.utils import ModelMode, load_model
from torchtitan.experiments.rl.vllm_compat.weights_vllm_compat import (
    torchtitan_to_vllm_compat,
)


@dataclass
class TrajectoryData:
    """
    Data from one generation batch.

    Attributes:
        policy_version: Version of policy that produced this batch
        completions: List of completion strings
        vllm_token_ids: List of token ID lists for each completion
        vllm_token_log_probs: List of per-token log prob lists
        prompt_token_ids: List of prompt token ID lists
        rewards: Computed rewards for each completion
        advantages: Computed advantages for each completion
    """

    policy_version: int
    completions: List[str]
    vllm_token_ids: List[List[int]]
    vllm_token_log_probs: List[List[float]]
    prompt_token_ids: List[List[int]]
    rewards: torch.Tensor
    advantages: torch.Tensor


class TrajectoryQueue(Actor):
    """Queue for trajectory data between Generator and Trainer."""

    def __init__(self):
        """Initialize an empty queue."""
        self.queue: asyncio.Queue[TrajectoryData] = asyncio.Queue()

    @endpoint
    async def put(self, trajectory: TrajectoryData) -> None:
        """Add trajectory data to the queue.

        Args:
            trajectory: The trajectory data to add
        """
        await self.queue.put(trajectory)

    @endpoint
    async def get(self) -> TrajectoryData:
        """Remove and return trajectory data from the queue.

        Returns:
            The next trajectory data in the queue
        """
        return await self.queue.get()


class VLLMRolloutEngine:
    """
    vLLM engine for fast rollouts with weight updates.

    Note: vLLM loads from model_config.model path, so we create a temporary
    directory with updated weights and restart the engine. This is faster than
    recreating temp dirs repeatedly and handles config/tokenizer files properly.

    Args:
        model_path: Path to HuggingFace model (for config/tokenizer)
        temp_checkpoint_dir: Directory to save temporary weight checkpoints
    """

    def __init__(self, model_path: str, temp_checkpoint_dir: str = "./converted", tp_size: int = 1):
        self.base_model_path = model_path
        self.temp_model_dir = os.path.abspath(
            os.path.join(temp_checkpoint_dir, "vllm_temp_model")
        )
        os.makedirs(self.temp_model_dir, exist_ok=True)

        import glob

        # Copy config/tokenizer files from base model to temp dir
        import shutil

        for file in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "merges.txt",
            "vocab.json",
        ]:
            src = os.path.join(model_path, file)
            if os.path.exists(src):
                shutil.copy2(src, self.temp_model_dir)

        # Copy the original model shard files if they exist
        # We'll overwrite these with our single model.safetensors later
        for shard_file in glob.glob(os.path.join(model_path, "model-*.safetensors")):
            dst = os.path.join(self.temp_model_dir, os.path.basename(shard_file))
            shutil.copy2(shard_file, dst)

        # Copy index file if it exists
        index_file = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            shutil.copy2(index_file, self.temp_model_dir)

        self.llm = None
        self.tp_size = tp_size
        print("vLLM rollout engine initialized (will load on first use)")

    def update_weights(self, vllm_compat_state: dict) -> None:
        """
        Update vLLM model weights from vLLM-compat state dict.

        This converts weights to vLLM format, saves them, and reloads using
        vLLM's reload_weights() API after updating the model path config.

        Args:
            vllm_compat_state: vLLM-compat model state dict (with gate_up_proj/down_proj)
        """
        # Convert vLLM-compat -> vLLM (torchtitan_to_vllm handles both formats)
        vllm_state = torchtitan_to_vllm(vllm_compat_state)

        # Save to temp model directory
        import os
        checkpoint_path = os.path.join(self.temp_model_dir, "model.safetensors")

        # Update the shard files that vLLM will actually load
        # We need to split our weights to match the original 2-shard structure
        import glob
        import json

        shard_files = sorted(
            glob.glob(os.path.join(self.temp_model_dir, "model-*.safetensors"))
        )
        index_file = os.path.join(self.temp_model_dir, "model.safetensors.index.json")

        if len(shard_files) == 2 and os.path.exists(index_file):
            # Load the index to see which weights go in which shard
            with open(index_file, "r") as f:
                index_data = json.load(f)

            weight_map = index_data["weight_map"]

            # Split weights according to the index
            shard1_weights = {}
            shard2_weights = {}

            for key, value in vllm_state.items():
                shard_file = weight_map.get(key, shard_files[0])
                if "model-00001-of-00002" in shard_file:
                    shard1_weights[key] = value
                else:
                    shard2_weights[key] = value

            # Ensure weights stay in bfloat16
            shard1_weights = {
                k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                for k, v in shard1_weights.items()
            }
            shard2_weights = {
                k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                for k, v in shard2_weights.items()
            }

            # Save to the shard files
            save_file(shard1_weights, shard_files[0])
            save_file(shard2_weights, shard_files[1])
        else:
            # Ensure weights stay in bfloat16
            vllm_state = {
                k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                for k, v in vllm_state.items()
            }
            # Fallback: save as single file
            save_file(vllm_state, checkpoint_path)

        # First time: create the engine
        if self.llm is None:
            # Disable distributed execution to avoid NCCL conflicts in Monarch actors
            # Use single GPU mode
            import os
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

            self.llm = LLM(
                model=self.temp_model_dir,
                trust_remote_code=True,
                max_model_len=2048,
                dtype="bfloat16",
                gpu_memory_utilization=0.1,  # Reduced from 0.5
                seed=42,  # Fixed seed for determinism
                enforce_eager=True,
                tensor_parallel_size=self.tp_size,  # Explicitly single GPU
            )
            print("✓ Created new vLLM engine")
        else:
            # Use collective_rpc to call reload_weights on all workers
            # This reloads weights from temp_model_dir without recreating the engine
            self.llm.collective_rpc("reload_weights")

    @torch.no_grad()
    def generate(
        self,
        prompt_texts: list[str],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        n_samples_per_prompt: int = 4,
    ) -> tuple[
        list[str], torch.Tensor, list[list[int]], list[list[float]], list[list[int]]
    ]:
        """
        Generate samples using vLLM.

        Args:
            prompt_texts: List of prompt strings
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            n_samples_per_prompt: Number of samples per prompt

        Returns:
            completions: List of completion strings
            log_probs: [batch] - Sum of log probs for each completion
            token_ids: List of token ID lists for each completion (generated tokens only)
            token_log_probs: List of per-token log prob lists for each completion
            prompt_token_ids: List of prompt token ID lists for each completion
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=n_samples_per_prompt,
            seed=42,
            logprobs=1,
            prompt_logprobs=1,  # Also get prompt log probs to access prompt token IDs
        )

        outputs = self.llm.generate(prompt_texts, sampling_params)

        # Extract completions and log probs
        completions = []
        log_probs_list = []
        token_ids_list = []
        token_log_probs_list = []
        prompt_token_ids_list = []

        for output in outputs:
            # Extract prompt token IDs from the output
            prompt_token_ids = output.prompt_token_ids

            for sample in output.outputs:
                completions.append(sample.text)

                # Store prompt tokens for this sample
                prompt_token_ids_list.append(prompt_token_ids)

                # Extract token IDs (generated tokens only)
                token_ids = sample.token_ids
                token_ids_list.append(token_ids)

                # Extract per-token log probs
                per_token_log_probs = [
                    list(logprob_dict.values())[0].logprob
                    for logprob_dict in sample.logprobs
                ]
                token_log_probs_list.append(per_token_log_probs)

                # Sum log probs across generated tokens
                total_log_prob = sum(per_token_log_probs)
                log_probs_list.append(total_log_prob)

        log_probs = torch.tensor(log_probs_list, dtype=torch.float32)

        return (
            completions,
            log_probs,
            token_ids_list,
            token_log_probs_list,
            prompt_token_ids_list,
        )

    def __del__(self):
        """Cleanup vLLM engine."""
        if hasattr(self, "llm"):
            del self.llm
            torch.cuda.empty_cache()


class GeneratorState:
    """States for the Generator's state machine."""

    READY_TO_GENERATE = "READY_TO_GENERATE"
    READY_TO_UPDATE = "READY_TO_UPDATE"


class Generator(Actor):
    """
    Generates rollouts using vLLM engine.

    Maintains a vLLM engine that is synchronized with the Trainer
    via RDMA buffers. Generates completions for given prompts and
    computes rewards/advantages.
    """

    def __init__(
        self,
        weight_buffers: dict,
        trajectory_queue: Any,
        model_path: str,
        prompt_texts: List[str],
        expected_answers: List[str],
        group_size: int = 8,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        use_real_dataset: bool = False,
        grpo_beta: float = 0.1,
        use_stable_grpo: bool = False,
        tp_size: int = 1,
    ):
        """Initialize the generator.

        Args:
            weight_buffers: RDMA buffers for policy weights
            trajectory_queue: Queue to put generated trajectories in
            model_path: Path to HuggingFace model
            prompt_texts: List of prompt strings
            expected_answers: List of expected answers
            group_size: Number of samples per prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            use_real_dataset: Whether using real dataset (GSM8K)
            grpo_beta: Beta for GRPO advantages
            use_stable_grpo: Whether to use stable GRPO
            tp_size: Tensor Paralell size
        """
        self.weight_buffers = weight_buffers
        self.trajectory_queue = trajectory_queue
        self.model_path = model_path
        self.prompt_texts = prompt_texts
        self.expected_answers = expected_answers
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_real_dataset = use_real_dataset
        self.grpo_beta = grpo_beta
        self.use_stable_grpo = use_stable_grpo
        self.tp_size = tp_size

        # Initialize vLLM engine
        self.vllm_engine = VLLMRolloutEngine(model_path, tp_size=self.tp_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # State machine
        self.state = GeneratorState.READY_TO_UPDATE
        self.cond = asyncio.Condition()
        self.policy_version = 0

        # Reward function
        self.reward_fn = (
            math_reward_function if use_real_dataset else trivial_reward_function
        )

        print("Generator initialized with vLLM engine")

    @endpoint
    async def generate(self) -> None:
        """Generate trajectories and compute rewards/advantages."""
        async with self.cond:
            # Wait until ready to generate (weights have been updated)
            await self.cond.wait_for(
                lambda: self.state == GeneratorState.READY_TO_GENERATE
            )

            print(f"Generating rollouts (policy v{self.policy_version})...")

            # Generate samples using vLLM
            (
                completions,
                vllm_log_probs,
                vllm_token_ids,
                vllm_token_log_probs,
                prompt_token_ids,
            ) = self.vllm_engine.generate(
                self.prompt_texts,
                self.max_new_tokens,
                self.temperature,
                n_samples_per_prompt=self.group_size,
            )

            # Compute rewards
            if self.reward_fn == trivial_reward_function:
                rewards = self.reward_fn(
                    completions, self.tokenizer, self.expected_answers, self.group_size
                )
            else:
                rewards = self.reward_fn(
                    completions, self.expected_answers, self.group_size
                )

            # Normalize rewards
            reward_mean = rewards.mean()
            reward_std = rewards.std()
            if reward_std > 1e-8:
                rewards_normalized = (rewards - reward_mean) / reward_std
            else:
                rewards_normalized = rewards - reward_mean

            # Compute advantages using GRPO
            if self.use_stable_grpo:
                advantages = compute_grpo_advantages_stable(
                    rewards_normalized, self.group_size
                )
            else:
                advantages = compute_grpo_advantages(
                    rewards_normalized, self.group_size, beta=self.grpo_beta
                )

            # Create trajectory data
            trajectory = TrajectoryData(
                policy_version=self.policy_version,
                completions=completions,
                vllm_token_ids=vllm_token_ids,
                vllm_token_log_probs=vllm_token_log_probs,
                prompt_token_ids=prompt_token_ids,
                rewards=rewards,
                advantages=advantages,
            )

        # Send to trajectory queue
        await self.trajectory_queue.put.call(trajectory)

        async with self.cond:
            # Signal ready for update
            self.state = GeneratorState.READY_TO_UPDATE
            self.cond.notify_all()

    @endpoint
    async def update(self, version: int, vllm_compat_state: dict) -> None:
        """Update generate weights.

        Args:
            version: New policy version number
            vllm_compat_state: vLLM-compatible state dict
        """
        async with self.cond:
            print(f"Geneartor updating weights to policy v{version}...")
            self.vllm_engine.update_weights(vllm_compat_state)

            # Update version and state
            self.policy_version = version
            self.state = GeneratorState.READY_TO_GENERATE
            self.cond.notify_all()


class Trainer(Actor):
    """
    Updates policy based on collected trajectories.

    Pulls trajectories from queue, computes loss, and updates model.
    Notifies generators of weight updates via RDMA.
    """

    def __init__(
        self,
        trajectory_queue: Any,
        titan_checkpoint_path: str,
        model_path: str,
        learning_rate: float = 1e-5,
        model_mode: str = ModelMode.BATCH_INVARIANT,
    ):
        """Initialize the trainer.

        Args:
            trajectory_queue: Queue to pull trajectories from
            titan_checkpoint_path: Path to TorchTitan checkpoint
            model_path: Path to HuggingFace model
            learning_rate: Learning rate for optimizer
            model_mode: Inidcates which model to use. Train inferece unified model, batch invariant Torchtitan model,
                or plain Torchtitan model
        """
        self.trajectory_queue = trajectory_queue

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(
            titan_checkpoint_path, model_path, model_mode=model_mode
        )
        self.model = self.model.to(device)
        self.model.train()

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.policy_version = 0
        self.generator: Optional[Any] = None

        print("Trainer initialized with TorchTitan model")

    @endpoint
    async def init_generator(self, generator: Any) -> None:
        """Set the generator service for weight updates.

        Args:
            generator: Service to notify of policy updates
        """
        self.generator = generator

    @endpoint
    async def get_initial_weights(self) -> dict:
        """Get initial vLLM-compatible weights for generator.

        Returns:
            vLLM-compatible state dict
        """
        titan_state = self.model.state_dict()
        vllm_compat_state = torchtitan_to_vllm_compat(titan_state)
        return vllm_compat_state


    @endpoint
    async def step(self) -> dict:
        """Perform one training step.

        Returns:
            Training metrics
        """
        print(f"Trainer step {self.policy_version}:")

        # Pull trajectory from queue
        trajectory = await self.trajectory_queue.get.call_one()

        # Compute loss
        loss, loss_metrics = compute_policy_gradient_loss_vllm(
            self.model,
            trajectory.vllm_token_ids,
            trajectory.vllm_token_log_probs,
            trajectory.prompt_token_ids,
            trajectory.advantages,
            kl_coef=0.1,
        )

        # Update weights
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.policy_version += 1

        # Notify generator of updated weights
        if self.generator:
            titan_state = self.model.state_dict()
            vllm_compat_state = torchtitan_to_vllm_compat(titan_state)
            await self.generator.update.call(self.policy_version, vllm_compat_state)

        # Return metrics
        metrics = {
            "loss": loss.item(),
            "reward_mean": trajectory.rewards.mean().item(),
            "reward_std": trajectory.rewards.std().item(),
            "advantage_mean": trajectory.advantages.mean().item(),
            "advantage_std": trajectory.advantages.std().item(),
            "sample_completion": trajectory.completions[0][:80],
            **loss_metrics,
        }

        return metrics


async def main():
    """Run the distributed RL training loop."""
    # ========== Config ==========
    model_name = "Qwen/Qwen3-1.7B"
    cache_dir = "./models"
    output_dir = "./converted"

    # Training config
    group_size = 8
    num_steps = 10
    learning_rate = 1e-5
    max_new_tokens = 20

    # GRPO config
    use_stable_grpo = False
    grpo_beta = 0.1

    # Dataset config
    use_real_dataset = False
    num_dataset_samples = 5

    # vLLM parallelism sizes
    tp_size = 4

    # Check batch invariance
    from vllm.model_executor.layers.batch_invariant import (
        init_batch_invariance,
        vllm_is_batch_invariant,
    )

    init_batch_invariance()
    use_vllm_compat = vllm_is_batch_invariant()
    mode = ModelMode.BATCH_INVARIANT

    if use_vllm_compat:
        print("Batch invariance detected - using vLLM-compatible model")
        from torchtitan.experiments.rl.vllm_compat.batch_invariant_backward import (
            enable_batch_invariant_backward_mode,
        )

        enable_batch_invariant_backward_mode()
    else:
        print("Batch invariance NOT detected - using standard model")

    # Download and convert model
    titan_checkpoint_path, model_path = download_and_convert_model(
        model_name, cache_dir, output_dir
    )

    # Load dataset
    if use_real_dataset:
        print(f"Loading GSM8K dataset ({num_dataset_samples} samples)...")
        prompt_texts, expected_answers = load_gsm8k_dataset(
            split="train", num_samples=num_dataset_samples
        )
        if prompt_texts is None or len(prompt_texts) == 0:
            use_real_dataset = False

    if not use_real_dataset:
        print("Using default prompts")
        prompts_with_answers = [
            ("The capital of France is", "paris"),
            ("What is 7 times 8?", "56"),
            ("The first president of the United States was", "washington"),
            ("The chemical symbol for water is", "h2o"),
            ("The largest planet in our solar system is", "jupiter"),
        ]
        prompt_texts = [p[0] for p in prompts_with_answers]
        expected_answers = [p[1] for p in prompts_with_answers]

    print(f"Loaded {len(prompt_texts)} prompts")

    # ========== Create process meshes ==========
    trainer_mesh = this_host().spawn_procs(per_host={"gpus": 1})
    gen_mesh = this_host().spawn_procs(per_host={"gpus": 1})

    # Spawn actors on trainer mesh
    traj_queue = trainer_mesh.spawn("traj_queue", TrajectoryQueue)
    trainer = trainer_mesh.spawn(
        "trainer",
        Trainer,
        traj_queue,
        titan_checkpoint_path,
        model_path,
        learning_rate,
        mode,
    )

    # Get initial weights and spawn generator
    initial_weights = await trainer.get_initial_weights.call_one()
    generator = gen_mesh.spawn(
        "generator",
        Generator,
        {},  # weight_buffers (not using RDMA in this simple version)
        traj_queue,
        model_path,
        prompt_texts,
        expected_answers,
        group_size,
        max_new_tokens,
        1.0,  # temperature
        use_real_dataset,
        grpo_beta,
        use_stable_grpo,
        tp_size,
    )
    await trainer.init_generator.call(generator)

    # Initialize generator with weights
    await generator.update.call(0, initial_weights)
    
    # ========== Training loop ==========
    print("\n" + "=" * 80)
    print(f"Starting RL training for {num_steps} steps")
    print("=" * 80)

    for step in range(num_steps):
        # Generate and train in parallel
        _, metrics = await asyncio.gather(
            generator.generate.call(),
            trainer.step.call_one(),
        )

        print(
            f"\nStep {step:3d} | Loss: {metrics['loss']:.4f} | "
            f"Reward: {metrics['reward_mean']:+.3f}"
        )
        print(f"  Sample: {metrics['sample_completion']}...")

        # Check for divergence
        if not torch.isfinite(torch.tensor(metrics["loss"])):
            print("\n" + "!" * 80)
            print("ERROR: Loss is NaN/Inf! Training diverged.")
            print("!" * 80)
            break

    print("\n" + "=" * 80)
    print("Training complete")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
