# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os

from dataclasses import dataclass
from typing import List

import torch
from monarch.actor import Actor, endpoint
from safetensors.torch import save_file
from torchtitan.config import CommConfig
from torchtitan.distributed import utils as dist_utils

# Import unified module - this automatically registers TorchTitan models with vLLM
from torchtitan.experiments.rl import unified  # noqa: F401

from torchtitan.experiments.rl.vllm_compat.simple_rl import (
    compute_grpo_advantages,
    compute_grpo_advantages_stable,
    math_reward_function,
    trivial_reward_function,
)
from torchtitan.experiments.rl.vllm_compat.weights.converter import torchtitan_to_vllm
from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        model_path: str,
        temp_checkpoint_dir: str = "./converted",
        tp_size: int = 1,
    ):
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
        logger.info("vLLM rollout engine initialized (will load on first use)")

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

        # TODO: need to replace this with Torchtitan's checkpoint save and load
        # right now we hardcoded to work with 2 safe tensor files which we only
        # tested on Qwen3 0.6B model. In the longer term, need to use TorchStore
        # to achieve the weight communication.
        # only generator rank 0 saves the weight
        if torch.distributed.get_rank() == 0:
            logger.info(f"Saving weights to {checkpoint_path}")
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

        # Synchronize all ranks before reloading to ensure rank 0 finished writing
        torch.distributed.barrier()
        logger.info(
            f"[Rank {torch.distributed.get_rank()}] Synchronized after weight save"
        )

        # First time: create the engine
        if self.llm is None:
            self.llm = LLM(
                model=self.temp_model_dir,
                hf_overrides={
                    # Override architectures to use our registered TorchTitan model class
                    "architectures": ["Qwen3TorchTitanForCausalLM"],
                },
                trust_remote_code=True,
                max_model_len=2048,
                dtype="bfloat16",
                gpu_memory_utilization=0.1,  # Reduced from 0.5
                distributed_executor_backend="external_launcher",  # vllm do not spawn processes
                seed=42,  # Fixed seed for determinism
                enforce_eager=True,
                tensor_parallel_size=self.tp_size,
                attention_config=AttentionConfig(
                    backend=AttentionBackendEnum.FLASH_ATTN,
                ),
            )
            logger.info("Created new vLLM engine")
        else:
            # Direct parameter copy into model tensors.
            # This bypasses vLLM's reload_weights() which uses a layerwise
            # reload mechanism that moves params to meta device
            from torchtitan.experiments.rl.vllm_compat.weights_vllm_compat import (
                vllm_compat_to_torchtitan,
            )

            titan_state = vllm_compat_to_torchtitan(vllm_compat_state)
            self._direct_weight_update(titan_state)

    def _direct_weight_update(self, titan_state: dict) -> None:
        """Update model weights by copying directly into GPU parameters.

        Args:
            titan_state: TorchTitan format state dict (w1/w2/w3, wq/wk/wv/wo, etc.)
        """

        # Access model from vLLM engine
        model = self.llm.llm_engine.model_executor.driver_worker.get_model()
        params = dict(model.named_parameters())

        for name, new_weight in titan_state.items():
            # TorchTitanVLLMModelWrapper stores the model as self.model,
            # so parameters have a "model." prefix
            param_name = f"model.{name}"
            if param_name in params:
                param = params[param_name]
                new_w = new_weight.to(device=param.device, dtype=param.dtype)
                param.data.copy_(new_w)

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
    via weight sync. Generates completions for given prompts and
    computes rewards/advantages.

    Args:
        model_path: Path to HuggingFace model
        prompt_texts: List of prompt strings
        expected_answers: List of expected answers
        group_size: Number of samples per prompt
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        use_real_dataset: Whether using real dataset (GSM8K)
        grpo_beta: Beta for GRPO advantages
        use_stable_grpo: Whether to use stable GRPO
        tp_size: Tensor Parallel size
    """

    def __init__(
        self,
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

        # Initialize distributed environment for SPMD generator
        world_size = dist_utils.init_distributed(
            CommConfig(),
        )
        # Initialize vLLM engine
        self.vllm_engine = VLLMRolloutEngine(model_path, tp_size=self.tp_size)

        # State machine
        self.state = GeneratorState.READY_TO_UPDATE
        self.cond = asyncio.Condition()
        self.policy_version = 0

        # Reward function
        self.reward_fn = (
            math_reward_function if use_real_dataset else trivial_reward_function
        )

        logger.info("Generator initialized with vLLM engine")

    @endpoint
    async def generate(self) -> None:
        """Generate trajectories and compute rewards/advantages."""
        logger.info(
            f"{os.getpid()=} Generating start generate (policy v{self.policy_version})..."
        )
        async with self.cond:
            # Wait until ready to generate (weights have been updated)
            await self.cond.wait_for(
                lambda: self.state == GeneratorState.READY_TO_GENERATE
            )

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

            # Signal ready for update
            self.state = GeneratorState.READY_TO_UPDATE
            self.cond.notify_all()

            logger.info(
                f"{os.getpid()=} Generating finish generate (policy v{self.policy_version})..."
            )
            return trajectory

    @endpoint
    async def update(self, version: int, vllm_compat_state: dict) -> None:
        """Update generate weights.

        Args:
            version: New policy version number
            vllm_compat_state: vLLM-compatible state dict
        """
        async with self.cond:
            self.vllm_engine.update_weights(vllm_compat_state)
            # Update version and state
            self.policy_version = version
            self.state = GeneratorState.READY_TO_GENERATE
            self.cond.notify_all()
            logger.info(
                f"{os.getpid()=} Generator updating weights to policy v{version}..."
            )
