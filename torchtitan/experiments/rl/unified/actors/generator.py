# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os

from dataclasses import dataclass, field
from typing import List

import torch
from monarch.actor import Actor, endpoint
from safetensors.torch import save_file

# TODO: Replace with ``from torchtitan.config.configs import ParallelismConfig``
# once the config branch lands.
from torchtitan.config.job_config import Parallelism as ParallelismConfig
from torchtitan.distributed import utils as dist_utils

# Import unified module - this automatically registers TorchTitan models with vLLM
from torchtitan.experiments.rl import unified  # noqa: F401

from torchtitan.experiments.rl.unified.configs import RLTrainer, VLLMSamplingConfig

# TODO: Replace with ``from torchtitan.config import Configurable``
# once the config branch lands.
from torchtitan.experiments.rl.unified.configurable import Configurable
from torchtitan.experiments.rl.vllm_compat.simple_rl import (
    compute_grpo_advantages,
    compute_grpo_advantages_stable,
    trivial_reward_function,
)
from vllm import EngineArgs, LLMEngine, SamplingParams

from vllm.config import AttentionConfig
from vllm.model_executor.layers.batch_invariant import init_batch_invariance
from vllm.sampling_params import RequestOutputKind

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


class VLLMEngine(Configurable):
    """
    vLLM engine for fast rollouts with weight updates.

    Note: vLLM loads from model_config.model path, so we create a temporary
    directory with updated weights and restart the engine. This is faster than
    recreating temp dirs repeatedly and handles config/tokenizer files properly.

    Constructed via ``config.build(model_path=..., dump_folder=...)``
    which calls ``VLLMEngine(config=..., model_path=..., dump_folder=...)``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """vLLM engine configuration for rollout generation."""

        dtype: str = "bfloat16"
        """Data type for model weights (auto, float16, bfloat16, float32)."""

        gpu_memory_limit: float = 0.5
        """Fraction of GPU memory to use for the vLLM engine (0.0 to 1.0)."""

        enforce_eager: bool = True
        """Disable CUDA graphs in vLLM (use eager execution)."""

        seed: int = 42
        """Random seed for reproducible generation."""

        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        """Parallelism configuration for the vLLM engine."""

        sampling: VLLMSamplingConfig = field(default_factory=VLLMSamplingConfig)
        """Default sampling parameters for generation."""

    def __init__(
        self,
        config: Config,
        *,
        model_path: str,
        dump_folder: str,
    ) -> None:
        self.config = config
        self.base_model_path = model_path

        self.temp_model_dir = os.path.abspath(
            os.path.join(dump_folder, "vllm_temp_model")
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

        self.engine = None
        logger.info("vLLM rollout engine initialized (will load on first use)")

    def update_weights(self, vllm_state: dict) -> None:
        """
        Update vLLM model weights from vLLM-compat state dict.

        This converts weights to vLLM format, saves them, and reloads using
        vLLM's reload_weights() API after updating the model path config.

        Args:
            vllm_state: vLLM model state dict
        """
        # Save to temp model directory
        import os

        checkpoint_path = os.path.join(self.temp_model_dir, "model.safetensors")

        # Update the shard files that vLLM will actually load
        # We need to split our weights to match the original 2-shard structure
        import glob

        shard_files = sorted(
            glob.glob(os.path.join(self.temp_model_dir, "model-*.safetensors"))
        )
        index_file = os.path.join(self.temp_model_dir, "model.safetensors.index.json")

        # TODO: need to replace this with Torchtitan's checkpoint save and load
        # right now we hardcoded to work with 1 safetensor files which we only
        # tested on Qwen3 0.6B model. In the longer term, need to use TorchStore
        # to achieve the weight communication.
        # only generator rank 0 saves the weight
        if torch.distributed.get_rank() == 0:
            logger.info(f"Saving weights to {checkpoint_path}")

            # Ensure weights stay in bfloat16
            vllm_state = {
                k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                for k, v in vllm_state.items()
            }
            save_file(vllm_state, checkpoint_path)

        # Synchronize all ranks before reloading to ensure rank 0 finished writing
        torch.distributed.barrier()
        logger.info(
            f"[Rank {torch.distributed.get_rank()}] Synchronized after weight save"
        )

        # First time: create the engine using LLMEngine and EngineArgs
        if self.engine is None:
            cfg = self.config

            engine_args = EngineArgs(
                # Model configuration
                model=self.temp_model_dir,
                trust_remote_code=True,
                dtype=cfg.dtype,
                # Parallelism configuration
                tensor_parallel_size=cfg.parallelism.tensor_parallel_degree,
                distributed_executor_backend="external_launcher",
                # Memory and performance
                gpu_memory_utilization=cfg.gpu_memory_limit,
                enforce_eager=cfg.enforce_eager,
                # Seed
                seed=cfg.seed,
                # HuggingFace overrides to use TorchTitan model.
                # TODO: make this field configurable and align with model registration
                hf_overrides={"architectures": ["Qwen3TorchTitanForCausalLM"]},
                attention_config=AttentionConfig(
                    backend=AttentionBackendEnum.FLASH_ATTN,
                ),
            )

            logger.info("Initializing LLMEngine from EngineArgs...")
            self.engine = LLMEngine.from_engine_args(engine_args)
            logger.info("Created new vLLM LLMEngine")
        else:
            # Direct parameter copy into model tensors.
            # This bypasses vLLM's reload_weights() which uses a layerwise
            # reload mechanism that moves params to meta device
            from torchtitan.experiments.rl.vllm_compat.weights import vllm_to_torchtitan

            titan_state = vllm_to_torchtitan(vllm_state)
            self._direct_weight_update(titan_state)

    def _direct_weight_update(self, titan_state: dict) -> None:
        """Update model weights by copying directly into GPU parameters.

        Args:
            titan_state: TorchTitan format state dict (w1/w2/w3, wq/wk/wv/wo, etc.)
        """

        # Access model from vLLM engine
        model = self.engine.model_executor.driver_worker.get_model()
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
        Generate samples using vLLM LLMEngine.

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
        logger.info(
            f"Starting generation: {len(prompt_texts)} prompts, "
            f"n_samples_per_prompt={n_samples_per_prompt}, "
            f"max_tokens={max_new_tokens}, temp={temperature}"
        )

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=n_samples_per_prompt,
            seed=42,
            logprobs=1,
            prompt_logprobs=1,  # Also get prompt log probs to access prompt token IDs
            output_kind=RequestOutputKind.FINAL_ONLY,  # Only return completed outputs
        )

        # Add one request per prompt; vLLM handles n_samples_per_prompt via n=
        for request_id, prompt in enumerate(prompt_texts):
            self.engine.add_request(str(request_id), prompt, sampling_params)

        # Step through engine until all requests are finished
        all_outputs = []
        while self.engine.has_unfinished_requests():
            request_outputs = self.engine.step()
            all_outputs.extend(request_outputs)

        # Extract completions and log probs
        completions = []
        log_probs_list = []
        token_ids_list = []
        token_log_probs_list = []
        prompt_token_ids_list = []

        for output in all_outputs:
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
        if hasattr(self, "engine"):
            del self.engine
            torch.cuda.empty_cache()


class GeneratorState:
    """States for the Generator's state machine."""

    READY_TO_GENERATE = "READY_TO_GENERATE"
    READY_TO_UPDATE = "READY_TO_UPDATE"


class Generator(Actor, Configurable):
    """
    Generates rollouts using vLLM engine.

    Maintains a vLLM engine that is synchronized with the Trainer
    via weight sync. Generates completions for given prompts and
    computes rewards/advantages.

    Constructed via ``config.build(rl_config=..., prompt_texts=..., expected_answers=...)``
    which calls ``Generator(config=..., rl_config=..., prompt_texts=..., expected_answers=...)``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Generator actor configuration."""

        vllm_engine: VLLMEngine.Config = field(default_factory=VLLMEngine.Config)
        """vLLM rollout engine configuration."""

        vllm_attention_backend: str = "FLASH_ATTN"
        """vLLM attention backend to use (e.g., FLASH_ATTN, XFORMERS)."""

    def __init__(
        self,
        config: Config,
        *,
        rl_config: RLTrainer.Config,
        prompt_texts: list[str],
        expected_answers: list[str],
    ):
        self.config = config
        self.rl_config = rl_config

        # Set vLLM environment variables from config before any vLLM initialization
        if rl_config.batch_invariant_mode:
            os.environ["VLLM_BATCH_INVARIANT"] = "1"
            init_batch_invariance(AttentionBackendEnum.FLASH_ATTN)

        os.environ["VLLM_ATTENTION_BACKEND"] = config.vllm_attention_backend

        self.prompt_texts = prompt_texts
        self.expected_answers = expected_answers

        # Extract needed fields from configs
        self.model_path = rl_config.trainer.checkpoint.initial_load_path
        self.max_new_tokens = config.vllm_engine.sampling.max_tokens
        self.temperature = config.vllm_engine.sampling.temperature
        self.group_size = rl_config.policy_optimization.group_size
        self.grpo_beta = rl_config.policy_optimization.beta
        self.use_stable_grpo = rl_config.policy_optimization.use_stable

        # Initialize distributed environment for SPMD generator
        # TODO: Replace rl_config.trainer.comm with rl_config.trainer.comm
        # once the config branch lands (Comm -> CommConfig).
        from torchtitan.config.job_config import Comm

        world_size = dist_utils.init_distributed(Comm())

        # Build vLLM engine from its config
        self.vllm_engine = config.vllm_engine.build(
            model_path=self.model_path,
            dump_folder=rl_config.dump_folder,
        )

        # State machine
        self.state = GeneratorState.READY_TO_UPDATE
        self.cond = asyncio.Condition()
        self.policy_version = 0

        # Reward function. TODO: Use a real reward function
        self.reward_fn = trivial_reward_function

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
            logger.info(
                f"Computing rewards: {len(completions)} completions, "
                f"{len(self.expected_answers)} expected answers, "
                f"group_size={self.group_size}"
            )
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
