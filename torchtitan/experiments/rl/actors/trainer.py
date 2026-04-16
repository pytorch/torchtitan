# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
import torchstore as ts
from monarch.actor import Actor, endpoint
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import CommConfig, Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.utils import set_batch_invariance
from torchtitan.experiments.rl.actors.utils import (
    compute_logprobs,
    create_positions_from_seq_lens,
    create_varlen_metadata,
    extract_response_logprobs,
    verify_logprob_identity,
)
from torchtitan.experiments.rl.types import TrainBatch
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils

logger = logging.getLogger(__name__)


class PolicyTrainer(Actor, Configurable):
    """
    Updates policy based on collected Episode using TorchTitan components.

    Uses ModelSpec for model construction, parallelization, and weight loading.

    Args:
        config: PolicyTrainer.Config for model/optimizer/parallelism settings.
        model_spec: Model specification (model config, parallelize_fn, state_dict_adapter).
        hf_assets_path: Path to HF assets folder for checkpoint loading.
        transfer_dtype: DType to cast weights to before transfer. If None, no cast is performed.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """PolicyTrainer configuration for optimizer, training, and parallelism."""

        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )
        lr_scheduler: LRSchedulersContainer.Config = field(
            default_factory=LRSchedulersContainer.Config
        )
        training: TrainingConfig = field(default_factory=TrainingConfig)
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        comm: CommConfig = field(default_factory=CommConfig)
        compile: CompileConfig = field(default_factory=CompileConfig)
        debug: DebugConfig = field(default_factory=DebugConfig)
        loss: Configurable.Config = field(default_factory=Configurable.Config)

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        hf_assets_path: str = "",
        transfer_dtype: str = "",
    ):
        self.config = config
        self.loss_fn = config.loss.build()
        self.model_spec = model_spec
        # Only cast if transfer dtype differs from training dtype, otherwise
        # staging buffers would be allocated for a no-op cast.
        training_dtype = TORCH_DTYPE_MAP[config.training.dtype]
        requested = TORCH_DTYPE_MAP[transfer_dtype] if transfer_dtype else None
        self._transfer_dtype = requested if requested != training_dtype else None

        # The policy and ref models share code objects, so dynamo's
        # per-code-object cache must hold entries for both grad modes
        # (grad for policy, no_grad for ref). The default limit of 8
        # is not enough; 16 accommodates both without recompile storms.
        # TODO: @Lucaskabela fix recompiles in general as these increase startup
        torch._dynamo.config.cache_size_limit = 16

        # Device setup
        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)

        # Enable batch-invariant mode BEFORE init_distributed
        set_batch_invariance(config.debug.batch_invariant)

        world_size = dist_utils.init_distributed(config.comm)

        self.parallel_dims = ParallelDims.from_config(config.parallelism, world_size)

        # Set determinism flags and seed via core torchtitan utility
        dist_utils.set_determinism(
            self.parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["pp"],
        )

        # Initialize state dict adapter for HF checkpoint loading
        if model_spec.state_dict_adapter is not None:
            self.sd_adapter = model_spec.state_dict_adapter(
                model_spec.model, hf_assets_path
            )
        else:
            self.sd_adapter = None

        # Create training policy model
        model = self._build_model(model_spec, config, device_type, hf_assets_path)
        model.train()
        self.model = model
        self.model_parts = [model]

        # Conditionally build frozen reference model for KL penalty
        # TODO: @joecummings remove ref entirely, this is hacky and we don't need it
        if getattr(config.loss, "kl_coef", 0) > 0:
            ref_model = self._build_model(
                model_spec, config, device_type, hf_assets_path
            )
            ref_model.eval()
            ref_model.requires_grad_(False)
            self.ref_model = ref_model
            logger.info("Built frozen reference model for KL penalty")
        else:
            self.ref_model = None

        # Build optimizer and LR scheduler
        self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )

        self.policy_version = 0
        self.generator: Any | None = None

        # Data parallelism: mesh is available after _build_model triggers build_mesh
        self.dp_enabled = self.parallel_dims.dp_enabled
        batch_mesh = self.parallel_dims.get_optional_mesh("batch")
        if batch_mesh is not None:
            self.dp_size = batch_mesh.size()
            self.dp_rank = batch_mesh.get_local_rank()
        else:
            self.dp_size = 1
            self.dp_rank = 0

        logger.debug(
            f"PolicyTrainer initialized (dp_rank={self.dp_rank}, dp_size={self.dp_size})"
        )

    def _load_initial_hf_weights(self, model, checkpoint_path: str) -> None:
        """Load model weights from HF checkpoint using DCP and state_dict_adapter.

        Args:
            model: The model to load weights into.
            checkpoint_path: Path to HF checkpoint directory.
        """
        if self.sd_adapter is None:
            logger.warning(
                "No state_dict_adapter available, skipping initial weight load"
            )
            return

        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint path '{checkpoint_path}' does not exist. "
                "Please provide a valid path to a HuggingFace checkpoint directory."
            )

        storage_reader = self.sd_adapter.get_hf_storage_reader(checkpoint_path)
        hf_state_dict = self.sd_adapter.to_hf(model.state_dict())
        dcp.load(hf_state_dict, storage_reader=storage_reader)
        torchtitan_state_dict = self.sd_adapter.from_hf(hf_state_dict)

        set_model_state_dict(
            model=model,
            model_state_dict=torchtitan_state_dict,
            options=StateDictOptions(strict=True),
        )
        logger.info(
            f"Loaded initial weights from {checkpoint_path} "
            f"({len(torchtitan_state_dict)} parameters)"
        )

    def _build_model(
        self,
        model_spec: ModelSpec,
        config: Config,
        device_type: str,
        hf_assets_path: str,
    ):
        """Build, parallelize, and initialize a model from checkpoint.
        Will be used to build trainer's policy model and reference model.

        Args:
            model_spec: Model specification for building and parallelizing.
            config: Trainer config (used for dtype, parallelism, checkpoint path, etc.).
            device_type: Device type string (e.g. "cuda").
            hf_assets_path: Path to HF assets folder for checkpoint loading.

        Returns:
            Initialized model with weights loaded from checkpoint.
        """

        # TODO Also support flex attention backend later.
        from torchtitan.models.common.attention import VarlenAttention

        assert isinstance(
            model_spec.model.layers[0].attention.inner_attention, VarlenAttention.Config
        ), "Only varlen attention backend is allowed."

        with torch.device("meta"):
            with utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]):
                model = model_spec.model.build()

        model = model_spec.parallelize_fn(
            model,
            parallel_dims=self.parallel_dims,
            parallelism=config.parallelism,
            compile_config=config.compile,
        )

        model.to_empty(device=device_type)
        with torch.no_grad():
            model.init_weights(buffer_device=None)

        # Load initial weights from HF
        self._load_initial_hf_weights(model, hf_assets_path)

        return model

    @endpoint
    async def push_model_state_dict(self) -> None:
        """Publish model weights for generator consumption via TorchStore.

        When ``direct_rdma=True``, weights are transferred directly from
        GPU to GPU via one-sided RDMA reads, bypassing StorageVolumes
        entirely. When ``False``, data goes through StorageVolumes
        (which may themselves use RDMA as a transport internally).

        Note: we couple ``is_rdma_available()`` with ``direct_rdma`` here,
        but the two concepts are not identical — StorageVolumes can also
        use RDMA as their transport layer. ``direct_rdma`` specifically
        means "skip StorageVolumes and let the destination read directly
        from the source's GPU memory".
        """
        from monarch.rdma import is_rdma_available

        await ts.put_state_dict(
            self.model.state_dict(),
            "model_state_dict",
            direct_rdma=is_rdma_available(),
            transfer_dtype=self._transfer_dtype,
        )

    @endpoint
    async def step(self, train_data: list[TrainBatch]) -> dict:
        """Perform one training step.

        Args:
            train_data (list[TrainBatch]): List of batches, one per DP rank.

        Returns:
            dict: Training metrics (loss, policy version, etc.).
        """
        # The policy and ref models share code objects, so dynamo's
        # per-code-object cache must hold entries for both grad modes
        # (grad for policy, no_grad for ref). The default limit of
        # is not enough; 16 accommodates both without recompile storms.
        # TODO: @Lucaskabela fix recompiles in general as these increase startup
        torch._dynamo.config.recompile_limit = 16

        logger.debug(
            f"{os.getpid()=} PolicyTrainer starting step {self.policy_version} "
        )

        local_batch = train_data[self.dp_rank]
        device = self.device

        token_ids = local_batch.token_ids.to(device)
        seq_lens = local_batch.seq_lens
        prompt_lens = local_batch.prompt_lens
        response_lens = local_batch.response_lens
        advantages = local_batch.advantages.to(device)

        max_seq_len = max(seq_lens)
        rope_cache_len = self.model.freqs_cis.shape[0]
        if max_seq_len > rope_cache_len:
            raise ValueError(
                f"Episode length {max_seq_len} exceeds rope cache size "
                f"{rope_cache_len}. Increase model max_seq_len or reduce "
                f"generation max_tokens."
            )

        attention_masks = create_varlen_metadata(seq_lens, device)
        positions = create_positions_from_seq_lens(seq_lens, device)

        logits = self.model(
            token_ids, attention_masks=attention_masks, positions=positions
        )
        all_policy_logprobs = compute_logprobs(logits, token_ids)
        policy_logprobs = extract_response_logprobs(
            all_policy_logprobs, seq_lens, prompt_lens, response_lens
        )

        ref_logprobs = None
        if self.ref_model is not None:
            with torch.no_grad():
                ref_logits = self.ref_model(
                    token_ids, attention_masks=attention_masks, positions=positions
                )
                all_ref_logprobs = compute_logprobs(ref_logits, token_ids)
                ref_logprobs = extract_response_logprobs(
                    all_ref_logprobs, seq_lens, prompt_lens, response_lens
                )

        loss, loss_metrics = self.loss_fn(
            policy_logprobs=policy_logprobs,
            advantages=advantages,
            ref_logprobs=ref_logprobs,
        )

        verification_result = verify_logprob_identity(
            local_batch.token_logprobs,
            policy_logprobs,
        )

        logger.debug(
            f"Logprob verification: bitwise_identical={verification_result['logprob_bitwise_identical']}, "
            f"max_delta={verification_result['logprob_max_delta']:.6e}, "
            f"diff_mean={verification_result['logprob_diff_mean']:.6e}, "
            f"diff_max={verification_result['logprob_diff_max']:.6e}, "
            f"tokens_checked={verification_result['total_tokens_checked']}"
        )

        # Update weights
        self.optimizers.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
        )

        self.optimizers.step()
        self.lr_schedulers.step()

        self.policy_version += 1

        # Return metrics
        metrics = {
            "loss": loss.item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            "policy_version": self.policy_version,
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
            # Trainer vs generator log prob divergence
            "logprob_diff_mean": verification_result["logprob_diff_mean"],
            "logprob_diff_max": verification_result["logprob_diff_max"],
            "logprob_max_delta": verification_result["logprob_max_delta"],
            "logprob_bitwise_identical": verification_result[
                "logprob_bitwise_identical"
            ],
            **loss_metrics,
        }
        logger.debug(
            f"{os.getpid()=} PolicyTrainer finished step {self.policy_version}"
        )
        return metrics
