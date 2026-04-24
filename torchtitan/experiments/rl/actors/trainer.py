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
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import CommConfig, Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
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
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger

logger = logging.getLogger(__name__)


class PolicyTrainer(Actor, Configurable):
    """Updates policy based on collected Episode using TorchTitan components.

    Exposes separate ``forward_backward`` and ``optim_step`` endpoints, called
    explicitly by the controller.

    Args:
        config: PolicyTrainer.Config with all model/optimizer/parallelism settings.
        model_spec: TorchTitan model specification.
        hf_assets_path: Path to HF assets folder for checkpoint loading.
            Shared with the generator (both load from the same HF checkpoint).
        generator_dtype: Generator dtype (e.g. "bfloat16"). Needed to cast weights to generator dtype
            if generator dtype differs from training dtype. If None, no cast is performed.
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
        ac_config: ActivationCheckpointConfig = field(
            default_factory=lambda: ActivationCheckpointConfig(mode="none")
        )
        model_converters: ModelConvertersContainer.Config = field(
            default_factory=ModelConvertersContainer.Config
        )
        dump_folder: str = ""
        """Folder for AC debug dumps when using memory_budget mode."""

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        hf_assets_path: str = "",
        generator_dtype: str = "",
    ):

        init_logger()

        self.config = config
        self.loss_fn = config.loss.build()

        # Only cast if generator dtype differs from training dtype, otherwise
        # staging buffers would be allocated for a no-op cast.
        training_dtype = TORCH_DTYPE_MAP[config.training.dtype]
        gen_dtype = TORCH_DTYPE_MAP[generator_dtype] if generator_dtype else None
        self._transfer_dtype = gen_dtype if gen_dtype != training_dtype else None

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

        # Build model converters (e.g. float8). No-op if converters list is empty.
        model_compile_enabled = (
            config.compile.enable and "model" in config.compile.components
        )
        model_converters = config.model_converters.build(
            parallel_dims=self.parallel_dims,
            model_compile_enabled=model_compile_enabled,
        )

        # Create training policy model
        model = self._build_model(
            model_spec, config, model_converters, device_type, hf_assets_path
        )
        model.train()
        self.model = model
        self.model_parts = [model]

        # Build optimizer and LR scheduler
        self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )

        # Post optimizer step model converters hook.
        # e.g. calculate float8 dynamic amax/scale for FSDP2
        self.optimizers.register_step_post_hook(
            lambda *args, **kwargs: model_converters.post_optimizer_hook(
                self.model_parts
            )
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
        model_converters: ModelConvertersContainer,
        device_type: str,
        hf_assets_path: str,
    ):
        """Build, parallelize, and initialize a model from checkpoint.

        Args:
            model_spec: Model specification for building and parallelizing.
            config: Trainer config (used for dtype, parallelism, checkpoint path, etc.).
            model_converters: Built model converters instance for pre-parallelize conversion.
            device_type: Device type string (e.g. "cuda").
            hf_assets_path: Path to HF assets folder for checkpoint loading.

        Returns:
            Initialized model with weights loaded from checkpoint.
        """

        # TODO: Also support flex attention backend later.
        from torchtitan.models.common.attention import VarlenAttention

        assert isinstance(
            model_spec.model.layers[0].attention.inner_attention, VarlenAttention.Config
        ), "Only varlen attention backend is allowed."

        # Fill sharding configs on the config BEFORE build via the
        # model-agnostic ``update_from_config`` hook (RL's trainer bypasses
        # ``torchtitan.Trainer``'s call, so we invoke it directly).
        model_spec.model.update_from_config(trainer_config=config)

        with torch.device("meta"):
            with utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]):
                model = model_spec.model.build()

        model_converters.convert(model)

        model = model_spec.parallelize_fn(
            model,
            parallel_dims=self.parallel_dims,
            training=config.training,
            model_converters=config.model_converters,
            parallelism=config.parallelism,
            compile_config=config.compile,
            ac_config=config.ac_config,
            dump_folder=config.dump_folder,
        )

        model.to_empty(device=device_type)
        with torch.no_grad():
            model.init_weights(buffer_device=None)

        # Load initial weights from HF
        self._load_initial_hf_weights(model, hf_assets_path)

        return model

    @endpoint
    async def forward_backward(self, train_data: list[TrainBatch]) -> dict:
        """Run forward pass, compute loss, and call backward.

        Args:
            train_data: List of batches, one per DP rank.

        Returns:
            dict with loss metrics, advantage stats, and logprob verification.
        """
        logger.debug(
            f"{os.getpid()=} PolicyTrainer forward_backward "
            f"step {self.policy_version}"
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

        loss, loss_metrics = self.loss_fn(
            policy_logprobs=policy_logprobs,
            advantages=advantages,
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

        # Backward pass
        self.optimizers.zero_grad()
        loss.backward()

        return {
            "loss": loss.item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            "logprob_diff_mean": verification_result["logprob_diff_mean"],
            "logprob_diff_max": verification_result["logprob_diff_max"],
            "logprob_max_delta": verification_result["logprob_max_delta"],
            "logprob_bitwise_identical": verification_result[
                "logprob_bitwise_identical"
            ],
            **loss_metrics,
        }

    @endpoint
    async def optim_step(self) -> dict:
        """Clip gradients, step optimizer and LR scheduler.

        Returns:
            dict with grad_norm and policy_version.
        """
        # TODO: Accept optional optimizer params (e.g. learning rate)
        # to allow controller-owned schedules (see Tinker API).

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
        )

        self.optimizers.step()
        self.lr_schedulers.step()

        self.policy_version += 1

        logger.debug(
            f"{os.getpid()=} PolicyTrainer optim_step done, "
            f"policy_version={self.policy_version}"
        )

        return {
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
            "policy_version": self.policy_version,
        }

    @endpoint
    async def save_checkpoint(self, path: str) -> None:
        """Save model state dict to disk via DCP.

        Args:
            path: Directory to save the checkpoint to.
        """
        # TODO: Reuse torchtitan's CheckpointManager for async saves, HF export,
        # and checkpoint loading for resume support.
        state_dict = {"model": get_model_state_dict(self.model)}
        dcp.save(state_dict, checkpoint_id=path)
        logger.info(f"Saved checkpoint to {path}")

    @endpoint
    async def push_model_state_dict(self) -> None:
        """Publish model weights for generator consumption via TorchStore.

        When ``direct_rdma=True``, weights are transferred directly from
        GPU to GPU via one-sided RDMA reads, bypassing StorageVolumes
        entirely. When ``False``, data goes through StorageVolumes
        (which may themselves use RDMA as a transport internally).

        Note: we couple ``is_rdma_available()`` with ``direct_rdma`` here,
        but the two concepts are not identical -- StorageVolumes can also
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
