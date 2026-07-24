# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.checkpoint_utils import canonical_fqn
from torchtitan.components.loss import BaseLoss, ChunkedLossWrapper
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    apply_overrides,
    CommConfig,
    CompileConfig,
    Configurable,
    DebugConfig,
    OverrideConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointingConfig,
    SelectiveAC,
)
from torchtitan.distributed.utils import set_batch_invariance
from torchtitan.experiments.rl.losses import GRPOLoss
from torchtitan.experiments.rl.types import OptimStepOutput, TrainingMicrobatch
from torchtitan.models.common.attention import FlexAttention
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger

logger = logging.getLogger(__name__)

_TRAIN_STATE_VERSION = 1


class PolicyTrainer(Actor, Configurable):
    """Updates policy based on collected TrainingSample using TorchTitan components.

    Exposes separate `forward_backward` and `optim_step` endpoints, called
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
        debug: DebugConfig = field(default_factory=DebugConfig)
        loss: BaseLoss.Config = field(default_factory=GRPOLoss.Config)
        ac_config: ActivationCheckpointingConfig = field(
            default_factory=SelectiveAC.Config
        )
        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        override: OverrideConfig = field(default_factory=OverrideConfig)
        """Config overrides (e.g. ``torchtitan.overrides.fused_swiglu.fused_swiglu``)
        applied to this trainer's model spec after ``update_from_config`` and before build.
        Separate from the generator's override so the two can differ."""
        dump_folder: str = ""
        """Folder for AC debug dumps when using memory_budget mode."""

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        compile_config: CompileConfig,
        hf_assets_path: str = "",
        generator_dtype: str = "",
        output_dir: str,
    ):
        init_logger()
        # Quiet torchstore's per-op transport-resolve INFO spam (very noisy in CI).
        logging.getLogger("torchstore.transport").setLevel(logging.WARNING)
        if not config.dump_folder:
            config.dump_folder = output_dir
        sl.init_structured_logger(
            source="rl_trainer",
            output_dir=output_dir,
            rank=current_rank().rank,
            enable=config.debug.enable_structured_logging,
        )
        sl.log_trace_instant("structured_logger_started")

        self.config = config
        self.compile_config = compile_config
        self.loss_fn = config.loss.build()
        # TODO: add support to compile the loss.

        # Only cast if generator dtype differs from training dtype, otherwise
        # staging buffers would be allocated for a no-op cast.
        training_dtype = TORCH_DTYPE_MAP[config.training.dtype]
        gen_dtype = TORCH_DTYPE_MAP[generator_dtype] if generator_dtype else None
        self._transfer_dtype = gen_dtype if gen_dtype != training_dtype else None

        # Device setup
        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)
        self.grad_scaler = torch.amp.GradScaler(
            device=device_type,
            enabled=config.training.mixed_precision_param == "float16",
        )

        # Enable batch-invariant mode BEFORE init_distributed
        set_batch_invariance(config.debug.batch_invariant)

        with sl.log_trace_span("torch_distributed_init"):
            world_size = dist_utils.init_distributed(
                config.comm,
                base_folder=output_dir,
            )

        self.parallel_dims = ParallelDims.from_config(config.parallelism, world_size)
        dist_utils.set_spmd_backend(config.parallelism.spmd_backend)
        self.train_context = dist_utils.get_spmd_context(
            parallel_dims=self.parallel_dims,
            spmd_typechecking=False,
        )

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
        model = self._build_model(model_spec, config, device_type)
        model.train()
        self.model = model
        self.model_parts = [model]

        if isinstance(self.loss_fn, ChunkedLossWrapper):
            lm_head = model.lm_head
            assert lm_head is not None, "Model must have lm_head for ChunkedLossWrapper"
            self.loss_fn.set_lm_head(lm_head)
            model._skip_lm_head = True

        # Build optimizer and LR scheduler
        self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )

        self.policy_version = 0

        # Always build CheckpointManager; enable is a field on the config.
        # When enable=False (CI/debug), load() is a no-op and random init stands.
        self.checkpointer = config.checkpoint.build(
            dataloader=None,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            sd_adapter=self.sd_adapter,
            base_folder=config.dump_folder,
        )
        self.checkpointer.load()
        if not self.checkpointer.enable:
            logger.warning(
                "Checkpoint disabled, skip weight loading and use random-initialized weights. "
                "Set checkpoint.enable=True to load from a checkpoint."
            )

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

    def state_dict(self) -> dict[str, Any]:
        # Checkpoint "train_state": policy_version == completed trainer steps, so it
        # doubles as the resume step counter.
        if not self.grad_scaler.is_enabled():
            return {"policy_version": self.policy_version}

        # DCP requires every requested nested key to exist in older checkpoints.
        # Keep AMP state in the existing leaf so pre-AMP checkpoints remain loadable.
        return {
            "policy_version": (
                _TRAIN_STATE_VERSION,
                self.policy_version,
                self.grad_scaler.state_dict(),
            )
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        policy_state = state_dict["policy_version"]
        grad_scaler_state = None
        if isinstance(policy_state, tuple):
            state_version, policy_version, grad_scaler_state = policy_state
            if state_version != _TRAIN_STATE_VERSION:
                raise ValueError(
                    f"Unsupported PolicyTrainer state version {state_version}."
                )
            self.policy_version = policy_version
        else:
            # Checkpoints written before fp16 support stored only the integer version.
            self.policy_version = policy_state

        if grad_scaler_state is not None and self.grad_scaler.is_enabled():
            self.grad_scaler.load_state_dict(grad_scaler_state)

    @endpoint
    async def get_policy_version(self) -> int:
        """Current policy version: after load(), the step a resume restored from
        (0 if fresh). The controller uses it to resume and re-sync generators."""
        return self.policy_version

    @endpoint
    async def close(self) -> None:
        """Close actor-local resources before the process mesh stops.

        The trainer does not own the distributed process group lifecycle here:
        Monarch created it for the actor mesh, and ``ProcMesh.stop()`` performs
        the final teardown. Destroying it from this endpoint can race with mesh
        shutdown and hang at process exit.
        """
        logger.debug("PolicyTrainer close requested; ProcMesh.stop owns PG teardown.")

    @sl.log_trace_span("build_model")
    def _build_model(
        self,
        model_spec: ModelSpec,
        config: Config,
        device_type: str,
    ):
        """Build, parallelize, and initialize a model with random weights.

        Checkpoint loading (e.g. from HF) is handled separately by
        CheckpointManager after model and optimizer construction.

        Args:
            model_spec: Model specification for building and parallelizing.
            config: Trainer config (used for dtype, parallelism, etc.).
            device_type: Device type string (e.g. "cuda").

        Returns:
            Model with random-initialized weights.
        """

        from torchtitan.models.common.attention import VarlenAttention

        assert isinstance(
            model_spec.model.layers[0].attention.inner_attention,
            (VarlenAttention.Config, FlexAttention.Config),
        ), "Only varlen and flex attention backends are allowed."

        # Fill sharding configs on the config BEFORE build via the
        # model-agnostic `update_from_config` hook (RL's trainer bypasses
        # `torchtitan.Trainer's` call, so we invoke it directly).
        model_spec.model.update_from_config(config=config)

        # Check if seq_length passed the max_seq_len
        max_seq_len = model_spec.model.max_seq_len
        seq_len = config.training.seq_len
        if seq_len > max_seq_len:
            raise ValueError(
                f"Training sequence length {seq_len} exceeds "
                f"attention RoPE maximum supported sequence "
                f"length {max_seq_len}."
            )

        for layer_cfg in model_spec.model.layers:
            attention_cfg = getattr(layer_cfg, "attention", None)
            if attention_cfg is not None:
                attention_cfg.rope = replace(attention_cfg.rope, max_seq_len=seq_len)

        # Apply this trainer's config overrides after update_from_config (which
        # sets the sharding configs the override factories read) and before build
        if config.override.imports:
            apply_overrides(config.override, model_spec.model)

        with torch.device("meta"):
            with utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]):
                model = model_spec.model.build()

        model = model_spec.parallelize_fn(
            model,
            parallel_dims=self.parallel_dims,
            training=config.training,
            parallelism=config.parallelism,
            compile_config=self.compile_config,
            ac_config=config.ac_config,
            dump_folder=config.dump_folder,
        )

        model.to_empty(device=device_type)
        with torch.no_grad():
            model.init_weights(buffer_device=None)

        return model

    @endpoint
    async def sync_log_step(self, step: int, relative_step: int | None = None) -> None:
        """Sync the structured-logger step counter from the controller."""
        sl.set_step(step, relative_step=relative_step)

    def reduce_forward_backward_metrics(
        self,
        *,
        sum_reduced_metrics: dict[str, torch.Tensor],
        max_reduced_metrics: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Reduce forward/backward metrics across the loss mesh.

        Args:
            sum_reduced_metrics: Per-rank shares to be SUM-reduced. Each
                value must be pre-normalized so that summing across ranks
                reconstructs the global metric.
            max_reduced_metrics: Per-rank values to be MAX-reduced.

        Returns:
            {key: float} after collective reduction.
        """
        # TODO: switch from plain tensors to DTensor / spmd_types so the
        # reduction op is encoded in the placement instead of split across
        # `sum_reduced_metrics` / `max_reduced_metrics` dicts.
        loss_mesh = self.parallel_dims.get_optional_mesh("loss")

        out: dict[str, float] = {
            key: dist_utils.dist_sum(value.detach(), loss_mesh)
            for key, value in sum_reduced_metrics.items()
        }
        out.update(
            {
                key: dist_utils.dist_max(value.detach(), loss_mesh)
                for key, value in max_reduced_metrics.items()
            }
        )
        return out

    @endpoint
    @sl.log_trace_span("forward_backward")
    async def forward_backward(
        self,
        training_data: list[TrainingMicrobatch],
        num_global_valid_tokens: int,
    ) -> dict[str, float]:
        """Run forward pass, compute loss, call backward, and reduce metrics.

        Args:
            training_data: List of TrainingMicrobatch, one per DP rank. Local rank
                picks training_data[self.dp_rank].
            num_global_valid_tokens: Total response tokens across all DP
                ranks for this step. The controller computes this before
                sharding training_samples.

        Returns:
            dict[str, float]: Globally-reduced metrics.
        """
        logger.debug(
            f"{os.getpid()=} PolicyTrainer forward_backward "
            f"step {self.policy_version}"
        )

        # RL does not support pipeline parallelism yet, so the trainer
        # owns one model part.
        if len(self.model_parts) != 1:
            raise ValueError(
                f"PolicyTrainer expects exactly one model part, got "
                f"{len(self.model_parts)} (pipeline parallelism is not yet "
                "supported in RL)."
            )
        model = self.model_parts[0]

        local_batch = training_data[self.dp_rank]
        device = self.device
        token_ids = local_batch.token_ids.to(device)
        labels = local_batch.labels.to(device)
        positions = local_batch.positions.to(device)
        loss_mask = local_batch.loss_mask.to(device)
        generator_logprobs = local_batch.generator_logprobs.to(device)
        advantages = local_batch.advantages.to(device)

        attention_masks = model.get_attention_masks(positions)

        with self.train_context():
            with sl.log_trace_span("model_forward"):
                pred = model(
                    token_ids, attention_masks=attention_masks, positions=positions
                )

            with sl.log_trace_span("loss_fn"):
                backward_scale: float | torch.Tensor = 1.0
                if (
                    isinstance(self.loss_fn, ChunkedLossWrapper)
                    and self.grad_scaler.is_enabled()
                ):
                    # Scaling a scala also lazily initializes GradScaler's device-side tensor
                    backward_scale = self.grad_scaler.scale(
                        torch.ones((), device=device, dtype=torch.float32)
                    )
                loss_kwargs: dict[str, Any] = {
                    "generator_logprobs": generator_logprobs,
                    "advantages": advantages,
                    "loss_mask": loss_mask,
                }
                if isinstance(self.loss_fn, ChunkedLossWrapper):
                    loss_kwargs["backward_scale"] = backward_scale
                loss, loss_metrics = self.loss_fn(
                    pred,
                    labels,
                    num_global_valid_tokens,
                    **loss_kwargs,
                )

            with sl.log_trace_span("model_backward"):
                if isinstance(self.loss_fn, ChunkedLossWrapper):
                    loss.backward()
                else:
                    self.grad_scaler.scale(loss).backward()

        sum_reduced_metrics = {
            key: value
            for key, value in loss_metrics.items()
            if not key.endswith("/max")
        }
        max_reduced_metrics = {
            key: value for key, value in loss_metrics.items() if key.endswith("/max")
        }

        return self.reduce_forward_backward_metrics(
            sum_reduced_metrics=sum_reduced_metrics,
            max_reduced_metrics=max_reduced_metrics,
        )

    def _step_optimizer(self) -> bool:
        scale_before = self.grad_scaler.get_scale()
        self.grad_scaler.step(self.optimizers)
        self.grad_scaler.update()
        return self.grad_scaler.get_scale() < scale_before

    @endpoint
    @sl.log_trace_span("optim_step")
    async def optim_step(self) -> OptimStepOutput:
        """Clip gradients, step optimizer + LR scheduler, return updated state."""
        # TODO: Accept optional optimizer params (e.g. learning rate)
        # to allow controller-owned schedules.

        # capture LR before step
        current_lrs = self.lr_schedulers.schedulers[0].get_last_lr()
        if len(current_lrs) != 1:
            raise ValueError(
                "RL metrics only support a single optimizer LR for "
                f"trainer/lr; got {current_lrs}"
            )
        current_lr = float(current_lrs[0])

        with sl.log_trace_span("grad_clip"):
            self.grad_scaler.unscale_(self.optimizers)
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in self.model_parts for p in m.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
                ep_enabled=self.parallel_dims.ep_enabled,
            )

        with sl.log_trace_span("optim"):
            step_skipped = self._step_optimizer()
            if not step_skipped:
                self.lr_schedulers.step()
            # The controller and checkpoint folders count training iterations.
            # A skipped AMP update still consumes one iteration and weight sync.
            self.policy_version += 1
            self.optimizers.zero_grad()

        logger.debug(
            f"{os.getpid()=} PolicyTrainer optim_step done, "
            f"policy_version={self.policy_version}"
        )

        return OptimStepOutput(
            policy_version=self.policy_version,
            metrics={
                "trainer/grad_norm/mean": float(grad_norm.item()),
                "trainer/lr": current_lr,
                "trainer/policy_version": float(self.policy_version),
                "trainer/amp/grad_scale": float(self.grad_scaler.get_scale()),
                "trainer/amp/step_skipped": float(step_skipped),
            },
        )

    @endpoint
    @sl.log_trace_span("save_checkpoint")
    async def save_checkpoint(self, step: int, last_step: bool = False) -> bool:
        """Save checkpoint via CheckpointManager.

        Args:
            step: Current training step number.
            last_step: Whether this is the final step of training.

        Returns:
            True if a checkpoint was saved.
        """
        return self.checkpointer.save(step, last_step=last_step)

    @endpoint
    @sl.log_trace_span("push_model_state_dict")
    async def push_model_state_dict(self) -> None:
        """Stage model weights to a CPU StorageVolume for the generators to pull (TorchStore).

        `direct_rdma=False` copies the state dict GPU->CPU, so the trainer's GPU weights are free once
        this returns and any number of generators can read the staged copy.
        """
        state_dict = self.model.state_dict()
        if self._transfer_dtype is not None:
            # torchstore only applies `transfer_dtype` on the RDMA path, so under direct_rdma=False
            # cast to the generator dtype here (else the generator reads fp32 into its bf16 state dict).
            # Exclude buffers from the cast: FSDP mixed precision casts params to the compute dtype but
            # leaves buffers at their registered dtype (same as pretraining), e.g. the fp32
            # expert_bias_E load-balance bias in MoE. The generator keeps those buffers at the same
            # registered dtype, so casting them here would mismatch its state dict and fail torchstore's
            # dtype check on weight sync.
            # Strip the AC wrapper's `_checkpoint_wrapped_module` segment so buffer FQNs match state_dict() keys.
            # TODO(async-rl): remove this manual cast once torchstore applies transfer_dtype on the
            #   CPU-staged path.
            buffer_names = {
                canonical_fqn(name) for name, _ in self.model.named_buffers()
            }
            state_dict = {
                name: (
                    tensor if name in buffer_names else tensor.to(self._transfer_dtype)
                )
                for name, tensor in state_dict.items()
            }

        await ts.put_state_dict(
            state_dict,
            "model_state_dict",
            direct_rdma=False,
        )
