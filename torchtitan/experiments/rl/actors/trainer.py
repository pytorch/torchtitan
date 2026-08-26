# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torchstore as ts
from monarch.actor import Actor, concurrent_endpoint, current_rank

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.components.loss import BaseLoss, ChunkedLossWrapper
from torchtitan.components.optimizer import LRSchedulersContainer, OptimizersContainer
from torchtitan.config import (
    TORCH_DTYPE_MAP,
    CommConfig,
    CompileConfig,
    Configurable,
    DebugConfig,
    OverrideConfig,
    ParallelismConfig,
    TrainingConfig,
    apply_overrides,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointingConfig,
    SelectiveAC,
)
from torchtitan.distributed.utils import set_batch_invariance
from torchtitan.experiments.rl.losses import GRPOLoss
from torchtitan.experiments.rl.models.native_vllm_qwen3_5 import (
    qwen35_text_state_dict,
)
from torchtitan.experiments.rl.types import OptimStepOutput, TrainingMicrobatch
from torchtitan.models.common.attention import FlexAttention
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger

logger = logging.getLogger(__name__)


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
        enable_post_update_metrics: bool = False
        """Run an extra fixed-batch forward after each optimizer step.

        This emits selected-token post-update policy-vs-behavior metrics for
        correctness comparisons. It is disabled by default because the extra
        forward pass changes performance measurements.
        """
        enable_kl_artifact_logging: bool = False
        """Save fixed-batch full-vocabulary logits before and after each step."""
        kl_artifact_max_tokens_per_rank: int = 64
        """Maximum exact-KL token positions saved by each trainer rank."""

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        compile_config: CompileConfig,
        hf_assets_path: str = "",
        generator_dtype: str = "",
        native_vllm_generator: bool = False,
        output_dir: str,
    ):
        init_logger()
        # Quiet torchstore's per-op transport-resolve INFO spam (very noisy in CI).
        logging.getLogger("torchstore.transport").setLevel(logging.WARNING)
        if not config.dump_folder:
            config.dump_folder = output_dir
        actor_rank = current_rank().rank
        sl.init_structured_logger(
            source="rl_trainer",
            output_dir=output_dir,
            rank=actor_rank,
            enable=config.debug.enable_structured_logging,
        )
        sl.log_trace_instant("structured_logger_started")

        self.config = config
        self.actor_rank = actor_rank
        self.output_dir = Path(output_dir)
        self.model_name = model_spec.name
        self.model_flavor = model_spec.flavor
        self.hf_assets_path = hf_assets_path
        self.compile_config = compile_config
        self.loss_fn = config.loss.build()
        # TODO: add support to compile the loss.

        # Only cast if generator dtype differs from training dtype, otherwise
        # staging buffers would be allocated for a no-op cast.
        training_dtype = TORCH_DTYPE_MAP[config.training.dtype]
        gen_dtype = TORCH_DTYPE_MAP[generator_dtype] if generator_dtype else None
        self._transfer_dtype = gen_dtype if gen_dtype != training_dtype else None
        self._native_vllm_generator = native_vllm_generator
        if native_vllm_generator and model_spec.name != "qwen3_5":
            raise ValueError(
                "native_vllm_generator currently supports only qwen3_5, got "
                f"{model_spec.name!r}"
            )

        # Device setup
        device_module, device_type = utils.device_module, utils.device_type
        self.device = utils.get_local_device()
        device_module.set_device(self.device)

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
        # Checkpoint "train_state": policy_version == completed optim steps, so it
        # doubles as the resume step counter.
        return {"policy_version": self.policy_version}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.policy_version = state_dict["policy_version"]

    @concurrent_endpoint
    async def get_policy_version(self) -> int:
        """Current policy version: after load(), the step a resume restored from
        (0 if fresh). The controller uses it to resume and re-sync generators."""
        return self.policy_version

    @concurrent_endpoint
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

        attention_backend = model_spec.model.first_full_attention_backend
        assert isinstance(
            attention_backend,
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

    @concurrent_endpoint
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

    @concurrent_endpoint
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
            num_global_valid_tokens: Total response tokens with finite generator
                logprobs across all DP ranks and microbatches for this step.

        Returns:
            dict[str, float]: Globally-reduced metrics.
        """
        logger.debug(
            f"{os.getpid()=} PolicyTrainer forward_backward step {self.policy_version}"
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
                loss, loss_metrics = self.loss_fn(
                    pred,
                    labels,
                    num_global_valid_tokens,
                    generator_logprobs=generator_logprobs,
                    advantages=advantages,
                    loss_mask=loss_mask,
                )

            with sl.log_trace_span("model_backward"):
                loss.backward()

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

    @concurrent_endpoint
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
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in self.model_parts for p in m.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
                ep_enabled=self.parallel_dims.ep_enabled,
            )

        with sl.log_trace_span("optim"):
            self.optimizers.step()
            self.lr_schedulers.step()
            self.optimizers.zero_grad()

        self.policy_version += 1

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
            },
        )

    @concurrent_endpoint
    @sl.log_trace_span("post_update_metrics")
    async def post_update_metrics(
        self,
        training_data: list[TrainingMicrobatch],
        num_global_valid_tokens: int,
    ) -> dict[str, float]:
        """Evaluate the updated policy on the same fixed batch used for the step."""
        metrics, _ = self._comparison_snapshot(
            training_data=training_data,
            num_global_valid_tokens=num_global_valid_tokens,
            include_artifact=False,
        )
        return self._reduce_comparison_metrics(metrics)

    def _comparison_snapshot(
        self,
        *,
        training_data: list[TrainingMicrobatch],
        num_global_valid_tokens: int,
        include_artifact: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
        """Evaluate the current policy and optionally materialize a KL artifact."""
        if len(self.model_parts) != 1:
            raise ValueError(
                "PolicyTrainer expects exactly one model part for post-update metrics"
            )

        model = self.model_parts[0]
        local_batch = training_data[self.dp_rank]
        token_ids = local_batch.token_ids.to(self.device)
        labels = local_batch.labels.to(self.device)
        positions = local_batch.positions.to(self.device)
        loss_mask = local_batch.loss_mask.to(self.device)
        generator_logprobs = local_batch.generator_logprobs.to(self.device)
        attention_masks = model.get_attention_masks(positions)

        with torch.no_grad(), self.train_context():
            pred = model(
                token_ids, attention_masks=attention_masks, positions=positions
            )
            # Reuse the configured loss path for the all-token sampled-logprob
            # metrics. Materializing full-vocabulary logits for every rollout
            # token would make a production artifact tens of gigabytes.
            _, loss_metrics = self.loss_fn(
                pred,
                labels,
                num_global_valid_tokens,
                generator_logprobs=generator_logprobs,
                advantages=torch.zeros_like(generator_logprobs),
                loss_mask=loss_mask,
            )

        pre_update_prefix = "comparison/correctness/pre_update/"
        post_update_prefix = "comparison/correctness/post_update/"
        metrics = {
            key.replace(pre_update_prefix, post_update_prefix, 1): value
            for key, value in loss_metrics.items()
            if key.startswith(pre_update_prefix)
        }
        artifact = None
        if include_artifact:
            effective_flat_indices = (
                (loss_mask & torch.isfinite(generator_logprobs))
                .flatten()
                .nonzero(as_tuple=False)
                .flatten()
            )
            max_tokens = self.config.kl_artifact_max_tokens_per_rank
            if max_tokens <= 0:
                raise ValueError(
                    "kl_artifact_max_tokens_per_rank must be greater than zero"
                )
            if effective_flat_indices.numel() > max_tokens:
                sample_offsets = (
                    torch.linspace(
                        0,
                        effective_flat_indices.numel() - 1,
                        steps=max_tokens,
                        device=effective_flat_indices.device,
                    )
                    .round()
                    .long()
                )
                selected_flat_indices = effective_flat_indices[sample_offsets]
            else:
                selected_flat_indices = effective_flat_indices
            artifact_mask = torch.zeros_like(loss_mask, dtype=torch.bool).flatten()
            artifact_mask[selected_flat_indices] = True
            artifact_mask = artifact_mask.view_as(loss_mask)

            with torch.no_grad(), self.train_context():
                if isinstance(self.loss_fn, ChunkedLossWrapper):
                    selected_logits = self.loss_fn.compute_selected_logits(
                        pred, artifact_mask
                    )
                else:
                    selected_logits = pred[artifact_mask]
            artifact = {
                "format_version": 1,
                "framework": "torchtitan",
                "trainer_rank": self.actor_rank,
                "policy_version": self.policy_version,
                "model_name": self.model_name,
                "model_flavor": self.model_flavor,
                "hf_assets_path": self.hf_assets_path,
                "num_global_valid_tokens": num_global_valid_tokens,
                "token_ids": local_batch.token_ids.cpu(),
                "labels": local_batch.labels.cpu(),
                "positions": local_batch.positions.cpu(),
                "loss_mask": local_batch.loss_mask.cpu(),
                "generator_logprobs": local_batch.generator_logprobs.float().cpu(),
                "advantages": local_batch.advantages.float().cpu(),
                "selection_strategy": "evenly_spaced_effective_tokens_per_rank",
                "num_effective_tokens": effective_flat_indices.numel(),
                "selected_flat_indices": selected_flat_indices.cpu(),
                "selected_logits": selected_logits.float().cpu(),
            }
        return metrics, artifact

    def _reduce_comparison_metrics(
        self, metrics: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        sum_reduced_metrics = {
            key: value for key, value in metrics.items() if not key.endswith("/max")
        }
        max_reduced_metrics = {
            key: value for key, value in metrics.items() if key.endswith("/max")
        }
        return self.reduce_forward_backward_metrics(
            sum_reduced_metrics=sum_reduced_metrics,
            max_reduced_metrics=max_reduced_metrics,
        )

    @concurrent_endpoint
    @sl.log_trace_span("capture_kl_artifact")
    async def capture_kl_artifact(
        self,
        training_data: list[TrainingMicrobatch],
        num_global_valid_tokens: int,
        step: int,
        microbatch_index: int,
        phase: str,
    ) -> dict[str, float]:
        """Save full-vocabulary logits and their exact input batch for KL0/KL1."""
        if phase not in {"kl0", "kl1"}:
            raise ValueError(f"phase must be kl0 or kl1, got {phase!r}")
        metrics, artifact = self._comparison_snapshot(
            training_data=training_data,
            num_global_valid_tokens=num_global_valid_tokens,
            include_artifact=True,
        )
        assert artifact is not None
        artifact.update(
            {"step": step, "microbatch_index": microbatch_index, "phase": phase}
        )
        artifact_dir = self.output_dir / "kl_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / (
            f"step_{step:06d}_microbatch_{microbatch_index:04d}_"
            f"rank_{self.actor_rank:05d}_{phase}.pt"
        )
        temporary_path = path.with_suffix(".tmp")
        torch.save(artifact, temporary_path)
        os.replace(temporary_path, path)
        logger.info("Saved %s artifact to %s", phase.upper(), path)

        return self._reduce_comparison_metrics(metrics)

    @concurrent_endpoint
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

    @concurrent_endpoint
    @sl.log_trace_span("push_model_state_dict")
    async def push_model_state_dict(self) -> None:
        """Stage model weights to a CPU StorageVolume for the generators to pull (TorchStore).

        `direct_rdma=False` copies the state dict GPU->CPU, so the trainer's GPU weights are free once
        this returns and any number of generators can read the staged copy.
        """
        state_dict = self.model.state_dict()
        if self._native_vllm_generator:
            # Native vLLM loads the vision tower once from the original HF
            # checkpoint. SWE-rebench is text-only, so only publish the trained
            # language-model weights. Keeping TorchTitan names lets TorchStore
            # reshard directly into native packed-parameter views.
            state_dict = qwen35_text_state_dict(state_dict)
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
