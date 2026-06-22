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
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
import torch.nn.functional as F
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from torch.utils.checkpoint import checkpoint

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    CommConfig,
    CompileConfig,
    Configurable,
    DebugConfig,
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
from torchtitan.experiments.rl.types import OptimStepOutput, TrainingBatch
from torchtitan.models.common.attention import FlexAttention
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger

logger = logging.getLogger(__name__)


# Row chunk for compute_logprobs' cross-entropy. The fp32 upcast + log_softmax
# transient scales with rows * vocab and is saved for backward; chunking + recompute
# bounds it to this many rows * vocab at a time (2048 * 151936 * 4B ~ 1.2GB).
_LOGPROB_CHUNK_ROWS = 2048


@sl.log_trace_span("compute_logprobs")
def compute_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-position logprobs from logits and pre-shifted labels.

    ``labels`` is pre-shifted per episode in the batcher
    (``labels[i] = raw_token_ids[i+1]``), matching the pre-training
    dataloader convention.  No internal shift is needed.
    Output shape matches input: ``[batch, seq_len]``.

    The cross-entropy is computed in row chunks, each recomputed in the backward
    (activation checkpointing) rather than saving the fp32 upcast + log_softmax
    for the whole sequence. That transient scales with rows * vocab and is saved
    for backward; at a long seq_len and a 150K+ vocab (e.g. Qwen3 V=151936,
    S=24576 -> ~15GB saved + ~15GB its gradient) it OOMs the 32B/FSDP-8 backward.
    cross_entropy(reduction="none") is independent per row, so chunking + recompute
    is numerically identical.
    """
    from torch.distributed.tensor import DTensor, Replicate, Shard

    if isinstance(logits, DTensor):
        # TODO: pass `grad_placements=[Replicate(), ...]` to make the autograd
        # contract explicit (see .claude/rules/distributed.md).
        # Gather vocab-sharded TP logits before computing per-token logprobs.
        placements = tuple(
            (
                Replicate()
                if isinstance(p, Shard) and p.dim in (-1, logits.ndim - 1)
                else p
            )
            for p in logits.placements
        )
        logits = logits.redistribute(placements=placements).to_local()

    B, S, V = logits.shape
    logits_flat = logits.reshape(B * S, V)
    labels_flat = labels.reshape(B * S)

    def _row_logprobs(lg: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
        return -F.cross_entropy(
            lg.float(), lb, reduction="none", ignore_index=IGNORE_INDEX
        )

    rows = B * S
    if rows <= _LOGPROB_CHUNK_ROWS:
        logprobs = _row_logprobs(logits_flat, labels_flat)
    else:
        logprobs = torch.cat(
            [
                checkpoint(
                    _row_logprobs,
                    logits_flat[i : i + _LOGPROB_CHUNK_ROWS],
                    labels_flat[i : i + _LOGPROB_CHUNK_ROWS],
                    use_reentrant=False,
                )
                for i in range(0, rows, _LOGPROB_CHUNK_ROWS)
            ],
            dim=0,
        )
    return logprobs.reshape(B, S)


@dataclass(frozen=True, slots=True)
class PartialLogprobDrift:
    """Per-rank generator-vs-trainer logprob drift awaiting reduction across the loss-mesh."""

    logprob_diff_mean: torch.Tensor
    logprob_diff_max: torch.Tensor
    ratio_tokens_different: torch.Tensor


@torch.no_grad()
@sl.log_trace_span("verify_logprob_identity")
def verify_logprob_identity(
    generator_logprobs: torch.Tensor,
    trainer_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    num_global_valid_tokens: int,
) -> PartialLogprobDrift:
    """Compute per-rank drift between generator and trainer logprobs.

    Args:
        generator_logprobs: [B, L] generator logprobs from TrainingBatch.
        trainer_logprobs: [B, L] trainer-computed logprobs.
        loss_mask: [B, L] bool mask; True for response tokens.
        num_global_valid_tokens: Total response tokens across all DP ranks.

    Returns:
        PartialLogprobDrift.
    """
    ref_flat = generator_logprobs[loss_mask].float()
    policy_flat = trainer_logprobs[loss_mask].float()

    if ref_flat.numel() == 0:
        zero = torch.zeros((), dtype=torch.float32, device=generator_logprobs.device)
        return PartialLogprobDrift(zero, zero, zero)

    denom = max(num_global_valid_tokens, 1)
    diff = policy_flat - ref_flat
    return PartialLogprobDrift(
        logprob_diff_mean=diff.sum() / denom,
        logprob_diff_max=diff.abs().max(),
        ratio_tokens_different=(diff.abs() > 1e-6).sum() / denom,
    )


class PolicyTrainer(Actor, Configurable):
    """Updates policy based on collected Episode using TorchTitan components.

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
        loss: Configurable.Config = field(default_factory=Configurable.Config)
        ac_config: ActivationCheckpointingConfig = field(
            default_factory=SelectiveAC.Config
        )
        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        dump_folder: str = ""
        """Folder for AC debug dumps when using memory_budget mode."""

        weight_sync_direct_rdma: bool | None = None
        """Weight-sync transport for ``push_model_state_dict``: ``None`` uses
        ``is_rdma_available()``; ``True``/``False`` force it on/off. Must match the
        generator's ``weight_sync_direct_rdma`` (CPU-staged on both, or direct-RDMA on
        both). Set ``False`` with >1 generator (fanout-safe, avoids the GPU memory spike)."""

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

        with sl.log_trace_span("torch_distributed_init"):
            world_size = dist_utils.init_distributed(
                config.comm,
                base_folder=output_dir,
            )

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
        model = self._build_model(model_spec, config, device_type)
        model.train()
        self.model = model
        self.model_parts = [model]

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
        return {"policy_version": self.policy_version}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.policy_version = state_dict["policy_version"]

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

        inner_attn = model_spec.model.layers[0].attention.inner_attention
        assert isinstance(
            model_spec.model.layers[0].attention.inner_attention,
            (VarlenAttention.Config, FlexAttention.Config),
        ), "Only varlen and flex attention backends are allowed."

        # Fill sharding configs on the config BEFORE build via the
        # model-agnostic `update_from_config` hook (RL's trainer bypasses
        # `torchtitan.Trainer's` call, so we invoke it directly).
        model_spec.model.update_from_config(config=config)

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

        out: dict[str, float] = {}
        for values_by_key, op in [
            (sum_reduced_metrics, c10d.ReduceOp.SUM),
            (max_reduced_metrics, c10d.ReduceOp.MAX),
        ]:
            if not values_by_key:
                continue
            keys = list(values_by_key)
            stacked = torch.stack([values_by_key[key].detach() for key in keys])
            if loss_mesh is not None:
                stacked = funcol.all_reduce(stacked, reduceOp=op.name, group=loss_mesh)
            for key, value in zip(keys, stacked.cpu().tolist(), strict=True):
                out[key] = float(value)
        return out

    @endpoint
    @sl.log_trace_span("forward_backward")
    async def forward_backward(
        self,
        training_data: list[TrainingBatch],
        num_global_valid_tokens: int,
    ) -> dict[str, float]:
        """Run forward pass, compute loss, call backward, and reduce metrics.

        Args:
            training_data: List of TrainingBatch, one per DP rank. Local rank
                picks training_data[self.dp_rank].
            num_global_valid_tokens: Total response tokens across all DP
                ranks for this step. The controller computes this before
                sharding episodes.

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

        with sl.log_trace_span("model_forward"):
            logits = model(
                token_ids, attention_masks=attention_masks, positions=positions
            )
        trainer_logprobs = compute_logprobs(logits, labels)

        with sl.log_trace_span("loss_fn"):
            loss, loss_metrics = self.loss_fn(
                trainer_logprobs=trainer_logprobs,
                generator_logprobs=generator_logprobs,
                loss_mask=loss_mask,
                advantages=advantages,
                num_global_valid_tokens=num_global_valid_tokens,
            )

        with sl.log_trace_span("model_backward"):
            loss.backward()

        # Metrics for bitwise verification of policy logprobs.
        verification: PartialLogprobDrift = verify_logprob_identity(
            generator_logprobs=generator_logprobs,
            trainer_logprobs=trainer_logprobs,
            loss_mask=loss_mask,
            num_global_valid_tokens=num_global_valid_tokens,
        )

        # Per-rank pre-normalized metrics, so SUM-reducing reconstructs the global.
        sum_reduced_metrics = {
            **loss_metrics,
            "bit_wise/logprob_diff/mean": verification.logprob_diff_mean,
            "bit_wise/ratio_tokens_different/mean": verification.ratio_tokens_different,
        }
        max_reduced_metrics = {
            "bit_wise/logprob_diff/max": verification.logprob_diff_max,
        }

        return self.reduce_forward_backward_metrics(
            sum_reduced_metrics=sum_reduced_metrics,
            max_reduced_metrics=max_reduced_metrics,
        )

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
                f"train/lr; got {current_lrs}"
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
                "train/grad_norm/mean": float(grad_norm.item()),
                "train/lr": current_lr,
                "train/policy_version": float(self.policy_version),
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
        """Publish model weights for generator consumption via TorchStore.

        When `direct_rdma=True`, weights are transferred directly from
        GPU to GPU via one-sided RDMA reads, bypassing StorageVolumes
        entirely. When `False`, data goes through StorageVolumes
        (which may themselves use RDMA as a transport internally).

        Note: we couple `is_rdma_available()` with `direct_rdma` here,
        but the two concepts are not identical -- StorageVolumes can also
        use RDMA as their transport layer. `direct_rdma` specifically
        means "skip StorageVolumes and let the destination read directly
        from the source's GPU memory".

        """
        from monarch.rdma import is_rdma_available

        cfg_rdma = self.config.weight_sync_direct_rdma
        direct_rdma = is_rdma_available() if cfg_rdma is None else cfg_rdma
        state_dict = self.model.state_dict()
        # torchstore honors ``transfer_dtype`` only on the direct-RDMA path
        # (state_dict_utils._put_state_dict_direct_rdma); the CPU-staged
        # StorageVolume path ignores it. Under mixed-precision FSDP the master
        # weights are fp32, so on the CPU-staged path -- which >1 generator
        # requires (direct-RDMA does not fan out) -- cast to the generator dtype
        # here, else the generator's pull asserts a dtype mismatch (e.g.
        # float32 != bfloat16). For direct-RDMA leave it to transfer_dtype, whose
        # staging buffer avoids an extra full-precision copy.
        if self._transfer_dtype is not None and not direct_rdma:
            state_dict = {
                name: (
                    tensor.to(self._transfer_dtype)
                    if tensor.dtype.is_floating_point
                    else tensor
                )
                for name, tensor in state_dict.items()
            }
        await ts.put_state_dict(
            state_dict,
            "model_state_dict",
            direct_rdma=direct_rdma,
            transfer_dtype=self._transfer_dtype,
        )
