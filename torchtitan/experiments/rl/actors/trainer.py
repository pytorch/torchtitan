# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Policy trainer actor.

Consumes :class:`TrainBatch` (varlen-packed with ``loss_mask``) and
calls the configured loss function. Loss is computed per token; the
mask selects loss positions (response tokens) and zeroes everything
else. Compatible with multi-turn rollouts (prefix-break-aware
:class:`ReplaySample`s) without any prompt/response boundary
arithmetic in the trainer.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    Configurable,
    DebugConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.utils import set_batch_invariance
from torchtitan.experiments.rl.actors.utils import (
    compute_logprobs,
    PartialLogprobDrift,
    verify_logprob_identity,
)
from torchtitan.experiments.rl.types import OptimStepOutput, TrainBatch
from torchtitan.models.common.attention import create_varlen_metadata_for_document
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger

logger = logging.getLogger(__name__)


class PolicyTrainer(Actor, Configurable):
    """Updates policy weights from :class:`TrainBatch`es.

    Exposes ``forward_backward`` + ``optim_step`` as separate endpoints
    so the controller can interleave weight-sync work (push to
    torchstore, signal generator to pull) between them.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
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
        """Loss function config (typically :class:`DAPOLoss.Config`)."""
        ac_config: ActivationCheckpointConfig = field(
            default_factory=lambda: ActivationCheckpointConfig(mode="none")
        )
        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
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
    ) -> None:
        init_logger()
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

        training_dtype = TORCH_DTYPE_MAP[config.training.dtype]
        gen_dtype = TORCH_DTYPE_MAP[generator_dtype] if generator_dtype else None
        self._transfer_dtype = gen_dtype if gen_dtype != training_dtype else None

        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)

        # Enable batch-invariant mode BEFORE init_distributed.
        set_batch_invariance(config.debug.batch_invariant)

        with sl.log_trace_span("torch_distributed_init"):
            world_size = dist_utils.init_distributed(config.comm)

        self.parallel_dims = ParallelDims.from_config(config.parallelism, world_size)
        dist_utils.set_determinism(
            self.parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["pp"],
        )

        if model_spec.state_dict_adapter is not None:
            self.sd_adapter = model_spec.state_dict_adapter(
                model_spec.model, hf_assets_path
            )
        else:
            self.sd_adapter = None

        model = self._build_model(model_spec, config, device_type)
        model.train()
        self.model = model
        self.model_parts = [model]

        self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )

        self.policy_version = 0

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
                "Checkpoint disabled; using random-initialized weights. "
                "Set checkpoint.enable=True to load from a checkpoint."
            )

        self.dp_enabled = self.parallel_dims.dp_enabled
        batch_mesh = self.parallel_dims.get_optional_mesh("batch")
        if batch_mesh is not None:
            self.dp_size = batch_mesh.size()
            self.dp_rank = batch_mesh.get_local_rank()
        else:
            self.dp_size = 1
            self.dp_rank = 0

        logger.debug(
            "PolicyTrainer initialized (dp_rank=%d, dp_size=%d)",
            self.dp_rank,
            self.dp_size,
        )

    def state_dict(self) -> dict[str, Any]:
        return {"policy_version": self.policy_version}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.policy_version = state_dict["policy_version"]

    @endpoint
    async def close(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    @sl.log_trace_span("build_model")
    def _build_model(
        self,
        model_spec: ModelSpec,
        config: Config,
        device_type: str,
    ):
        """Build, parallelize, and initialize a model with random weights."""
        from torchtitan.models.common.attention import VarlenAttention

        assert isinstance(
            model_spec.model.layers[0].attention.inner_attention,
            VarlenAttention.Config,
        ), "Only varlen attention backend is allowed."

        model_spec.model.update_from_config(trainer_config=config)
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

    def _reduce_forward_backward_metrics(
        self,
        *,
        sum_reduced_metrics: dict[str, torch.Tensor],
        max_reduced_metrics: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Sum- or max-reduce per-rank metrics across the loss mesh."""
        loss_mesh = self.parallel_dims.get_optional_mesh("loss")
        out: dict[str, float] = {}
        for values_by_key, op in [
            (sum_reduced_metrics, c10d.ReduceOp.SUM),
            (max_reduced_metrics, c10d.ReduceOp.MAX),
        ]:
            if not values_by_key:
                continue
            keys = list(values_by_key)
            stacked = torch.stack([values_by_key[k].detach() for k in keys])
            if loss_mesh is not None:
                stacked = funcol.all_reduce(stacked, reduceOp=op.name, group=loss_mesh)
            for key, value in zip(keys, stacked.cpu().tolist(), strict=True):
                out[key] = float(value)
        return out

    @endpoint
    @sl.log_trace_span("forward_backward")
    async def forward_backward(
        self,
        train_data: list[TrainBatch],
        *,
        num_global_valid_tokens: int,
    ) -> dict[str, float]:
        """Forward, loss, backward — all per-token + mask.

        Args:
            train_data: one :class:`TrainBatch` per DP rank.
            num_global_valid_tokens: total ``loss_mask`` sum across all
                DP ranks. Normalizes the DAPO global-token-mean loss so
                gradients are independent of how the batch is sharded.
        """
        if len(self.model_parts) != 1:
            raise ValueError(
                f"PolicyTrainer expects exactly one model part, got "
                f"{len(self.model_parts)} (pipeline parallelism not supported)."
            )
        model = self.model_parts[0]
        device = self.device

        local_batch = train_data[self.dp_rank]
        token_ids = local_batch.token_ids.to(device)  # [1, T_total]
        seq_lens = local_batch.seq_lens
        loss_mask = local_batch.loss_mask.to(device)  # [1, T-1]
        behavior_logprobs = local_batch.behavior_logprobs.to(device)  # [1, T-1]
        advantages_per_token = local_batch.advantages_per_token.to(device)

        max_seq_len = max(seq_lens) if seq_lens else 0
        rope_cache_len = self.model.freqs_cis.shape[0]
        if max_seq_len > rope_cache_len:
            raise ValueError(
                f"Sample length {max_seq_len} exceeds rope cache {rope_cache_len}. "
                "Increase model max_seq_len or reduce generation max_tokens."
            )

        num_global_valid_tokens_t = torch.tensor(
            float(max(num_global_valid_tokens, 1)),
            device=device,
            dtype=torch.float32,
        )

        # Varlen positions: per-sample arange concatenated.
        positions = torch.cat(
            [torch.arange(L, device=device) for L in seq_lens]
        ).unsqueeze(0)
        attention_masks = create_varlen_metadata_for_document(positions)

        with sl.log_trace_span("model_forward"):
            logits = model(
                token_ids, attention_masks=attention_masks, positions=positions
            )
        policy_logprobs = compute_logprobs(logits, token_ids)  # [1, T-1]

        # TODO(nan-debug): remove once AlphabetSort 4B is reliably stable.
        # Pinpoint which input is non-finite when the loss explodes.
        if (
            logger.isEnabledFor(logging.DEBUG)
            or not torch.isfinite(policy_logprobs).all()
        ):
            with torch.no_grad():
                logger.warning(
                    "nan-debug: logits finite=%s min=%s max=%s | "
                    "policy_logprobs finite=%s min=%s max=%s | "
                    "behavior_logprobs finite=%s min=%s max=%s | "
                    "advantages_per_token finite=%s min=%s max=%s | "
                    "loss_mask sum=%d | seq_lens[0:3]=%s max_seq_len=%d",
                    torch.isfinite(logits).all().item(),
                    float(logits.min()),
                    float(logits.max()),
                    torch.isfinite(policy_logprobs).all().item(),
                    float(policy_logprobs.min()),
                    float(policy_logprobs.max()),
                    torch.isfinite(behavior_logprobs).all().item(),
                    float(behavior_logprobs.min()),
                    float(behavior_logprobs.max()),
                    torch.isfinite(advantages_per_token).all().item(),
                    float(advantages_per_token.min()),
                    float(advantages_per_token.max()),
                    int(loss_mask.sum().item()),
                    seq_lens[:3],
                    max_seq_len,
                )

        with sl.log_trace_span("loss_fn"):
            loss_out = self.loss_fn(
                policy_logprobs=policy_logprobs,
                behavior_logprobs=behavior_logprobs,
                advantages_per_token=advantages_per_token,
                loss_mask=loss_mask,
                num_global_valid_tokens=num_global_valid_tokens_t,
            )

        self.optimizers.zero_grad()
        with sl.log_trace_span("model_backward"):
            loss_out.loss.backward()

        drift: PartialLogprobDrift = verify_logprob_identity(
            behavior_logprobs=behavior_logprobs,
            policy_logprobs=policy_logprobs,
            loss_mask=loss_mask,
            num_global_valid_tokens=num_global_valid_tokens_t,
        )

        # Peak memory after forward+backward — useful to size batch.
        mem_alloc_gb = torch.tensor(
            torch.cuda.max_memory_allocated(device) / 1e9,
            device=device,
            dtype=torch.float32,
        )
        mem_reserved_gb = torch.tensor(
            torch.cuda.max_memory_reserved(device) / 1e9,
            device=device,
            dtype=torch.float32,
        )
        torch.cuda.reset_peak_memory_stats(device)

        sum_reduced_metrics = {
            **loss_out.sum_metrics,
            "bit_wise/logprob_diff/mean": drift.logprob_diff_mean,
            "bit_wise/ratio_tokens_different/mean": drift.ratio_tokens_different,
        }
        max_reduced_metrics = {
            **loss_out.max_metrics,
            "bit_wise/logprob_diff/max": drift.logprob_diff_max,
            "memory/trainer/peak_allocated_gb": mem_alloc_gb,
            "memory/trainer/peak_reserved_gb": mem_reserved_gb,
        }
        return self._reduce_forward_backward_metrics(
            sum_reduced_metrics=sum_reduced_metrics,
            max_reduced_metrics=max_reduced_metrics,
        )

    @endpoint
    @sl.log_trace_span("optim_step")
    async def optim_step(self) -> OptimStepOutput:
        """Grad clip + optimizer step + LR scheduler step."""
        current_lrs = self.lr_schedulers.schedulers[0].get_last_lr()
        if len(current_lrs) != 1:
            raise ValueError(
                "RL metrics only support a single optimizer LR for train/lr; "
                f"got {current_lrs}"
            )
        current_lr = float(current_lrs[0])

        with sl.log_trace_span("grad_clip"):
            grad_norm = dist_utils.clip_grad_norm_(
                [p for mp in self.model_parts for p in mp.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
            )

        with sl.log_trace_span("optim"):
            self.optimizers.step()
            self.lr_schedulers.step()

        self.policy_version += 1
        return OptimStepOutput(
            policy_version=self.policy_version,
            metrics={
                "train/grad_norm/mean": float(grad_norm.item()),
                "train/lr": current_lr,
                "train/policy_version": float(self.policy_version),
            },
        )

    @endpoint
    @sl.log_trace_span("push_model_state_dict")
    async def push_model_state_dict(self) -> None:
        """Publish weights for generator consumption via TorchStore.

        See the matching docstring on
        :meth:`VLLMGenerator.pull_model_state_dict` for why we disable
        ``direct_rdma`` under TCP fallback; ``MONARCH_FORCE_DIRECT_RDMA=1``
        overrides.
        """
        from monarch.rdma import is_rdma_available

        tcp_fallback = (
            os.environ.get("MONARCH_RDMA_DISABLE_IBVERBS") == "1"
            and os.environ.get("MONARCH_RDMA_ALLOW_TCP_FALLBACK") == "1"
        )
        force_direct = os.environ.get("MONARCH_FORCE_DIRECT_RDMA") == "1"
        use_direct = is_rdma_available() and (not tcp_fallback or force_direct)
        await ts.put_state_dict(
            self.model.state_dict(),
            "model_state_dict",
            direct_rdma=use_direct,
            transfer_dtype=self._transfer_dtype,
        )
