# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import copy
import logging
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torchstore as ts
from torch.distributed.tensor import DTensor
from torchstore import RankRole

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.config import apply_overrides, Configurable, TORCH_DTYPE_MAP
from torchtitan.config.validation import validate_model_training_config
from torchtitan.distributed import maybe_apply_numa_binding, utils as dist_utils
from torchtitan.distributed.utils import get_local_tensor
from torchtitan.models.common.aux_loss import collect_aux_loss_metrics
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.observability.metrics import compute_training_performance_metrics
from torchtitan.protocols.model import BaseModel
from torchtitan.rl.components.checkpointer import CONTROLLER_STATE_KEY, MirroredState
from torchtitan.rl.observability.controller import combine_microbatch_metrics
from torchtitan.rl.types import OptimizerStepOutput, TrainingMicrobatch
from torchtitan.tools import utils
from torchtitan.training_engine import TrainingEngine

logger = logging.getLogger(__name__)


class Trainer(Configurable):
    """Updates policy based on collected TrainingSample using TorchTitan components.

    Exposes separate `forward_backward` and `optim_step` endpoints, called
    explicitly by the controller.

    Args:
        config: Trainer.Config with all model/optimizer/parallelism settings.
        model_config: TorchTitan model configuration.
        max_num_documents: Fixed varlen metadata capacity configured by the batcher.
        hf_assets_path: Path to HF assets folder for checkpoint loading.
            Shared with the generator (both load from the same HF checkpoint).
        generator_dtype: Generator dtype (e.g. "bfloat16"). Needed to cast weights to generator dtype
            if generator dtype differs from training dtype. If None, no cast is performed.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TrainingEngine.Config):
        """Trainer configuration for optimizer, training, and parallelism."""

        def __post_init__(self) -> None:
            TrainingEngine.Config.__post_init__(self)
            if self.parallelism.pipeline_parallel_degree > 1:
                raise ValueError(
                    "RL pipeline parallelism is temporarily disabled because "
                    "TorchStore cannot publish a complete model state from "
                    "stage-local state dictionaries."
                )

    def __init__(
        self,
        config: Config,
        *,
        model_config: BaseModel.Config,
        max_num_documents: int | None,
        hf_assets_path: str = "",
        generator_dtype: str = "",
        output_dir: str,
    ):
        init_logger()
        # Quiet torchstore's per-op transport-resolve INFO spam (very noisy in CI).
        logging.getLogger("torchstore.transport").setLevel(logging.WARNING)
        sl.init_structured_logger(
            source="rl_trainer",
            output_dir=output_dir,
            rank=int(os.environ.get("RANK", "0")),
            enable=config.debug.enable_structured_logging,
        )
        sl.log_trace_instant("structured_logger_started")

        self.config = config
        model_config = copy.deepcopy(model_config)
        model_config.set_sharding_(config.parallelism)

        if config.override.imports:
            apply_overrides(config.override, model_config)

        validate_model_training_config(
            model_config,
            parallelism=config.parallelism,
            training=config.training,
            debug=config.debug,
            activation_checkpoint=config.activation_checkpoint,
            max_num_documents=max_num_documents,
        )

        self.engine = TrainingEngine(
            config,
            model_config=model_config,
            max_num_documents=max_num_documents,
            output_dir=output_dir,
        )
        engine = self.engine
        self._controller_state = MirroredState()
        maybe_apply_numa_binding(engine.device.index, engine.device.type)

        # Only cast when the generator and training dtypes differ. A CPU
        # staging snapshot is still required when no cast is needed.
        training_dtype = TORCH_DTYPE_MAP[config.training.dtype]
        gen_dtype = TORCH_DTYPE_MAP[generator_dtype] if generator_dtype else None
        self._transfer_dtype = gen_dtype if gen_dtype != training_dtype else None

        self.gpu_peak_flops = utils.get_peak_flops(
            engine.device_memory_monitor.device_name
        )
        engine.initialize(
            hf_assets_path=hf_assets_path,
            extra_states={CONTROLLER_STATE_KEY: self._controller_state},
        )

        logger.info(f"Peak FLOPS used for computing MFU: {self.gpu_peak_flops:.3e}")
        logger.info(
            f"{engine.device.type.upper()} memory usage for model: "
            f"{engine.model_device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({engine.model_device_mem_stats.max_reserved_pct:.2f}%)"
        )
        self.model = engine.model_parts[0]

        engine.load_checkpoint()
        self._staging_state_dict = self._setup_staging_model_state_dict()
        self._staging_device_module = torch.get_device_module(engine.device)
        if engine.device.type == "cpu":
            self._staging_model_state_dict_stream = None
        else:
            self._staging_model_state_dict_stream = self._staging_device_module.Stream(
                device=engine.device
            )
            self._mark_model_state_dict_ready_for_staging()
        if config.checkpointer is None:
            logger.warning(
                "Checkpoint disabled, skip weight loading and use random-initialized weights. "
                "Configure checkpointer to load from a checkpoint."
            )

        engine.start_profiler()
        engine.device_memory_monitor.reset_peak_stats()

        # Data parallelism: mesh is available after model construction builds it.
        self.dp_enabled = engine.parallelism_context.dp_enabled
        dp_mesh = engine.parallelism_context.get_optional_mesh("dp")
        if dp_mesh is not None:
            self.dp_size = dp_mesh.size()
            self.dp_rank = dp_mesh.get_local_rank()
        else:
            self.dp_size = 1
            self.dp_rank = 0

    @property
    def policy_version(self) -> int:
        """Number of completed optimizer steps, restored with engine state."""
        return self.engine.num_completed_steps

    async def get_controller_state(self) -> dict[str, Any] | None:
        """Return controller state restored with the trainer checkpoint."""
        return self._controller_state.loaded

    def _mark_model_state_dict_ready_for_staging(self) -> None:
        """Order the staging stream after the latest model-weight updates."""
        if self._staging_model_state_dict_stream is None:
            return

        # Capture the dependency when weights become final: staging must wait
        # for optimizer updates, but not for next-step forward/backward kernels
        # that may be queued before the concurrent push starts.
        self._staging_model_state_dict_stream.wait_stream(
            self._staging_device_module.current_stream(self.engine.device)
        )

    async def close(self) -> None:
        """Close actor-local resources before the process mesh stops.

        The trainer does not own the distributed process group lifecycle here:
        Monarch created it for the actor mesh, and ``ProcMesh.stop()`` performs
        the final teardown. Destroying it from this endpoint can race with mesh
        shutdown and hang at process exit.
        """
        self.engine.close()
        logger.debug("Trainer close requested; ProcMesh.stop owns PG teardown.")

    async def sync_log_step(self, step: int, relative_step: int | None = None) -> None:
        """Sync the structured-logger step counter from the controller."""
        sl.set_step(step, relative_step=relative_step)

    def _reduce_forward_backward_metrics(
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
        loss_mesh = self.engine.parallelism_context.get_optional_mesh("loss")

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

    @sl.log_trace_span("forward_backward")
    async def forward_backward(
        self,
        training_data: list[list[TrainingMicrobatch]],
        global_loss_token_counts: torch.Tensor,
        global_routing_token_counts: torch.Tensor,
    ) -> dict[str, float]:
        """Run one optimizer step's forward/backward microbatches.

        Args:
            training_data: Microbatch-major grid with shape
                ``[num_microbatches][dp_degree]``.
            global_loss_token_counts: Per-objective loss-token counts across the
                global batch.
            global_routing_token_counts: Per-depth non-padding routing-token
                counts across the global batch.

        Returns:
            dict[str, float]: Globally-reduced metrics.
        """
        logger.debug(
            f"{os.getpid()=} Trainer forward_backward "
            f"policy_version={self.policy_version}"
        )
        engine = self.engine
        self._step_compute_start = time.perf_counter()
        self._step_num_tokens_per_dp_rank = sum(
            rank_batches[self.dp_rank].labels.numel() for rank_batches in training_data
        )
        result = engine.forward_backward(
            microbatch_groups=[
                [rank_batches[self.dp_rank]] for rank_batches in training_data
            ],
            global_loss_token_counts=global_loss_token_counts,
            global_routing_token_counts=global_routing_token_counts,
        )
        microbatch_metrics: list[dict[str, float]] = []
        for loss_metrics in result.loss_metrics:
            microbatch_metrics.append(
                self._reduce_forward_backward_metrics(
                    sum_reduced_metrics={
                        key: value
                        for key, value in loss_metrics.items()
                        if not key.endswith("/max")
                    },
                    max_reduced_metrics={
                        key: value
                        for key, value in loss_metrics.items()
                        if key.endswith("/max")
                    },
                )
            )

        return combine_microbatch_metrics(microbatch_metrics)

    @sl.log_trace_span("optim_step")
    async def optim_step(
        self, *, controller_state: dict[str, Any], last_step: bool = False
    ) -> OptimizerStepOutput:
        """Clip gradients, step optimizer + LR scheduler, return updated state."""
        # TODO: Accept optional optimizer params (e.g. learning rate)
        # to allow controller-owned schedules.

        engine = self.engine
        # Capture the learning rates used by this optimizer update before the
        # scheduler advances in engine.optim_step().
        lr_metrics = engine.optim.lr_schedulers.get_metrics()

        grad_norm = engine.optim_step()
        self._mark_model_state_dict_ready_for_staging()
        self._controller_state.value = controller_state

        # TODO: Move performance, LR, and auxiliary-loss reporting into a shared
        # trainer metrics interface while preserving controller-side aggregation.
        performance = compute_training_performance_metrics(
            num_tokens=self._step_num_tokens_per_dp_rank,
            elapsed_time=time.perf_counter() - self._step_compute_start,
            non_data_parallel_size=engine.parallelism_context.non_data_parallel_size,
            num_flops_per_token=engine.num_flops_per_token,
            gpu_peak_flops=self.gpu_peak_flops,
            has_quantization=engine.has_quantization,
        )

        engine.save_checkpoint(last_step=last_step)
        engine.step_profiler()
        device_mem_stats = engine.device_memory_monitor.get_peak_stats()
        engine.device_memory_monitor.reset_peak_stats()

        logger.debug(
            f"{os.getpid()=} Trainer optim_step done, "
            f"policy_version={self.policy_version}"
        )

        return OptimizerStepOutput(
            policy_version=self.policy_version,
            metrics={
                "trainer/grad_norm/mean": float(grad_norm.item()),
                **{f"trainer/{key}": value for key, value in lr_metrics.items()},
                "trainer/policy_version": float(self.policy_version),
                "trainer/tflops": performance["tflops"],
                "trainer/memory/max_active_gib": device_mem_stats.max_active_gib,
                "trainer/memory/max_active_percent": device_mem_stats.max_active_pct,
                "trainer/memory/max_reserved_gib": device_mem_stats.max_reserved_gib,
                "trainer/memory/max_reserved_percent": (
                    device_mem_stats.max_reserved_pct
                ),
                "trainer/memory/num_alloc_retries": float(
                    device_mem_stats.num_alloc_retries
                ),
                "trainer/memory/num_ooms": float(device_mem_stats.num_ooms),
                **collect_aux_loss_metrics(engine.parallelism_context),
                **(
                    {"trainer/mfu_percent": performance["mfu_percent"]}
                    if "mfu_percent" in performance
                    else {}
                ),
            },
        )

    @sl.log_trace_span("push_model_state_dict")
    async def push_model_state_dict(self) -> None:
        """Publish a CPU snapshot of model weights for generators to pull."""

        model_state_dict = self.model.state_dict()
        # Today we copy the model into reusable pinned CPU buffers before calling
        # TorchStore. Parameter destinations use the generator dtype, so copy_()
        # performs the cast; buffers keep their registered dtype.
        # TODO(async-rl): add a TorchStore-owned staging API so the trainer can
        # copy directly into the StorageVolume, eliminating the duplicate CPU
        # snapshot while preserving this overlap and making buffer lifetime
        # explicit.
        staging_complete = await self._stage_model_state_dict(model_state_dict)
        await asyncio.to_thread(
            self._wait_for_staging_and_put_model_state_dict,
            staging_complete,
        )

    @sl.log_trace_span("stage_model_state_dict")
    async def _stage_model_state_dict(
        self, model_state_dict: dict[str, torch.Tensor]
    ) -> torch.Event | None:
        """Enqueue model-state copies into pinned CPU buffers."""
        stream = self._staging_model_state_dict_stream
        stream_context = (
            self._staging_device_module.stream(stream)
            if stream is not None
            else nullcontext()
        )
        staging_complete: torch.Event | None = None
        with stream_context:
            for name, destination in self._staging_state_dict.items():
                source = model_state_dict[name]
                get_local_tensor(destination).copy_(
                    get_local_tensor(source), non_blocking=stream is not None
                )
            if stream is not None:
                staging_complete = self._staging_device_module.Event()
                staging_complete.record(stream)

        return staging_complete

    def _wait_for_staging_and_put_model_state_dict(
        self, staging_complete: torch.Event | None
    ) -> None:
        """Wait for staging and publish without blocking the actor event loop."""
        if staging_complete is not None:
            staging_complete.synchronize()
        asyncio.run(
            ts.put_state_dict(
                self._staging_state_dict,
                "model_state_dict",
                direct_rdma=False,
            )
        )

    def _setup_staging_model_state_dict(self) -> dict[str, torch.Tensor]:
        """Allocate a final-dtype pinned CPU mirror of the model state dict."""
        buffer_names = {canonical_fqn(name) for name, _ in self.model.named_buffers()}
        model_state_dict = self.model.state_dict()
        staging_state_dict = {}
        for name, tensor in model_state_dict.items():
            dtype = (
                self._transfer_dtype
                if self._transfer_dtype is not None and name not in buffer_names
                else tensor.dtype
            )
            staging_tensor = torch.empty_like(
                tensor,
                dtype=dtype,
                device="cpu",
                pin_memory=not isinstance(tensor, DTensor),
            )
            if isinstance(staging_tensor, DTensor):
                # DTensor does not support pin_memory(), so pin its local shard.
                staging_tensor._local_tensor = torch.empty_like(
                    get_local_tensor(staging_tensor), pin_memory=True
                )
            staging_state_dict[name] = staging_tensor
        return staging_state_dict

    async def initialize_torchstore_client(self) -> None:
        """Initialize this process as a TorchStore routing publisher."""
        await ts.client(role=RankRole.PUBLISHER)
