# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torchstore as ts
from torchstore import RankRole

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.config.validation import validate_model_training_config
from torchtitan.distributed import maybe_apply_numa_binding, utils as dist_utils
from torchtitan.models.base.model import BaseModel
from torchtitan.models.common.aux_loss import collect_aux_loss_metrics
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.observability.metrics import compute_training_performance_metrics
from torchtitan.rl.controller.metrics import combine_microbatch_metrics
from torchtitan.rl.trainer.batcher import TrainingMicrobatch
from torchtitan.rl.trainer.controller_state import CONTROLLER_STATE_KEY, MirroredState
from torchtitan.rl.trainer.rdma_microbatch import (
    RDMAMicrobatch,
    RDMAMicrobatchPrefetcher,
)
from torchtitan.tools import utils
from torchtitan.training_engine import TrainingEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OptimizerStepOutput:
    """Result returned by `Trainer.optim_step` to the controller."""

    policy_version: int
    metrics: dict[str, float]


class Trainer(Configurable):
    """Updates policy based on collected TrainingSample using TorchTitan components.

    Exposes separate `forward_backward` and `optim_step` endpoints, called
    explicitly by the controller.

    Args:
        config: Trainer.Config with all model/optimizer/parallelism settings.
        max_num_documents: Fixed varlen metadata capacity configured by the batcher.
        hf_assets_path: Path to HF assets folder for checkpoint loading.
            Shared with the generator (both load from the same HF checkpoint).
        generator_dtype: Generator dtype (e.g. "bfloat16"). Needed to cast weights to generator dtype
            if generator dtype differs from training dtype. If None, no cast is performed.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TrainingEngine.Config):
        """Trainer configuration for model, optimizer, training, and parallelism."""

        model: BaseModel.Config
        """Trainer model config, built for ``parallelism``."""

    def __init__(
        self,
        config: Config,
        *,
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
        validate_model_training_config(
            config.model,
            parallelism=config.parallelism,
            training=config.training,
            debug=config.debug,
            activation_remat=config.activation_remat,
            max_num_documents=max_num_documents,
        )

        self.engine = TrainingEngine(
            config,
            model_config=config.model,
            max_num_documents=max_num_documents,
            output_dir=output_dir,
        )
        engine = self.engine
        self._controller_state = MirroredState()
        maybe_apply_numa_binding(engine.device.index, engine.device.type)

        # Only cast when the generator and training dtypes differ.
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

        engine.load_checkpoint()
        if engine.device.type == "cpu":
            self._push_model_state_dict_stream = None
        else:
            self._push_model_state_dict_stream = torch.Stream(device=engine.device)
            self._mark_model_state_dict_ready_for_push()
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
        # Batches read over RDMA so far; the next batch's `batch_index` must equal it.
        self._num_rdma_batches_read = 0

    @property
    def policy_version(self) -> int:
        """Number of completed optimizer steps, restored with engine state."""
        return self.engine.num_completed_steps

    async def get_controller_state(self) -> dict[str, Any] | None:
        """Return controller state restored with the trainer checkpoint."""
        return self._controller_state.loaded

    def _mark_model_state_dict_ready_for_push(self) -> None:
        """Order the push stream after the latest model-weight updates."""
        if self._push_model_state_dict_stream is None:
            return

        # Capture the dependency when weights become final: the push must wait
        # for optimizer updates, but not for next-step forward/backward kernels
        # that may be queued before the concurrent push starts.
        self._push_model_state_dict_stream.wait_stream(
            torch.accelerator.current_stream(self.engine.device)
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
        microbatch_groups: (
            list[list[TrainingMicrobatch]] | list[list[RDMAMicrobatch]]
        ),
        global_loss_token_counts: torch.Tensor,
        global_routing_token_counts: torch.Tensor,
    ) -> dict[str, float] | None:
        """Run one optimizer step's forward/backward microbatches.

        Args:
            microbatch_groups: This rank's column of the batcher's
                `[num_groups][num_microbatches_per_group][dp_degree]` grid. Under PP, each group runs
                as one pipeline step. `RDMAMicrobatch`es when `rdma_min_tensor_bytes` is set,
                read a few ahead of training.
            global_loss_token_counts: Per-objective loss-token counts across the
                global batch.
            global_routing_token_counts: Per-depth non-padding routing-token
                counts across the global batch.

        Returns:
            dict[str, float] | None: Globally-reduced metrics, or ``None`` on
                ranks without the last pipeline stage, which compute no loss.

        Example:
            # dp_degree=2, tp=2, one microbatch per group,
            # grid = [[[mb_00, mb_01]], [[mb_10, mb_11]]]
            # ranks 0, 1 (DP rank 0) get microbatch_groups = [[mb_00], [mb_10]]
            # ranks 2, 3 (DP rank 1) get microbatch_groups = [[mb_01], [mb_11]]
        """
        logger.debug(
            f"{os.getpid()=} Trainer forward_backward "
            f"policy_version={self.policy_version}"
        )
        engine = self.engine
        self._step_compute_start = time.perf_counter()
        microbatches = [
            microbatch for group in microbatch_groups for microbatch in group
        ]
        # The batcher pads every microbatch to the same token count.
        self._step_num_tokens_per_dp_rank = (
            len(microbatches)
            * self.config.training.num_tokens_per_microbatch_per_dp_rank
        )
        uses_rdma = bool(microbatches) and isinstance(microbatches[0], RDMAMicrobatch)
        if uses_rdma:
            # A wrong column, or a batch read twice or out of order, fails here instead of training.
            # Both counters start at 0 in their own process, and every registered batch trains in order.
            for microbatch_index, microbatch in enumerate(microbatches):
                identity = (
                    microbatch.batch_index,
                    microbatch.dp_rank,
                    microbatch.microbatch_index,
                )
                expected = (self._num_rdma_batches_read, self.dp_rank, microbatch_index)
                if identity != expected:
                    raise ValueError(
                        "RDMA microbatch (batch_index, dp_rank, microbatch_index) is "
                        f"{identity}, expected {expected}."
                    )
            self._num_rdma_batches_read += 1
            # TODO: the cp * tp ranks of a DP replica read the same column. Read it once per replica
            # and broadcast it, to cut the trainer's reads by cp * tp. Needs a per-replica group.
            # The controller rejects RDMA with PP, so every group holds one microbatch.
            microbatch_groups = RDMAMicrobatchPrefetcher(microbatches)
        result = engine.forward_backward(
            microbatch_groups=microbatch_groups,
            global_loss_token_counts=global_loss_token_counts,
            global_routing_token_counts=global_routing_token_counts,
        )
        if not engine.pp_has_last_stage:
            return None
        microbatch_metrics: list[dict[str, float]] = []
        for group_metrics in result.loss_metrics:
            for loss_metrics in group_metrics:
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

        metrics = combine_microbatch_metrics(microbatch_metrics)
        if uses_rdma:
            # The slowest rank's total read time this step, hidden or not (max over the loss mesh).
            read_microbatches_s = torch.tensor(
                microbatch_groups.read_s, device=engine.device
            )
            metrics.update(
                self._reduce_forward_backward_metrics(
                    sum_reduced_metrics={},
                    max_reduced_metrics={
                        "timing/trainer/read_microbatches/max": read_microbatches_s
                    },
                )
            )
        return metrics

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
        self._mark_model_state_dict_ready_for_push()
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
                "trainer/tokens_per_second": performance["tokens_per_second"],
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
        """Publish model weights for generators to pull.

        The live GPU tensors are put directly, so the trainer keeps no CPU
        snapshot of its own. With the colocated SharedMemory transport,
        TorchStore copies (and casts) each tensor straight into StorageVolume
        memory; other transports make a GPU cast copy of each batch first.

        The controller overlaps this push with the next forward/backward (see
        `Controller._trainer_loop`), which is safe because:
        - The worker synchronizes the push stream and runs the put with it
          current, so TorchStore reads weights after the last optimizer step
          without waiting on later forward/backward work, and forward/backward
          does not mutate persistent model state.
        - The TorchStore put runs in an `asyncio.to_thread` worker, so it
          progresses while a synchronous forward/backward blocks this actor's
          event loop.
        - The controller awaits this push before the next optimizer step mutates
          the model, and awaits the generators' pull of it before the next push
          overwrites the TorchStore key.
        """
        model_state_dict = self._model_state_dict()
        await asyncio.to_thread(
            self._put_model_state_dict,
            model_state_dict,
            self._model_buffer_names(model_state_dict),
        )

    def _put_model_state_dict(
        self,
        model_state_dict: dict[str, torch.Tensor],
        buffer_names: frozenset[str],
    ) -> None:
        """Publish without blocking the actor event loop."""
        # The current stream is thread-local; set it in this worker so TorchStore
        # copies wait on the optimizer step rather than on later default-stream
        # work. Synchronize because RDMA transports read outside stream order.
        stream = self._push_model_state_dict_stream
        with stream if stream is not None else nullcontext():
            if stream is not None:
                stream.synchronize()
            asyncio.run(
                ts.put_state_dict(
                    model_state_dict,
                    "model_state_dict",
                    direct_rdma=False,
                    transfer_dtype=self._transfer_dtype,
                    # Buffers keep their registered dtype when weights are cast.
                    preserve_dtype_keys=buffer_names,
                )
            )

    def _model_state_dict(self) -> dict[str, torch.Tensor]:
        """Merge the state dicts of this rank's model parts.

        Under PP each rank holds only its own stages, so every stage publishes
        a partial state dict into the same TorchStore key.
        """
        return {
            name: tensor
            for model_part in self.engine.model_parts
            for name, tensor in model_part.state_dict().items()
        }

    def _model_buffer_names(
        self, model_state_dict: dict[str, torch.Tensor]
    ) -> frozenset[str]:
        """Return the `model_state_dict` keys of this rank's buffers.

        Non-persistent buffers (e.g. RoPE caches) are not in the state dict.
        """
        return frozenset(
            canonical_fqn(name)
            for model_part in self.engine.model_parts
            for name, _ in model_part.named_buffers()
        ).intersection(model_state_dict)

    async def initialize_torchstore_client(self) -> None:
        """Initialize this process as a TorchStore routing publisher."""
        await ts.client(role=RankRole.PUBLISHER)
