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
from torch.distributed.tensor import DTensor
from torch.numa import binding as numa_binding
from torchstore import RankRole

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.config.validation import validate_model_training_config
from torchtitan.distributed import maybe_apply_numa_binding, utils as dist_utils
from torchtitan.distributed.utils import get_local_tensor
from torchtitan.models.base.model import BaseModel
from torchtitan.models.common.aux_loss import collect_aux_loss_metrics
from torchtitan.models.common.moe import collect_routing_mismatch_metrics
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

        log_routing_mismatch_per_layer: bool = False
        """With router replay, also log each MoE layer's `moe_routing/set_mismatch_frac/layer_<i>`
        (the aggregate keys are always logged)."""

        def __post_init__(self) -> None:
            TrainingEngine.Config.__post_init__(self)
            # TODO: support CP by sharding the per-token loss kwargs along the
            # sequence like the labels.
            if self.parallelism.context_parallel_degree > 1:
                raise ValueError(
                    "RL context parallelism is not supported yet: per-token "
                    "loss arguments are not sharded along the sequence."
                )

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
        if engine.device.type == "cuda":
            # torch defaults to one thread per physical core of the host, which oversubscribes the
            # bound node: the weight push takes 1.0 s instead of 0.08 s at Qwen3-0.6B. Set it before
            # the push's worker thread starts: a thread reads the count on its first parallel op.
            torch.set_num_threads(
                _num_cpu_threads_per_rank(device_index=engine.device.index)
            )

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
        # Batches read over RDMA so far; the next batch's `batch_index` must equal it.
        self._num_rdma_batches_read = 0

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
        self, *, controller_state: dict[str, Any]
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
                **collect_routing_mismatch_metrics(
                    engine.model_parts,
                    engine.parallelism_context,
                    per_layer=self.config.log_routing_mismatch_per_layer,
                ),
                **(
                    {"trainer/mfu_percent": performance["mfu_percent"]}
                    if "mfu_percent" in performance
                    else {}
                ),
            },
        )

    @sl.log_trace_span("save_checkpoint")
    async def save_checkpoint(self, *, last_step: bool) -> None:
        """Save a checkpoint if the checkpointer is due one at this step, or `last_step` is True."""
        self.engine.save_checkpoint(last_step=last_step)

    @sl.log_trace_span("push_model_state_dict")
    async def push_model_state_dict(self) -> None:
        """Publish a CPU snapshot of model weights for generators to pull.

        The controller overlaps this push with the next forward/backward (see
        `Controller._trainer_loop`), which is safe because:
        - The staging stream waits for the last optimizer step before copying, and
          forward/backward does not mutate persistent model state.
        - The CUDA-event wait and the TorchStore put run together in an
          `asyncio.to_thread` worker, so they progress while a synchronous
          forward/backward blocks this actor's event loop.
        - The controller awaits this push before the next optimizer step mutates
          the model, and awaits the generators' pull of it before the next push
          reuses the staging buffers and TorchStore key.
        """

        model_state_dict = self._model_state_dict()
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

    def _setup_staging_model_state_dict(self) -> dict[str, torch.Tensor]:
        """Allocate a final-dtype pinned CPU mirror of the model state dict."""
        buffer_names = {
            canonical_fqn(name)
            for model_part in self.engine.model_parts
            for name, _ in model_part.named_buffers()
        }
        model_state_dict = self._model_state_dict()
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


def _num_cpu_threads_per_rank(device_index: int) -> int:
    """Return this rank's share of its NUMA node's physical cores.

    Assumes one trainer rank per visible GPU, as `train.py` launches them.

    Example (2 sockets x 96 cores x 2 threads; GPUs 0-3 on NUMA node 0, GPUs 4-7 on node 1):
        trainer on GPU 0     -> 192 logical CPUs / 2 per core / 1 rank on node 0  = 96 threads
        trainer on GPUs 0-3  -> 192 logical CPUs / 2 per core / 4 ranks on node 0 = 24 threads
        trainer on GPUs 0-7  -> 192 logical CPUs / 2 per core / 4 ranks per node  = 24 threads
    """
    cpus = os.sched_getaffinity(0)
    threads_per_core = len(
        numa_binding._get_logical_cpu_indices_sharing_same_physical_core_as(
            logical_cpu_index=min(cpus)
        )
    )
    numa_node_index = numa_binding._get_numa_node_index_for_device_index(
        device_index=device_index
    )
    num_ranks_on_node = len(
        numa_binding._get_device_indices_for_numa_node(numa_node_index=numa_node_index)
    )
    return max(1, len(cpus) // threads_per_core // num_ranks_on_node)
