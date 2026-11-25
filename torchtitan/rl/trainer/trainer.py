# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os
import pickle
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
import torchstore as ts
from torch.distributed.checkpoint.stateful import Stateful
from torchstore import RankRole

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.config import Configurable, PrecisionConfig, TORCH_DTYPE_MAP
from torchtitan.config.validation import validate_model_training_config
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.base.model import BaseModel
from torchtitan.models.common.aux_loss import collect_aux_loss_metrics
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.observability.metrics import compute_training_performance_metrics
from torchtitan.rl.controller.metrics import combine_microbatch_metrics
from torchtitan.rl.model_state_dict_transport import (
    ModelStateDictTransport,
    TensorShard,
    TrainerSource,
)
from torchtitan.rl.trainer.batcher import MicrobatchPacker
from torchtitan.rl.trainer.microbatch_group_reader import (
    MicrobatchGroupReader,
    ReplicaColumn,
)
from torchtitan.rl.trainer.training_sample_builder import TrainingSampleBuilder
from torchtitan.tools import utils
from torchtitan.training_engine import TrainingEngine

logger = logging.getLogger(__name__)


class MirroredState(Stateful):
    """Checkpoint state produced elsewhere and mirrored into the trainer.

    ``value`` is replaced by the controller before saving the next trainer
    checkpoint. ``loaded`` holds the value restored by the checkpointer so the
    controller can retrieve it after actor startup. Separate fields prevent a
    fresh value from being mistaken for restored state.
    """

    def __init__(self) -> None:
        self.value: Any = None
        self.loaded: Any = None

    def state_dict(self) -> dict[str, bytes]:
        return {"state": pickle.dumps(self.value)}

    def load_state_dict(self, state_dict: dict[str, bytes]) -> None:
        self.loaded = pickle.loads(state_dict["state"])


def build_dp_replica_group(
    parallelism_context: ParallelismContext, *, timeout: timedelta
) -> tuple[dist.ProcessGroup | None, bool]:
    """This rank's DP replica group, and whether this rank leads it. Every trainer rank calls
    it (`new_subgroups_by_enumeration` is collective).

    A DP replica is the pp * cp * tp ranks at one DP rank of the dense mesh:
    they train the same microbatches, which the replica's leader (its first rank: pipeline
    stage 0, CP and TP rank 0) packs and broadcasts. The group is gloo, so the reader thread's
    broadcasts stay off the NCCL communicators the main thread's collectives use. A replica of
    one rank has no group.

    Example:
        # pp=2, dp=2, tp=2: ranks by DP rank [[0, 1, 4, 5], [2, 3, 6, 7]]
        build_dp_replica_group(parallelism_context, timeout=timeout)
        # on rank 4 -> (the group of ranks 0, 1, 4, 5, False); on rank 0 -> (that group, True)
    """
    dense_mesh = parallelism_context.get_optional_mesh(
        ["pp", "dp_replicate", "dp_shard", "cp", "tp"], include_singleton_axes=True
    )
    assert dense_mesh is not None
    # [dp_replicate * dp_shard][pp * cp * tp] global ranks: the DP rank folds dp_replicate
    # and dp_shard, and each row is in (pp, cp, tp) order.
    ranks_by_dp_rank = (
        dense_mesh.mesh.permute(1, 2, 0, 3, 4)
        .reshape(parallelism_context.dp_replicate * parallelism_context.dp_shard, -1)
        .tolist()
    )
    [replica_ranks] = [ranks for ranks in ranks_by_dp_rank if dist.get_rank() in ranks]
    if len(replica_ranks) == 1:
        return None, True
    replica_group, _ = dist.new_subgroups_by_enumeration(
        ranks_by_dp_rank, timeout=timeout, backend="gloo"
    )
    return replica_group, dist.get_rank() == replica_ranks[0]


@dataclass(frozen=True, slots=True)
class OptimizerStepOutput:
    """Result returned by `Trainer.optim_step` to the controller."""

    policy_version: int
    metrics: dict[str, float]


class Trainer(Configurable):
    """Updates policy based on collected TrainingSample using TorchTitan components.

    Exposes separate `forward_backward` and `optim_step` endpoints, called
    explicitly by the controller.

    Each DP replica's leader (its first rank) packs the replica's microbatches from the step
    plan and broadcasts them to the replica's other ranks (`MicrobatchGroupReader`).

    Args:
        config: Trainer.Config with all model/optimizer/parallelism settings.
        precision: Parameter, compute, and gradient-reduction dtypes, from
            ``Controller.Config.precision``.
        max_num_documents: Fixed varlen metadata capacity configured by the batcher.
        hf_assets_path: Path to HF assets folder for checkpoint loading.
            Shared with the generator (both load from the same HF checkpoint).
        training_sample_builder: The batcher's builder config; leaders materialize the
            planned samples with it.
        microbatch_packer: The batcher's packer config; leaders pack each planned
            microbatch with it.
        pad_id: Token id of the packed padding (the tokenizer's EOS).
        temperature: The generators' sampling temperature, which the loss divides logits by.
        vision_processor: Turns each microbatch's image files into pixel values; None
            without images.
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
        precision: PrecisionConfig,
        max_num_documents: int | None,
        hf_assets_path: str = "",
        output_dir: str,
        training_sample_builder: TrainingSampleBuilder.Config,
        microbatch_packer: MicrobatchPacker.Config,
        pad_id: int,
        temperature: float,
        vision_processor: VisionProcessor.Config | None,
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
            precision=precision,
            max_num_documents=max_num_documents,
            output_dir=output_dir,
        )
        engine = self.engine
        self._resume_state = MirroredState()

        # Send parameters in the dtype the generator computes with, as FSDP casts
        # them for the trainer's forward. Only cast when it differs.
        self._transfer_dtype = (
            TORCH_DTYPE_MAP[precision.param_dtype]
            if precision.param_dtype != precision.dtype
            else None
        )
        # Set by `register_model_state_dict_source`; None pushes through TorchStore.
        self._model_state_dict_transport: ModelStateDictTransport | None = None

        self.gpu_peak_flops = utils.get_peak_flops(
            engine.device_memory_monitor.device_name
        )
        engine.initialize(
            hf_assets_path=hf_assets_path,
            extra_states={"resume": self._resume_state},
        )

        if self.gpu_peak_flops is not None:
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

        self._replica_group, self._is_replica_leader = build_dp_replica_group(
            engine.parallelism_context,
            timeout=timedelta(seconds=config.comm.train_timeout_seconds),
        )
        parallelism = config.parallelism
        self._num_microbatches_per_group = (
            parallelism.num_pp_microbatches
            if parallelism.pipeline_parallel_degree > 1
            else 1
        )
        self._training_sample_builder = training_sample_builder.build()
        self._microbatch_packer = microbatch_packer.build(
            num_tokens_per_microbatch_per_dp_rank=(
                config.training.num_tokens_per_microbatch_per_dp_rank
            ),
            max_context_length=config.model.max_context_length,
            pad_id=pad_id,
            temperature=temperature,
            vision_processor=vision_processor,
        )
        # Set on the leader by `receive_replica_column` before each `forward_backward`.
        self._replica_column: ReplicaColumn | None = None

    @property
    def policy_version(self) -> int:
        """Number of completed optimizer steps, restored with engine state."""
        return self.engine.num_completed_steps

    async def get_resume_state(self) -> dict[str, Any] | None:
        """Return controller resume state restored with the trainer checkpoint."""
        return self._resume_state.loaded

    def _mark_model_state_dict_ready_for_push(self) -> None:
        """Order the push stream after the latest model-weight updates."""
        if self._push_model_state_dict_stream is None:
            return

        # Capture the dependency when weights become final: the push must wait
        # for the optimizer updates. optim_step syncs the host before the push
        # starts, so the stream saves nothing today; it is kept for the TODO in
        # `Controller._trainer_loop` that overlaps the push with forward/backward.
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

    async def receive_replica_column(self, column: ReplicaColumn) -> None:
        """On a DP replica's leader: keep the replica's column of the next step's plan and the
        rollouts it names, for the next `forward_backward` to pack."""
        assert self._is_replica_leader and self._replica_column is None
        self._replica_column = column

    @sl.log_trace_span("forward_backward")
    async def forward_backward(
        self,
        *,
        num_microbatch_groups: int,
        global_loss_token_counts: torch.Tensor,
        global_routing_token_counts: torch.Tensor,
    ) -> dict[str, float] | None:
        """Run one optimizer step's forward/backward microbatches.

        The replica's leader packs its column (from `receive_replica_column`) one microbatch
        group at a time and broadcasts each group to the replica's other ranks
        (`MicrobatchGroupReader`, which says how much of that training hides).

        Args:
            num_microbatch_groups: Groups in this replica's column. Under PP, each group runs
                as one pipeline step.
            global_loss_token_counts: Per-objective loss-token counts across the
                global batch.
            global_routing_token_counts: Per-depth non-padding routing-token
                counts across the global batch.

        Returns:
            dict[str, float] | None: Globally-reduced metrics, or ``None`` on
                ranks without the last pipeline stage, which compute no loss.

        Example:
            # dp_degree=2, tp=2, one microbatch per group, plan grid
            #   [[[specs_00, specs_01]], [[specs_10, specs_11]]]
            # rank 0 (leader of DP rank 0) packs [[mb_00], [mb_10]] and sends them to rank 1;
            # rank 2 (leader of DP rank 1) packs [[mb_01], [mb_11]] and sends them to rank 3
        """
        logger.debug(
            f"{os.getpid()=} Trainer forward_backward "
            f"policy_version={self.policy_version}"
        )
        engine = self.engine
        self._step_compute_start = time.perf_counter()
        column, self._replica_column = self._replica_column, None
        # The controller sends each leader its column before every forward_backward.
        assert (column is not None) == self._is_replica_leader
        reader = MicrobatchGroupReader(
            num_microbatch_groups=num_microbatch_groups,
            replica_group=self._replica_group,
            column=column,
            training_sample_builder=self._training_sample_builder,
            microbatch_packer=self._microbatch_packer,
        )
        # The reader frees each rollout group once its samples exist.
        del column
        # The packer pads every microbatch to the same token count.
        self._step_num_tokens_per_dp_rank = (
            num_microbatch_groups
            * self._num_microbatches_per_group
            * self.config.training.num_tokens_per_microbatch_per_dp_rank
        )
        result = engine.forward_backward(
            microbatch_groups=reader,
            global_loss_token_counts=global_loss_token_counts,
            global_routing_token_counts=global_routing_token_counts,
        )
        self._num_flops = result.num_flops
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
        # The slowest rank's times this step (max over the loss mesh), totals, hidden or not:
        # its replica leader's packing (the leader sends it with each group, so the last
        # pipeline stage reports it too) and its own broadcast time.
        max_reduced_timing = {
            "timing/trainer/pack_microbatches/max": torch.tensor(
                reader.pack_s, device=engine.device
            ),
            "timing/trainer/broadcast_microbatches/max": torch.tensor(
                reader.broadcast_s, device=engine.device
            ),
        }
        metrics.update(
            self._reduce_forward_backward_metrics(
                sum_reduced_metrics={}, max_reduced_metrics=max_reduced_timing
            )
        )
        return metrics

    @sl.log_trace_span("optim_step")
    async def optim_step(
        self, *, resume_state: dict[str, Any], last_step: bool = False
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
        self._resume_state.value = resume_state

        # TODO: Move performance, LR, and auxiliary-loss reporting into a shared
        # trainer metrics interface while preserving controller-side aggregation.
        mean_num_flops = self._num_flops
        if engine.parallelism_context.dp_enabled:
            mean_num_flops = dist_utils.dist_mean(
                torch.tensor(mean_num_flops, dtype=torch.float64, device=engine.device),
                engine.parallelism_context.get_mesh("dp"),
            )

        performance = compute_training_performance_metrics(
            num_tokens=self._step_num_tokens_per_dp_rank,
            elapsed_time=time.perf_counter() - self._step_compute_start,
            non_data_parallel_size=engine.parallelism_context.non_data_parallel_size,
            num_flops=mean_num_flops,
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

        With TorchStore, the live GPU tensors are put directly, so the trainer
        keeps no CPU snapshot of its own. With the colocated SharedMemory transport,
        TorchStore copies (and casts) each tensor straight into StorageVolume
        memory; other transports make a GPU cast copy of each batch first.

        Ordering (see `Controller._trainer_loop`):
        - The worker synchronizes the push stream and runs the put with it
          current, so TorchStore reads weights after the last optimizer step.
        - The TorchStore put runs in an `asyncio.to_thread` worker, so it
          doesn't block this actor's event loop.
        - The controller awaits this push before the next forward/backward,
          which blocks this actor's event loop and would delay the push's reply.
        - The controller awaits the generators' pull of this push before the
          next push overwrites the TorchStore key.

        Every push carries the trainable parameters and the buffers. Generators load the
        frozen parameters from `generator.checkpointer`: for LoRA the base, so a push
        carries only the adapters (`lora_a`, `lora_b`) and the buffers.

        With a ``ModelStateDictTransport``, the push instead stages the tensors in
        its pinned staging buffer, which generators read directly; the same
        ordering applies, and the next push overwrites the staging buffer.
        Staging enqueues its copies on the push stream from this thread; only
        the wait for them is off the event loop.
        """
        transport = self._model_state_dict_transport
        if transport is not None:
            stream = self._push_model_state_dict_stream
            with stream if stream is not None else nullcontext():
                transport.stage(self._trainable_model_state_dict())
            # Generators read the staging buffer outside stream order.
            if stream is not None:
                await asyncio.to_thread(stream.synchronize)
            return
        model_state_dict = self._trainable_model_state_dict()
        await asyncio.to_thread(
            self._put_model_state_dict,
            model_state_dict,
            self._preserve_dtype_keys(model_state_dict),
        )

    def _put_model_state_dict(
        self,
        model_state_dict: dict[str, torch.Tensor],
        preserve_dtype_keys: frozenset[str],
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
                    # Buffers and `preserve_dtype_params` keep their dtype when
                    # weights are cast.
                    preserve_dtype_keys=preserve_dtype_keys,
                )
            )

    def register_model_state_dict_source(
        self,
        config: ModelStateDictTransport.Config,
        *,
        generator_rank_layouts: list[dict[str, TensorShard]],
    ) -> TrainerSource:
        """Allocate and register the staging buffer generators read pushes from.

        Args:
            config: The transport config.
            generator_rank_layouts: The layout of each rank of a generator, in
                rank order.
        """
        self._model_state_dict_transport = config.build()
        model_state_dict = self._trainable_model_state_dict()
        return self._model_state_dict_transport.register_trainer_source(
            model_state_dict,
            generator_rank_layouts=generator_rank_layouts,
            transfer_dtype=self._transfer_dtype,
            preserve_dtype_keys=self._preserve_dtype_keys(model_state_dict),
        )

    def _trainable_model_state_dict(self) -> dict[str, torch.Tensor]:
        """Merge the state dicts of this rank's model parts, leaving out the parameters
        that do not train.

        Under PP each rank holds only its own stages, so every stage publishes
        a partial state dict.
        """
        frozen = frozenset(
            canonical_fqn(name)
            for model_part in self.engine.model_parts
            for name, param in model_part.named_parameters(remove_duplicate=False)
            if not param.requires_grad
        )
        return {
            name: tensor
            for model_part in self.engine.model_parts
            for name, tensor in model_part.state_dict().items()
            if canonical_fqn(name) not in frozen
        }

    def _preserve_dtype_keys(
        self, model_state_dict: dict[str, torch.Tensor]
    ) -> frozenset[str]:
        """Return the `model_state_dict` keys sent in their own dtype: this rank's
        buffers and `preserve_dtype_params`.

        Non-persistent buffers (e.g. RoPE caches) are not in the state dict.
        """
        return frozenset(
            canonical_fqn(name)
            for model_part in self.engine.model_parts
            for name, _ in (
                *model_part.named_buffers(),
                *model_part.named_preserve_dtype_params(),
            )
        ).intersection(model_state_dict)

    async def initialize_torchstore_client(self) -> None:
        """Initialize this process as a TorchStore routing publisher."""
        await ts.client(role=RankRole.PUBLISHER)
