# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from dataclasses import dataclass

import torch
import torchstore as ts

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.config import (
    apply_overrides,
    CompileConfig,
    Configurable,
    TORCH_DTYPE_MAP,
)
from torchtitan.config.validation import validate_model_training_config
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.common.aux_loss import collect_aux_loss_metrics
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.observability.metrics import compute_training_performance_metrics
from torchtitan.protocols.model import BaseModel
from torchtitan.rl.observability.controller import combine_microbatch_metrics
from torchtitan.rl.types import OptimizerStepOutput, TrainingMicrobatch
from torchtitan.tools import utils
from torchtitan.training_engine import TrainingEngine

logger = logging.getLogger(__name__)


class Trainer(Configurable):
    """Updates policy based on collected TrainingSample using TorchTitan components.

    Exposes separate `forward_backward_steps` and `optimizer_step` endpoints, called
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
        compile_config: CompileConfig | None,
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

        model_config.update_from_config(config=config)
        if config.override.imports:
            apply_overrides(config.override, model_config)
        config.__post_init__()

        validate_model_training_config(
            model_config,
            parallelism=config.parallelism,
            training=config.training,
            debug=config.debug,
            activation_checkpoint=config.activation_checkpoint,
            compile_config=compile_config,
            max_num_documents=max_num_documents,
        )

        self.engine = TrainingEngine(
            config,
            model_config=model_config,
            max_num_documents=max_num_documents,
            output_dir=output_dir,
        )
        engine = self.engine

        # Only cast if generator dtype differs from training dtype, otherwise
        # staging buffers would be allocated for a no-op cast.
        training_dtype = TORCH_DTYPE_MAP[config.training.dtype]
        gen_dtype = TORCH_DTYPE_MAP[generator_dtype] if generator_dtype else None
        self._transfer_dtype = gen_dtype if gen_dtype != training_dtype else None

        self.gpu_peak_flops = utils.get_peak_flops(
            engine.device_memory_monitor.device_name
        )
        engine.initialize(
            compile_config=compile_config,
            hf_assets_path=hf_assets_path,
        )

        logger.info(f"Peak FLOPS used for computing MFU: {self.gpu_peak_flops:.3e}")
        logger.info(
            f"{engine.device.type.upper()} memory usage for model: "
            f"{engine.model_device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({engine.model_device_mem_stats.max_reserved_pct:.2f}%)"
        )
        self.model = engine.model_parts[0]

        engine.load_checkpoint()
        if config.checkpointer is None:
            logger.warning(
                "Checkpoint disabled, skip weight loading and use random-initialized weights. "
                "Configure checkpointer to load from a checkpoint."
            )

        engine.start_profiler()
        engine.device_memory_monitor.reset_peak_stats()

        # Data parallelism: mesh is available after model construction builds it.
        self.dp_enabled = engine.parallel_dims.dp_enabled
        dp_mesh = engine.parallel_dims.get_optional_mesh("dp")
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

    async def get_policy_version(self) -> int:
        """Current policy version: after load(), the step a resume restored from
        (0 if fresh). The controller uses it to resume and re-sync generators."""
        return self.policy_version

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
        loss_mesh = self.engine.parallel_dims.get_optional_mesh("loss")

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

    @sl.log_trace_span("forward_backward_steps")
    async def forward_backward_steps(
        self,
        training_data: list[list[TrainingMicrobatch]],
        num_global_valid_tokens: int,
    ) -> dict[str, float]:
        """Run one optimizer step's forward/backward microbatches.

        Args:
            training_data: Microbatch-major grid with shape
                ``[num_microbatches][dp_degree]``.
            num_global_valid_tokens: Total response tokens with finite generator
                logprobs across all DP ranks and microbatches for this step.

        Returns:
            dict[str, float]: Globally-reduced metrics.
        """
        logger.debug(
            f"{os.getpid()=} Trainer forward_backward_steps "
            f"policy_version={self.policy_version}"
        )
        engine = self.engine
        self._step_compute_start = time.perf_counter()
        self._step_num_tokens_per_dp_rank = sum(
            rank_batches[self.dp_rank].labels.numel() for rank_batches in training_data
        )
        microbatch_metrics: list[dict[str, float]] = []
        num_accumulation_steps = len(training_data)
        prepared_global_valid_tokens = engine.prepare_step(
            num_global_valid_tokens,
            num_accumulation_steps=num_accumulation_steps,
        )

        for microbatch_index, rank_batches in enumerate(training_data):
            local_batch = rank_batches[self.dp_rank]

            engine.forward_backward_microbatch(
                microbatch_group=[local_batch],
                global_valid_tokens=prepared_global_valid_tokens,
                accumulation_index=microbatch_index,
            )
            microbatch_metrics.append(
                self._reduce_forward_backward_metrics(
                    sum_reduced_metrics={
                        key: value
                        for key, value in engine.loss_metrics.items()
                        if not key.endswith("/max")
                    },
                    max_reduced_metrics={
                        key: value
                        for key, value in engine.loss_metrics.items()
                        if key.endswith("/max")
                    },
                )
            )

        return combine_microbatch_metrics(microbatch_metrics)

    @sl.log_trace_span("optimizer_step")
    async def optimizer_step(self, *, last_step: bool = False) -> OptimizerStepOutput:
        """Clip gradients, step optimizer + LR scheduler, return updated state."""
        # TODO: Accept optional optimizer params (e.g. learning rate)
        # to allow controller-owned schedules.

        engine = self.engine
        # Capture the learning rates used by this optimizer update before the
        # scheduler advances in engine.optimizer_step().
        lr_metrics = engine.lr_schedulers.get_metrics()

        grad_norm = engine.optimizer_step()

        # TODO: Move performance, LR, and auxiliary-loss reporting into a shared
        # trainer metrics interface while preserving controller-side aggregation.
        performance = compute_training_performance_metrics(
            num_tokens=self._step_num_tokens_per_dp_rank,
            elapsed_time=time.perf_counter() - self._step_compute_start,
            non_data_parallel_size=engine.parallel_dims.non_data_parallel_size,
            num_flops_per_token=engine.num_flops_per_token,
            gpu_peak_flops=self.gpu_peak_flops,
            has_quantization=engine.has_quantization,
        )

        engine.save_checkpoint(last_step=last_step)
        engine.step_profiler()
        device_mem_stats = engine.device_memory_monitor.get_peak_stats()
        engine.device_memory_monitor.reset_peak_stats()

        logger.debug(
            f"{os.getpid()=} Trainer optimizer_step done, "
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
                **collect_aux_loss_metrics(engine.parallel_dims),
                **(
                    {"trainer/mfu_percent": performance["mfu_percent"]}
                    if "mfu_percent" in performance
                    else {}
                ),
            },
        )

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
