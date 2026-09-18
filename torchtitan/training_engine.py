# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Annotated, Any, cast, NamedTuple

import spmd_types as spmd
import torch
import torch.distributed.checkpoint.stateful
import tyro
from torch.distributed.fsdp import FSDPModule
from torch.distributed.pipelining.schedules import (
    _PipelineScheduleRuntime,
    get_schedule_class,
    PipelineScheduleMulti,
)
from torch.distributed.tensor import DTensor

from torchtitan.components.checkpointer import BaseCheckpointManager, CheckpointManager
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.components.loss import BaseLoss, ChunkedLossWrapper
from torchtitan.components.optimizer import LRSchedulersContainer, OptimizersContainer
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import (
    CommConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.config.override import OverrideConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointingConfig,
    SelectiveAC,
)
from torchtitan.distributed.cuda_graph import (
    cuda_graph_teardown,
    run_eager_on_cuda_graph_stream,
    wrap_with_cuda_graph,
)
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.metrics import (
    build_device_memory_monitor,
    DeviceMemoryMonitor,
    DeviceMemStats,
)
from torchtitan.observability.profiler import Profiler
from torchtitan.observability.sdc_replayer import SDCReplayer
from torchtitan.protocols import BaseModel
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.quantization.utils import has_quantization
from torchtitan.tools import utils


logger = logging.getLogger(__name__)

_NUM_CUDA_GRAPH_WARMUP_STEPS = 2


class ForwardBackwardResult(NamedTuple):
    loss: torch.Tensor
    loss_metrics: list[dict[str, torch.Tensor]]


class TrainingEngine(Configurable, torch.distributed.checkpoint.stateful.Stateful):
    """Shared distributed training engine.

    A microbatch is one data-parallel-rank-local forward/backward input. An
    accumulation step contains one microbatch without pipeline parallelism and
    all pipeline microbatches for one schedule step with pipeline parallelism.
    One or more accumulation steps contribute to each optimizer step.

    Forward and backward calls follow this flow::

        TrainingEngine.initialize
        |
        +-- _initialize_forward_backward
            |
            +-- _initialize_gradient_accumulation

        Trainer.train_step
        |
        +-- TrainingEngine.forward_backward_step
            |
            +-- _configure_fsdp_gradient_accumulation
            |
            +-- _preprocess_accumulation_step_inputs
            |
            +-- _run_gradient_accumulation =
                _gradient_accumulation_body (maybe_wrapped_with_cuda_graph)
                |
                +-- _pp_forward_backward_body
                |
                +-- _non_pp_forward_backward_body
    """

    _supports_deferred_fsdp_gradient_reduction = True

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
        checkpointer: Annotated[
            CheckpointManager.Config | None, tyro.conf.AvoidSubcommands
        ] = None
        activation_checkpoint: ActivationCheckpointingConfig = field(
            default_factory=SelectiveAC.Config
        )
        profiler: Profiler.Config = field(default_factory=Profiler.Config)
        # Suppressed because replay is enabled programmatically in recipes.
        sdc_replayer: Annotated[SDCReplayer.Config | None, tyro.conf.Suppress] = None
        comm: CommConfig = field(default_factory=CommConfig)
        debug: DebugConfig = field(default_factory=DebugConfig)
        override: OverrideConfig = field(default_factory=OverrideConfig)
        loss: BaseLoss.Config = field(default_factory=BaseLoss.Config)

        def __post_init__(self) -> None:
            if (
                self.debug.spmd_typechecking
                and self.parallelism.pipeline_parallel_degree > 1
            ):
                # TODO(sanketpurandare): Enable SPMD typechecking under PP.
                raise ValueError(
                    "SPMD typechecking is not supported with pipeline parallelism. "
                    "Validate the same config without PP "
                    "(--parallelism.pipeline_parallel_degree 1)."
                )

            if (
                not self.training.disable_cuda_graphs
                and self.parallelism.pipeline_parallel_degree > 1
            ):
                pp_schedule_class = (
                    _PipelineScheduleRuntime
                    if self.parallelism.pipeline_parallel_schedule_csv
                    else get_schedule_class(self.parallelism.pipeline_parallel_schedule)
                )
                if issubclass(pp_schedule_class, PipelineScheduleMulti):
                    raise ValueError(
                        "CUDA graphs do not support looped pipeline schedules yet. "
                        "Use a single-stage pipeline schedule or disable CUDA graphs."
                    )

            if self.parallelism.num_pp_microbatches <= 0:
                raise ValueError(
                    "parallelism.num_pp_microbatches must be greater than 0."
                )
            num_tokens = self.training.num_tokens_per_microbatch_per_dp_rank
            sequence_parallel_degree = (
                self.parallelism.tensor_parallel_degree
                if self.parallelism.enable_sequence_parallel
                else 1
            )
            context_parallel_degree = self.parallelism.context_parallel_degree
            activation_shard_degree = sequence_parallel_degree * (
                2 * context_parallel_degree if context_parallel_degree > 1 else 1
            )
            if num_tokens % activation_shard_degree != 0:
                raise ValueError(
                    "The number of tokens per pipeline microbatch "
                    f"({num_tokens}) must be divisible by "
                    f"{activation_shard_degree} for the configured "
                    "sequence/context parallelism."
                )
            if self.sdc_replayer is not None:
                if not self.debug.deterministic:
                    raise ValueError("SDC replay requires debug.deterministic=True.")
                if self.debug.deterministic_warn_only:
                    raise ValueError(
                        "SDC replay requires debug.deterministic_warn_only=False."
                    )
                if (
                    not self.training.disable_cuda_graphs
                    and self.sdc_replayer.num_replays > 1
                ):
                    raise ValueError(
                        "SDC replay supports at most one replay when CUDA graphs "
                        "are enabled: set sdc_replayer.num_replays=1 or "
                        "training.disable_cuda_graphs=True."
                    )

    config: Config
    device: torch.device
    parallel_dims: ParallelDims
    model_parts: list[torch.nn.Module]
    model_config: BaseModel.Config
    output_dir: str
    loss_fn: BaseLoss
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer
    checkpointer: BaseCheckpointManager
    pp_has_last_stage: bool
    max_num_documents: int | None
    num_accumulation_steps: int
    num_completed_steps: int
    ntokens_seen: int
    sdc_replayer: SDCReplayer | None
    model_param_count: int
    num_flops_per_token: int
    has_quantization: bool
    loss_is_finite: torch.Tensor
    loss_metrics: dict[str, torch.Tensor]
    _fsdp_root: FSDPModule | None
    device_memory_monitor: DeviceMemoryMonitor
    model_device_mem_stats: DeviceMemStats

    def __init__(
        self,
        config: Config,
        *,
        model_config: BaseModel.Config,
        max_num_documents: int | None,
        output_dir: str,
    ) -> None:
        torch._C._log_api_usage_once("torchtitan.train")

        self.config = config
        self.model_config = model_config
        self.max_num_documents = max_num_documents
        self.output_dir = output_dir
        self.has_quantization = has_quantization(model_config)
        self.num_accumulation_steps = 1
        self.num_completed_steps = 0
        self.ntokens_seen = 0
        self._num_optimizer_steps_since_cuda_graph_init = 0
        self.sdc_replayer = None
        self.preprocess_inputs_kwargs: dict[str, Any] = {}
        self.loss_metrics = {}
        self._fsdp_root = None
        self._defer_fsdp_gradient_reduction = False
        self._initialize_distributed_runtime()

    def _initialize_distributed_runtime(self) -> None:
        """Initialize the device, distributed meshes, GC, and deterministic RNGs."""
        device_module, device_type = utils.device_module, utils.device_type
        # pyrefly: ignore [read-only]
        self.device = utils.get_local_device()
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)
        config = self.config
        dist_utils.set_batch_invariance(config.debug.batch_invariant)
        with sl.log_trace_span("torch_distributed_init"):
            world_size = dist_utils.init_distributed(
                config.comm,
                enable_cpu_backend=config.training.enable_cpu_offload,
                base_folder=self.output_dir,
            )
        self.parallel_dims = ParallelDims.from_config(config.parallelism, world_size)
        self.gc_handler = utils.GarbageCollection(
            gc_freq=config.training.gc_freq,
            debug=config.training.gc_debug,
        )
        dist_utils.set_determinism(
            self.parallel_dims,
            self.device,
            self.config.debug,
            distinct_seed_mesh_dims=["pp"],
        )
        self.device_memory_monitor = build_device_memory_monitor()

    @sl.log_trace_span("initialize")
    def initialize(
        self,
        model_spec: ModelSpec,
        *,
        compile_config: CompileConfig | None,
        sd_adapter: Any | None,
        dataloader: BaseDataLoader | None = None,
        create_seed_checkpoint: bool = False,
    ) -> None:
        """Initialize model execution and the state required to train it."""
        self._initialize_model(
            model_spec,
            compile_config=compile_config,
            create_seed_checkpoint=create_seed_checkpoint,
        )
        self.model_device_mem_stats = self.device_memory_monitor.get_peak_stats()
        self._initialize_optimizer(model_spec)
        self._initialize_checkpointer(dataloader=dataloader, sd_adapter=sd_adapter)
        self._initialize_forward_backward()

    def _initialize_model(
        self,
        model_spec: ModelSpec,
        *,
        compile_config: CompileConfig | None,
        create_seed_checkpoint: bool = False,
    ) -> None:
        """Build the loss and model execution state."""
        self.loss_fn = self.config.loss.build(compile_config=compile_config)
        if create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif self.config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = torch.device(self.device.type)
        else:
            init_device = self.device.type
            buffer_device = None

        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[self.config.training.dtype]),
        ):
            model = self.model_config.build()
        model.verify_module_protocol()
        (
            self.model_param_count,
            self.num_flops_per_token,
        ) = self.model_config.get_nparams_and_flops(
            model, self.config.training.max_context_length
        )
        config = self.config
        if self.parallel_dims.pp_enabled:
            if model_spec.pipelining_fn is None:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {model_spec.name} "
                    "does not support pipelining"
                )
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = model_spec.pipelining_fn(
                model,
                parallel_dims=self.parallel_dims,
                training=config.training,
                parallelism=config.parallelism,
                compile_config=compile_config,
                ac_config=config.activation_checkpoint,
                dump_folder=self.output_dir,
                device=self.device,
                model_config=self.model_config,
                parallelize_fn=model_spec.parallelize_fn,
                loss_fn=self.loss_fn,
            )
            del model
        else:
            if not create_seed_checkpoint:
                model = model_spec.parallelize_fn(
                    model,
                    parallel_dims=self.parallel_dims,
                    training=config.training,
                    parallelism=config.parallelism,
                    compile_config=compile_config,
                    ac_config=config.activation_checkpoint,
                    dump_folder=self.output_dir,
                )
            self.model_parts = [model]
            self.pp_has_first_stage = True
            self.pp_has_last_stage = True

        for model_part in self.model_parts:
            model_part.to_empty(device=init_device)
            with torch.no_grad():
                cast(BaseModel, model_part).init_weights(buffer_device=buffer_device)
            model_part.train()

        if isinstance(self.loss_fn, ChunkedLossWrapper) and (
            not self.parallel_dims.pp_enabled or self.pp_has_last_stage
        ):
            if self.parallel_dims.pp_enabled:
                model = self.model_parts[-1]
                error_message = "Last PP stage must have lm_head for ChunkedLossWrapper"
            else:
                assert len(self.model_parts) == 1
                model = self.model_parts[0]
                error_message = "Model must have lm_head for ChunkedLossWrapper"

            model_with_lm_head = cast(Any, model)
            lm_head = model_with_lm_head.lm_head
            assert lm_head is not None, error_message
            self.loss_fn.set_lm_head(lm_head)
            model_with_lm_head._skip_lm_head = True

        logger.info(
            f"Model {model_spec.name} {model_spec.flavor} size: "
            f"{self.model_param_count:,} total parameters"
        )

    def _configure_fsdp_gradient_accumulation(self) -> None:
        """Configure FSDP for CUDA-graphed gradient accumulation."""
        self._defer_fsdp_gradient_reduction = (
            self.parallel_dims.fsdp_enabled
            and self.num_accumulation_steps > 1
            and not self.config.training.disable_cuda_graphs
        )
        if not self._defer_fsdp_gradient_reduction:
            return

        if not self._supports_deferred_fsdp_gradient_reduction:
            raise ValueError(
                f"{type(self).__name__} does not support deferred gradient reduction."
            )
        if self.config.parallelism.fsdp_reshard_after_forward != "never":
            raise ValueError(
                "FSDP CUDA graph gradient accumulation requires "
                "fsdp_reshard_after_forward='never'."
            )

        if not self.parallel_dims.pp_enabled:
            fsdp_root = self.model_parts[0]
            assert isinstance(fsdp_root, FSDPModule)
            fsdp_root.set_manual_backward_finalization(True)
            self._fsdp_root = fsdp_root

    def _initialize_optimizer(self, model_spec: ModelSpec) -> None:
        """Construct optimizers and learning-rate schedulers."""
        self.optimizers = self.config.optimizer.build(model_parts=self.model_parts)
        if model_spec.post_optimizer_build_fn is not None:
            model_spec.post_optimizer_build_fn(
                self.optimizers, self.model_parts, self.parallel_dims
            )
        self.lr_schedulers = self.config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=self.config.training.steps,
        )

    def _initialize_checkpointer(
        self,
        *,
        dataloader: BaseDataLoader | None,
        sd_adapter: Any | None,
    ) -> None:
        """Build checkpointing around core and optional owner-provided state."""
        checkpointer_config = self.config.checkpointer
        if checkpointer_config is None:
            return
        self.checkpointer = checkpointer_config.build(
            dataloader=dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            sd_adapter=sd_adapter,
            base_folder=self.output_dir,
        )

    def _initialize_forward_backward(self) -> None:
        """Build SDC replay and the gradient accumulation execution path."""
        sdc_config = self.config.sdc_replayer
        self.sdc_replayer = None
        if sdc_config is not None:
            self.sdc_replayer = sdc_config.build(
                modules=self.model_parts,
                device=self.device,
            )

        self._num_optimizer_steps_since_cuda_graph_init = 0
        self.train_context = dist_utils.get_spmd_context(
            parallel_dims=self.parallel_dims,
            spmd_typechecking=self.config.debug.spmd_typechecking,
        )
        if self.parallel_dims.pp_enabled:
            self._pp_loss_sentinel_on_non_last_stage = torch.full(
                (1,), -1.0, device=self.device
            )

        self._initialize_gradient_accumulation()

    def _initialize_gradient_accumulation(self) -> None:
        """Select eager or CUDA graph execution for gradient accumulation."""

        self._run_gradient_accumulation = self._gradient_accumulation_body
        if self.config.training.disable_cuda_graphs:
            return

        cuda_graph_gradient_accumulation_fn = wrap_with_cuda_graph(
            self._run_gradient_accumulation
        )
        # The wrapper returns its input when CUDA graph capture is unavailable.
        if cuda_graph_gradient_accumulation_fn is self._run_gradient_accumulation:
            return

        def run_with_cuda_graph(
            accumulation_step_inputs: list[tuple[Any, ...]],
            global_valid_tokens: torch.Tensor,
        ) -> ForwardBackwardResult:
            if (
                self._num_optimizer_steps_since_cuda_graph_init
                < _NUM_CUDA_GRAPH_WARMUP_STEPS
            ):
                return run_eager_on_cuda_graph_stream(
                    self._gradient_accumulation_body,
                    accumulation_step_inputs,
                    global_valid_tokens,
                )
            return cuda_graph_gradient_accumulation_fn(
                accumulation_step_inputs,
                global_valid_tokens,
            )

        self._run_gradient_accumulation = run_with_cuda_graph

    @sl.log_trace_span("forward_backward_step")
    def forward_backward_step(
        self,
        *,
        accumulation_step_inputs: list[list[TrainingMicrobatch]],
        global_valid_tokens: int | torch.Tensor,
    ) -> ForwardBackwardResult:
        """Run every gradient accumulation step for one optimizer update."""
        if not accumulation_step_inputs:
            raise ValueError("accumulation_step_inputs must not be empty.")
        self.num_accumulation_steps = len(accumulation_step_inputs)
        self._configure_fsdp_gradient_accumulation()
        self.gc_handler.run(self.num_completed_steps + 1)
        self.optimizers.zero_grad(set_to_none=self.config.training.disable_cuda_graphs)
        if isinstance(global_valid_tokens, int):
            global_valid_tokens = torch.tensor(
                global_valid_tokens,
                dtype=torch.int64,
                device=self.device,
            )
        # TODO(sdmyzlp): Each MTP depth can have a different valid-token count
        # after shifting and should use its own auxiliary-loss denominator.
        AuxLoss.set_step_denominator(global_valid_tokens)

        preprocessed_accumulation_step_inputs = [
            self._preprocess_accumulation_step_inputs(accumulation_step)
            for accumulation_step in accumulation_step_inputs
        ]

        result: ForwardBackwardResult | None = None

        def run_gradient_accumulation() -> torch.Tensor:
            nonlocal result
            result = self._run_gradient_accumulation(
                preprocessed_accumulation_step_inputs, global_valid_tokens
            )
            return result.loss

        if self.sdc_replayer is not None:
            loss = self.sdc_replayer.run_fwd_bwd(
                run_gradient_accumulation,
                step=self.num_completed_steps + 1,
            )
            assert result is not None
            result = ForwardBackwardResult(loss, result.loss_metrics)
        else:
            result = self._run_gradient_accumulation(
                preprocessed_accumulation_step_inputs, global_valid_tokens
            )

        local_loss = (
            result.loss.to_local() if isinstance(result.loss, DTensor) else result.loss
        )
        # int32 is supported by NCCL reductions, unlike bool.
        self.loss_is_finite = torch.isfinite(local_loss).all().to(torch.int32)
        return result

    def _preprocess_accumulation_step_inputs(
        self,
        accumulation_step: list[TrainingMicrobatch],
    ) -> tuple[Any, ...]:
        """Move and preprocess one accumulation step outside CUDA capture."""
        prepared_microbatches: list[tuple[Any, ...]] = []
        for microbatch in accumulation_step:
            with sl.log_trace_span("preprocess_inputs"):
                inputs, labels, model_kwargs = cast(
                    BaseModel, self.model_parts[0]
                ).preprocess_inputs(
                    microbatch.to_input_dict(self.device, non_blocking=True),
                    parallel_dims=self.parallel_dims,
                    parallelism=self.config.parallelism,
                    max_num_documents=self.max_num_documents,
                    max_context_length=self.config.training.max_context_length,
                    **self.preprocess_inputs_kwargs,
                )
            self.ntokens_seen += (
                self.config.training.num_tokens_per_microbatch_per_dp_rank
                // self.parallel_dims.cp
            )
            prepared_microbatches.append(
                (
                    inputs,
                    labels,
                    model_kwargs,
                    microbatch.to_loss_kwargs(self.device, non_blocking=True),
                )
            )

        if not self.parallel_dims.pp_enabled:
            assert len(prepared_microbatches) == 1
            return prepared_microbatches[0]

        if any(loss_kwargs for *_, loss_kwargs in prepared_microbatches):
            raise ValueError(
                "Per-microbatch loss arguments are not supported with "
                "pipeline parallelism yet."
            )
        arg_mbs = (
            [(inputs,) for inputs, *_ in prepared_microbatches]
            if self.pp_has_first_stage
            else None
        )
        kwarg_mbs = [model_kwargs for _, _, model_kwargs, _ in prepared_microbatches]
        target_mbs = (
            [labels for _, labels, _, _ in prepared_microbatches]
            if self.pp_has_last_stage
            else None
        )
        return arg_mbs, kwarg_mbs, target_mbs

    def _gradient_accumulation_body(
        self,
        accumulation_step_inputs: list[tuple[Any, ...]],
        global_valid_tokens: torch.Tensor,
    ) -> ForwardBackwardResult:
        """Run all gradient accumulation steps in one graphable call."""
        defer_reduction = self._defer_fsdp_gradient_reduction
        defer_non_pp_reduction = defer_reduction and not self.parallel_dims.pp_enabled
        if defer_non_pp_reduction:
            assert self._fsdp_root is not None
            self._fsdp_root.set_reshard_after_backward(False)
            self._fsdp_root.set_requires_gradient_sync(False)

        accumulated_loss: torch.Tensor | None = None
        loss_metrics: list[dict[str, torch.Tensor]] = []
        last_index = len(accumulation_step_inputs) - 1
        for index, prepared_inputs in enumerate(accumulation_step_inputs):
            is_last = index == last_index
            if not defer_reduction and self.parallel_dims.dp_replicate_enabled:
                for part in self.model_parts:
                    cast(FSDPModule, part).set_requires_all_reduce(is_last)

            self.loss_metrics = {}
            if self.parallel_dims.pp_enabled:
                arg_mbs, kwarg_mbs, target_mbs = prepared_inputs
                loss = self._pp_forward_backward_body(
                    inputs=arg_mbs,
                    model_kwargs=kwarg_mbs,
                    labels=target_mbs,
                    loss_kwargs={"global_valid_tokens": global_valid_tokens},
                    finalize_gradients=not defer_reduction or is_last,
                )
            else:
                inputs, labels, model_kwargs, loss_kwargs = prepared_inputs
                loss = self._non_pp_forward_backward_body(
                    inputs=inputs,
                    labels=labels,
                    model_kwargs=model_kwargs,
                    loss_kwargs={
                        **loss_kwargs,
                        "global_valid_tokens": global_valid_tokens,
                    },
                )

            detached_loss = loss.detach()
            if accumulated_loss is None:
                accumulated_loss = detached_loss.clone()
            else:
                accumulated_loss.add_(detached_loss)
            loss_metrics.append(
                {
                    key: value.detach().clone()
                    for key, value in self.loss_metrics.items()
                }
            )

        if defer_non_pp_reduction:
            assert self._fsdp_root is not None
            self._fsdp_root.set_requires_gradient_sync(True)
            self._fsdp_root.set_reshard_after_backward(True)
            self._fsdp_root.finalize_backward()

        assert accumulated_loss is not None
        return ForwardBackwardResult(accumulated_loss, loss_metrics)

    def _non_pp_forward_backward_body(
        self,
        *,
        inputs: torch.Tensor | tuple[torch.Tensor, ...],
        labels: torch.Tensor | tuple[torch.Tensor, ...],
        model_kwargs: dict[str, Any],
        loss_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        with self.train_context():
            pred = self.model_parts[0](inputs, **model_kwargs)
            loss, self.loss_metrics = self.loss_fn(
                pred,
                labels,  # pyrefly: ignore[bad-argument-type]
                **loss_kwargs,
            )
            del pred
            with spmd.no_typecheck():
                loss.backward()
        return loss

    def _pp_forward_backward_body(
        self,
        *,
        inputs: list[tuple[torch.Tensor, ...]] | None,
        labels: list[torch.Tensor] | None,
        model_kwargs: list[dict[str, Any]],
        loss_kwargs: dict[str, Any],
        finalize_gradients: bool = True,
    ) -> torch.Tensor:
        """Run one PP schedule and optionally finalize its FSDP gradients."""
        with self.train_context():
            losses = [] if self.pp_has_last_stage else None
            self.pp_schedule.step(
                arg_mbs=inputs,
                kwarg_mbs=model_kwargs,
                target_mbs=labels,
                losses=losses,
                loss_kwargs=loss_kwargs,
                return_outputs=False,
                finalize_gradients=finalize_gradients,
            )

        if self.pp_has_last_stage:
            assert losses is not None
            detached_losses = [loss.detach() for loss in losses]
            losses.clear()
            return torch.sum(torch.stack(detached_losses)).to(self.device)
        return self._pp_loss_sentinel_on_non_last_stage

    @sl.log_trace_span("optimizer_step")
    def optimizer_step(self) -> torch.Tensor:
        """Validate gradients, then advance optimizer and learning-rate scheduler."""
        current_step = self.num_completed_steps + 1
        grad_norm = dist_utils.clip_grad_norm_(
            [p for model in self.model_parts for p in model.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
            ep_enabled=self.parallel_dims.ep_enabled,
        )
        if not self.parallel_dims.pp_enabled or self.pp_has_last_stage:
            loss_mesh = self.parallel_dims.get_optional_mesh("loss")
            if loss_mesh is not None:
                torch.distributed.all_reduce(
                    self.loss_is_finite,
                    op=torch.distributed.ReduceOp.MIN,
                    group=loss_mesh.get_group(),
                )
        pp_mesh = self.parallel_dims.get_optional_mesh("pp")
        if pp_mesh is not None:
            torch.distributed.all_reduce(
                self.loss_is_finite,
                op=torch.distributed.ReduceOp.MIN,
                group=pp_mesh.get_group(),
            )
        step_is_finite = self.loss_is_finite.logical_and(
            torch.isfinite(grad_norm).all()
        )
        torch._assert_async(
            step_is_finite,
            "Loss or gradient norm is not finite on at least one rank at "
            f"step {current_step}. Stopping training before the optimizer update.",
        )
        if hasattr(self, "checkpointer"):
            self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()
        self.num_completed_steps = current_step
        self._num_optimizer_steps_since_cuda_graph_init += 1
        return grad_norm

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.num_completed_steps, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.num_completed_steps = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]
        if self.sdc_replayer is not None:
            self.sdc_replayer.reset_schedule()

    def load_checkpoint(self) -> bool:
        checkpointer_config = self.config.checkpointer
        if checkpointer_config is None:
            return False
        return self.checkpointer.load(step=checkpointer_config.load_step)

    def save_checkpoint(self, *, last_step: bool = False) -> bool:
        if not hasattr(self, "checkpointer"):
            return False
        return self.checkpointer.save(self.num_completed_steps, last_step=last_step)

    def start_profiler(self) -> None:
        self.profiler = self.config.profiler.build(
            global_step=self.num_completed_steps,
            base_folder=self.output_dir,
        )
        self.profiler.__enter__()

    def step_profiler(self) -> None:
        """Signal that one training-loop iteration has completed."""
        self.profiler.step()

    def close_profiler(self) -> None:
        """Close the profiler if it was started."""
        if hasattr(self, "profiler"):
            self.profiler.__exit__(None, None, None)
            del self.profiler

    def close(self) -> None:
        """Release CUDA graph and checkpoint resources owned by the trainer."""
        self.close_profiler()
        if not self.config.training.disable_cuda_graphs:
            cuda_graph_teardown()
        if hasattr(self, "checkpointer"):
            self.checkpointer.close()
