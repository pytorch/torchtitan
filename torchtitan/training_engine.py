# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any, cast, NamedTuple, TYPE_CHECKING, TypeAlias

import spmd_types as spmd
import torch
import torch.distributed.checkpoint.stateful
from torch.distributed.fsdp import FSDPModule

from torchtitan.components.checkpointer import BaseCheckpointManager, CheckpointManager
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.components.loss import BaseLoss, ChunkedLossWrapper
from torchtitan.components.optim import Optim
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import CommConfig, DebugConfig, TrainingConfig
from torchtitan.config.override import OverrideConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext, utils as dist_utils
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointingConfig,
    SelectiveAC,
)
from torchtitan.distributed.batch_invariant import set_batch_invariance
from torchtitan.distributed.cuda_graph import (
    cuda_graph_teardown,
    cuda_graphs_supported,
    NUM_CUDA_GRAPH_WARMUP_STEPS,
    wrap_fwd_bwd_with_cuda_graph,
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
from torchtitan.quantization.utils import has_quantization
from torchtitan.tools import utils
from torchtitan.tools.garbage_collector import GarbageCollector


if TYPE_CHECKING:
    from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime


logger = logging.getLogger(__name__)


class ForwardBackwardResult(NamedTuple):
    loss: torch.Tensor
    loss_metrics: list[dict[str, torch.Tensor]]


_ForwardBackwardFn: TypeAlias = Callable[
    [list[tuple[Any, ...]], torch.Tensor], ForwardBackwardResult
]


class TrainingEngine(Configurable, torch.distributed.checkpoint.stateful.Stateful):
    """Shared distributed training engine.

    A microbatch is one data-parallel-rank-local forward/backward input. A
    microbatch group contains one microbatch without pipeline parallelism and
    all pipeline microbatches for one schedule step with pipeline parallelism.
    One or more microbatch groups contribute to each optimizer step.

    Forward and backward calls follow this flow::

        TrainingEngine.initialize
        |
        +-- _initialize_forward_backward

        Trainer.train_step
        |
        +-- TrainingEngine.forward_backward
            |
            +-- _preprocess_microbatch_groups
            |
            +-- _run_forward_backward =
                _forward_backward_body (maybe_wrapped_with_cuda_graph)
                |
                +-- _pp_forward_backward_microbatch_group
                |
                +-- _non_pp_forward_backward_microbatch

        Trainer.train_step
        |
        +-- TrainingEngine.optim_step
            |
            +-- Optim.step
                |
                +-- _update (maybe_wrapped_with_cuda_graph)
                +-- lr_schedulers.step
                +-- ema.step
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        optim: Optim.Config = field(default_factory=Optim.Config)
        training: TrainingConfig = field(default_factory=TrainingConfig)
        garbage_collector: GarbageCollector.Config = field(
            default_factory=GarbageCollector.Config
        )
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        checkpointer: CheckpointManager.Config | None = None
        activation_checkpoint: ActivationCheckpointingConfig = field(
            default_factory=SelectiveAC.Config
        )
        profiler: Profiler.Config = field(default_factory=Profiler.Config)
        sdc_replayer: SDCReplayer.Config | None = None
        comm: CommConfig = field(default_factory=CommConfig)
        debug: DebugConfig = field(default_factory=DebugConfig)
        override: OverrideConfig = field(default_factory=OverrideConfig)
        loss: BaseLoss.Config = field(default_factory=BaseLoss.Config)
        dist_moe: Configurable.Config | None = None
        """Optional rank-wide Dist-MoE memory and pipeline-slot policy."""

        def __post_init__(self) -> None:
            if (
                self.dist_moe is not None
                and self.training.mixed_precision_param != "bfloat16"
            ):
                raise ValueError("Dist-MoE requires mixed_precision_param='bfloat16'")
            if (
                self.debug.spmd_typechecking
                and self.parallelism.pipeline_parallel_degree > 1
            ):
                # TODO(sanketpurandare): Enable SPMD typechecking under PP.
                raise ValueError(
                    "SPMD typechecking is not supported with pipeline parallelism. "
                    "Validate the same config without PP "
                    "(parallelism.pipeline_parallel_degree=1)."
                )

            if (
                self.training.enable_cpu_offload
                and "dp_shard" not in self.debug.distinct_seed_mesh_axes
            ):
                raise ValueError(
                    "training.enable_cpu_offload requires 'dp_shard' in "
                    "debug.distinct_seed_mesh_axes; otherwise every FSDP shard "
                    "is initialized with the same CPU RNG stream."
                )

            if self.parallelism.num_pp_microbatches <= 0:
                raise ValueError(
                    "parallelism.num_pp_microbatches must be greater than 0."
                )
            if self.optim.enable_cuda_graph and self.training.disable_cuda_graphs:
                raise ValueError(
                    "The optimization CUDA graph requires CUDA graphs to be enabled."
                )
            num_tokens = self.training.num_tokens_per_microbatch_per_dp_rank
            sequence_parallel_degree = (
                self.parallelism.tensor_parallel_degree
                if self.parallelism.enable_sequence_parallel
                else 1
            )
            context_parallel_degree = self.parallelism.context_parallel_degree
            activation_shard_degree = sequence_parallel_degree * context_parallel_degree
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
                    and cuda_graphs_supported()
                    and self.sdc_replayer.num_replays > 1
                ):
                    raise ValueError(
                        "SDC replay supports at most one replay when CUDA graphs "
                        "are enabled: set sdc_replayer.num_replays=1 or "
                        "training.disable_cuda_graphs=True."
                    )

    config: Config
    device: torch.device
    parallelism_context: ParallelismContext
    model_parts: list[BaseModel]
    model_config: BaseModel.Config
    output_dir: str
    loss_fn: BaseLoss
    optim: Optim
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
    loss: torch.Tensor
    loss_metrics: dict[str, torch.Tensor]
    device_memory_monitor: DeviceMemoryMonitor
    model_device_mem_stats: DeviceMemStats
    _run_forward_backward: _ForwardBackwardFn
    _dist_moe_runtime: "DistMoeRuntime | None"

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
        self.sdc_replayer = None
        self._dist_moe_runtime = None
        self.preprocess_inputs_kwargs: dict[str, Any] = {}
        self.loss_metrics = {}
        self._initialize_distributed_runtime()

    def _initialize_distributed_runtime(self) -> None:
        """Initialize the device, distributed meshes, GC, and deterministic RNGs."""
        device_module, device_type = utils.device_module, utils.device_type
        # pyrefly: ignore [read-only]
        self.device = utils.get_local_device()
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)
        config = self.config
        set_batch_invariance(config.debug.batch_invariant)
        with sl.log_trace_span("torch_distributed_init"):
            topology = dist_utils.init_distributed(
                config.comm,
                enable_cpu_backend=config.training.enable_cpu_offload,
                base_folder=self.output_dir,
                pipeline_parallel_degree=config.parallelism.pipeline_parallel_degree,
            )
        self.parallelism_context = ParallelismContext.from_config(
            config.parallelism, topology
        )
        self.garbage_collector = config.garbage_collector.build()
        dist_utils.set_determinism(
            self.parallelism_context,
            self.device,
            config.debug,
            distinct_seed_mesh_axes=config.debug.distinct_seed_mesh_axes,
        )
        self.device_memory_monitor = build_device_memory_monitor()

    @sl.log_trace_span("initialize")
    def initialize(
        self,
        *,
        hf_assets_path: str,
        dataloader: BaseDataLoader | None = None,
        create_seed_checkpoint: bool = False,
    ) -> None:
        """Initialize model execution and the state required to train it."""
        self._initialize_model(
            hf_assets_path=hf_assets_path,
            create_seed_checkpoint=create_seed_checkpoint,
        )
        self.model_device_mem_stats = self.device_memory_monitor.get_peak_stats()
        self._initialize_optim()
        self._initialize_checkpointer(
            dataloader=dataloader,
            sd_adapter=self.state_dict_adapter,
        )
        if create_seed_checkpoint:
            return
        self._initialize_forward_backward()

    def _initialize_model(
        self,
        *,
        hf_assets_path: str,
        create_seed_checkpoint: bool = False,
    ) -> None:
        """Build the loss and model execution state."""
        self.loss_fn = self.config.loss.build()
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
            self.parallelism_context.activate_spmd(),
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[self.config.training.dtype]),
        ):
            model = self.model_config.build()
        self.model_cls = type(model)
        adapter_cls = type(model).state_dict_adapter_cls
        self.state_dict_adapter = (
            adapter_cls(self.model_config, hf_assets_path)
            if adapter_cls is not None
            else None
        )
        (
            self.model_param_count,
            self.num_flops_per_token,
        ) = self.model_config.get_nparams_and_flops(
            model, self.config.training.max_context_length
        )
        config = self.config
        if self.parallelism_context.pp_enabled:
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = model.pipeline(
                parallelism_context=self.parallelism_context,
                training=config.training,
                parallelism=config.parallelism,
                ac_config=config.activation_checkpoint,
                dump_folder=self.output_dir,
                device=self.device,
                model_config=self.model_config,
                loss_fn=self.loss_fn,
                **self._parallelize_compile_kwargs(),
            )
            del model
        else:
            if not create_seed_checkpoint:
                model = model.parallelize(
                    parallelism_context=self.parallelism_context,
                    training=config.training,
                    parallelism=config.parallelism,
                    ac_config=config.activation_checkpoint,
                    dump_folder=self.output_dir,
                    **self._parallelize_compile_kwargs(),
                )
            self.model_parts = [model]
            self.pp_has_first_stage = True
            self.pp_has_last_stage = True

        with self.parallelism_context.activate_spmd():
            for model_part in self.model_parts:
                model_part.to_empty(device=init_device)
                with torch.no_grad():
                    model_part.init_weights(buffer_device=buffer_device)
                model_part.train()

        if isinstance(self.loss_fn, ChunkedLossWrapper) and (
            not self.parallelism_context.pp_enabled or self.pp_has_last_stage
        ):
            if self.parallelism_context.pp_enabled:
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
            f"Model {type(self.model_config).__qualname__} size: "
            f"{self.model_param_count:,} total parameters"
        )

    def _parallelize_compile_kwargs(self) -> dict[str, Any]:
        """Return the compile kwargs ``model.parallelize``/``model.pipeline`` expect for this engine's models.

        Regular models expect ``local_compile_regions`` (compiled regions); GraphTrainer overrides this to
        return ``compile_config`` (whole-step compile).
        """
        # TODO: apply local compile outside parallelize/pipeline (#5026 review). That needs
        # GraphTrainer to stop reading its compile config there first (apply_compile,
        # EP-overlap chunking, enable_autoparallel, GraphPP runtime); then delete this hook.
        return {"local_compile_regions": self.model_config.local_compile_regions}

    def _initialize_optim(self) -> None:
        """Construct the parameter update and its state."""
        self.optim = self.config.optim.build(
            model_parts=self.model_parts,
            parallelism_context=self.parallelism_context,
            training_steps=self.config.training.steps,
            pp_has_last_stage=self.pp_has_last_stage,
        )
        self.model_cls._register_optimizer_hooks(
            self.optim.optimizers,
            self.model_parts,
            self.parallelism_context,
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
            optimizers=self.optim.optimizers,
            lr_schedulers=self.optim.lr_schedulers,
            ema=self.optim.ema,
            states={"train_state": self},
            sd_adapter=sd_adapter,
            base_folder=self.output_dir,
        )

    def _initialize_forward_backward(self) -> None:
        """Build SDC replay and the gradient accumulation execution path."""
        if self.config.dist_moe is not None:
            self._dist_moe_runtime = self.config.dist_moe.build(
                model_parts=self.model_parts,
                parallelism_context=self.parallelism_context,
                device=self.device,
                num_tokens_per_microbatch_per_dp_rank=(
                    self.config.training.num_tokens_per_microbatch_per_dp_rank
                ),
                pp_schedule=(
                    self.pp_schedule if self.parallelism_context.pp_enabled else None
                ),
                functional_wgrad_dtype=TORCH_DTYPE_MAP[
                    self.config.training.mixed_precision_param
                ],
            )

        sdc_config = self.config.sdc_replayer
        self.sdc_replayer = None
        if sdc_config is not None:
            self.sdc_replayer = sdc_config.build(
                modules=self.model_parts,
                device=self.device,
            )

        if self.parallelism_context.pp_enabled:
            self._pp_loss_sentinel_on_non_last_stage = torch.full(
                (1,), -1.0, device=self.device
            )

        eager_forward_backward_fn = partial(
            self._forward_backward_body,
            defer_fsdp_gradient_reduction=(
                self.config.parallelism.fsdp_defer_gradient_reduction
            ),
        )
        self._run_forward_backward = eager_forward_backward_fn
        if self.config.training.disable_cuda_graphs or not cuda_graphs_supported():
            return

        self._run_forward_backward = wrap_fwd_bwd_with_cuda_graph(
            eager_forward_backward_fn,
            parameters=(
                parameter
                for model_part in self.model_parts
                for parameter in model_part.parameters()
            ),
            num_warmup_iterations=NUM_CUDA_GRAPH_WARMUP_STEPS,
        )

    @sl.log_trace_span("forward_backward")
    def forward_backward(
        self,
        *,
        microbatch_groups: list[list[TrainingMicrobatch]],
        global_valid_tokens: int | torch.Tensor,
    ) -> ForwardBackwardResult:
        """Run all microbatch groups for one optimizer update."""
        if not microbatch_groups:
            raise ValueError("microbatch_groups must not be empty.")
        self.num_accumulation_steps = len(microbatch_groups)
        self.garbage_collector.run(self.num_completed_steps + 1)
        self.optim.zero_grad(set_to_none=True)
        if isinstance(global_valid_tokens, int):
            global_valid_tokens = torch.tensor(
                global_valid_tokens,
                dtype=torch.int64,
                device=self.device,
            )
        # TODO(sdmyzlp): Each MTP depth can have a different valid-token count
        # after shifting and should use its own auxiliary-loss denominator.
        AuxLoss.set_step_denominator(global_valid_tokens)

        preprocessed_microbatch_groups = self._preprocess_microbatch_groups(
            microbatch_groups
        )

        if self.sdc_replayer is not None:
            result = self.sdc_replayer.run_fwd_bwd(
                partial(
                    self._run_forward_backward,
                    preprocessed_microbatch_groups,
                    global_valid_tokens,
                ),
                step=self.num_completed_steps + 1,
                get_loss=lambda result: result.loss,
            )
        else:
            result = self._run_forward_backward(
                preprocessed_microbatch_groups,
                global_valid_tokens,
            )

        self.loss = result.loss
        return result

    def _preprocess_microbatch_groups(
        self,
        microbatch_groups: list[list[TrainingMicrobatch]],
    ) -> list[tuple[Any, ...]]:
        """Preprocess all microbatch groups outside CUDA capture."""
        preprocessed_microbatch_groups: list[tuple[Any, ...]] = []
        for microbatch_group in microbatch_groups:
            prepared_microbatches: list[tuple[Any, ...]] = []
            for microbatch in microbatch_group:
                with (
                    sl.log_trace_span("preprocess_inputs"),
                    self.parallelism_context.activate_spmd(),
                ):
                    inputs, labels, model_kwargs = self.model_parts[
                        0
                    ].preprocess_inputs(
                        microbatch.to_input_dict(self.device, non_blocking=True),
                        parallelism_context=self.parallelism_context,
                        parallelism=self.config.parallelism,
                        max_num_documents=self.max_num_documents,
                        max_context_length=self.config.training.max_context_length,
                        **self.preprocess_inputs_kwargs,
                    )
                self.ntokens_seen += (
                    self.config.training.num_tokens_per_microbatch_per_dp_rank
                    // self.parallelism_context.cp
                )
                prepared_microbatches.append(
                    (
                        inputs,
                        labels,
                        model_kwargs,
                        microbatch.to_loss_kwargs(self.device, non_blocking=True),
                    )
                )

            if not self.parallelism_context.pp_enabled:
                assert len(prepared_microbatches) == 1
                preprocessed_microbatch_groups.append(prepared_microbatches[0])
                continue

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
            kwarg_mbs = [
                model_kwargs for _, _, model_kwargs, _ in prepared_microbatches
            ]
            target_mbs = (
                [labels for _, labels, _, _ in prepared_microbatches]
                if self.pp_has_last_stage
                else None
            )
            preprocessed_microbatch_groups.append((arg_mbs, kwarg_mbs, target_mbs))
        return preprocessed_microbatch_groups

    def _forward_backward_body(
        self,
        microbatch_groups: list[tuple[Any, ...]],
        global_valid_tokens: torch.Tensor,
        *,
        defer_fsdp_gradient_reduction: bool,
    ) -> ForwardBackwardResult:
        """Run all microbatch groups in one graphable call."""

        accumulated_loss: torch.Tensor | None = None
        loss_metrics: list[dict[str, torch.Tensor]] = []
        num_accumulation_steps = len(microbatch_groups)
        for accumulation_index, prepared_inputs in enumerate(microbatch_groups):
            is_last_accumulation_step = accumulation_index == num_accumulation_steps - 1

            if self.parallelism_context.dp_replicate_enabled:
                # All-reduce HSDP replicas only with the final accumulated gradient.
                for model_part in self.model_parts:
                    fsdp_root = cast(FSDPModule, model_part)
                    fsdp_root.set_requires_all_reduce(is_last_accumulation_step)

            self.loss_metrics = {}
            if self.parallelism_context.pp_enabled:
                arg_mbs, kwarg_mbs, target_mbs = prepared_inputs
                # Finalization runs after the group's last PP microbatch.
                loss = self._pp_forward_backward_microbatch_group(
                    inputs=arg_mbs,
                    model_kwargs=kwarg_mbs,
                    labels=target_mbs,
                    loss_kwargs={"global_valid_tokens": global_valid_tokens},
                    finalize_gradients=(
                        not defer_fsdp_gradient_reduction or is_last_accumulation_step
                    ),
                )
            else:
                if defer_fsdp_gradient_reduction:
                    fsdp_root = cast(FSDPModule, self.model_parts[0])
                    fsdp_root.set_is_last_backward(is_last_accumulation_step)
                    fsdp_root.set_reshard_after_backward(is_last_accumulation_step)
                    fsdp_root.set_requires_gradient_sync(is_last_accumulation_step)
                inputs, labels, model_kwargs, loss_kwargs = prepared_inputs
                loss = self._non_pp_forward_backward_microbatch(
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

        assert accumulated_loss is not None
        return ForwardBackwardResult(accumulated_loss, loss_metrics)

    def _non_pp_forward_backward_microbatch(
        self,
        *,
        inputs: torch.Tensor | tuple[torch.Tensor, ...],
        labels: torch.Tensor | tuple[torch.Tensor, ...],
        model_kwargs: dict[str, Any],
        loss_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        with self.parallelism_context.activate_spmd(
            typechecking=self.config.debug.spmd_typechecking,
        ):
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

    def _pp_forward_backward_microbatch_group(
        self,
        *,
        inputs: list[tuple[torch.Tensor, ...]] | None,
        labels: list[torch.Tensor] | None,
        model_kwargs: list[dict[str, Any]],
        loss_kwargs: dict[str, Any],
        finalize_gradients: bool = True,
    ) -> torch.Tensor:
        """Run one PP microbatch group.

        The input lists contain one pipeline schedule step. ``finalize_gradients``
        controls whether that step finishes FSDP gradient reduction.
        """
        with self.parallelism_context.activate_spmd(
            typechecking=self.config.debug.spmd_typechecking,
        ):
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

    @sl.log_trace_span("optim_step")
    def optim_step(self) -> torch.Tensor:
        """Run one optimization update and advance its eager state."""
        current_step = self.num_completed_steps + 1
        if hasattr(self, "checkpointer"):
            self.checkpointer.maybe_wait_for_staging()
        grad_norm = self.optim.step(
            self.loss,
            current_step=current_step,
        )
        self.num_completed_steps = current_step
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
        if self._dist_moe_runtime is not None:
            self._dist_moe_runtime.close()
            self._dist_moe_runtime = None
        if hasattr(self, "checkpointer"):
            self.checkpointer.close()
