# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, cast

import spmd_types as spmd
import torch
import torch.distributed.checkpoint.stateful
import tyro

from torchtitan.components.checkpointer import BaseCheckpointManager, CheckpointManager
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.components.loss import BaseLoss, ChunkedLossWrapper
from torchtitan.components.optimizer import (
    EMA,
    LRSchedulersContainer,
    OptimizersContainer,
)
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
    cuda_graphs_supported,
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
from torchtitan.observability.sdc_replayer import ScalarStateAccessor, SDCReplayer
from torchtitan.protocols import BaseModel
from torchtitan.quantization.utils import has_quantization
from torchtitan.tools import utils


logger = logging.getLogger(__name__)

_NUM_CUDA_GRAPH_WARMUP_STEPS = 2


class TrainingEngine(Configurable, torch.distributed.checkpoint.stateful.Stateful):
    """Shared distributed training engine.

    A microbatch is one data-parallel-rank-local forward/backward input. A
    microbatch group is one gradient-accumulation unit: it contains one
    microbatch without pipeline parallelism and all pipeline microbatches for
    one schedule step with pipeline parallelism. One or more microbatch groups
    contribute to each optimizer step.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )
        lr_scheduler: LRSchedulersContainer.Config = field(
            default_factory=LRSchedulersContainer.Config
        )
        ema: EMA.Config | None = None
        """Online EMA of model weights, e.g. for cheap mid-WSD-training eval
        without a full LR decay. Unset (None) means EMA is disabled."""
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
    parallel_dims: ParallelDims
    model_parts: list[BaseModel]
    model_config: BaseModel.Config
    output_dir: str
    loss_fn: BaseLoss
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer
    ema: EMA | None
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
            topology = dist_utils.init_distributed(
                config.comm,
                enable_cpu_backend=config.training.enable_cpu_offload,
                base_folder=self.output_dir,
                pipeline_parallel_degree=config.parallelism.pipeline_parallel_degree,
            )
        self.parallel_dims = ParallelDims.from_config(config.parallelism, topology)
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
        *,
        compile_config: CompileConfig | None,
        hf_assets_path: str,
        dataloader: BaseDataLoader | None = None,
        create_seed_checkpoint: bool = False,
    ) -> None:
        """Initialize model execution and the state required to train it."""
        self._initialize_model(
            compile_config=compile_config,
            hf_assets_path=hf_assets_path,
            create_seed_checkpoint=create_seed_checkpoint,
        )
        self.model_device_mem_stats = self.device_memory_monitor.get_peak_stats()
        self._initialize_optimizer()
        self._initialize_checkpointer(
            dataloader=dataloader,
            sd_adapter=self.state_dict_adapter,
        )
        self._initialize_forward_backward()

    def _initialize_model(
        self,
        *,
        compile_config: CompileConfig | None,
        hf_assets_path: str,
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
        if self.parallel_dims.pp_enabled:
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = model.pipeline(
                parallel_dims=self.parallel_dims,
                training=config.training,
                parallelism=config.parallelism,
                compile_config=compile_config,
                ac_config=config.activation_checkpoint,
                dump_folder=self.output_dir,
                device=self.device,
                model_config=self.model_config,
                loss_fn=self.loss_fn,
            )
            del model
        else:
            if not create_seed_checkpoint:
                model = model.parallelize(
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

        with dist_utils.get_spmd_context(parallel_dims=self.parallel_dims):
            for model_part in self.model_parts:
                model_part.to_empty(device=init_device)
                with torch.no_grad():
                    model_part.init_weights(buffer_device=buffer_device)
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
            f"Model {type(self.model_config).__qualname__} size: "
            f"{self.model_param_count:,} total parameters"
        )

    def _initialize_optimizer(self) -> None:
        """Construct optimizers, learning-rate schedulers and the weight EMA."""
        self.optimizers = self.config.optimizer.build(model_parts=self.model_parts)
        self.model_cls._register_optimizer_hooks(
            self.optimizers,
            self.model_parts,
            self.parallel_dims,
        )
        self.lr_schedulers = self.config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=self.config.training.steps,
        )
        self.ema = (
            self.config.ema.build(model_parts=self.model_parts)
            if self.config.ema is not None
            else None
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
            ema=self.ema,
            states={"train_state": self},
            sd_adapter=sd_adapter,
            base_folder=self.output_dir,
        )

    def _initialize_forward_backward(self) -> None:
        """Build SDC replay and select PP/eager or CUDA graph execution."""
        sdc_config = self.config.sdc_replayer
        self.sdc_replayer = None
        if sdc_config is not None:
            self.sdc_replayer = sdc_config.build(
                modules=self.model_parts,
                device=self.device,
                scalar_state={
                    "ntokens_seen": ScalarStateAccessor(
                        get=lambda: self.ntokens_seen,
                        set=lambda value: setattr(self, "ntokens_seen", value),
                    )
                },
            )

        self._num_optimizer_steps_since_cuda_graph_init = 0
        if self.parallel_dims.pp_enabled:
            self.forward_backward_body_fn = cast(
                Callable[..., torch.Tensor], self._pp_forward_backward_body
            )
            self._pp_loss_sentinel_on_non_last_stage = torch.full(
                (1,), -1.0, device=self.device
            )
        else:
            self.forward_backward_body_fn = cast(
                Callable[..., torch.Tensor], self._non_pp_forward_backward_body
            )

        if not self.config.training.disable_cuda_graphs:
            eager_forward_backward_body_fn = self.forward_backward_body_fn
            cuda_graph_forward_backward_body_fn = wrap_with_cuda_graph(
                eager_forward_backward_body_fn
            )

            if (
                cuda_graph_forward_backward_body_fn
                is not eager_forward_backward_body_fn
            ):

                def run_with_cuda_graph(**kwargs: Any) -> torch.Tensor:
                    # Count complete optimizer steps instead of forward/backward
                    # calls: accumulation and SDC replay can produce a variable
                    # call count.
                    if (
                        self._num_optimizer_steps_since_cuda_graph_init
                        < _NUM_CUDA_GRAPH_WARMUP_STEPS
                    ):
                        return run_eager_on_cuda_graph_stream(
                            eager_forward_backward_body_fn, **kwargs
                        )
                    return cuda_graph_forward_backward_body_fn(**kwargs)

                self.forward_backward_body_fn = run_with_cuda_graph

    @sl.log_trace_span("prepare_step")
    def prepare_step(
        self,
        global_valid_tokens: int | torch.Tensor,
        *,
        num_accumulation_steps: int = 1,
    ) -> torch.Tensor:
        """Prepare one optimizer step and record its accumulation plan."""
        if num_accumulation_steps < 1:
            raise ValueError("num_accumulation_steps must be greater than 0.")
        self.num_accumulation_steps = num_accumulation_steps
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
        return global_valid_tokens

    @sl.log_trace_span("forward_backward_microbatch")
    def forward_backward_microbatch(
        self,
        *,
        microbatch_group: list[TrainingMicrobatch],
        global_valid_tokens: torch.Tensor,
        accumulation_index: int = 0,
    ) -> torch.Tensor:
        """Preprocess and execute one gradient-accumulation unit.

        A microbatch group forms one complete PP step. Without PP, the group
        contains exactly one data-parallel-rank-local microbatch.

        Args:
            microbatch_group: Microbatches forming one complete pipeline step.
            global_valid_tokens: Valid-token count used to normalize the loss.
            accumulation_index: Index of this gradient-accumulation unit.

        Returns:
            The detached loss for this gradient-accumulation unit.
        """
        if accumulation_index == 0:
            # int32 is supported by NCCL reductions, unlike bool.
            self.loss_is_finite = torch.ones((), dtype=torch.int32, device=self.device)

        # HSDP replicate all-reduce is a no-op until the last accum group.
        # Do not toggle under CUDA graphs when accum > 1: the graph is
        # captured on the first group and replayed for later groups.
        if self.parallel_dims.dp_replicate_enabled and (
            self.num_accumulation_steps == 1 or self.config.training.disable_cuda_graphs
        ):
            is_last = accumulation_index == self.num_accumulation_steps - 1
            for part in self.model_parts:
                part.set_requires_all_reduce(is_last)  # pyrefly: ignore[not-callable]

        def forward_backward() -> torch.Tensor:
            if self.parallel_dims.pp_enabled:
                if any(microbatch.loss_kwargs() for microbatch in microbatch_group):
                    raise ValueError(
                        "Per-microbatch loss arguments are not supported with "
                        "pipeline parallelism yet."
                    )
                arg_mbs: list[tuple[torch.Tensor, ...]] = []
                kwarg_mbs: list[dict[str, Any]] = []
                target_mbs: list[torch.Tensor] | None = (
                    [] if self.pp_has_last_stage else None
                )
                for microbatch in microbatch_group:
                    input_dict = microbatch.to_input_dict(
                        self.device, non_blocking=True
                    )
                    with sl.log_trace_span("preprocess_inputs"):
                        inputs_mb, labels_mb, extra_kwargs_mb = self.model_parts[
                            0
                        ].preprocess_inputs(
                            input_dict,
                            parallel_dims=self.parallel_dims,
                            parallelism=self.config.parallelism,
                            max_num_documents=self.max_num_documents,
                            max_context_length=(
                                self.config.training.max_context_length
                            ),
                            **self.preprocess_inputs_kwargs,
                        )
                        assert isinstance(inputs_mb, torch.Tensor)
                        assert isinstance(labels_mb, torch.Tensor)
                        # Count logical input tokens instead of target elements:
                        # model preprocessing can change the target shape, as in
                        # Flux latent targets. CP shards the logical tokens.
                        self.ntokens_seen += (
                            self.config.training.num_tokens_per_microbatch_per_dp_rank
                            // self.parallel_dims.cp
                        )
                    if self.pp_has_first_stage:
                        arg_mbs.append((inputs_mb,))
                    kwarg_mbs.append(extra_kwargs_mb)
                    if target_mbs is not None:
                        target_mbs.append(labels_mb)

                return self.forward_backward_body_fn(
                    inputs=arg_mbs if self.pp_has_first_stage else None,
                    model_kwargs=kwarg_mbs,
                    labels=target_mbs,
                    loss_kwargs={"global_valid_tokens": global_valid_tokens},
                )

            assert len(microbatch_group) == 1
            microbatch = microbatch_group[0]
            input_dict = microbatch.to_input_dict(self.device, non_blocking=True)
            with sl.log_trace_span("preprocess_inputs"):
                inputs, labels, extra_kwargs = self.model_parts[0].preprocess_inputs(
                    input_dict,
                    parallel_dims=self.parallel_dims,
                    parallelism=self.config.parallelism,
                    max_num_documents=self.max_num_documents,
                    max_context_length=self.config.training.max_context_length,
                    **self.preprocess_inputs_kwargs,
                )
                # Count logical input tokens instead of target elements: model
                # preprocessing can change the target shape or structure, as in
                # Flux latent targets and MTP label tuples. CP shards the tokens.
                self.ntokens_seen += (
                    self.config.training.num_tokens_per_microbatch_per_dp_rank
                    // self.parallel_dims.cp
                )

            return self.forward_backward_body_fn(
                inputs=inputs,
                labels=labels,
                model_kwargs=extra_kwargs,
                loss_kwargs={
                    **microbatch.to_loss_kwargs(self.device, non_blocking=True),
                    "global_valid_tokens": global_valid_tokens,
                },
            )

        if self.sdc_replayer is not None and accumulation_index == 0:
            loss = self.sdc_replayer.run_fwd_bwd(
                forward_backward, step=self.num_completed_steps + 1
            )
        else:
            loss = forward_backward()
        detached_loss = loss.detach()
        self.loss_is_finite.logical_and_(torch.isfinite(detached_loss).all())
        return detached_loss

    def _non_pp_forward_backward_body(
        self,
        *,
        inputs: torch.Tensor | tuple[torch.Tensor, ...],
        labels: torch.Tensor | tuple[torch.Tensor, ...],
        model_kwargs: dict[str, Any],
        loss_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        with dist_utils.get_spmd_context(
            parallel_dims=self.parallel_dims,
            spmd_typechecking=self.config.debug.spmd_typechecking,
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

    def _pp_forward_backward_body(
        self,
        *,
        inputs: list[tuple[torch.Tensor, ...]] | None,
        labels: list[torch.Tensor] | None,
        model_kwargs: list[dict[str, Any]],
        loss_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        with dist_utils.get_spmd_context(
            parallel_dims=self.parallel_dims,
            spmd_typechecking=self.config.debug.spmd_typechecking,
        ):
            losses = [] if self.pp_has_last_stage else None
            self.pp_schedule.step(
                arg_mbs=inputs,
                kwarg_mbs=model_kwargs,
                target_mbs=labels,
                losses=losses,
                loss_kwargs=loss_kwargs,
                return_outputs=False,
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
        if self.ema is not None:
            # current_step is the step just optimized, which is what the EMA
            # schedule's start_step/update_every_n_steps are defined against.
            self.ema.step(current_step)
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
