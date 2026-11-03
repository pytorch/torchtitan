# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial
from typing import Any, cast, NamedTuple, Protocol, TYPE_CHECKING

import spmd_types as spmd
import torch
import torch.distributed.checkpoint.stateful
from torch.distributed.fsdp import FSDPModule

from torchtitan.components.checkpointer import BaseCheckpointManager, CheckpointManager
from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.components.loss import ChunkedLossWrapper, Loss
from torchtitan.components.optim import Optim
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import CommConfig, DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext, utils as dist_utils
from torchtitan.distributed.accelerator_graph import (
    accelerator_graph_manager,
    accelerator_graph_teardown,
    accelerator_graphs_supported,
    NUM_ACCELERATOR_GRAPH_WARMUP_STEPS,
    StructuredAcceleratorGraph,
    wrap_fwd_bwd_with_accelerator_graph,
    wrap_with_accelerator_graph,
)
from torchtitan.distributed.activation_remat import (
    ActivationRematConfig,
    DefaultActivationRematConfig,
)
from torchtitan.distributed.batch_invariant import set_batch_invariance
from torchtitan.distributed.pipeline_parallel import (
    prepare_pipeline_schedule_for_accelerator_graph,
)
from torchtitan.models.base import BaseModel
from torchtitan.models.common.aux_loss import aux_loss_denominators
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.metrics import (
    build_device_memory_monitor,
    DeviceMemoryMonitor,
    DeviceMemStats,
)
from torchtitan.observability.profiler import Profiler
from torchtitan.observability.sdc_replayer import SDCReplayer
from torchtitan.quantization.utils import has_quantization
from torchtitan.tools import utils
from torchtitan.tools.garbage_collector import GarbageCollector


if TYPE_CHECKING:
    from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime


logger = logging.getLogger(__name__)


class ForwardBackwardResult(NamedTuple):
    loss: torch.Tensor
    """This rank's loss summed over the call's loss tokens."""
    loss_metrics: list[list[dict[str, torch.Tensor]]]
    """Loss metrics, one list per microbatch group with one dict per
    microbatch. Under PP, they are empty on ranks without the last stage."""


class _ForwardBackwardFn(Protocol):
    def __call__(
        self,
        microbatch_groups: list[Any],
        *,
        preprocess_microbatch_groups_lazily: bool,
    ) -> ForwardBackwardResult:
        """Run the groups; with ``preprocess_microbatch_groups_lazily`` they are raw microbatches."""


class TrainingEngine(Configurable, torch.distributed.checkpoint.stateful.Stateful):
    """Shared distributed training engine.

    A microbatch is one data-parallel-rank-local forward/backward input. A
    microbatch group contains one microbatch without pipeline parallelism and
    all pipeline microbatches for one schedule step with pipeline parallelism.
    One or more microbatch groups contribute to each optimizer step. Losses
    return sums; ``optim_step`` divides every gradient by the step's
    loss-token count.

    Forward and backward calls follow this flow::

        TrainingEngine.initialize
        |
        +-- _initialize_forward_backward

        Trainer.train_step
        |
        +-- TrainingEngine.forward_backward
            |
            +-- _preprocess_microbatch_groups (unless lazy)
            |
            +-- _run_forward_backward =
                _forward_backward_body, or with accelerator graphs
                _run_forward_backward_segments (one graph replay per segment)
                |
                +-- _preprocess_microbatch_group (if lazy)
                |
                +-- _pp_forward_backward_microbatch_group
                |
                +-- _non_pp_forward_backward_microbatch

        Trainer.train_step
        |
        +-- TrainingEngine.optim_step
            |
            +-- divide gradients by the step's loss-token count (all-reduced over DP)
            +-- Optim.step
                |
                +-- _update (maybe_wrapped_with_accelerator_graph)
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
        activation_remat: ActivationRematConfig | None = field(
            default_factory=DefaultActivationRematConfig
        )
        profiler: Profiler.Config = field(default_factory=Profiler.Config)
        sdc_replayer: SDCReplayer.Config | None = None
        comm: CommConfig = field(default_factory=CommConfig)
        debug: DebugConfig = field(default_factory=DebugConfig)
        loss: Loss.Config = field(default_factory=Loss.Config)
        dist_moe_runtime: Configurable.Config | None = None
        """Optional rank-wide Dist-MoE memory and pipeline-slot policy."""

        def __post_init__(self) -> None:
            if (
                self.dist_moe_runtime is not None
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
            if (
                self.optim.enable_accelerator_graph
                and self.training.disable_accelerator_graphs
            ):
                raise ValueError(
                    "The optimization accelerator graph requires accelerator graphs to be enabled."
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

    config: Config
    device: torch.device
    parallelism_context: ParallelismContext
    model_parts: list[BaseModel]
    model_config: BaseModel.Config
    output_dir: str
    loss_fn: Loss[Any]
    optim: Optim
    checkpointer: BaseCheckpointManager
    pp_has_last_stage: bool
    max_num_documents: int | None
    num_completed_steps: int
    ntokens_seen: int
    sdc_replayer: SDCReplayer | None
    model_param_count: int
    num_flops_per_token: int
    has_quantization: bool
    loss: torch.Tensor
    """The step's summed loss over its ``forward_backward`` calls; ``optim_step`` checks it
    is finite."""
    loss_metrics: dict[str, torch.Tensor]
    local_loss_token_count: int
    """This rank's loss tokens in the current step, counted by ``forward_backward``."""
    global_loss_token_count: torch.Tensor
    """The step's loss tokens over the ranks whose gradients it sums (int64), set on
    device by ``optim_step``. A CPU zero before the first step."""
    _aux_loss_denominators: torch.Tensor
    """The current ``forward_backward`` call's ``aux_loss_denominators``, on device."""
    _step_started: bool
    _forward_backward_captures_whole_step: bool
    """One ``forward_backward`` call per optimizer step: set by subclasses whose graph or
    runtime covers the whole step (GraphTrainer)."""
    device_memory_monitor: DeviceMemoryMonitor
    model_device_mem_stats: DeviceMemStats
    _run_forward_backward: _ForwardBackwardFn
    _forward_backward_accelerator_graph_enabled: bool
    graph_segment_is_whole_call: bool
    """With accelerator graphs, whether a segment (the microbatch groups between two
    gradient reductions) is a whole ``forward_backward`` call, as under HSDP or deferred
    gradient reduction. Then the step's first call must keep its group count across steps."""
    _first_segment_graph: StructuredAcceleratorGraph
    _later_segment_graphs: defaultdict[int, StructuredAcceleratorGraph]
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
        self.num_completed_steps = 0
        self.ntokens_seen = 0
        self.local_loss_token_count = 0
        self.global_loss_token_count = torch.zeros((), dtype=torch.int64)
        # Set by forward_backward and cleared by optim_step: the step's first forward_backward call
        # zeroes the gradients, and optim_step divides each step's gradients once.
        self._step_started = False
        self._forward_backward_captures_whole_step = False
        self.sdc_replayer = None
        self._forward_backward_accelerator_graph_enabled = False
        self.graph_segment_is_whole_call = False
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
        # TODO: Remove this workaround once supported PyTorch builds ship
        # NCCL >= 2.32.3.
        dist_utils.configure_nccl_for_ulysses(
            self.model_config,
            device_type=self.device.type,
            comm_backend=config.comm.backend,
        )
        with sl.log_trace_span("torch_distributed_init"):
            topology = dist_utils.init_distributed(
                config.comm,
                enable_cpu_backend=config.training.enable_cpu_offload,
                base_folder=self.output_dir,
                pipeline_parallel_degree=config.parallelism.pipeline_parallel_degree,
            )
        self.parallelism_context = ParallelismContext.from_config(
            config.parallelism, topology, dump_folder=self.output_dir
        )
        # DeviceMesh subgroups do not inherit the world group's timeout.
        dist_utils.set_pg_timeouts(
            timeout=timedelta(seconds=config.comm.init_timeout_seconds),
            parallelism_context=self.parallelism_context,
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
        extra_states: dict[str, torch.distributed.checkpoint.stateful.Stateful]
        | None = None,
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
            extra_states=extra_states,
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
                activation_remat=config.activation_remat,
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
                    activation_remat=config.activation_remat,
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

        Regular models expect ``local_compile`` (compiled regions); GraphTrainer overrides this to
        return ``compile_config`` (whole-step compile).
        """
        # TODO: apply local compile outside parallelize/pipeline (#5026 review). That needs
        # GraphTrainer to stop reading its compile config there first (apply_compile,
        # EP-overlap chunking, enable_autoparallel, GraphPP runtime); then delete this hook.
        return {"local_compile": self.model_config.local_compile}

    def _initialize_optim(self) -> None:
        """Construct the parameter update and its state."""
        self.optim = self.config.optim.build(
            model_parts=self.model_parts,
            parallelism_context=self.parallelism_context,
            training_steps=self.config.training.steps,
            pp_has_last_stage=self.pp_has_last_stage,
            cuda_graph_annotation_config=self.config.profiler.maybe_cuda_graph_annotation_config(),
        )
        self.model_cls._register_optimizer_hooks(
            self.optim.optimizers,
            self.model_parts,
            self.parallelism_context,
        )

    def _initialize_checkpointer(
        self,
        *,
        extra_states: dict[str, torch.distributed.checkpoint.stateful.Stateful] | None,
        sd_adapter: Any | None,
    ) -> None:
        """Build checkpointing around core and optional owner-provided state."""
        checkpointer_config = self.config.checkpointer
        if checkpointer_config is None:
            return
        self.checkpointer = checkpointer_config.build(
            model_parts=self.model_parts,
            optimizers=self.optim.optimizers,
            lr_schedulers=self.optim.lr_schedulers,
            ema=self.optim.ema,
            extra_states={"train_state": self, **(extra_states or {})},
            sd_adapter=sd_adapter,
            base_folder=self.output_dir,
        )

    def _initialize_forward_backward(self) -> None:
        """Build SDC replay and the gradient accumulation execution path."""
        if self.config.dist_moe_runtime is not None:
            self._dist_moe_runtime = self.config.dist_moe_runtime.build(
                model_parts=self.model_parts,
                parallelism_context=self.parallelism_context,
                device=self.device,
                num_tokens_per_microbatch_per_dp_rank=(
                    self.config.training.num_tokens_per_microbatch_per_dp_rank
                ),
                pp_schedule=(
                    self.pp_schedule if self.parallelism_context.pp_enabled else None
                ),
            )

        sdc_config = self.config.sdc_replayer
        self.sdc_replayer = None
        if sdc_config is not None:
            self.sdc_replayer = sdc_config.build(
                modules=self.model_parts,
                device=self.device,
            )
        self._forward_backward_captures_whole_step = False

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
        self._forward_backward_accelerator_graph_enabled = (
            not self.config.training.disable_accelerator_graphs
            and accelerator_graphs_supported()
        )
        if not self._forward_backward_accelerator_graph_enabled:
            return

        def prepare_for_capture(microbatch_groups) -> None:
            """Prepare training PP metadata outside capture.

            Eval can change the schedule mode after training warmup. Rebuilding
            metadata can synchronize with the CPU, which capture does not allow.
            """
            arg_mbs, kwarg_mbs, target_mbs, loss_kwarg_mbs = microbatch_groups[0]
            with self.parallelism_context.activate_spmd(
                typechecking=self.config.debug.spmd_typechecking,
            ):
                # Initialization calls the loss on the first microbatch, so it
                # needs that microbatch's loss arguments, as in the real step.
                prepare_pipeline_schedule_for_accelerator_graph(
                    self.pp_schedule,
                    args=arg_mbs[0] if arg_mbs is not None else (),
                    kwargs=kwarg_mbs[0],
                    target=target_mbs[0] if target_mbs is not None else None,
                    loss_kwargs=loss_kwarg_mbs[0],
                )

        graph_kwargs: dict[str, Any] = {
            "num_warmup_iterations": NUM_ACCELERATOR_GRAPH_WARMUP_STEPS,
            "prepare_for_capture": prepare_for_capture
            if self.parallelism_context.pp_enabled
            else None,
            "annotation_config": self.config.profiler.maybe_cuda_graph_annotation_config(),
        }
        # The first-segment graph starts each step from None gradients and allocates them in the
        # graph pool. Later-segment graphs add into them, one graph per segment length.
        self._first_segment_graph = wrap_fwd_bwd_with_accelerator_graph(
            eager_forward_backward_fn,
            parameters=(
                parameter
                for model_part in self.model_parts
                for parameter in model_part.parameters()
            ),
            name="forward_backward",
            **graph_kwargs,
        )
        # TODO: with eval graphs, a later-segment length that still has warmup calls left
        # keeps all_wrappers_warmed_up False. Then eval and training don't capture until that
        # length has run twice: training stays eager and validation runs after every update.
        # No config reaches it today (RL builds no validator; the core trainer sends a fixed
        # group count per step).
        self._later_segment_graphs = defaultdict(
            partial(
                wrap_with_accelerator_graph,
                eager_forward_backward_fn,
                name="forward_backward_later_segment",
                **graph_kwargs,
            )
        )
        # HSDP all-reduces replicas, and deferral reduce-scatters, once per call.
        self.graph_segment_is_whole_call = (
            self.parallelism_context.dp_replicate_enabled
            or self.config.parallelism.fsdp_defer_gradient_reduction
        )
        self._run_forward_backward = self._run_forward_backward_segments

    @property
    def step_started(self) -> bool:
        """Whether a ``forward_backward`` call ran since the last ``optim_step``."""
        return self._step_started

    @sl.log_trace_span("forward_backward")
    def forward_backward(
        self,
        *,
        microbatch_groups: list[list[TrainingMicrobatch]],
        aux_loss_denominators: torch.Tensor,
    ) -> ForwardBackwardResult:
        """Run microbatch groups and add their gradients to this optimizer step's gradients.

        The step's first call (the first after ``optim_step``) zeroes the gradients;
        later calls add to them. Each loss is a sum over its loss tokens, and
        ``optim_step`` divides the gradients by the step's count.
        ``aux_loss_denominators`` (``aux_loss.aux_loss_denominators``) divides this
        call's auxiliary losses.

        Example, with d01 and d012 the R / N of mb0-mb1 and of mb0-mb2:
            engine.forward_backward(microbatch_groups=[[mb0], [mb1]], aux_loss_denominators=d01)
            # zeroes, then adds
            engine.forward_backward(microbatch_groups=[[mb2]], aux_loss_denominators=d012)
            # adds to the same step
            # engine.local_loss_token_count: the loss tokens of mb0, mb1 and mb2
            grad_norm = engine.optim_step()  # divides by that count summed over DP ranks
        """
        if not microbatch_groups:
            raise ValueError("microbatch_groups must not be empty.")
        if self._step_started and self._forward_backward_captures_whole_step:
            raise ValueError(
                "This engine runs each optimizer step in one forward_backward call: "
                "call optim_step before the next forward_backward."
            )
        if not self._step_started:
            self.garbage_collector.run(self.num_completed_steps + 1)
            # The first-segment graph needs None gradients: its capture allocates them,
            # and its replays put them back.
            self.optim.zero_grad(set_to_none=True)
            self.local_loss_token_count = 0
        # non_blocking: a pageable copy would wait for the work queued before this call.
        self._aux_loss_denominators = aux_loss_denominators.to(
            self.device, non_blocking=True
        )
        # A checked call runs several times, so it preprocesses its groups once, up front.
        preprocess_lazily = (
            self.config.training.preprocess_microbatch_groups_lazily
            and not (
                self.sdc_replayer is not None and self.sdc_replayer.checks_current_step
            )
        )
        if preprocess_lazily:
            # _forward_backward_body preprocesses each group just before its forward/backward.
            step_inputs: list[Any] = microbatch_groups
        else:
            step_inputs = self._preprocess_microbatch_groups(microbatch_groups)

        run_forward_backward = partial(
            self._run_forward_backward,
            step_inputs,
            preprocess_microbatch_groups_lazily=preprocess_lazily,
        )
        if self.sdc_replayer is not None:
            result = self.sdc_replayer.run_fwd_bwd(
                run_forward_backward,
                step=self.num_completed_steps + 1,
                get_loss=lambda result: result.loss,
            )
        else:
            result = run_forward_backward()

        self.loss = self.loss + result.loss if self._step_started else result.loss
        self._step_started = True
        return result

    def _preprocess_microbatch_groups(
        self,
        microbatch_groups: list[list[TrainingMicrobatch]],
    ) -> list[tuple[Any, ...]]:
        """Preprocess all microbatch groups outside CUDA capture."""
        return [
            self._preprocess_microbatch_group(microbatch_group)
            for microbatch_group in microbatch_groups
        ]

    def _preprocess_microbatch_group(
        self,
        microbatch_group: list[TrainingMicrobatch],
    ) -> tuple[Any, ...]:
        """Preprocess one group: ``(inputs, labels, model_kwargs, loss_kwargs)``, or with PP
        ``(arg_mbs, kwarg_mbs, target_mbs, loss_kwarg_mbs)``."""
        prepared_microbatches: list[tuple[Any, ...]] = []
        for microbatch in microbatch_group:
            with (
                sl.log_trace_span("preprocess_inputs"),
                self.parallelism_context.activate_spmd(),
            ):
                prepared_microbatches.append(
                    self.model_parts[0].preprocess_inputs(
                        microbatch.to_input_dict(self.device, non_blocking=True),
                        parallelism_context=self.parallelism_context,
                        parallelism=self.config.parallelism,
                        max_num_documents=self.max_num_documents,
                        max_context_length=self.config.training.max_context_length,
                        aux_loss_denominators=self._aux_loss_denominators,
                        **self.preprocess_inputs_kwargs,
                    )
                )
            self.ntokens_seen += (
                self.config.training.num_tokens_per_microbatch_per_dp_rank
                // self.parallelism_context.cp
            )
            self.local_loss_token_count += int(microbatch.loss_token_count)

        if not self.parallelism_context.pp_enabled:
            assert len(prepared_microbatches) == 1
            return prepared_microbatches[0]

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
        loss_kwarg_mbs = [loss_kwargs for *_, loss_kwargs in prepared_microbatches]
        return (arg_mbs, kwarg_mbs, target_mbs, loss_kwarg_mbs)

    def _run_forward_backward_segments(
        self,
        microbatch_groups: list[Any],
        *,
        preprocess_microbatch_groups_lazily: bool,
    ) -> ForwardBackwardResult:
        """Replay one accelerator graph per segment: the microbatch groups between two
        gradient reductions.

        The step's first segment replays the first-segment graph, which allocates the
        gradients. Each later segment replays the later-segment graph of its length, which
        adds into them.

        Example (one reduction per group, plain FSDP or PP):
            step's first call [g0, g1, g2] -> first_segment_graph([g0]),
                                              later_segment_graphs[1]([g1]), then ([g2])
            next call [g3, g4]             -> later_segment_graphs[1]([g3]), then ([g4])
        Example (one reduction per call, HSDP or deferral):
            step's first call [g0]         -> first_segment_graph([g0])
            next call [g1, g2]             -> later_segment_graphs[2]([g1, g2])
        """
        if self.graph_segment_is_whole_call:
            segments: Iterable[list[Any]] = [microbatch_groups]
        else:
            # A generator: lazy preprocessing reads each group just before its replay.
            segments = ([group] for group in microbatch_groups)
        loss: torch.Tensor | None = None
        loss_metrics: list[list[dict[str, torch.Tensor]]] = []
        for segment_index, segment in enumerate(segments):
            if preprocess_microbatch_groups_lazily:
                # Preprocessing waits for the host, which capture does not allow.
                segment = self._preprocess_microbatch_groups(segment)
            if segment_index == 0 and not self._step_started:
                result = self._first_segment_graph(segment)
            elif self._first_segment_graph.captured:
                result = self._later_segment_graphs[len(segment)](segment)
            else:
                # A later-segment graph records adds into the gradient addresses that the
                # first-segment graph's replays put back, so it captures only after that graph.
                # Until then it warms up on the graph stream.
                capture_allowed = accelerator_graph_manager.capture_allowed
                accelerator_graph_manager.capture_allowed = False
                try:
                    result = self._later_segment_graphs[len(segment)](segment)
                finally:
                    accelerator_graph_manager.capture_allowed = capture_allowed
            # The next replay overwrites graph outputs.
            segment_loss = result.loss.clone()
            loss = segment_loss if loss is None else loss.add_(segment_loss)
            loss_metrics.extend(
                [
                    {key: value.clone() for key, value in metrics.items()}
                    for metrics in group
                ]
                for group in result.loss_metrics
            )
        assert loss is not None
        return ForwardBackwardResult(loss, loss_metrics)

    def _forward_backward_body(
        self,
        microbatch_groups: list[Any],
        *,
        defer_fsdp_gradient_reduction: bool,
        preprocess_microbatch_groups_lazily: bool = False,
    ) -> ForwardBackwardResult:
        """Run all microbatch groups in one graphable call."""

        accumulated_loss: torch.Tensor | None = None
        loss_metrics: list[list[dict[str, torch.Tensor]]] = []
        num_accumulation_steps = len(microbatch_groups)
        for accumulation_index, microbatch_group in enumerate(microbatch_groups):
            is_last_accumulation_step = accumulation_index == num_accumulation_steps - 1
            prepared_inputs = microbatch_group
            if preprocess_microbatch_groups_lazily:
                # TODO: this waits ~4 ms per microbatch on pageable inputs and varlen's host sync
                # (+0.77% at 16K tokens, Qwen3-0.6B, H100). Pinned inputs + max_num_documents fix it.
                prepared_inputs = self._preprocess_microbatch_group(microbatch_group)

            if self.parallelism_context.dp_replicate_enabled:
                # All-reduce HSDP replicas only with each call's final accumulated gradient.
                for model_part in self.model_parts:
                    fsdp_root = cast(FSDPModule, model_part)
                    fsdp_root.set_requires_all_reduce(is_last_accumulation_step)

            self.loss_metrics = {}
            if self.parallelism_context.pp_enabled:
                arg_mbs, kwarg_mbs, target_mbs, loss_kwarg_mbs = prepared_inputs
                # Finalization runs after the group's last PP microbatch.
                loss, group_loss_metrics = self._pp_forward_backward_microbatch_group(
                    inputs=arg_mbs,
                    model_kwargs=kwarg_mbs,
                    labels=target_mbs,
                    loss_kwargs=loss_kwarg_mbs,
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
                    loss_kwargs=loss_kwargs,
                )
                group_loss_metrics = [self.loss_metrics]

            detached_loss = loss.detach()
            if accumulated_loss is None:
                accumulated_loss = detached_loss.clone()
            else:
                accumulated_loss.add_(detached_loss)
            loss_metrics.append(
                [
                    {key: value.detach().clone() for key, value in metrics.items()}
                    for metrics in group_loss_metrics
                ]
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
                labels,
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
        loss_kwargs: list[dict[str, Any]],
        finalize_gradients: bool = True,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """Run one PP microbatch group.

        The input lists, including ``loss_kwargs``, hold one entry per
        microbatch of one pipeline schedule step. ``finalize_gradients``
        controls whether that step finishes FSDP gradient reduction. Returns
        the group loss and one loss metrics dict per microbatch.
        """
        # Each microbatch's loss fills its own metrics dict.
        loss_metrics: list[dict[str, torch.Tensor]] = [{} for _ in loss_kwargs]
        loss_kwarg_mbs = [
            {**mb_loss_kwargs, "loss_metrics": mb_loss_metrics}
            for mb_loss_kwargs, mb_loss_metrics in zip(
                loss_kwargs, loss_metrics, strict=True
            )
        ]
        with self.parallelism_context.activate_spmd(
            typechecking=self.config.debug.spmd_typechecking,
        ):
            losses = [] if self.pp_has_last_stage else None
            self.pp_schedule.step(
                arg_mbs=inputs,
                kwarg_mbs=model_kwargs,
                target_mbs=labels,
                losses=losses,
                loss_kwarg_mbs=loss_kwarg_mbs,
                return_outputs=False,
                finalize_gradients=finalize_gradients,
            )

        if self.pp_has_last_stage:
            assert losses is not None
            detached_losses = [loss.detach() for loss in losses]
            losses.clear()
            return torch.sum(torch.stack(detached_losses)).to(self.device), loss_metrics
        return self._pp_loss_sentinel_on_non_last_stage, loss_metrics

    @sl.log_trace_span("optim_step")
    def optim_step(self) -> torch.Tensor:
        """Divide every gradient by the step's loss-token count, then run one
        optimization update (clip, finiteness check, step) and advance its eager state.
        """
        if not self._step_started:
            raise ValueError(
                "optim_step needs a forward_backward call since the last optim_step: "
                "the gradients are already divided by their loss-token count."
            )
        current_step = self.num_completed_steps + 1
        if hasattr(self, "checkpointer"):
            self.checkpointer.maybe_wait_for_staging()
        # non_blocking: a pageable copy would wait for the whole backward before launching the update.
        local_count = torch.tensor(self.local_loss_token_count, dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        self.global_loss_token_count = self._sum_loss_token_count(local_count)
        # A step with no loss tokens has no loss gradients; dividing by 1 keeps the auxiliary-loss
        # gradients as they are. The update still runs, so AdamW's momentum and weight decay still
        # move the weights.
        divisor = self.global_loss_token_count.clamp_min(1).to(torch.float32)
        gradients_by_device: dict[torch.device, list[torch.Tensor]] = {}
        for parameter in self.optim.parameters:
            if parameter.grad is not None:
                gradient = dist_utils.get_local_tensor(parameter.grad)
                gradients_by_device.setdefault(gradient.device, []).append(gradient)
        # CPU offload keeps gradients on the CPU, where a device divisor is rejected.
        for device, gradients in gradients_by_device.items():
            torch._foreach_div_(gradients, divisor.to(device))
        grad_norm = self.optim.step(
            self.loss,
            current_step=current_step,
        )
        if self.sdc_replayer is not None:
            self.sdc_replayer.step_completed()
        self.num_completed_steps = current_step
        self._step_started = False
        return grad_norm

    def step_aux_loss_denominators(
        self, microbatch_groups: list[list[TrainingMicrobatch]]
    ) -> torch.Tensor:
        """Return ``aux_loss_denominators`` for a step made of ``microbatch_groups``
        on every data-parallel rank: one all-reduce of the token counts.

        Example, 2 DP ranks with one microbatch each:
            rank 0: routing_token_counts [8], loss_token_count 2
            rank 1: routing_token_counts [8], loss_token_count 6
            -> tensor([2.0]) on both ranks: 16 routed tokens per 8 loss tokens
        """
        # [N, R_0, ..., R_d]: the loss tokens, then the routed tokens of every depth.
        local_counts = torch.stack(
            [
                torch.cat(
                    (
                        microbatch.loss_token_count.reshape(1),
                        microbatch.routing_token_counts.reshape(-1),
                    )
                )
                for microbatch_group in microbatch_groups
                for microbatch in microbatch_group
            ]
        ).sum(dim=0)
        # non_blocking: a pageable copy would wait for the previous step's update.
        counts = local_counts.to(self.device, non_blocking=True)
        if self.parallelism_context.dp_enabled:
            counts = dist_utils.dist_sum_tensor(
                counts, self.parallelism_context.get_mesh("dp")
            )
        return aux_loss_denominators(counts[1:], counts[0])

    def _sum_loss_token_count(self, loss_token_count: torch.Tensor) -> torch.Tensor:
        """Sum this rank's loss-token count over the ranks whose gradients the step sums."""
        if not self.parallelism_context.dp_enabled:
            return loss_token_count
        return dist_utils.dist_sum_tensor(
            loss_token_count, self.parallelism_context.get_mesh("dp")
        )

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
        """Finish pending work and release accelerator graph resources."""
        self.close_profiler()
        if self._dist_moe_runtime is not None:
            self._dist_moe_runtime.close()
            self._dist_moe_runtime = None
        if hasattr(self, "checkpointer"):
            self.checkpointer.close()
        accelerator_graph_teardown()
