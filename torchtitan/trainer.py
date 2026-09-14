# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import logging
import os
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Annotated, Any, cast

import spmd_types as spmd
import torch
import torch.distributed.checkpoint.stateful
import tyro
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.pipelining.schedules import (
    _PipelineScheduleRuntime,
    get_schedule_class,
    PipelineScheduleMulti,
)

from torchtitan.components.checkpointer import BaseCheckpointManager, CheckpointManager
from torchtitan.components.data.loader import BaseDataLoader, DataloaderExhaustedError
from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.components.loss import BaseLoss, ChunkedLossWrapper
from torchtitan.components.optimizer import LRSchedulersContainer, OptimizersContainer
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.components.validate import BaseValidator, Validator
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import (
    CommConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.config.override import apply_overrides, OverrideConfig
from torchtitan.config.validation import validate_context_parallel
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointingConfig,
    MemoryBudgetAC,
    SelectiveAC,
)
from torchtitan.distributed.cudagraph import cudagraph_teardown, wrap_with_cuda_graph
from torchtitan.models.common.attention import FlexInnerAttention, VarlenInnerAttention
from torchtitan.models.common.aux_loss import AuxLoss, collect_aux_loss_metrics
from torchtitan.models.common.token_dispatcher import (
    HybridEPTokenDispatcher,
    LocalTokenDispatcher,
)
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.metrics import ensure_pp_loss_visible, MetricsProcessor
from torchtitan.observability.profiler import Profiler
from torchtitan.observability.sdc_replayer import ScalarStateAccessor, SDCReplayer
from torchtitan.protocols import BaseModel
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.quantization.utils import has_quantization
from torchtitan.tools import utils


logger = logging.getLogger(__name__)


class TrainingEngine(Configurable, torch.distributed.checkpoint.stateful.Stateful):
    """Shared distributed training engine."""

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
        checkpoint: BaseCheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
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
        dump_folder: str = "./outputs"

        def __post_init__(self) -> None:
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
    compile_config: CompileConfig
    loss_fn: BaseLoss
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer
    checkpointer: BaseCheckpointManager
    pp_has_last_stage: bool
    max_num_documents: int | None
    ntokens_seen: int
    sdc_replayer: SDCReplayer | None
    model_param_count: int
    num_flops_per_token: int
    has_quantization: bool
    loss_is_finite: torch.Tensor

    def __init__(
        self,
        config: Config,
        *,
        model_config: BaseModel.Config,
        compile_config: CompileConfig,
        max_num_documents: int | None,
    ) -> None:
        torch._C._log_api_usage_once("torchtitan.train")

        self.config = config
        self.model_config = model_config
        self.compile_config = compile_config
        self.max_num_documents = max_num_documents
        self.loss_fn = config.loss.build(compile_config=compile_config)
        self.has_quantization = has_quantization(model_config)
        self.step = 0
        self.ntokens_seen = 0
        self._num_optimizer_steps_since_cuda_graph_init = 0
        self.sdc_replayer = None
        self.preprocess_inputs_kwargs: dict[str, Any] = {}

    def initialize_checkpointer(
        self,
        *,
        dataloader: BaseDataLoader | None,
        sd_adapter: Any | None,
    ) -> None:
        """Build checkpointing around core and optional owner-provided state."""
        self.checkpointer = self.config.checkpoint.build(
            dataloader=dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            sd_adapter=sd_adapter,
            base_folder=self.config.dump_folder,
        )

    def initialize_distributed_runtime(
        self, *, distinct_seed_mesh_axes: list[str] | None = None
    ) -> None:
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
                base_folder=config.dump_folder,
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
            distinct_seed_mesh_dims=(
                ["pp"] if distinct_seed_mesh_axes is None else distinct_seed_mesh_axes
            ),
        )

    def initialize_model(
        self,
        model_spec: ModelSpec,
        *,
        create_seed_checkpoint: bool = False,
    ) -> None:
        """Build, measure, parallelize, materialize, and configure the model."""
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
                compile_config=self.compile_config,
                ac_config=config.activation_checkpoint,
                dump_folder=config.dump_folder,
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
                    compile_config=self.compile_config,
                    ac_config=config.activation_checkpoint,
                    dump_folder=config.dump_folder,
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

    def initialize_optimizer(self, model_spec: ModelSpec) -> None:
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

    def initialize_forward_backward(
        self,
        *,
        enable_cuda_graphs: bool,
        num_warmup_steps: int = 2,
    ) -> None:
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
        self.train_context = dist_utils.get_spmd_context(
            parallel_dims=self.parallel_dims,
            spmd_typechecking=self.config.debug.spmd_typechecking,
        )
        if self.parallel_dims.pp_enabled:
            self.fwd_bwd_fn = cast(
                Callable[..., torch.Tensor], self._pp_forward_backward_body
            )
            self._pp_loss_sentinel_on_non_last_stage = torch.full(
                (1,), -1.0, device=self.device
            )
        else:
            self.fwd_bwd_fn = cast(
                Callable[..., torch.Tensor], self._forward_backward_body
            )

        if enable_cuda_graphs:
            self.fwd_bwd_fn = wrap_with_cuda_graph(
                self.fwd_bwd_fn,
                num_warmup_steps=num_warmup_steps,
                optimizer_steps_completed=(
                    lambda: self._num_optimizer_steps_since_cuda_graph_init
                ),
            )

    @sl.log_trace_span("fwd_bwd")
    def forward_backward_microbatch(
        self,
        *,
        microbatch_group: list[TrainingMicrobatch],
        global_valid_tokens: int | torch.Tensor | None = None,
        loss_kwargs: dict[str, Any] | None = None,
        accumulation_index: int = 0,
        num_accumulation_steps: int = 1,
        compute_forward_backward: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Preprocess and execute one gradient-accumulation unit.

        A microbatch group forms one complete PP step. Without PP, the group
        contains exactly one data-parallel-rank-local microbatch.
        """
        assert global_valid_tokens is not None
        if accumulation_index == 0:
            # int32 is supported by NCCL reductions, unlike bool.
            self.loss_is_finite = torch.ones((), dtype=torch.int32, device=self.device)

        # HSDP replicate all-reduce is a no-op until the last accum group.
        # Do not toggle under CUDA graphs when accum > 1: the graph is
        # captured on the first group and replayed for later groups.
        if self.parallel_dims.dp_replicate_enabled and (
            num_accumulation_steps == 1 or self.config.training.disable_cuda_graphs
        ):
            is_last = accumulation_index == num_accumulation_steps - 1
            for part in self.model_parts:
                part.set_requires_all_reduce(is_last)  # pyrefly: ignore[not-callable]

        def fwd_bwd() -> torch.Tensor:
            if compute_forward_backward is not None:
                return compute_forward_backward()

            if self.parallel_dims.pp_enabled:
                if loss_kwargs:
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
                        inputs_mb, labels_mb, extra_kwargs_mb = cast(
                            BaseModel, self.model_parts[0]
                        ).preprocess_inputs(
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
                        self.ntokens_seen += (
                            self.config.training.num_tokens_per_microbatch_per_dp_rank
                            // self.parallel_dims.cp
                        )
                    if self.pp_has_first_stage:
                        arg_mbs.append((inputs_mb,))
                    kwarg_mbs.append(extra_kwargs_mb)
                    if target_mbs is not None:
                        target_mbs.append(labels_mb)

                return self.fwd_bwd_fn(
                    arg_mbs if self.pp_has_first_stage else None,
                    kwarg_mbs,
                    target_mbs,
                    global_valid_tokens,
                )

            assert len(microbatch_group) == 1
            input_dict = microbatch_group[0].to_input_dict(
                self.device, non_blocking=True
            )
            with sl.log_trace_span("preprocess_inputs"):
                inputs, labels, extra_kwargs = cast(
                    BaseModel, self.model_parts[0]
                ).preprocess_inputs(
                    input_dict,
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

            return self.fwd_bwd_fn(
                inputs,
                labels,
                global_valid_tokens,
                extra_kwargs,
                loss_kwargs or {},
            )

        if self.sdc_replayer is not None and accumulation_index == 0:
            loss = self.sdc_replayer.run_fwd_bwd(fwd_bwd, step=self.step)
        else:
            loss = fwd_bwd()
        detached_loss = loss.detach()
        self.loss_is_finite.logical_and_(torch.isfinite(detached_loss).all())
        return detached_loss

    def _forward_backward_body(
        self,
        inputs: torch.Tensor | tuple[torch.Tensor, ...],
        labels: torch.Tensor | tuple[torch.Tensor, ...],
        global_valid_tokens: int | torch.Tensor,
        model_kwargs: dict[str, Any],
        loss_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        with self.train_context():
            pred = self.model_parts[0](inputs, **model_kwargs)
            loss, self._last_loss_metrics = self.loss_fn(
                pred,
                labels,  # pyrefly: ignore[bad-argument-type]
                global_valid_tokens,
                **loss_kwargs,
            )
            del pred
            with spmd.no_typecheck():
                loss.backward()
        return loss

    def _pp_forward_backward_body(
        self,
        arg_mbs: list[tuple[torch.Tensor, ...]] | None,
        kwarg_mbs: list[dict[str, Any]],
        target_mbs: list[torch.Tensor] | None,
        global_valid_tokens: int | torch.Tensor,
    ) -> torch.Tensor:
        loss_kwargs = {"global_valid_tokens": global_valid_tokens}
        with self.train_context():
            losses = [] if self.pp_has_last_stage else None
            self.pp_schedule.step(
                arg_mbs=arg_mbs,
                kwarg_mbs=kwarg_mbs,
                target_mbs=target_mbs,
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

    def prepare_step(
        self, global_valid_tokens: int | torch.Tensor | None, *, step: int
    ) -> torch.Tensor | None:
        """Prepare GC, gradients, and token-normalized auxiliary losses for a step."""
        self.gc_handler.run(step)
        self.optimizers.zero_grad(set_to_none=self.config.training.disable_cuda_graphs)
        if global_valid_tokens is None:
            return None
        if isinstance(global_valid_tokens, int):
            global_valid_tokens = torch.tensor(
                global_valid_tokens,
                dtype=torch.int64,
                device=self.device,
            )
        AuxLoss.set_step_denominator(global_valid_tokens)
        return global_valid_tokens

    def optimizer_step(self) -> torch.Tensor:
        """Validate gradients, then advance optimizer and learning-rate scheduler."""
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
            "Loss or gradient norm is not finite on at least one rank. "
            "Stopping training before the optimizer update.",
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()
        self._num_optimizer_steps_since_cuda_graph_init += 1
        return grad_norm

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]
        if self.sdc_replayer is not None:
            self.sdc_replayer.reset_schedule()

    def load_checkpoint(self) -> bool:
        return self.checkpointer.load(step=self.config.checkpoint.load_step)

    def save_checkpoint(self, *, last_step: bool = False) -> bool:
        return self.checkpointer.save(self.step, last_step=last_step)

    def start_profiler(self) -> None:
        self.profiler = self.config.profiler.build(
            global_step=self.step,
            base_folder=self.config.dump_folder,
        )
        self.profiler.__enter__()

    def complete_step(self, *, last_step: bool = False) -> None:
        """Advance step-scoped services after a successful optimizer update."""
        self.profiler.step()
        self.save_checkpoint(last_step=last_step)

    def close(self) -> None:
        """Release CUDA graph and checkpoint resources owned by the trainer."""
        if hasattr(self, "profiler"):
            self.profiler.__exit__(None, None, None)
        if not self.config.training.disable_cuda_graphs:
            cudagraph_teardown()
        self.checkpointer.close()


class Trainer(Configurable):
    """Dataset-driven training application that owns a :class:`TrainingEngine`.

    The trainer owns input, validation, and reporting policy. Its training engine
    owns model execution, optimization, checkpointing, profiling, and replay.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TrainingEngine.Config):
        """
        Default container for training configuration.
        """

        # model_spec is always set by the registry. The unused string constructor
        # keeps Tyro from traversing the model config before applying Suppress.
        model_spec: Annotated[
            ModelSpec,
            tyro.conf.Suppress,
            tyro.conf.arg(constructor=str),
        ]

        hf_assets_path: str = "./tests/assets/tokenizer"
        """
        Path to HF assets folder. This folder contains local copies of Hugging Face assets,
        including model weights in .safetensors format, the model.safetensor.index.json file
        (fqn to file mapping), the config.json file, generation_config.json, and tokenizer files.
        """

        metrics: MetricsProcessor.Config = field(
            default_factory=MetricsProcessor.Config
        )
        tokenizer: BaseTokenizer.Config = field(
            default_factory=HuggingFaceTokenizer.Config
        )
        dataloader: BaseDataLoader.Config = field(default_factory=BaseDataLoader.Config)
        compile: CompileConfig = field(default_factory=CompileConfig)
        validator: Validator.Config = field(default_factory=Validator.Config)

        create_seed_checkpoint: bool = False
        """Initialize and save an unsharded model-only checkpoint, then exit."""

        def __post_init__(self):
            TrainingEngine.Config.__post_init__(self)
            if self.debug.batch_invariant:
                raise ValueError("Batch-invariant mode is not supported in Trainer.")

            self._validate_cuda_graphs()

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
                self.debug.spmd_typechecking
                and isinstance(self.activation_checkpoint, SelectiveAC.Config)
                and any(self.model_spec.model.traverse(FlexInnerAttention.Config))
            ):
                # TODO(pianpwk): Enable SAC with FlexInnerAttention under SPMD typechecking.
                raise ValueError(
                    "Selective activation checkpointing (SAC) is not supported "
                    "with FlexInnerAttention while SPMD typechecking is enabled. "
                    "Use full activation checkpointing, disable activation "
                    "checkpointing, or switch to a non-Flex attention backend."
                )

            if isinstance(self.activation_checkpoint, MemoryBudgetAC.Config) and not (
                self.compile.enable and "model" in self.compile.components
            ):
                raise ValueError(
                    "Memory budget activation checkpointing requires the model to be "
                    "compiled: set --compile.enable and include 'model' in "
                    "--compile.components."
                )

            if self.model_spec is not None:
                validate_context_parallel(self.model_spec.model, self.parallelism)

        def _validate_cuda_graphs(self) -> None:
            if self.training.disable_cuda_graphs:
                return

            pp_enabled = self.parallelism.pipeline_parallel_degree > 1
            if pp_enabled and self.validator.enable:
                raise ValueError(
                    "CUDA graphs with pipeline parallelism do not support "
                    "validation because validation reinitializes the shared "
                    "pipeline schedule. Disable validation or CUDA graphs."
                )

            if pp_enabled:
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

            if self.dataloader.max_num_documents is None:
                for fqn, _, _, _ in self.model_spec.model.traverse(
                    VarlenInnerAttention.Config
                ):
                    raise ValueError(
                        "CUDA graphs require fixed-shape varlen document "
                        f"metadata for {fqn}, but "
                        "dataloader.max_num_documents is unset. Set it to "
                        "an upper bound on documents per local token microbatch, "
                        "or set --training.disable_cuda_graphs."
                    )

            if self.parallelism.expert_parallel_degree == 1:
                return

            for _, dispatcher_config, _, _ in self.model_spec.model.traverse(
                LocalTokenDispatcher.Config
            ):
                if (
                    isinstance(dispatcher_config, HybridEPTokenDispatcher.Config)
                    and dispatcher_config.non_blocking_capacity_factor is not None
                ):
                    continue

                raise ValueError(
                    "CUDA graphs support only expert parallel token dispatcher "
                    "configurations without CPU synchronization. "
                    "Set HybridEP non_blocking_capacity_factor, or set "
                    "--training.disable_cuda_graphs. "
                    "Unsupported token "
                    f"dispatcher: {type(dispatcher_config).__qualname__}."
                )

        def to_dict(self) -> dict[str, Any]:
            d = {}
            for f in dataclasses.fields(self):
                if f.name == "model_spec":
                    # ModelSpec contains callables that can't be serialized
                    d["model_spec"] = {
                        "name": self.model_spec.name,
                        "flavor": self.model_spec.flavor,
                        "model": self.model_spec.model.to_dict(),
                    }
                else:
                    val = getattr(self, f.name)
                    if hasattr(val, "to_dict"):
                        d[f.name] = val.to_dict()
                    elif dataclasses.is_dataclass(val):
                        d[f.name] = asdict(val)
                    else:
                        d[f.name] = val
            return d

        def maybe_log(self) -> None:
            if self.debug.print_config:
                logger.info(
                    f"Running with configs: {json.dumps(self.to_dict(), indent=2, ensure_ascii=False)}"
                )

            if self.debug.save_config_file is not None:
                config_file = os.path.join(
                    self.dump_folder, self.debug.save_config_file
                )
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        os.makedirs(os.path.dirname(config_file), exist_ok=True)
                        with open(config_file, "w") as f:
                            json.dump(self.to_dict(), f, indent=2)
                    logger.info(f"Saved job configs to {config_file}")
                else:
                    logger.warning(
                        "Job configs logging is disabled due to torch.distributed not initialized."
                    )

    config: Config
    engine: TrainingEngine

    tokenizer: BaseTokenizer
    dataloader: BaseDataLoader
    validator: BaseValidator
    metrics_processor: MetricsProcessor
    gradient_accumulation_steps: int
    num_pp_microbatches: int

    def _forward_backward_microbatch(
        self,
        *,
        microbatch_group: list[TrainingMicrobatch],
        global_valid_tokens: int | torch.Tensor | None = None,
        loss_kwargs: dict[str, Any] | None = None,
        accumulation_index: int = 0,
        num_accumulation_steps: int = 1,
    ) -> torch.Tensor:
        return self.engine.forward_backward_microbatch(
            microbatch_group=microbatch_group,
            global_valid_tokens=global_valid_tokens,
            loss_kwargs=loss_kwargs,
            accumulation_index=accumulation_index,
            num_accumulation_steps=num_accumulation_steps,
        )

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, config: Config):
        self.config = config
        model_spec = config.model_spec
        model_config = model_spec.model
        model_config.update_from_config(config=config)

        # Apply overrides to the full config tree, before any component is
        # built. The model config is reached via ModelSpec.traverse. Model
        # overrides must run after update_from_config above (it sets sharding
        # config on the pre-override modules); all other components (optimizer,
        # loss, dataloader, …) are built later in __init__.
        if config.override.imports:
            apply_overrides(config.override, config)
        # Overrides may change any config field; re-run the full validation.
        # __post_init__ only raises (no mutation), so re-running is safe.
        config.__post_init__()

        self.engine = TrainingEngine(
            config,
            model_config=model_config,
            compile_config=config.compile,
            max_num_documents=config.dataloader.max_num_documents,
        )
        engine = self.engine
        engine.initialize_distributed_runtime()
        parallel_dims = engine.parallel_dims

        # Logging needs to happen after distributed initialized
        config.maybe_log()

        if parallel_dims.dp_enabled:
            dp_mesh = parallel_dims.get_mesh("dp")
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0

        # metrics logging
        self.metrics_processor = config.metrics.build(
            parallel_dims=parallel_dims,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            config_dict=config.to_dict(),
            has_quantization=engine.has_quantization,
        )
        color = self.metrics_processor.color

        self.num_pp_microbatches = (
            config.parallelism.num_pp_microbatches if parallel_dims.pp_enabled else 1
        )
        num_tokens_per_dp_rank = (
            config.training.num_tokens_per_microbatch_per_dp_rank
            * self.num_pp_microbatches
        )
        num_tokens_per_train_step = config.training.num_tokens_per_train_step
        if num_tokens_per_train_step < 0:
            num_tokens_per_train_step = num_tokens_per_dp_rank * dp_degree
        if num_tokens_per_train_step % (num_tokens_per_dp_rank * dp_degree) != 0:
            raise ValueError(
                "training.num_tokens_per_train_step "
                f"({num_tokens_per_train_step}) must be divisible by the number "
                "of tokens processed globally in one gradient accumulation "
                f"iteration ({num_tokens_per_dp_rank * dp_degree})."
            )
        self.gradient_accumulation_steps = num_tokens_per_train_step // (
            num_tokens_per_dp_rank * dp_degree
        )
        # Build and initialize the model and its parallel execution state.
        with sl.log_trace_span("model_parallelism_init"):
            engine.initialize_model(
                model_spec,
                create_seed_checkpoint=config.create_seed_checkpoint,
            )
            if parallel_dims.pp_enabled:
                ensure_pp_loss_visible(
                    parallel_dims=parallel_dims,
                    pp_schedule=config.parallelism.pipeline_parallel_schedule,
                    color=color,
                )
        self.metrics_processor.num_flops_per_token = engine.num_flops_per_token

        device_memory_monitor = self.metrics_processor.device_memory_monitor
        logger.info(
            "Peak FLOPS used for computing MFU: "
            f"{self.metrics_processor.gpu_peak_flops:.3e}"
        )
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{engine.device.type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        engine.initialize_optimizer(model_spec)
        self.metrics_processor.optimizers = engine.optimizers
        self.metrics_processor.model_parts = engine.model_parts

        # build tokenizer
        self.tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)

        # build dataloader
        num_tokens_per_microbatch = (
            config.training.num_tokens_per_microbatch_per_dp_rank
        )
        self.dataloader = config.dataloader.build(
            dp_world_size=dp_degree,
            dp_rank=dp_rank,
            tokenizer=self.tokenizer,
            max_context_length=config.training.max_context_length,
            num_tokens_per_microbatch=num_tokens_per_microbatch,
        )
        # build checkpointer
        engine.initialize_checkpointer(
            dataloader=self.dataloader,
            sd_adapter=(
                model_spec.state_dict_adapter(model_config, config.hf_assets_path)
                if model_spec.state_dict_adapter
                else None
            ),
        )

        engine.initialize_forward_backward(
            enable_cuda_graphs=not config.training.disable_cuda_graphs,
        )

        # Build validator if validation is configured
        if config.validator.enable:
            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    engine.pp_schedule,
                    engine.pp_has_first_stage,
                    engine.pp_has_last_stage,
                )
                if parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = config.validator.build(
                parallelism=config.parallelism,
                dp_world_size=dp_degree,
                dp_rank=dp_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                loss_fn=engine.loss_fn,
                validation_context=engine.train_context,
                metrics_processor=self.metrics_processor,
                seq_len=config.training.max_context_length,
                num_tokens_per_microbatch=num_tokens_per_microbatch,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        logger.info(
            "Trainer is initialized with "
            f"{num_tokens_per_dp_rank} tokens per DP rank, "
            f"{num_tokens_per_train_step} tokens per train step, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"maximum context length {config.training.max_context_length}, "
            f"total steps {config.training.steps} "
            f"(warmup {config.lr_scheduler.warmup_steps})"
        )

    def microbatch_generator(
        self, data_iterable: Iterable[TrainingMicrobatch]
    ) -> Iterator[TrainingMicrobatch]:
        """Return microbatches while recording data-loading metrics.

        Note: Tensors are yielded on CPU. The caller is responsible for moving
        them to GPU when needed. This allows for more efficient memory usage
        when doing gradient accumulation.
        """
        data_iterator = iter(data_iterable)

        while True:
            data_load_start = time.perf_counter()
            try:
                microbatch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise DataloaderExhaustedError() from ex
            ntokens_microbatch = (
                self.config.training.num_tokens_per_microbatch_per_dp_rank
            )
            self.metrics_processor.ntokens_since_last_log += ntokens_microbatch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            # Tensors stay on CPU; moved to GPU per-microbatch during training
            yield microbatch

    def train_step(self, data_iterator: Iterator[TrainingMicrobatch]):
        engine = self.engine
        # Save per-optimizer-group learning rates for logging
        lr_metrics = engine.lr_schedulers.get_metrics()
        should_log = self.metrics_processor.should_log(engine.step)

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = engine.parallel_dims
        # All groups form one optimizer step. Each microbatch group forms one
        # complete PP step, or one local forward/backward when PP is disabled.
        microbatch_groups: list[list[TrainingMicrobatch]] = []
        local_valid_tokens = 0
        for _ in range(self.gradient_accumulation_steps):
            microbatch_group = []
            for _ in range(self.num_pp_microbatches):
                with sl.log_trace_span("fetching_batch"):
                    microbatch = next(data_iterator)
                local_valid_tokens += microbatch.num_valid_tokens
                microbatch_group.append(microbatch)
            microbatch_groups.append(microbatch_group)
        sl.log_trace_scalar({"local_valid_tokens": local_valid_tokens})

        # Keep the global token count on device so loss normalization does not
        # introduce a CPU synchronization in the training path.
        local_valid_tokens_tensor = torch.tensor(
            local_valid_tokens,
            dtype=torch.int64,
            device=engine.device,
        )
        if parallel_dims.dp_enabled:
            dp_mesh = parallel_dims.get_mesh("dp")
            global_valid_tokens = dist_utils.dist_sum_tensor(
                local_valid_tokens_tensor, dp_mesh
            )
        else:
            global_valid_tokens = local_valid_tokens_tensor

        # Auxiliary losses normalize by the same per-step token count as the
        # main loss, so their scale is independent of parallelism degrees.
        prepared_valid_tokens = engine.prepare_step(
            global_valid_tokens, step=engine.step
        )
        assert prepared_valid_tokens is not None
        global_valid_tokens = prepared_valid_tokens

        # Process each gradient accumulation step, then free its inputs.
        accumulated_loss: torch.Tensor | None = None
        for fwd_bwd_index, microbatch_group in enumerate(microbatch_groups):
            detached_loss = self._forward_backward_microbatch(
                microbatch_group=microbatch_group,
                global_valid_tokens=global_valid_tokens,
                accumulation_index=fwd_bwd_index,
                num_accumulation_steps=self.gradient_accumulation_steps,
            )
            if should_log:
                if accumulated_loss is None:
                    # Take ownership before the next replay overwrites the
                    # graph-owned output. Later losses accumulate in place.
                    accumulated_loss = detached_loss.clone()
                else:
                    accumulated_loss.add_(detached_loss)

        with sl.log_trace_span("optim"):
            grad_norm = engine.optimizer_step()

        # log metrics
        if not should_log:
            return

        assert accumulated_loss is not None

        with sl.log_trace_span("collect_dist_metrics"):
            sl.log_trace_scalar({"global_valid_tokens": int(global_valid_tokens)})

            if parallel_dims.dp_cp_enabled:
                loss_mesh = parallel_dims.get_optional_mesh("loss")

                # For global_avg_loss, we want the average loss across all ranks:
                # accumulated_loss = local_loss_sum / global_valid_tokens
                # global_avg_loss = sum(local_loss_sum) / global_valid_tokens
                #                 = sum(accumulated_loss)
                #
                # For global_max_loss, we want the max of local average losses across ranks:
                # local_avg_loss = local_loss_sum / local_valid_tokens
                #                = (accumulated_loss * global_valid_tokens) / local_valid_tokens
                # global_max_loss = max(local_avg_loss)
                local_avg_loss = (
                    accumulated_loss * global_valid_tokens / local_valid_tokens
                )
                global_avg_loss, global_max_loss, global_ntokens_seen = (
                    dist_utils.dist_sum(accumulated_loss, loss_mesh),
                    dist_utils.dist_max(local_avg_loss, loss_mesh),
                    dist_utils.dist_sum(
                        torch.tensor(
                            engine.ntokens_seen,
                            dtype=torch.int64,
                            device=engine.device,
                        ),
                        loss_mesh,
                    ),
                )
            else:
                global_avg_loss = global_max_loss = float(accumulated_loss.item())
                global_ntokens_seen = engine.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            **lr_metrics,
            **collect_aux_loss_metrics(parallel_dims),
        }
        self.metrics_processor.log(
            engine.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )

    @record
    def train(self):
        config = self.config
        engine = self.engine

        sl.log_trace_instant("training_start")

        engine.load_checkpoint()

        # Capture loaded step for relative_step calculation.
        # After checkpoint load: self.step = restored step (e.g. 100), or 0 if fresh.
        loaded_step = engine.step

        logger.info(f"Training starts at step {engine.step + 1}")

        engine.start_profiler()
        try:
            data_iterator = self.microbatch_generator(self.dataloader)
            while self.should_continue_training():
                engine.step += 1
                sl.set_step(engine.step, relative_step=engine.step - loaded_step)

                with sl.log_trace_span("step"):
                    try:
                        self.train_step(data_iterator)
                    except DataloaderExhaustedError:
                        logger.warning("Ran out of data; last step was canceled.")
                        break

                    engine.complete_step(
                        last_step=(engine.step == config.training.steps)
                    )

                    # Run validation if validator is available
                    if self.config.validator.enable and self.validator.should_validate(
                        engine.step
                    ):
                        self.validator.validate(engine.model_parts, engine.step)

                    # Reduce timeout after the first train step of THIS process
                    # (assuming lazy init and compilation are finished). Use the
                    # relative step so this fires on resumed runs too.
                    if engine.step - loaded_step == 1:
                        dist_utils.set_pg_timeouts(
                            timeout=timedelta(
                                seconds=config.comm.train_timeout_seconds
                            ),
                            parallel_dims=engine.parallel_dims,
                        )
        finally:
            # The entry point also calls close() for checkpoint and graph
            # cleanup; close the profiler here so direct train() callers get
            # balanced lifecycle handling.
            engine.profiler.__exit__(None, None, None)
            del engine.profiler

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")

    def should_continue_training(self) -> bool:
        return self.engine.step < self.config.training.steps

    def close(self) -> None:
        if hasattr(self, "dataloader") and self.dataloader:
            self.dataloader.close()
        if hasattr(self, "engine"):
            self.engine.close()
        if hasattr(self, "metrics_processor") and self.metrics_processor:
            self.metrics_processor.close()
