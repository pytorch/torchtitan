# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Annotated, Any, cast

import spmd_types as spmd
import torch
import torch.distributed.checkpoint.stateful
import tyro
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor import DTensor

from torchtitan.components.checkpointer import BaseCheckpointManager, CheckpointManager
from torchtitan.components.data.collators import TrainerBatch
from torchtitan.components.data.loader import BaseDataLoader, DataloaderExhaustedError
from torchtitan.components.loss import BaseLoss, ChunkedLossWrapper
from torchtitan.components.metrics import ensure_pp_loss_visible, MetricsProcessor
from torchtitan.components.optimizer import LRSchedulersContainer, OptimizersContainer
from torchtitan.components.quantization.utils import has_quantization
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
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointingConfig,
    MemoryBudgetAC,
    SelectiveAC,
)
from torchtitan.distributed.cudagraph import (
    cudagraph_teardown,
    ForwardBackwardFn,
    wrap_with_cuda_graph,
)
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.token_dispatcher import (
    HybridEPTokenDispatcher,
    LocalTokenDispatcher,
    MinimalAsyncEPTokenDispatcher,
)
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.sdc_replayer import ScalarStateAccessor, SDCReplayer
from torchtitan.protocols import BaseModel
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.profiler import Profiler


class Trainer(torch.distributed.checkpoint.stateful.Stateful, Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """
        Default container for training configuration.
        """

        # NOTE: model_spec is suppressed from tyro CLI parsing and is always
        # set programmatically by the model registry before Trainer construction.
        model_spec: Annotated[ModelSpec | None, tyro.conf.Suppress] = None

        hf_assets_path: str = "./tests/assets/tokenizer"
        """
        Path to HF assets folder. This folder contains local copies of Hugging Face assets,
        including model weights in .safetensors format, the model.safetensor.index.json file
        (fqn to file mapping), the config.json file, generation_config.json, and tokenizer files.
        """

        dump_folder: str = "./outputs"
        """Folder to dump job outputs"""

        profiler: Profiler.Config = field(default_factory=Profiler.Config)
        metrics: MetricsProcessor.Config = field(
            default_factory=MetricsProcessor.Config
        )
        tokenizer: BaseTokenizer.Config = field(
            default_factory=HuggingFaceTokenizer.Config
        )
        dataloader: BaseDataLoader.Config = field(default_factory=BaseDataLoader.Config)
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
        compile: CompileConfig = field(default_factory=CompileConfig)
        comm: CommConfig = field(default_factory=CommConfig)
        validator: Validator.Config = field(default_factory=Validator.Config)
        debug: DebugConfig = field(default_factory=DebugConfig)
        # NOTE: sdc_replayer is suppressed from tyro CLI parsing; enable it
        # programmatically in a config/recipe by assigning a config
        # (config.sdc_replayer = SDCReplayer.Config()). None disables replay.
        sdc_replayer: Annotated[SDCReplayer.Config | None, tyro.conf.Suppress] = None
        override: OverrideConfig = field(default_factory=OverrideConfig)
        loss: BaseLoss.Config = field(default_factory=BaseLoss.Config)

        def __post_init__(self):
            if self.debug.batch_invariant:
                raise ValueError(
                    "Batch-invariant mode is not supported in pre-training."
                )

            self._validate_sdc_replay()

            num_pp_microbatches = self.parallelism.num_pp_microbatches
            if num_pp_microbatches <= 0:
                raise ValueError(
                    "parallelism.num_pp_microbatches must be greater than 0."
                )

            self._validate_cuda_graphs()

            if (
                self.parallelism.spmd_backend == "spmd_types"
                and self.debug.spmd_typechecking
                and self.parallelism.pipeline_parallel_degree > 1
            ):
                # TODO(sanketpurandare): Enable SPMD typechecking under PP.
                raise ValueError(
                    "SPMD typechecking is not supported with pipeline parallelism. "
                    "Validate the same config without PP "
                    "(--parallelism.pipeline_parallel_degree 1)."
                )

            if (
                self.parallelism.spmd_backend == "spmd_types"
                and self.debug.spmd_typechecking
                and isinstance(self.activation_checkpoint, SelectiveAC.Config)
                and self.model_spec is not None
                and any(self.model_spec.model.traverse(FlexAttention.Config))
            ):
                # TODO(pianpwk): Enable SAC with FlexAttention under SPMD typechecking.
                raise ValueError(
                    "Selective activation checkpointing (SAC) is not supported "
                    "with FlexAttention while SPMD typechecking is enabled. "
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

        def _validate_cuda_graphs(self) -> None:
            if self.training.disable_cuda_graphs:
                return

            if self.parallelism.pipeline_parallel_degree > 1:
                raise ValueError(
                    "CUDA graphs do not support pipeline parallelism yet. "
                    "Set --training.disable_cuda_graphs."
                )

            if self.parallelism.expert_parallel_degree == 1 or self.model_spec is None:
                return

            for _, dispatcher_config, _, _ in self.model_spec.model.traverse(
                LocalTokenDispatcher.Config
            ):
                if isinstance(
                    dispatcher_config, MinimalAsyncEPTokenDispatcher.Config
                ) or (
                    isinstance(dispatcher_config, HybridEPTokenDispatcher.Config)
                    and dispatcher_config.non_blocking_capacity_factor is not None
                ):
                    continue

                raise ValueError(
                    "CUDA graphs support only expert parallel token dispatcher "
                    "configurations without CPU synchronization. "
                    "Set HybridEP non_blocking_capacity_factor, or use "
                    "MinimalAsyncEP, or set --training.disable_cuda_graphs. "
                    "Unsupported token "
                    f"dispatcher: {type(dispatcher_config).__qualname__}."
                )

        def _validate_sdc_replay(self) -> None:
            if self.sdc_replayer is None:
                return
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
                # TODO: Support additional replays after CUDA graph capture has
                # established stable gradient and buffer identities, or make
                # replay state restoration aware of graph-owned storage.
                raise ValueError(
                    "SDC replay supports at most one replay when CUDA graphs "
                    "are enabled: set sdc_replayer.num_replays=1 or "
                    "training.disable_cuda_graphs=True."
                )

        def to_dict(self) -> dict[str, Any]:
            d = {}
            for f in dataclasses.fields(self):
                if f.name == "model_spec":
                    assert self.model_spec is not None
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

    # core configs
    config: Config
    parallel_dims: ParallelDims

    # swappable training components
    tokenizer: BaseTokenizer
    dataloader: BaseDataLoader
    model_config: BaseModel.Config
    # TODO: we should make this list[BaseModel / Decoder] but this will affect many components.
    # will do this in a separate PR
    model_parts: list[torch.nn.Module]
    loss_fn: BaseLoss
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer
    validator: BaseValidator
    metrics_processor: MetricsProcessor
    checkpointer: BaseCheckpointManager

    # runtime utilities
    device: torch.device
    gc_handler: utils.GarbageCollection
    train_context: dist_utils.SpmdContext
    fwd_bwd_fn: ForwardBackwardFn
    gradient_accumulation_steps: int
    num_pp_microbatches: int
    pp_has_first_stage: bool
    pp_has_last_stage: bool
    sdc_replayer: SDCReplayer | None

    # additional training states
    step: int
    ntokens_seen: int

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, config: Config):
        torch._C._log_api_usage_once("torchtitan.train")

        self.config = config
        assert (
            config.model_spec is not None
        ), "model_spec must be set before creating Trainer"
        model_spec = config.model_spec

        device_module, device_type = utils.device_module, utils.device_type
        # pyrefly: ignore [read-only]
        self.device = utils.get_local_device()
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes
        self.parallel_dims = parallel_dims = self.init_distributed()

        # Validate dense activation token-count evenness.
        num_tokens_per_pp_microbatch = (
            config.training.num_tokens_per_microbatch_per_dp_rank
        )
        seq_len_divisor = (
            parallel_dims.tp if config.parallelism.enable_sequence_parallel else 1
        ) * (2 * parallel_dims.cp if parallel_dims.cp > 1 else 1)
        if num_tokens_per_pp_microbatch % seq_len_divisor != 0:
            raise ValueError(
                "The number of tokens per pipeline microbatch "
                f"({num_tokens_per_pp_microbatch}) must be "
                f"divisible by {seq_len_divisor} for the configured "
                "sequence/context parallelism."
            )

        # TODO(pianpwk): Transitional until the local-SPMD and full-DTensor
        # backends share one runtime mesh/type mechanism.
        dist_utils.set_spmd_backend(config.parallelism.spmd_backend)

        # Logging needs to happen after distributed initialized
        config.maybe_log()

        if parallel_dims.dp_enabled:
            dp_mesh = parallel_dims.get_mesh("batch")
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=config.training.gc_freq, debug=config.training.gc_debug
        )

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["pp"],
        )

        # build model (using meta init)
        model_config = model_spec.model
        # set the model args from training job configs
        model_config.update_from_config(
            config=config,
        )
        self.model_config = model_config

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

        logger.info(f"Building {model_spec.name} {model_spec.flavor}")

        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            model = model_config.build()

        # Verify all submodules satisfy the Module protocol
        # TODO: move this to module validate().
        # This is current put here to verify module build and
        # converter, which should guanrantee Module protocol.
        # On the other hand, some parallelism wrappers don't
        # have this guanrantee, e.g., fully_shard.
        model.verify_module_protocol()

        # metrics logging
        self.metrics_processor = config.metrics.build(
            parallel_dims=parallel_dims,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            config_dict=config.to_dict(),
            has_quantization=has_quantization(model_config),
        )
        color = self.metrics_processor.color

        # calculate model size and flops per token
        (
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = model_config.get_nparams_and_flops(
            model, config.training.max_context_length
        )

        logger.info(
            f"{color.blue}Model {model_spec.name} {model_spec.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        buffer_device: torch.device | None
        if config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = torch.device(device_type)
        else:
            init_device = device_type
            buffer_device = None

        self.loss_fn = config.loss.build(
            compile_config=config.compile,
        )

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
        # apply parallelisms and initialization
        with sl.log_trace_span("model_parallelism_init"):
            if parallel_dims.pp_enabled:
                if not model_spec.pipelining_fn:
                    raise RuntimeError(
                        f"Pipeline Parallel is enabled but {model_spec.name} "
                        f"does not support pipelining"
                    )

                # apply both Pipeline Parallel and SPMD-style scaling techniques
                (
                    self.pp_schedule,
                    self.model_parts,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                ) = model_spec.pipelining_fn(
                    model,
                    parallel_dims=parallel_dims,
                    training=config.training,
                    parallelism=config.parallelism,
                    compile_config=config.compile,
                    ac_config=config.activation_checkpoint,
                    dump_folder=config.dump_folder,
                    device=self.device,
                    model_config=model_config,
                    parallelize_fn=model_spec.parallelize_fn,
                    loss_fn=self.loss_fn,
                )
                # when PP is enabled, `model` obj is no longer used after this point,
                # model_parts is used instead
                del model

                for m in self.model_parts:
                    m.to_empty(device=init_device)
                    with torch.no_grad():
                        # TODO: Change this back to init_weights once
                        # autoparallel contains the wrap_init_states
                        cast(BaseModel, m).init_weights(buffer_device=buffer_device)
                    m.train()

                # confirm that user will be able to view loss metrics on the console
                ensure_pp_loss_visible(
                    parallel_dims=parallel_dims,
                    pp_schedule=config.parallelism.pipeline_parallel_schedule,
                    color=color,
                )
            else:
                if not config.checkpoint.create_seed_checkpoint:
                    # Skip parallelize_fn for seed checkpoints — nothing from
                    # it is needed (AC, compile, nD parallelism, mixed precision, etc.).
                    model = model_spec.parallelize_fn(
                        model,
                        parallel_dims=parallel_dims,
                        training=config.training,
                        parallelism=config.parallelism,
                        compile_config=config.compile,
                        ac_config=config.activation_checkpoint,
                        dump_folder=config.dump_folder,
                    )

                model.to_empty(device=init_device)
                with torch.no_grad():
                    # TODO: Change this back to init_weights once
                    # autoparallel contains the wrap_init_states
                    cast(BaseModel, model).init_weights(buffer_device=buffer_device)
                model.train()

                self.model_parts = [model]

        # Set lm_head reference for ChunkedLossWrapper after model construction.
        # Non-PP: single model part always has lm_head.
        # PP: only the last stage has lm_head; non-last stages skip this.
        if isinstance(self.loss_fn, ChunkedLossWrapper):
            if parallel_dims.pp_enabled:
                if self.pp_has_last_stage:
                    lm_head = self.model_parts[-1].lm_head
                    assert (
                        lm_head is not None
                    ), "Last PP stage must have lm_head for ChunkedLossWrapper"
                    self.loss_fn.set_lm_head(
                        lm_head  # pyrefly: ignore[bad-argument-type]
                    )
                    self.model_parts[
                        -1
                    ]._skip_lm_head = True  # pyrefly: ignore[bad-argument-type]
            else:
                assert len(self.model_parts) == 1
                lm_head = self.model_parts[0].lm_head
                assert (
                    lm_head is not None
                ), "Model must have lm_head for ChunkedLossWrapper"
                self.loss_fn.set_lm_head(lm_head)  # pyrefly: ignore[bad-argument-type]
                self.model_parts[
                    0
                ]._skip_lm_head = True  # pyrefly: ignore[bad-argument-type]

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        if model_spec.post_optimizer_build_fn is not None:
            model_spec.post_optimizer_build_fn(
                self.optimizers, self.model_parts, parallel_dims
            )
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )
        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.ntokens_seen = 0

        # SDC replay state is process-local and not checkpointed; its check
        # schedule restarts after every checkpoint load (see load_state_dict).
        self.sdc_replayer = None
        if config.sdc_replayer is not None:
            self.sdc_replayer = config.sdc_replayer.build(
                modules=self.model_parts,
                device=self.device,
                # ntokens_seen is the only trainer scalar the replayed
                # forward/backward mutates; self.step is incremented outside
                # the replay boundary and needs no capture.
                scalar_state={
                    "ntokens_seen": ScalarStateAccessor(
                        get=lambda: self.ntokens_seen,
                        set=lambda value: setattr(self, "ntokens_seen", value),
                    )
                },
            )

        # build tokenizer
        self.tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)

        # build dataloader
        num_tokens_per_batch = config.training.num_tokens_per_microbatch_per_dp_rank
        self.dataloader = config.dataloader.build(
            dp_world_size=dp_degree,
            dp_rank=dp_rank,
            tokenizer=self.tokenizer,
            max_context_length=config.training.max_context_length,
            num_tokens_per_batch=num_tokens_per_batch,
        )

        # build checkpointer
        self.checkpointer = config.checkpoint.build(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            sd_adapter=(
                model_spec.state_dict_adapter(model_config, config.hf_assets_path)
                if model_spec.state_dict_adapter
                else None
            ),
            base_folder=config.dump_folder,
        )

        self.train_context = dist_utils.get_spmd_context(
            parallel_dims=parallel_dims,
            spmd_typechecking=(
                config.parallelism.spmd_backend == "spmd_types"
                and config.debug.spmd_typechecking
            ),
        )
        self.fwd_bwd_fn = self._forward_backward_body
        if not config.training.disable_cuda_graphs:
            self.fwd_bwd_fn = wrap_with_cuda_graph(self.fwd_bwd_fn)

        # Build validator if validation is configured
        if config.validator.enable:
            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    self.pp_schedule,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
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
                loss_fn=self.loss_fn,
                validation_context=self.train_context,
                metrics_processor=self.metrics_processor,
                seq_len=config.training.max_context_length,
                num_tokens_per_batch=num_tokens_per_batch,
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

    @sl.log_trace_span("torch_distributed_init")
    def init_distributed(self) -> ParallelDims:
        config = self.config
        world_size = dist_utils.init_distributed(
            config.comm,
            enable_cpu_backend=config.training.enable_cpu_offload,
            base_folder=config.dump_folder,
        )

        return ParallelDims.from_config(config.parallelism, world_size)

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Returns an iterator that processes batches from the data iterator.

        Note: Tensors are yielded on CPU. The caller is responsible for moving
        them to GPU when needed. This allows for more efficient memory usage
        when doing gradient accumulation.
        """
        data_iterator = iter(data_iterable)

        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise DataloaderExhaustedError() from ex
            input_dict, labels = batch
            ntokens_batch = labels.numel()
            self.metrics_processor.ntokens_since_last_log += ntokens_batch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            # Tensors stay on CPU; moved to GPU per-microbatch during training
            yield input_dict, labels

    @sl.log_trace_span("fwd_bwd")
    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
        labels: torch.Tensor | list[torch.Tensor],
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        if parallel_dims.pp_enabled:
            assert isinstance(input_dict, list)
            assert isinstance(labels, list)
            return self.pp_forward_backward_step(
                input_dict_mbs=input_dict,
                label_mbs=labels,
                global_valid_tokens=global_valid_tokens,
            )

        assert isinstance(input_dict, dict)
        assert isinstance(labels, torch.Tensor)
        with sl.log_trace_span("preprocess_inputs"):
            inputs, labels, extra_kwargs = cast(
                BaseModel, self.model_parts[0]
            ).preprocess_inputs(
                {**input_dict, "labels": labels},
                parallel_dims=self.parallel_dims,
                parallelism=self.config.parallelism,
            )
            self.ntokens_seen += labels.numel()

        assert len(model_parts) == 1
        return self.fwd_bwd_fn(inputs, labels, global_valid_tokens, extra_kwargs)

    def _forward_backward_body(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
        extra_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        with self.train_context():
            pred = self.model_parts[0](inputs, **extra_kwargs)
            loss_kwargs = {}
            if "positions" in extra_kwargs:
                loss_kwargs["positions"] = extra_kwargs["positions"]
            loss, _ = self.loss_fn(
                pred,
                labels,
                global_valid_tokens,
                **loss_kwargs,
            )
            del pred
            with spmd.no_typecheck():
                # this propagates types through BWD, causing unnecessary conflicts
                # between torch_function and internals (e.g. AC). FWD is sufficient.
                loss.backward()

        # The returned loss here is local SUM loss / global_valid_tokens
        return loss

    def pp_forward_backward_step(
        self,
        *,
        input_dict_mbs: list[dict[str, torch.Tensor]],
        label_mbs: list[torch.Tensor],
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        arg_mbs: list[tuple[torch.Tensor, ...]] = []
        kwarg_mbs: list[dict[str, Any]] = []
        target_mbs: list[torch.Tensor] | None = [] if self.pp_has_last_stage else None
        for input_dict, labels in zip(input_dict_mbs, label_mbs, strict=True):
            with sl.log_trace_span("preprocess_inputs"):
                inputs, labels, extra_kwargs = cast(
                    BaseModel, self.model_parts[0]
                ).preprocess_inputs(
                    {**input_dict, "labels": labels},
                    parallel_dims=self.parallel_dims,
                    parallelism=self.config.parallelism,
                )
                self.ntokens_seen += labels.numel()
            if self.pp_has_first_stage:
                arg_mbs.append((inputs,))
            kwarg_mbs.append(extra_kwargs)
            if target_mbs is not None:
                target_mbs.append(labels)

        loss_kwargs = {"global_valid_tokens": global_valid_tokens}
        with self.train_context():
            losses = [] if self.pp_has_last_stage else None
            self.pp_schedule.step(
                arg_mbs=arg_mbs if self.pp_has_first_stage else None,
                kwarg_mbs=kwarg_mbs,
                target_mbs=target_mbs,
                losses=losses,
                loss_kwargs=loss_kwargs,
                return_outputs=False,
            )

        # TODO: PP+FSDP unexpectedly puts the loss back to the CPU.
        if self.pp_has_last_stage:
            assert losses is not None
            return torch.sum(torch.stack(losses)).to(self.device)
        return torch.tensor([-1.0], device=self.device)

    def train_step(self, data_iterator: Iterator[TrainerBatch]):
        self.optimizers.zero_grad(set_to_none=self.config.training.disable_cuda_graphs)
        # Save per-optimizer-group learning rates for logging
        lr_metrics = self.lr_schedulers.get_metrics()
        should_log = self.metrics_processor.should_log(self.step)

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims
        # All groups form one optimizer step; each group feeds one fwd-bwd call.
        microbatch_groups: list[list[TrainerBatch]] = []
        local_valid_tokens = 0
        for _ in range(self.gradient_accumulation_steps):
            microbatches = []
            for _ in range(self.num_pp_microbatches):
                with sl.log_trace_span("fetching_batch"):
                    input_dict, labels = next(data_iterator)
                # Popped so the batch reaching the model holds only its kwargs.
                local_valid_tokens += input_dict.pop("num_valid_tokens")
                microbatches.append((input_dict, labels))
            microbatch_groups.append(microbatches)
        sl.log_trace_scalar({"local_valid_tokens": local_valid_tokens})

        # Keep the global token count on device so loss normalization does not
        # introduce a CPU synchronization in the training path.
        local_valid_tokens_tensor = torch.tensor(
            local_valid_tokens,
            dtype=torch.int64,
            device=self.device,
        )
        if parallel_dims.dp_enabled:
            dp_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum_tensor(
                local_valid_tokens_tensor, dp_mesh
            )
        else:
            global_valid_tokens = local_valid_tokens_tensor

        # Process each gradient accumulation step, then free its inputs.
        accumulated_loss: torch.Tensor | None = None
        # int32 is supported by NCCL reductions, unlike bool.
        loss_is_finite = torch.ones((), dtype=torch.int32, device=self.device)
        for fwd_bwd_index, microbatches in enumerate(microbatch_groups):
            input_dict_mbs = []
            label_mbs = []
            for input_dict, labels in microbatches:
                for key, value in input_dict.items():
                    if isinstance(value, torch.Tensor):
                        input_dict[key] = value.to(self.device, non_blocking=True)
                input_dict_mbs.append(input_dict)
                label_mbs.append(labels.to(self.device, non_blocking=True))

            if parallel_dims.pp_enabled:
                fwd_bwd_input_dict = input_dict_mbs
                fwd_bwd_labels = label_mbs
            else:
                assert len(input_dict_mbs) == len(label_mbs) == 1
                fwd_bwd_input_dict = input_dict_mbs[0]
                fwd_bwd_labels = label_mbs[0]

            def fwd_bwd() -> torch.Tensor:
                return self.forward_backward_step(
                    input_dict=fwd_bwd_input_dict,
                    labels=fwd_bwd_labels,
                    global_valid_tokens=global_valid_tokens,
                )

            if self.sdc_replayer is not None and fwd_bwd_index == 0:
                # Only the step's first gradient-accumulation group is
                # replay-checked; under PP one group is a complete pipeline
                # schedule. Later groups exercise the same compute and
                # communication paths, so checking them too would only add
                # overhead.
                loss = self.sdc_replayer.run_fwd_bwd(fwd_bwd, step=self.step)
            else:
                loss = fwd_bwd()
            detached_loss = loss.detach()
            local_loss = (
                detached_loss.to_local()
                if isinstance(detached_loss, DTensor)
                else detached_loss
            )
            loss_is_finite.logical_and_(torch.isfinite(local_loss).all())
            if should_log:
                if accumulated_loss is None:
                    # Take ownership before the next replay overwrites the
                    # graph-owned output. Later losses accumulate in place.
                    accumulated_loss = detached_loss.clone()
                else:
                    accumulated_loss.add_(detached_loss)

        with sl.log_trace_span("optim"):
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in self.model_parts for p in m.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=parallel_dims.get_optional_mesh("pp"),
                ep_enabled=parallel_dims.ep_enabled,
            )
            # Only the last PP stage owns the loss. First combine its DP/CP
            # replicas, then propagate the result across PP. TP replicas have
            # identical loss values, and grad_norm is already world-reduced by
            # clip_grad_norm_.
            if not parallel_dims.pp_enabled or self.pp_has_last_stage:
                loss_mesh = parallel_dims.get_optional_mesh("loss")
                if loss_mesh is not None:
                    torch.distributed.all_reduce(
                        loss_is_finite,
                        op=torch.distributed.ReduceOp.MIN,
                        group=loss_mesh.get_group(),
                    )
            pp_mesh = parallel_dims.get_optional_mesh("pp")
            if pp_mesh is not None:
                torch.distributed.all_reduce(
                    loss_is_finite,
                    op=torch.distributed.ReduceOp.MIN,
                    group=pp_mesh.get_group(),
                )

            step_is_finite = loss_is_finite.logical_and(torch.isfinite(grad_norm).all())
            # Keep the check and optimizer kernels ordered on the device without
            # synchronizing the host on every step. The RuntimeError is catchable
            # on CPU, while a failed CUDA assertion invalidates the process.
            torch._assert_async(
                step_is_finite,
                "Loss or gradient norm is not finite on at least one rank at "
                f"step {self.step}. Stopping training before the optimizer update.",
            )
            self.checkpointer.maybe_wait_for_staging()
            self.optimizers.step()
            self.lr_schedulers.step()

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
                            self.ntokens_seen, dtype=torch.int64, device=self.device
                        ),
                        loss_mesh,
                    ),
                )
            else:
                global_avg_loss = global_max_loss = float(accumulated_loss.item())
                global_ntokens_seen = self.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            **lr_metrics,
        }
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )

    @record
    def train(self):
        config = self.config

        sl.log_trace_instant("training_start")

        self.checkpointer.load(step=config.checkpoint.load_step)

        # Capture loaded step for relative_step calculation.
        # After checkpoint load: self.step = restored step (e.g. 100), or 0 if fresh.
        loaded_step = self.step

        logger.info(f"Training starts at step {self.step + 1}")

        with config.profiler.build(
            global_step=self.step,
            base_folder=config.dump_folder,
        ) as profiler:
            data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                self.step += 1
                sl.set_step(self.step, relative_step=self.step - loaded_step)

                with sl.log_trace_span("step"):
                    self.gc_handler.run(self.step)

                    try:
                        self.train_step(data_iterator)
                    except DataloaderExhaustedError:
                        logger.warning("Ran out of data; last step was canceled.")
                        break

                    self.checkpointer.save(
                        self.step,
                        last_step=(self.step == config.training.steps),
                    )

                    # Run validation if validator is available
                    if self.config.validator.enable and self.validator.should_validate(
                        self.step
                    ):
                        self.validator.validate(self.model_parts, self.step)

                    # signal the profiler that the next profiling step has started
                    profiler.step()

                    # Reduce timeout after the first train step of THIS process
                    # (assuming lazy init and compilation are finished). Use the
                    # relative step so this fires on resumed runs too.
                    if self.step - loaded_step == 1:
                        dist_utils.set_pg_timeouts(
                            timeout=timedelta(
                                seconds=config.comm.train_timeout_seconds
                            ),
                            parallel_dims=self.parallel_dims,
                        )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")

    def should_continue_training(self) -> bool:
        return self.step < self.config.training.steps

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]
        if self.sdc_replayer is not None:
            self.sdc_replayer.reset_schedule()

    def close(self) -> None:
        if hasattr(self, "dataloader") and self.dataloader:
            self.dataloader.close()
        if not self.config.training.disable_cuda_graphs:
            cudagraph_teardown()
        if hasattr(self, "checkpointer") and self.checkpointer:
            self.checkpointer.close()
        if hasattr(self, "metrics_processor") and self.metrics_processor:
            self.metrics_processor.close()
