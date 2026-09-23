# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from torchtitan.components.data.types import TrainingMicrobatch

from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.cuda_graph import cuda_graph_teardown, CUDAGraphWrapper
from torchtitan.experiments.graph_trainer.common_utils import (
    accumulate_param_grads_,
    compute_annotated_loss,
    compute_parameter_gradients,
    log_timer,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    trace_input_preparer_keys,
)
from torchtitan.experiments.graph_trainer.gradient_accumulation import (
    GraphGradientState,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    validate_memory_policy_config,
)
from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    construct_default_graph_passes,
    construct_mandatory_graph_passes,
)
from torchtitan.experiments.graph_trainer.registry import (
    PASS_PIPELINE_REGISTRY,
    POST_INIT_HOOKS,
    PRE_TRAIN_STEP_HOOKS,
    TRACE_CALL_INPUT_PREPARERS,
    TRACE_INPUT_PREPARERS,
)
from torchtitan.experiments.graph_trainer.runner import GraphRunner
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols import BaseModel
from torchtitan.trainer import Trainer
from torchtitan.training_engine import TrainingEngine


logger = logging.getLogger(__name__)


def _maybe_apply_numa_binding(device_index: int, device_type: str) -> None:
    """Pin this process to the NUMA node of its GPU for local memory bandwidth.

    On multi-NUMA machines (e.g. GB200 NVLink-C2C), pinned-memory allocations
    that land on the GPU's local NUMA node get ~350 GB/s D2H bandwidth vs
    ~120 GB/s cross-NUMA. Must run before any pinned memory is allocated.
    """
    if device_type != "cuda":
        return
    from torch.numa.binding import (
        _maybe_apply_numa_binding_to_current_process,
        AffinityMode,
        NumaOptions,
    )

    _maybe_apply_numa_binding_to_current_process(
        device_index=device_index,
        numa_options=NumaOptions(
            affinity_mode=AffinityMode.NODE,
            should_fall_back_if_binding_fails=True,
        ),
    )
    logger.info("NUMA binding applied for GPU %d", device_index)


def make_fwd_bwd_step(model, loss_fn, *, accumulate_gradients: bool = False):
    """Return a plain function that traces the entire fwd+loss+bwd step.

    ``model`` and ``loss_fn`` are captured in the closure so neither shows up
    as a graph input. Pass ``model`` through ``minimal_fx_tracer(fn, module=model)``
    to thread its parameters/buffers as static graph inputs.
    """

    def compute_step(inputs, labels, global_valid_tokens, extra_kwargs):
        pred = model(inputs, **extra_kwargs)
        # The loss function is not a submodule of the model, so
        # annotate_module_fqns won't tag it. Annotate it here so that
        # downstream passes (bucketing, SAC, kernel annotations) can
        # attribute loss nodes in the traced graph.
        loss = compute_annotated_loss(
            loss_fn,
            pred,
            labels,
            {"global_valid_tokens": global_valid_tokens},
        )
        named_params = [
            (name, parameter)
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
        ]
        grads = compute_parameter_gradients(loss, named_params)
        return loss, named_params, grads

    if not accumulate_gradients:

        def fwd_bwd_step(inputs, labels, global_valid_tokens, extra_kwargs):
            loss, _named_params, grads = compute_step(
                inputs, labels, global_valid_tokens, extra_kwargs
            )
            return [loss, *grads]

        return fwd_bwd_step

    def fwd_bwd_step(
        gradient_buffers, inputs, labels, global_valid_tokens, extra_kwargs
    ):
        loss, named_params, grads = compute_step(
            inputs, labels, global_valid_tokens, extra_kwargs
        )
        for (fqn, _parameter), grad in zip(named_params, grads, strict=True):
            gradient_buffers[fqn].add_(grad)
        return [loss]

    return fwd_bwd_step


class GraphTrainingEngine(TrainingEngine):
    """Training engine for the experimental whole-step graph backend.

    TODO: Validate this as an optional execution backend for other workflows,
    such as RL training.
    """

    def __init__(
        self,
        config: "GraphTrainer.Config",
        *,
        model_config: BaseModel.Config,
        max_num_documents: int | None,
        output_dir: str,
    ) -> None:
        validate_memory_policy_config(config.compile)
        super().__init__(
            config,
            model_config=model_config,
            max_num_documents=max_num_documents,
            output_dir=output_dir,
        )
        # Lazy state for aot_fx_trace mode
        self._traced_step: TracedResult | None = None
        self._graph_runner: GraphRunner | None = None
        self._trainable_params: tuple[torch.Tensor, ...] | None = None
        self._graph_gradient_state: GraphGradientState | None = None
        self._pinned_pool_ctx = None

    def _initialize_forward_backward(self) -> None:
        super()._initialize_forward_backward()
        _maybe_apply_numa_binding(self.device.index, self.device.type)
        self._validate_inplace_graph_gradient_accumulation_config()
        if self.config.compile.enable_inplace_graph_gradient_accumulation:
            self._ensure_graph_gradient_state(self.model_parts[0])

        if self.config.compile.memory_policy == "sac_and_offload":
            from torch._functorch._activation_offloading.offload_ops import (
                pinned_memory_pool,
            )

            self._pinned_pool_ctx = pinned_memory_pool()
            self._pinned_pool_ctx.__enter__()
        else:
            self._pinned_pool_ctx = None

    def forward_backward_microbatch(
        self,
        *,
        microbatch_group: list[TrainingMicrobatch],
        global_valid_tokens: torch.Tensor,
        accumulation_index: int = 0,
    ) -> torch.Tensor:
        if self.parallel_dims.pp_enabled or self.config.compile.mode != "aot_fx_trace":
            return super().forward_backward_microbatch(
                microbatch_group=microbatch_group,
                global_valid_tokens=global_valid_tokens,
                accumulation_index=accumulation_index,
            )

        if any(microbatch.loss_kwargs() for microbatch in microbatch_group):
            raise ValueError(
                "GraphTrainingEngine does not support per-microbatch loss arguments."
            )

        # This intentionally duplicates the core engine's per-microbatch
        # execution envelope instead of adding graph-specific hooks to core.
        if accumulation_index == 0:
            self.loss_is_finite = torch.ones((), dtype=torch.int32, device=self.device)

        if self.parallel_dims.dp_replicate_enabled and (
            self.num_accumulation_steps == 1 or self.config.training.disable_cuda_graphs
        ):
            is_last = accumulation_index == self.num_accumulation_steps - 1
            for part in self.model_parts:
                part.set_requires_all_reduce(is_last)  # pyrefly: ignore[not-callable]

        def compute_forward_backward() -> torch.Tensor:
            assert len(microbatch_group) == 1
            microbatch = microbatch_group[0]
            assert len(self.model_parts) == 1
            model = self.model_parts[0]

            with sl.log_trace_span("preprocess_inputs"):
                inputs, labels, extra_kwargs = model.preprocess_inputs(
                    microbatch.to_input_dict(self.device, non_blocking=True),
                    parallel_dims=self.parallel_dims,
                    parallelism=self.config.parallelism,
                )
                # MTP returns one labels tensor per prediction; index 0 contains
                # the complete main-model labels used for token accounting.
                self.ntokens_seen += (
                    self.config.training.num_tokens_per_microbatch_per_dp_rank
                    // self.parallel_dims.cp
                )
            # remove_duplicate=False to preserve duplicate parameter entries
            # from weight tying (e.g. shared embedding/output weights).
            params = self._get_trainable_parameters(model)
            return self._make_fx_forward_backward_microbatch(
                model,
                inputs,
                labels,
                global_valid_tokens,
                params,
                extra_kwargs,
            )

        if self.sdc_replayer is not None and accumulation_index == 0:
            loss = self.sdc_replayer.run_fwd_bwd(
                compute_forward_backward, step=self.num_completed_steps + 1
            )
        else:
            loss = compute_forward_backward()
        detached_loss = loss.detach()
        self.loss_is_finite.logical_and_(torch.isfinite(detached_loss).all())
        return detached_loss

    def _load_precompiled_fx_trace(
        self,
        model: nn.Module,
        runtime_args: tuple[Any, ...],
    ) -> None:
        """Load a precompiled aot_fx_trace artifact from disk."""
        from torchtitan.experiments.graph_trainer.precompile import (
            _FX_TRACE_ARTIFACT_KEY,
            compute_config_fingerprint,
            flatten_runtime_inputs,
            get_spmd_precompile_meshes,
            precompile_fx_trace_load,
        )
        from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter

        compile_config = self.config.compile
        storage = DiskStorageAdapter(compile_config.precompile_artifact_dir)

        if not storage.exists(_FX_TRACE_ARTIFACT_KEY):
            raise ValueError(
                f"Precompiled fx_trace artifact not found at "
                f"'{compile_config.precompile_artifact_dir}/{_FX_TRACE_ARTIFACT_KEY}'. "
                f"Run precompile_main with --compile.mode aot_fx_trace first."
            )

        config_fingerprint = compute_config_fingerprint(
            model, compile_config, self.parallel_dims
        )
        precompile_meshes = get_spmd_precompile_meshes(self.parallel_dims)

        self._traced_step = precompile_fx_trace_load(
            storage,
            expected_fingerprint=config_fingerprint,
            example_inputs=flatten_runtime_inputs(
                model,
                runtime_args,
                {},
                precompile_meshes=precompile_meshes,
            ),
        )

    def _make_fx_forward_backward_microbatch(
        self,
        model: nn.Module,
        inputs: torch.Tensor | tuple[torch.Tensor, ...],
        labels: torch.Tensor | tuple[torch.Tensor, ...],
        global_valid_tokens: torch.Tensor,
        params: tuple[torch.Tensor, ...],
        extra_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        maybe_register_blockmask_pytree_node()
        gradient_state = self._graph_gradient_state
        if gradient_state is not None:
            gradient_state.prepare_for_backward()
        if self._traced_step is None:
            if self.config.compile.precompile_artifact_dir:
                self._load_precompiled_fx_trace(
                    model,
                    (inputs, labels, global_valid_tokens, extra_kwargs),
                )
            else:
                fwd_bwd_fn = make_fwd_bwd_step(
                    model,
                    self.loss_fn,
                    accumulate_gradients=gradient_state is not None,
                )
                with dist_utils.get_spmd_context(
                    parallel_dims=self.parallel_dims,
                    spmd_typechecking=False,
                ), log_timer("minimal_fx_tracer"):
                    self._traced_step = minimal_fx_tracer(
                        fwd_bwd_fn,
                        module=model,
                        graph_state=(
                            gradient_state.graph_state
                            if gradient_state is not None
                            else None
                        ),
                        prepare_inputs=self._prepare_trace_inputs,
                        prepare_call_inputs=self._prepare_trace_call_inputs,
                    )(
                        inputs,
                        labels,
                        global_valid_tokens,
                        extra_kwargs,
                    )
            if self.config.compile.enable_passes:
                pipeline_fn = PASS_PIPELINE_REGISTRY.get(
                    self.config.compile.pass_pipeline,
                    construct_default_graph_passes,
                )
                passes = pipeline_fn(
                    self._traced_step,
                    self.config,
                    parallel_dims=self.parallel_dims,
                )
            else:
                passes = construct_mandatory_graph_passes()
            self._traced_step.gm = apply_graph_passes(
                self._traced_step.gm,
                self._traced_step.example_inputs,
                passes,
                compile_config=self.config.compile,
                respect_disable_passes=self.config.compile.enable_passes,
            )
        assert self._traced_step is not None
        if self._graph_runner is None:
            runtime_meshes = ()
            if self.config.compile.precompile_artifact_dir:
                from torchtitan.experiments.graph_trainer.precompile import (
                    get_spmd_precompile_meshes,
                )

                runtime_meshes = tuple(get_spmd_precompile_meshes(self.parallel_dims))
            self._graph_runner = GraphRunner(
                self._traced_step,
                module=model,
                graph_state=(
                    gradient_state.graph_state if gradient_state is not None else None
                ),
                runtime_meshes=runtime_meshes,
            )
        with dist_utils.get_spmd_context(
            parallel_dims=self.parallel_dims,
            spmd_typechecking=self.config.debug.spmd_typechecking,
        ):
            outputs = self._graph_runner(
                inputs,
                labels,
                global_valid_tokens,
                extra_kwargs,
            )
        if gradient_state is None:
            loss = outputs[0]
            grads = outputs[1:]
            accumulate_param_grads_(
                params,
                grads,
                clone_grads_to_initialize_param_grad=isinstance(
                    self._traced_step.gm.forward, CUDAGraphWrapper
                ),
            )
            return loss

        if len(outputs) != 1:
            raise RuntimeError(
                "GraphTrainer directly traced gradient accumulation expected a "
                f"loss-only output, got {len(outputs)} outputs"
            )
        return outputs[0]

    def _ensure_graph_gradient_state(
        self,
        model: nn.Module,
    ) -> GraphGradientState:
        if self._graph_gradient_state is None:
            self._graph_gradient_state = GraphGradientState.create(
                model,
                self.optimizers,
            )
        return self._graph_gradient_state

    def _get_trainable_parameters(
        self,
        model: nn.Module,
    ) -> tuple[torch.Tensor, ...]:
        if self._graph_gradient_state is not None:
            return self._graph_gradient_state.parameters
        if self._trainable_params is None:
            # remove_duplicate=False preserves duplicate parameter entries from
            # weight tying (for example, shared embedding/output weights).
            self._trainable_params = tuple(
                parameter
                for _, parameter in model.named_parameters(remove_duplicate=False)
                if parameter.requires_grad
            )
        return self._trainable_params

    def _validate_inplace_graph_gradient_accumulation_config(self) -> None:
        if not self.config.compile.enable_inplace_graph_gradient_accumulation:
            return
        if self.config.compile.mode != "aot_fx_trace":
            raise ValueError(
                "GraphTrainer in-graph gradient accumulation requires "
                "compile.mode='aot_fx_trace'"
            )
        if self.parallel_dims.pp_enabled:
            raise ValueError(
                "GraphTrainer in-graph gradient accumulation does not yet "
                "support pipeline parallelism"
            )
        if len(self.model_parts) != 1:
            raise ValueError(
                "GraphTrainer in-graph gradient accumulation requires one model"
            )
        if self.config.compile.precompile_artifact_dir:
            raise ValueError(
                "GraphTrainer in-graph gradient accumulation does not yet "
                "support precompiled artifacts"
            )
        if self.config.compile.pass_pipeline in PASS_PIPELINE_REGISTRY:
            raise ValueError(
                "GraphTrainer in-graph gradient accumulation does not yet "
                "support custom pass pipelines"
            )

    def _prepare_trace_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        for pass_name in trace_input_preparer_keys(self.config.compile):
            prepare = TRACE_INPUT_PREPARERS.get(pass_name)
            if prepare is not None:
                prepare(self.config.compile, args, kwargs)

    def _prepare_trace_call_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        for pass_name in trace_input_preparer_keys(self.config.compile):
            prepare = TRACE_CALL_INPUT_PREPARERS.get(pass_name)
            if prepare is not None:
                prepared = prepare(self.config.compile, args, kwargs)
                if prepared is not None:
                    args, kwargs = prepared
        return args, kwargs

    def close(self) -> None:
        if self._pinned_pool_ctx is not None:
            self._pinned_pool_ctx.__exit__(None, None, None)
            self._pinned_pool_ctx = None

        super().close()

        self._graph_runner = None
        self._trainable_params = None

        # See Note [explicit CUDA graph teardown] in CUDA graph.py
        cuda_graph_teardown()


class GraphTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        compile: GraphTrainerCompileConfig = field(
            default_factory=GraphTrainerCompileConfig
        )

    engine_cls = GraphTrainingEngine
    engine: GraphTrainingEngine

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        POST_INIT_HOOKS.get(self.config.compile.pass_pipeline, lambda _: None)(self)

    def train_step(self, data_iterator: Iterator[dict[str, Any]]) -> None:
        PRE_TRAIN_STEP_HOOKS.get(self.config.compile.pass_pipeline, lambda _: None)(
            self
        )
        if self.engine._graph_gradient_state is not None:
            self.engine._graph_gradient_state.validate_grad_bindings()
        super().train_step(data_iterator)
