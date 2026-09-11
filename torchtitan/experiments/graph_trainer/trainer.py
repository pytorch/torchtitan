# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn as nn

from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.cudagraph import cudagraph_teardown, CUDAGraphWrapper
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
from torchtitan.experiments.graph_trainer.deferred_fsdp_reductions import (
    build_graph_with_deferred_fsdp_reductions,
    GraphWithDeferredFSDPReductions,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    deduplicate_fsdp_unshard_chains_pass,
)
from torchtitan.experiments.graph_trainer.gradient_accumulation import (
    GraphGradientState,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import flatten_graph_values
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    validate_memory_policy_config,
)
from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    compile_time_passes,
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
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    canonicalize_graph_pass,
    eliminate_dead_code_pass,
)
from torchtitan.experiments.graph_trainer.runner import GraphRunner
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols import BaseModel
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer


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


class GraphTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        compile: GraphTrainerCompileConfig = field(
            default_factory=GraphTrainerCompileConfig
        )

    def __init__(self, config):
        super().__init__(config)

        validate_memory_policy_config(self.config.compile)

        _maybe_apply_numa_binding(self.device.index, self.device.type)

        # Lazy state for aot_fx_trace mode
        self._traced_step: TracedResult | None = None
        self._graph_runner: GraphRunner | None = None
        self._trainable_params: tuple[torch.Tensor, ...] | None = None
        self._graph_gradient_state: GraphGradientState | None = None
        self._graph_with_deferred_fsdp_reductions: (
            GraphWithDeferredFSDPReductions | None
        ) = None
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

        # Run post-init hook for the active pass pipeline
        POST_INIT_HOOKS.get(self.config.compile.pass_pipeline, lambda _: None)(self)

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, Any] | list[dict[str, Any]],
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.parallel_dims.pp_enabled or self.config.compile.mode != "aot_fx_trace":
            return super().forward_backward_step(
                input_dict=input_dict,
                global_valid_tokens=global_valid_tokens,
            )

        assert isinstance(input_dict, dict)
        assert len(self.model_parts) == 1
        model = self.model_parts[0]

        inputs, labels, extra_kwargs = self._preprocess_fwd_bwd_inputs(
            model, input_dict
        )
        # remove_duplicate=False to preserve duplicate parameter entries
        # from weight tying (e.g. shared embedding/output weights).
        params = self._get_trainable_parameters(model)
        return self._make_fx_forward_backward_step(
            model,
            inputs,
            labels,
            global_valid_tokens,
            params,
            extra_kwargs,
        )

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
        precompile_meshes = (
            get_spmd_precompile_meshes(self.parallel_dims)
            if self.config.parallelism.spmd_backend == "spmd_types"
            else None
        )

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

    def _make_fx_forward_backward_step(
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
                self._traced_step = self._trace_fwd_bwd(
                    fwd_bwd_fn,
                    model,
                    (inputs, labels, global_valid_tokens, extra_kwargs),
                    graph_state=(
                        gradient_state.graph_state
                        if gradient_state is not None
                        else None
                    ),
                )
            passes = self._get_fwd_bwd_graph_passes(
                self._traced_step,
                finalize=True,
            )
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
            if (
                self.config.compile.precompile_artifact_dir
                and self.config.parallelism.spmd_backend == "spmd_types"
            ):
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
        with self.train_context():
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

    def _trace_fwd_bwd(
        self,
        fwd_bwd_fn: Callable[..., list[torch.Tensor]],
        model: nn.Module,
        runtime_args: tuple[Any, ...],
        *,
        graph_state: dict[str, torch.Tensor] | None = None,
    ) -> TracedResult:
        trace_context = dist_utils.get_spmd_context(
            parallel_dims=self.parallel_dims,
            spmd_typechecking=False,
        )
        with trace_context(), log_timer("minimal_fx_tracer"):
            return minimal_fx_tracer(
                fwd_bwd_fn,
                module=model,
                graph_state=graph_state,
                prepare_inputs=self._prepare_trace_inputs,
                prepare_call_inputs=self._prepare_trace_call_inputs,
            )(*runtime_args)

    def _get_fwd_bwd_graph_passes(
        self,
        traced_result: TracedResult,
        *,
        finalize: bool,
    ) -> list[Callable]:
        """Select passes for a complete graph or one that will be partitioned."""
        if not self.config.compile.enable_passes:
            passes = construct_mandatory_graph_passes()
            if not finalize:
                passes.extend(
                    [
                        eliminate_dead_code_pass,
                        canonicalize_graph_pass,
                        deduplicate_fsdp_unshard_chains_pass,
                    ]
                )
            return passes

        if not finalize:
            return compile_time_passes(
                traced_result,
                self.config,
                parallel_dims=self.parallel_dims,
                include_inductor=False,
            )

        pipeline_fn = PASS_PIPELINE_REGISTRY.get(
            self.config.compile.pass_pipeline,
            construct_default_graph_passes,
        )
        return pipeline_fn(
            traced_result,
            self.config,
            parallel_dims=self.parallel_dims,
        )

    def _preprocess_fwd_bwd_inputs(
        self,
        model: nn.Module,
        input_dict: dict[str, Any],
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, ...],
        torch.Tensor | tuple[torch.Tensor, ...],
        dict[str, Any],
    ]:
        with sl.log_trace_span("preprocess_inputs"):
            inputs, labels, extra_kwargs = cast(BaseModel, model).preprocess_inputs(
                input_dict,
                parallel_dims=self.parallel_dims,
                parallelism=self.config.parallelism,
            )
            # MTP returns one labels tensor per prediction; index 0 contains
            # the complete main-model labels used for token accounting.
            self.ntokens_seen += (
                labels[0].numel() if isinstance(labels, tuple) else labels.numel()
            )
        return inputs, labels, extra_kwargs

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
        if (
            self.config.compile.enable_deferred_fsdp_gradient_sync
            and not self.config.compile.enable_inplace_graph_gradient_accumulation
        ):
            raise ValueError(
                "Deferred FSDP gradient sync requires "
                "compile.enable_inplace_graph_gradient_accumulation"
            )
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

        if not self.config.compile.enable_deferred_fsdp_gradient_sync:
            return
        if self.gradient_accumulation_steps < 2:
            raise ValueError(
                "Deferred FSDP gradient sync requires at least two gradient "
                "accumulation microbatches"
            )
        if self.config.compile.memory_policy == "sac_and_offload":
            raise ValueError(
                "Deferred FSDP gradient sync does not yet support activation offload"
            )

    def _trace_fwd_bwd_with_deferred_fsdp_reductions(
        self,
        model: nn.Module,
        gradient_state: GraphGradientState,
        example_microbatch: tuple[
            torch.Tensor | tuple[torch.Tensor, ...],
            torch.Tensor | tuple[torch.Tensor, ...],
            dict[str, Any],
        ],
        global_valid_tokens: torch.Tensor,
    ) -> GraphWithDeferredFSDPReductions:
        maybe_register_blockmask_pytree_node()
        inputs, labels, extra_kwargs = example_microbatch
        fwd_bwd_fn = make_fwd_bwd_step(model, self.loss_fn)

        def deferred_fwd_bwd_step(
            _gradient_buffers: dict[str, torch.Tensor],
            inputs: torch.Tensor | tuple[torch.Tensor, ...],
            labels: torch.Tensor | tuple[torch.Tensor, ...],
            global_valid_tokens: torch.Tensor,
            extra_kwargs: dict[str, Any],
        ) -> list[torch.Tensor]:
            return fwd_bwd_fn(inputs, labels, global_valid_tokens, extra_kwargs)

        traced = self._trace_fwd_bwd(
            deferred_fwd_bwd_step,
            model,
            (inputs, labels, global_valid_tokens, extra_kwargs),
            graph_state=gradient_state.graph_state,
        )
        passes = self._get_fwd_bwd_graph_passes(traced, finalize=False)
        num_flat_parameters = len(
            flatten_graph_values(
                [
                    parameter
                    for _, parameter in model.named_parameters(remove_duplicate=False)
                ]
            )
        )
        enable_cudagraph = (
            not self.config.training.disable_cuda_graphs
            and "cudagraph_pass" not in self.config.compile.disable_passes
        )
        return build_graph_with_deferred_fsdp_reductions(
            traced,
            num_flat_parameters=num_flat_parameters,
            num_microbatches=self.gradient_accumulation_steps,
            compile_config=self.config.compile,
            enable_cudagraph=enable_cudagraph,
            reuse_unsharded_parameters=(
                self.config.parallelism.fsdp_reshard_after_forward == "never"
            ),
            graph_passes=passes,
        )

    def _run_microbatch_groups(
        self,
        microbatch_groups: list[list[dict[str, Any]]],
        global_valid_tokens: torch.Tensor,
    ) -> Iterator[torch.Tensor]:
        """Run the accumulation window normally or as one deferred-FSDP graph.

        The default trainer invokes forward/backward once per accumulation group
        and yields each loss separately.

        Deferred FSDP instead preprocesses every SPMD microbatch and executes one
        first/middle/final graph that accumulates pre-reduction gradients,
        synchronizes them after the final microbatch, and returns one summed loss.
        """
        if not self.config.compile.enable_deferred_fsdp_gradient_sync:
            return super()._run_microbatch_groups(
                microbatch_groups, global_valid_tokens
            )
        if self.sdc_replayer is not None:
            raise ValueError("Deferred FSDP gradient sync does not support SDC replay")

        assert len(self.model_parts) == 1
        model = self.model_parts[0]
        gradient_state = self._ensure_graph_gradient_state(model)
        gradient_state.prepare_for_backward()
        prepared = []
        for microbatches in microbatch_groups:
            if len(microbatches) != 1:
                # PP with deferred FSDP gradient sync is not supported yet.
                raise ValueError(
                    "Deferred FSDP gradient sync supports SPMD microbatches only"
                )
            input_dict = microbatches[0]
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.to(self.device, non_blocking=True)
            inputs, labels, extra_kwargs = self._preprocess_fwd_bwd_inputs(
                model, input_dict
            )
            prepared.append((inputs, labels, extra_kwargs))

        graph_runner = self._graph_runner
        if graph_runner is None:
            graph_with_deferred_fsdp_reductions = (
                self._trace_fwd_bwd_with_deferred_fsdp_reductions(
                    model,
                    gradient_state,
                    prepared[0],
                    global_valid_tokens,
                )
            )
            graph_runner = GraphRunner(
                graph_with_deferred_fsdp_reductions.traced_result,
                module=model,
                graph_state=gradient_state.graph_state,
                validate_user_inputs=True,
            )
            logger.info(
                "Built deferred FSDP graph for %d microbatches with %d "
                "parameter all-gathers and %d final gradient collectives",
                graph_with_deferred_fsdp_reductions.num_microbatches,
                graph_with_deferred_fsdp_reductions.num_all_gathers,
                graph_with_deferred_fsdp_reductions.num_gradient_collectives,
            )
            self._graph_with_deferred_fsdp_reductions = (
                graph_with_deferred_fsdp_reductions
            )
            self._traced_step = graph_with_deferred_fsdp_reductions.traced_result
            self._graph_runner = graph_runner
        runtime_calls = tuple(
            ((inputs, labels, global_valid_tokens, extra_kwargs), {})
            for inputs, labels, extra_kwargs in prepared
        )
        with self.train_context():
            loss = graph_runner(*runtime_calls)
        return iter((loss,))

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

    def train_step(self, data_iterator: Iterator[dict[str, Any]]):
        PRE_TRAIN_STEP_HOOKS.get(self.config.compile.pass_pipeline, lambda _: None)(
            self
        )
        if self._graph_gradient_state is not None:
            self._graph_gradient_state.validate_grad_bindings()
        if self._graph_with_deferred_fsdp_reductions is not None:
            assert self._graph_runner is not None
            self._graph_runner.validate_state()
        super().train_step(data_iterator)

    def close(self) -> None:
        if self._pinned_pool_ctx is not None:
            self._pinned_pool_ctx.__exit__(None, None, None)
            self._pinned_pool_ctx = None

        super().close()

        self._graph_runner = None
        self._trainable_params = None

        # See Note [explicit cudagraph teardown] in cudagraph.py
        cudagraph_teardown()
