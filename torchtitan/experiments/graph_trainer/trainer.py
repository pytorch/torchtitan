# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from torchtitan.experiments.graph_trainer.common_utils import (
    accumulate_param_grads_,
    compute_annotated_loss,
    log_timer,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.cudagraph import cudagraph_teardown
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    construct_default_graph_passes,
)
from torchtitan.experiments.graph_trainer.registry import (
    PASS_PIPELINE_REGISTRY,
    POST_INIT_HOOKS,
    PRE_TRAIN_STEP_HOOKS,
)
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer


def _maybe_apply_numa_binding(gpu_index: int, device_type: str) -> None:
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
        gpu_index=gpu_index,
        numa_options=NumaOptions(
            affinity_mode=AffinityMode.NODE,
            should_fall_back_if_binding_fails=True,
        ),
    )
    logger.info("NUMA binding applied for GPU %d", gpu_index)


def make_fwd_bwd_step(model, loss_fn):
    """Return a plain function that traces the entire fwd+loss+bwd step.

    ``model`` and ``loss_fn`` are captured in the closure so neither shows up
    as a graph input. Pass ``model`` through ``minimal_fx_tracer(fn, module=model)``
    to thread its parameters/buffers as static graph inputs.
    """

    def fwd_bwd_step(inputs, labels, global_valid_tokens, extra_kwargs):
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
        params = [
            p
            for _, p in model.named_parameters(remove_duplicate=False)
            if p.requires_grad
        ]
        grads = torch.autograd.grad(loss, params)
        return [loss] + list(grads)

    return fwd_bwd_step


class GraphTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        compile: GraphTrainerCompileConfig = field(
            default_factory=GraphTrainerCompileConfig
        )

    def __init__(self, config):
        super().__init__(config)

        _maybe_apply_numa_binding(self.device.index, self.device.type)

        # Lazy state for aot_fx_trace mode
        self._traced_step: TracedResult | None = None

        # Lazy state for GraphPP precompile load. The stage bundle loader is
        # built once on the first train step (when model_parts are materialized
        # so the fingerprint can be computed) and reused thereafter.
        self._graph_pp_stage_loader: Any = None
        self._graph_pp_stage_loader_loaded: bool = False

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
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: float,
    ) -> torch.Tensor:
        if self.config.compile.mode != "aot_fx_trace":
            return super().forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                global_valid_tokens=global_valid_tokens,
            )
        if self.parallel_dims.pp_enabled:
            from torchtitan.experiments.graph_trainer.graph_pp.runner import (
                GraphPPRunner,
            )

            if not isinstance(self.pp_schedule, GraphPPRunner):
                return super().forward_backward_step(
                    input_dict=input_dict,
                    labels=labels,
                    global_valid_tokens=global_valid_tokens,
                )
            return self._graph_pp_forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                global_valid_tokens=global_valid_tokens,
            )

        assert len(self.model_parts) == 1
        model = self.model_parts[0]

        inputs, labels, extra_kwargs = self.post_dataloading_process(input_dict, labels)
        # remove_duplicate=False to preserve duplicate parameter entries
        # from weight tying (e.g. shared embedding/output weights).
        params = [
            p
            for _, p in model.named_parameters(remove_duplicate=False)
            if p.requires_grad
        ]
        return self._make_fx_forward_backward_step(
            model,
            inputs,
            labels,
            global_valid_tokens,
            params,
            extra_kwargs,
        )

    def _graph_pp_forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: float,
    ) -> torch.Tensor:
        from torchtitan.experiments.graph_trainer.graph_pp.runner import (
            build_graph_pp_graph_bundles,
        )

        # When precompile is enabled, install saved stage bundles instead of
        # tracing. build_graph_pp_graph_bundles still runs _initialize_stages
        # (P2P comm setup); the loaded callables keep their saved compiled state,
        # so the per-stage Inductor compile and (for the supported non-overlap
        # schedules) the OVERLAP_F_B multiplex build are no-ops at load.
        stage_bundle_loader = self._maybe_graph_pp_stage_loader()

        inputs, labels, extra_kwargs = self.post_dataloading_process(input_dict, labels)
        loss_kwargs = {"global_valid_tokens": global_valid_tokens}
        with self.train_context():
            targets, losses = (labels, []) if self.pp_has_last_stage else (None, None)
            if self.pp_has_first_stage:
                build_graph_pp_graph_bundles(
                    self.pp_schedule.schedule,
                    inputs,
                    **extra_kwargs,
                    target=targets,
                    loss_kwargs=loss_kwargs,
                    stage_bundle_loader=stage_bundle_loader,
                )
                self.pp_schedule.step(
                    inputs,
                    **extra_kwargs,
                    target=targets,
                    losses=losses,
                    loss_kwargs=loss_kwargs,
                    return_outputs=False,
                )
            else:
                build_graph_pp_graph_bundles(
                    self.pp_schedule.schedule,
                    **extra_kwargs,
                    target=targets,
                    loss_kwargs=loss_kwargs,
                    stage_bundle_loader=stage_bundle_loader,
                )
                self.pp_schedule.step(
                    **extra_kwargs,
                    target=targets,
                    losses=losses,
                    loss_kwargs=loss_kwargs,
                    return_outputs=False,
                )

        if self.pp_has_last_stage:
            assert losses is not None
            return torch.sum(torch.stack(losses)).to(self.device)
        return torch.tensor([-1.0], device=self.device)

    def _maybe_graph_pp_stage_loader(self):
        """Return a GraphPP stage-bundle loader when precompile is enabled.

        Loads this PP rank's precompiled bundle once and caches the loader so
        later steps reuse it. Returns None when no precompile artifact directory
        is configured, in which case stage bundles are traced live.
        """
        if not self.config.compile.precompile_artifact_dir:
            return None
        if not self._graph_pp_stage_loader_loaded:
            self._graph_pp_stage_loader = self._load_precompiled_graph_pp()
            self._graph_pp_stage_loader_loaded = True
        return self._graph_pp_stage_loader

    def _load_precompiled_graph_pp(self):
        """Load this PP rank's GraphPP bundle and build a stage loader closure.

        The bundle is selected by this process's PP coordinate. The fingerprint
        is computed over only this rank's stage submodules (matching the save
        side, which fingerprints each rank's stages separately).
        """
        from torchtitan.experiments.graph_trainer.graph_pp.precompile import (
            build_graph_pp_stage_loader,
            compute_graph_pp_fingerprint,
            ensure_schedule_precompilable,
            graph_pp_rank_artifact_key,
            load_graph_pp_rank_bundle,
        )
        from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter

        compile_config = self.config.compile
        schedule_name = self.config.parallelism.pipeline_parallel_schedule
        ensure_schedule_precompilable(schedule_name)

        storage = DiskStorageAdapter(compile_config.precompile_artifact_dir)
        pp_rank = self.parallel_dims.get_mesh("pp").get_local_rank()

        key = graph_pp_rank_artifact_key(pp_rank)
        if not storage.exists(key):
            raise ValueError(
                f"GraphPP precompiled bundle not found for pp_rank {pp_rank} at "
                f"'{compile_config.precompile_artifact_dir}/{key}'. Run "
                f"precompile_main with --parallelism.pipeline_parallel_degree set."
            )

        fingerprint = compute_graph_pp_fingerprint(
            self.model_parts,
            compile_config,
            self.parallel_dims,
            schedule_name=schedule_name,
            parallelism=self.config.parallelism,
            training=self.config.training,
            loss_config=self.config.loss,
            debug_config=self.config.debug,
        )
        bundle = load_graph_pp_rank_bundle(
            storage, pp_rank=pp_rank, expected_fingerprint=fingerprint
        )
        return build_graph_pp_stage_loader(bundle)

    def _load_precompiled_fx_trace(self, model: nn.Module) -> None:
        """Load a precompiled aot_fx_trace artifact from disk."""
        from torchtitan.experiments.graph_trainer.precompile import (
            _FX_TRACE_ARTIFACT_KEY,
            compute_config_fingerprint,
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

        self._traced_step = precompile_fx_trace_load(
            storage,
            expected_fingerprint=config_fingerprint,
        )

    def _make_fx_forward_backward_step(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: float,
        params: list[torch.Tensor],
        extra_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        maybe_register_blockmask_pytree_node()
        if self._traced_step is None:
            if self.config.compile.precompile_artifact_dir:
                self._load_precompiled_fx_trace(model)
            else:
                fwd_bwd_fn = make_fwd_bwd_step(model, self.loss_fn)
                with self.train_context(), log_timer("minimal_fx_tracer"):
                    self._traced_step = minimal_fx_tracer(fwd_bwd_fn, module=model)(
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
                passes = pipeline_fn(self._traced_step, self.config)

                self._traced_step.gm = apply_graph_passes(
                    self._traced_step.gm,
                    self._traced_step.example_inputs,
                    passes,
                    compile_config=self.config.compile,
                )
        with self.train_context():
            outputs = run_traced(self._traced_step, module=model)(
                inputs,
                labels,
                global_valid_tokens,
                extra_kwargs,
            )
        loss = outputs[0]
        grads = outputs[1:]

        accumulate_param_grads_(params, grads)
        return loss

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        PRE_TRAIN_STEP_HOOKS.get(self.config.compile.pass_pipeline, lambda _: None)(
            self
        )
        super().train_step(data_iterator)

    def close(self) -> None:
        if self._pinned_pool_ctx is not None:
            self._pinned_pool_ctx.__exit__(None, None, None)
            self._pinned_pool_ctx = None

        super().close()

        # See Note [explicit cudagraph teardown] in cudagraph.py
        cudagraph_teardown()
