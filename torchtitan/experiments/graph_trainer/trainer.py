# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torchinsights.graph_estimation import (
    BENCHMARK,
    COST_MODEL,
    estimate_peak_memory,
    estimate_runtime,
    INTERPRETER,
    optimizer_state_bytes,
    RuntimeEstimator,
    UNKNOWN_OPTIMIZER_BYTES,
)
from torchinsights.graph_estimation._fx_utils import (
    ACT,
    GRAD,
    INPUT,
    PARAM,
    STATES_PER_PARAM,
    TEMP,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    accumulate_param_grads_,
    compute_annotated_loss,
    log_timer,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    trace_input_preparer_keys,
)
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
    TRACE_CALL_INPUT_PREPARERS,
    TRACE_INPUT_PREPARERS,
)
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


def _ratio(num: float, den: float) -> str:
    # A zero on either side means one mode did not price this node at all (the
    # per-node dicts drop zero-cost entries), which is not the same as "free".
    return "     n/a" if den <= 0 or num <= 0 else f"{num / den:7.2f}x"


def log_runtime_estimate_comparison(
    gm: torch.fx.GraphModule,
    cost_model_result,
    benchmark_result,
    interpreter_result=None,
    *,
    top_k: int = 12,
) -> None:
    """Log the cost-model, benchmark and (optional) interpreter estimates side by side.

    All three price the same graph: the roofline cost model derives each op's
    time from flops and bandwidth, benchmark mode times each real kernel in
    isolation, and the interpreter replays the whole graph. The ratio columns
    are the interesting part -- the ILP's recompute-vs-offload objective is
    built directly on these per-node numbers, so an op the analytical model
    misprices is an op the solver decides wrongly about.

    Read the totals carefully, the three columns do not mean the same thing:

    * cost_model / benchmark: ``fwd + bwd`` is the SUM of isolated per-op times.
      Neither costs collectives, so both are compute-only.
    * interpreter: ``fwd + bwd`` also comes from a serialized per-op replay, but
      its ``total_runtime_ms`` is the MEASURED end-to-end step time -- kernels
      overlap and communication hides behind compute, so it is smaller than the
      per-op sum and is the only figure here that includes collectives. It is
      reported on its own ``end-to-end`` row, not compared against the others.
    """
    cm, bm = cost_model_result, benchmark_result
    it = interpreter_result
    cm_ms, bm_ms = cm.node_runtimes_ms, bm.node_runtimes_ms
    it_ms = it.node_runtimes_ms if it is not None else {}

    logger.info("=" * 88)
    logger.info(
        "RUNTIME ESTIMATE  %s vs %s%s",
        cm.estimate_mode_type,
        bm.estimate_mode_type,
        f" vs {it.estimate_mode_type}" if it is not None else "",
    )
    if it is None:
        logger.info(
            "  %-12s %14s %14s %10s",
            "totals(ms)",
            "cost_model",
            "benchmark",
            "bench/cm",
        )
        for label, a, b in (
            ("forward", cm.fwd_runtime_ms, bm.fwd_runtime_ms),
            ("backward", cm.bwd_runtime_ms, bm.bwd_runtime_ms),
            ("sum(per-op)", cm.total_runtime_ms, bm.total_runtime_ms),
        ):
            logger.info("  %-12s %14.3f %14.3f %10s", label, a, b, _ratio(b, a))
    else:
        logger.info(
            "  %-12s %12s %12s %12s %9s %9s",
            "totals(ms)",
            "cost_model",
            "benchmark",
            "interp",
            "bench/cm",
            "interp/cm",
        )
        for label, a, b, c in (
            ("forward", cm.fwd_runtime_ms, bm.fwd_runtime_ms, it.fwd_runtime_ms),
            ("backward", cm.bwd_runtime_ms, bm.bwd_runtime_ms, it.bwd_runtime_ms),
            (
                "sum(per-op)",
                cm.total_runtime_ms,
                bm.total_runtime_ms,
                it.fwd_runtime_ms + it.bwd_runtime_ms,
            ),
        ):
            logger.info(
                "  %-12s %12.3f %12.3f %12.3f %9s %9s",
                label,
                a,
                b,
                c,
                _ratio(b, a),
                _ratio(c, a),
            )
        # Measured wall clock of the real replay. Lower than the per-op sum by
        # whatever the schedule manages to overlap, so the gap between these two
        # interpreter rows is the overlap the static modes cannot see at all.
        it_sum = it.fwd_runtime_ms + it.bwd_runtime_ms
        logger.info(
            "  %-12s %12s %12s %12.3f %9s %9s   <- measured, overlap-aware, "
            "includes collectives (overlap factor %s vs its own per-op sum)",
            "end-to-end",
            "n/a",
            "n/a",
            it.total_runtime_ms,
            "n/a",
            _ratio(it.total_runtime_ms, cm.total_runtime_ms),
            _ratio(it.total_runtime_ms, it_sum),
        )

    target_of = {
        n.name: str(getattr(n.target, "_overloadpacket", n.target))
        for n in gm.graph.nodes
        if n.op == "call_function"
    }

    # Aggregate by op family: a systematic per-family ratio is a model error,
    # whereas one slow node is usually just one slow node.
    fam_cm: dict = defaultdict(float)
    fam_bm: dict = defaultdict(float)
    fam_it: dict = defaultdict(float)
    fam_n: Counter = Counter()
    for name in set(cm_ms) | set(bm_ms) | set(it_ms):
        tgt = target_of.get(name, "<not-call_function>")
        fam_cm[tgt] += cm_ms.get(name, 0.0)
        fam_bm[tgt] += bm_ms.get(name, 0.0)
        fam_it[tgt] += it_ms.get(name, 0.0)
        fam_n[tgt] += 1
    logger.info("  by op family, top %d by benchmark time", top_k)
    if it is None:
        logger.info(
            "    %12s %12s %9s %6s  %s",
            "benchmark",
            "cost_model",
            "ratio",
            "count",
            "target",
        )
        for tgt, b in sorted(fam_bm.items(), key=lambda kv: -kv[1])[:top_k]:
            logger.info(
                "    %12.3f %12.3f %9s %6d  %s",
                b,
                fam_cm[tgt],
                _ratio(b, fam_cm[tgt]),
                fam_n[tgt],
                tgt,
            )
    else:
        logger.info(
            "    %12s %12s %12s %9s %9s %6s  %s",
            "benchmark",
            "cost_model",
            "interp",
            "bench/cm",
            "interp/cm",
            "count",
            "target",
        )
        for tgt, b in sorted(fam_bm.items(), key=lambda kv: -kv[1])[:top_k]:
            logger.info(
                "    %12.3f %12.3f %12.3f %9s %9s %6d  %s",
                b,
                fam_cm[tgt],
                fam_it[tgt],
                _ratio(b, fam_cm[tgt]),
                _ratio(fam_it[tgt], fam_cm[tgt]),
                fam_n[tgt],
                tgt,
            )

    logger.info("  slowest %d nodes by benchmark time", top_k)
    if it is None:
        logger.info(
            "    %12s %12s %9s  %s", "benchmark", "cost_model", "ratio", "node (target)"
        )
        for name, b in sorted(bm_ms.items(), key=lambda kv: -kv[1])[:top_k]:
            a = cm_ms.get(name, 0.0)
            logger.info(
                "    %12.4f %12.4f %9s  %s (%s)",
                b,
                a,
                _ratio(b, a),
                name,
                target_of.get(name, "?"),
            )
    else:
        logger.info(
            "    %12s %12s %12s %9s %9s  %s",
            "benchmark",
            "cost_model",
            "interp",
            "bench/cm",
            "interp/cm",
            "node (target)",
        )
        for name, b in sorted(bm_ms.items(), key=lambda kv: -kv[1])[:top_k]:
            a = cm_ms.get(name, 0.0)
            c = it_ms.get(name, 0.0)
            logger.info(
                "    %12.4f %12.4f %12.4f %9s %9s  %s (%s)",
                b,
                a,
                c,
                _ratio(b, a),
                _ratio(c, a),
                name,
                target_of.get(name, "?"),
            )

    # Nodes only one mode could price: benchmark cannot time a HOP (fused
    # inductor region, flex_attention), so those fall back to the roofline and
    # any decision resting on their cost is only as good as the roofline.
    cm_only = sorted(set(cm_ms) - set(bm_ms))
    bm_only = sorted(set(bm_ms) - set(cm_ms))
    logger.info(
        "  costed nodes: cost_model=%d benchmark=%d%s | cost_model-only=%d "
        "benchmark-only=%d",
        len(cm_ms),
        len(bm_ms),
        f" interpreter={len(it_ms)}" if it is not None else "",
        len(cm_only),
        len(bm_only),
    )
    if it is not None:
        # The interpreter replays the real graph, so it prices nodes the static
        # modes skip (collectives, HOPs) and skips nodes the replay elides.
        it_only = sorted(set(it_ms) - set(cm_ms))
        no_it = sorted(set(cm_ms) - set(it_ms))
        logger.info(
            "    interpreter-only=%d not-priced-by-interpreter=%d",
            len(it_only),
            len(no_it),
        )
        for label, names in (
            ("interpreter-only", it_only),
            ("not-priced-by-interpreter", no_it),
        ):
            for name in names[:5]:
                logger.info("    %s: %s (%s)", label, name, target_of.get(name, "?"))
    for label, names in (("cost_model-only", cm_only), ("benchmark-only", bm_only)):
        for name in names[:5]:
            logger.info("    %s: %s (%s)", label, name, target_of.get(name, "?"))

    if cm.mod_runtimes_ms and bm.mod_runtimes_ms:
        logger.info("  top %d modules by benchmark fw+bw (ms)", top_k // 2)
        it_mod = it.mod_runtimes_ms if it is not None else {}
        if it is None:
            logger.info(
                "    %8s %8s %8s %8s  %s", "cm_fw", "bm_fw", "cm_bw", "bm_bw", "module"
            )
        else:
            logger.info(
                "    %8s %8s %8s %8s %8s %8s  %s",
                "cm_fw",
                "bm_fw",
                "it_fw",
                "cm_bw",
                "bm_bw",
                "it_bw",
                "module",
            )
        ranked = sorted(
            bm.mod_runtimes_ms.items(),
            key=lambda kv: -(kv[1].get("fw", 0.0) + kv[1].get("bw", 0.0)),
        )
        for fqn, bt in ranked[: top_k // 2]:
            ct = cm.mod_runtimes_ms.get(fqn, {})
            if it is None:
                logger.info(
                    "    %8.3f %8.3f %8.3f %8.3f  %s",
                    ct.get("fw", 0.0),
                    bt.get("fw", 0.0),
                    ct.get("bw", 0.0),
                    bt.get("bw", 0.0),
                    fqn,
                )
            else:
                itt = it_mod.get(fqn, {})
                logger.info(
                    "    %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f  %s",
                    ct.get("fw", 0.0),
                    bt.get("fw", 0.0),
                    itt.get("fw", 0.0),
                    ct.get("bw", 0.0),
                    bt.get("bw", 0.0),
                    itt.get("bw", 0.0),
                    fqn,
                )
    logger.info("=" * 88)


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

        # Any policy that can emit ao.offload needs the pinned pool, not just
        # one named policy. Without it ao.offload does a fresh cudaHostAlloc per
        # tensor: measured 11.2 GB/s for 320 MB with a fresh pinned allocation
        # each copy vs 53.1 GB/s reusing one, and a real auto_perf_maxing step
        # aggregated only 32.3 GB/s of D2H against ~52 GB/s benchmarked at the
        # same transfer size. That gap is what made the offload plan's modeled
        # transfer window optimistic, and the per-tensor allocation cost is also
        # what caps usable pinned memory per rank. The pool is untouched when a
        # graph offloads nothing, so enabling it unconditionally is free.
        from torch._functorch._activation_offloading.offload_ops import (
            pinned_memory_pool,
        )

        self._pinned_pool_ctx = pinned_memory_pool()
        self._pinned_pool_ctx.__enter__()

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
        inputs, labels, extra_kwargs = self.post_dataloading_process(input_dict, labels)
        loss_kwargs = {"global_valid_tokens": global_valid_tokens}
        with self.train_context():
            targets, losses = (labels, []) if self.pp_has_last_stage else (None, None)
            schedule_args = (inputs,) if self.pp_has_first_stage else ()
            self.pp_schedule.step(
                *schedule_args,
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
                    self._traced_step = minimal_fx_tracer(
                        fwd_bwd_fn,
                        module=model,
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
                    model_parts=self.model_parts,
                )
                # # logger.info(f"passes[3]: {passes[3]}")
                # # 0. eliminate_dead_code_pass
                # # 1. canonicalize_graph_pass
                # # 2. deduplicate_fsdp_unshard_chains_pass
                # # 3. tag_with_memory_policy_pass
                # # 4. apply_cpu_offload_pass
                # # 5. selective_activation_remat_pass
                # # 6. joint_transformer_block_bucketing_reordering_pass
                # # 7. annotate_flex_attention_for_regional_inductor_pass
                # # 8. regional_inductor_pass

                # passes_1 = []
                # passes_1.append(passes[0])
                # passes_1.append(passes[1])
                # passes_1.append(passes[2])

                # self._traced_step.gm = apply_graph_passes(
                #     self._traced_step.gm,
                #     self._traced_step.example_inputs,
                #     passes_1,
                #     compile_config=self.config.compile,
                #     num_state_inputs=self._traced_step.num_static_inputs,
                # )

                # self._traced_step.gm = _tag_flex_attention(self._traced_step.gm)
                # passes_1 = []
                # passes_1.append(passes[5])
                # self._traced_step.gm = apply_graph_passes(
                #     self._traced_step.gm,
                #     self._traced_step.example_inputs,
                #     passes_1,
                #     compile_config=self.config.compile,
                #     num_state_inputs=self._traced_step.num_static_inputs,
                # )

                # passes_1 = []
                # passes_1.append(passes[7])
                # passes_1.append(passes[8])

                # self._traced_step.gm = apply_graph_passes(
                #     self._traced_step.gm,
                #     self._traced_step.example_inputs,
                #     passes_1,
                #     compile_config=self.config.compile,
                #     num_state_inputs=self._traced_step.num_static_inputs,
                # )

                # passes_2 = []
                # passes_2.append(passes[3])
                # passes_2.append(passes[4])
                # passes_2.append(passes[5])
                # passes_1.append(passes[6])
                # self._traced_step.gm = apply_graph_passes(
                #     self._traced_step.gm,
                #     self._traced_step.example_inputs,
                #     passes_2,
                #     compile_config=self.config.compile,
                #     num_state_inputs=self._traced_step.num_static_inputs,
                # )

                # --------------------------------------------------------------------------------
                # Once only: BENCHMARK runs every kernel for real (21 s on qwen3-14B)
                # and allocates while doing it, so per-step it would dominate the step
                # time and perturb the memory numbers measured just above.
                compare_runtime_est = (
                    False  # not getattr(self, "_runtime_est_compared", False)
                )
                if compare_runtime_est:
                    self._runtime_est_compared = True
                    # Static modes price the pre-pass graph, which is the graph
                    # the memory-policy ILP also prices, so these are the numbers
                    # its objective is built on.
                    cost_model_est_runtime = estimate_runtime(
                        gm=self._traced_step.gm,
                        mode=COST_MODEL,
                    )

                    benchmark_est_runtime = estimate_runtime(
                        gm=self._traced_step.gm,
                        mode=BENCHMARK,
                    )

                self._traced_step.gm = apply_graph_passes(
                    self._traced_step.gm,
                    self._traced_step.example_inputs,
                    passes,
                    compile_config=self.config.compile,
                    num_state_inputs=self._traced_step.num_static_inputs,
                )

                if compare_runtime_est:
                    # INTERPRETER must run AFTER the passes. It executes the real
                    # graph, and until regional_inductor_pass compiles the
                    # FlexAttention HOP the eager fallback is math_attention,
                    # which materializes the full batch x heads x seq x seq score
                    # matrix per layer (2 GiB at 16x32x1024x1024 fp32) and holds
                    # it for backward -- it OOMs around layer 28 of 32.
                    interpreter_est_runtime = RuntimeEstimator()(INTERPRETER).estimate(
                        self._traced_step,
                        model,
                        inputs,
                        labels,
                        global_valid_tokens,
                        extra_kwargs,
                        warmup=1,
                        reps=3,
                    )

                    # NOTE: the static modes measured the pre-pass graph and the
                    # interpreter the post-pass one, so per-node rows only line up
                    # for nodes the passes left untouched. Recompute duplicates
                    # (``*_recomputed``) and offload ops appear as
                    # interpreter-only.
                    log_runtime_estimate_comparison(
                        self._traced_step.gm,
                        cost_model_est_runtime,
                        benchmark_est_runtime,
                        interpreter_est_runtime,
                    )
        with self.train_context():
            # torch.cuda.synchronize()
            # torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_peak_memory_stats()
            outputs = run_traced(self._traced_step, module=model)(
                inputs,
                labels,
                global_valid_tokens,
                extra_kwargs,
            )
            real_peak = torch.cuda.max_memory_allocated()
            mem_factor = 1 << 30
            logger.info(f"Real Peak memory: {real_peak/mem_factor:.2f}")
            # --------------------------------------------------------------------------------
            est_mem = estimate_peak_memory(
                self._traced_step.gm,
                num_state_inputs=self._traced_step.num_static_inputs,
            )
            opt_bytes = optimizer_state_bytes(
                getattr(self.config, "optimizer", None), self.model_parts[0]
            )
            # Estimated peak includes optimizer state (resident from step 2 on).
            # This path only reports, so an unmodelled optimizer says so rather
            # than printing a number that silently omits its state.
            if opt_bytes == UNKNOWN_OPTIMIZER_BYTES:
                opt_bytes = 0
                logger.info(
                    "Estimated Peak memory: %.2f + unknown optimizer state "
                    "(graph only; this optimizer is not in STATES_PER_PARAM)",
                    est_mem.peak_bytes / mem_factor,
                )
            else:
                estimated = est_mem.peak_bytes + opt_bytes
                logger.info(f"Estimated Peak memory: {estimated/mem_factor:.2f}")

            logger.info(
                f"Categories: "
                f"Activation: {est_mem.per_category_at_peak[ACT]/mem_factor:.2f} "
                f"Grad: {est_mem.per_category_at_peak.get(GRAD, 0)/mem_factor:.2f} "
                f"INPUT: {est_mem.per_category_at_peak[INPUT]/mem_factor:.2f} "
                f"PARAM: {est_mem.per_category_at_peak[PARAM]/mem_factor:.2f} "
                f"TEMP: {est_mem.per_category_at_peak[TEMP]/mem_factor:.2f} "
                f"OPT: {opt_bytes/mem_factor:.2f} "
            )
            # --------------------------------------------------------------------------------
        loss = outputs[0]
        grads = outputs[1:]

        accumulate_param_grads_(params, grads)
        return loss

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
