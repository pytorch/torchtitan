# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_ac_regions,
    register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.cudagraph import cudagraph_teardown

register_blockmask_pytree_node()
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.passes import (
    apply_ac_remat_pass,
    apply_regional_inductor,
)
from torchtitan.models.common.attention import (
    annotate_flex_attention_for_regional_inductor,
)
from torchtitan.trainer import Trainer


def make_fwd_bwd_step(loss_fn):
    """Return a plain function that traces the entire fwd+loss+bwd step.

    ``loss_fn`` is captured in the closure so it is not a graph input.
    """

    def fwd_bwd_step(
        model, inputs, labels, global_valid_tokens, extra_inputs, extra_kwargs
    ):
        pred = model(inputs, **extra_inputs, **extra_kwargs)
        loss = loss_fn(pred, labels) / global_valid_tokens
        params = [p for p in model.parameters() if p.requires_grad]
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

        if self.config.compile.mode == "aot_fx_trace" and self.parallel_dims.pp_enabled:
            raise ValueError(
                "aot_fx_trace compile mode does not support Pipeline Parallel"
            )

        # Lazy state for aot_fx_trace mode
        self._traced_step: TracedResult | None = None

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.compile.mode != "aot_fx_trace":
            return super().forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                global_valid_tokens=global_valid_tokens,
            )

        assert len(self.model_parts) == 1
        model = self.model_parts[0]

        inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
            input_dict, labels
        )

        params = [p for p in model.parameters() if p.requires_grad]
        return self._make_fx_forward_backward_step(
            model,
            inputs,
            labels,
            global_valid_tokens,
            params,
            extra_inputs,
            extra_kwargs,
        )

    def _make_fx_forward_backward_step(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
        params: list[torch.Tensor],
        extra_inputs: dict[str, torch.Tensor],
        extra_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        import logging
        import time

        logger = logging.getLogger(__name__)

        if self._traced_step is None:
            fwd_bwd_fn = make_fwd_bwd_step(self.loss_fn)
            annotate_ac_regions(model)

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()

            with (
                self.train_context(),
                self.maybe_enable_amp,
                annotate_flex_attention_for_regional_inductor(),
            ):
                self._traced_step = minimal_fx_tracer(
                    fwd_bwd_fn,
                    (
                        model,
                        inputs,
                        labels,
                        global_valid_tokens,
                        extra_inputs,
                        extra_kwargs,
                    ),
                )

            torch.cuda.synchronize()
            trace_time = time.perf_counter() - t0
            trace_peak = torch.cuda.max_memory_allocated() / 1e9
            logger.info(
                f"[aot_fx_trace] minimal_fx_tracer took {trace_time:.2f}s, "
                f"peak memory during trace: {trace_peak:.2f} GB"
            )

            torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()
            self._traced_step.gm = apply_ac_remat_pass(self._traced_step.gm)
            ac_time = time.perf_counter() - t0
            logger.info(
                f"[aot_fx_trace] apply_ac_remat_pass took {ac_time:.2f}s"
            )

            t0 = time.perf_counter()
            apply_regional_inductor(self._traced_step)
            ri_time = time.perf_counter() - t0
            logger.info(
                f"[aot_fx_trace] apply_regional_inductor took {ri_time:.2f}s"
            )

            num_nodes = len(list(self._traced_step.gm.graph.nodes))
            recomputed = sum(
                1
                for n in self._traced_step.gm.graph.nodes
                if "_recomputed" in n.name
            )

            # Count forward vs backward nodes surviving DCE
            fwd_call_fn = 0
            bwd_call_fn = 0
            fwd_not_recomputed = 0
            for n in self._traced_step.gm.graph.nodes:
                if n.op != "call_function":
                    continue
                custom = n.meta.get("custom", {})
                if custom.get("remat_pass_tag") == "is_backward":
                    bwd_call_fn += 1
                else:
                    fwd_call_fn += 1
                    if "_recomputed" not in n.name:
                        fwd_not_recomputed += 1

            logger.info(
                f"[aot_fx_trace] Graph: {num_nodes} nodes, "
                f"{recomputed} recomputed nodes"
            )
            logger.info(
                f"[aot_fx_trace] Forward call_function: {fwd_call_fn} "
                f"({fwd_not_recomputed} original, "
                f"{fwd_call_fn - fwd_not_recomputed} recomputed), "
                f"Backward call_function: {bwd_call_fn}"
            )
            logger.info(
                f"[aot_fx_trace] Original forward nodes surviving DCE: "
                f"{fwd_not_recomputed} — these stay alive as locals "
                f"during backward execution"
            )

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        with self.train_context(), self.maybe_enable_amp:
            outputs = self._traced_step(
                model,
                inputs,
                labels,
                global_valid_tokens,
                extra_inputs,
                extra_kwargs,
            )
        loss = outputs[0]
        grads = outputs[1:]

        torch.cuda.synchronize()
        step_time = time.perf_counter() - t0
        step_peak = torch.cuda.max_memory_allocated() / 1e9
        grad_norm = torch.sqrt(
            sum(g.float().pow(2).sum() for g in grads)
        ).item()
        logger.info(
            f"[aot_fx_trace] step: {step_time:.3f}s, "
            f"loss: {loss.item():.6f}, grad_norm: {grad_norm:.6f}, "
            f"peak memory: {step_peak:.2f} GB"
        )

        for param, grad in zip(params, grads):
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad

        return loss

    def close(self) -> None:
        super().close()

        # See Note [explicit cudagraph teardown] in cudagraph.py
        cudagraph_teardown()
