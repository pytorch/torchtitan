# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Traced forward+loss+backward step for the RL PolicyTrainer (aot_fx_trace).

This module is intentionally free of any ``monarch`` import so it can be built
and unit-tested on a single GPU without the full actor/controller stack. The
``PolicyTrainer`` actor delegates its ``forward_backward`` to
``RLTracedStep.run`` when the compile mode is ``aot_fx_trace``.

It mirrors ``GraphTrainer._make_fx_forward_backward_step`` but adapts it to RL:

- The RL closure threads GRPO/DAPO's extra per-token tensors
  (``generator_logprobs``, ``advantages``, ``loss_mask``) and the attention
  masks as graph inputs, and returns a ``{"loss", "metrics", "grads"}`` pytree
  so the controller still gets the loss metrics (graph_trainer discards them).
- ``global_valid_tokens`` is passed as a 0-d float tensor graph input, NOT a
  Python int, so ``make_fx`` does not bake the first step's denominator into
  the graph as a constant (the value changes every step). This requires the RL
  loss to divide by a tensor (see ``losses/dapo.py``).
- Grads from the traced graph are written back into ``param.grad`` (assign on
  the first microbatch of a step, accumulate afterwards) so the separate
  ``optim_step`` endpoint, which reads ``param.grad`` and then zeroes it, is
  unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from torch.fx.traceback import annotate_fn

from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import (
    _maybe_materialize_grad_for_param_layout,
    _MODULE_FQN,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    construct_default_graph_passes,
)


def build_pass_config(
    *,
    compile_config: GraphTrainerCompileConfig,
    model_config,
    loss_config,
    parallelism,
    dump_folder: str,
) -> SimpleNamespace:
    """Build the lightweight config object the graph passes read.

    The graph passes only reach a fixed set of attributes (compile,
    model_spec.model.layers, loss, parallelism, activation_checkpoint), so a
    SimpleNamespace over the trainer's real config objects is sufficient. This
    mirrors graph_trainer's own test harness and is shared by the runtime path
    and the precompile driver.
    """
    return SimpleNamespace(
        compile=compile_config,
        model_spec=SimpleNamespace(model=model_config),
        loss=loss_config,
        parallelism=parallelism,
        activation_checkpoint=None,
        dump_folder=dump_folder,
    )


def trace_rl_fwd_bwd(
    model: nn.Module,
    loss_fn,
    *,
    token_ids: torch.Tensor,
    labels: torch.Tensor,
    global_valid_tokens: torch.Tensor,
    extra_kwargs: dict[str, Any],
    train_context,
) -> TracedResult:
    """Non-strict make_fx trace of the RL fwd+loss+bwd step.

    Registers the flex BlockMask pytree node so attention masks flatten as graph
    inputs, then traces under ``train_context``. Shared by the runtime JIT path
    and the single-process precompile driver.
    """
    maybe_register_blockmask_pytree_node()
    fwd_bwd_fn = make_rl_fwd_bwd_step(model, loss_fn)
    with train_context():
        return minimal_fx_tracer(fwd_bwd_fn, module=model)(
            token_ids, labels, global_valid_tokens, extra_kwargs
        )


def make_rl_fwd_bwd_step(model: nn.Module, loss_fn):
    """Return the fwd+loss+bwd function traced by ``minimal_fx_tracer``.

    ``model`` and ``loss_fn`` are captured in the closure so they are threaded
    through the tracer's ``module=`` state rather than appearing as graph
    inputs. The returned function takes only tensors / make_fx-safe primitives.

    Returns a pytree ``{"loss": Tensor, "metrics": dict[str, Tensor],
    "grads": list[Tensor]}``. The metric dict's keys live in the output
    ``TreeSpec``, so ``run_traced`` reconstructs the dict automatically.
    """

    def fwd_bwd_step(
        token_ids: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
        extra_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        pred = model(
            token_ids,
            attention_masks=extra_kwargs["attention_masks"],
            positions=extra_kwargs["positions"],
        )
        # The loss is not a submodule of the model, so annotate it here for
        # downstream passes (bucketing / SAC region attribution).
        loss, metrics = annotate_fn({_MODULE_FQN: "loss"})(loss_fn)(
            pred,
            labels,
            global_valid_tokens,
            generator_logprobs=extra_kwargs["generator_logprobs"],
            advantages=extra_kwargs["advantages"],
            loss_mask=extra_kwargs["loss_mask"],
        )
        params = [
            p
            for _, p in model.named_parameters(remove_duplicate=False)
            if p.requires_grad
        ]
        grads = torch.autograd.grad(loss, params)
        return {"loss": loss, "metrics": metrics, "grads": list(grads)}

    return fwd_bwd_step


class RLTracedStep:
    """Lazily trace (or load a precompiled) fwd+loss+bwd graph and run it.

    One instance lives on each PolicyTrainer actor for the process lifetime.
    """

    def __init__(
        self,
        *,
        compile_config: GraphTrainerCompileConfig,
        parallel_dims: ParallelDims,
        parallelism,
        model_config,
        loss_config,
        train_context,
        dump_folder: str,
    ) -> None:
        self.compile_config = compile_config
        self.parallel_dims = parallel_dims
        self.parallelism = parallelism
        self.model_config = model_config
        self.loss_config = loss_config
        self.train_context = train_context
        self.dump_folder = dump_folder
        self._traced: TracedResult | None = None

    def _pass_config(self) -> SimpleNamespace:
        return build_pass_config(
            compile_config=self.compile_config,
            model_config=self.model_config,
            loss_config=self.loss_config,
            parallelism=self.parallelism,
            dump_folder=self.dump_folder,
        )

    def _fingerprint(self, model: nn.Module):
        from torchtitan.experiments.graph_trainer.precompile import (
            compute_config_fingerprint,
        )

        return compute_config_fingerprint(
            model, self.compile_config, self.parallel_dims
        )

    def _load_precompiled(self, model: nn.Module) -> None:
        from torchtitan.experiments.graph_trainer.precompile import (
            _FX_TRACE_ARTIFACT_KEY,
            precompile_fx_trace_load,
        )
        from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter

        storage = DiskStorageAdapter(self.compile_config.precompile_artifact_dir)
        if not storage.exists(_FX_TRACE_ARTIFACT_KEY):
            raise ValueError(
                "Precompiled fx_trace artifact not found at "
                f"'{self.compile_config.precompile_artifact_dir}/"
                f"{_FX_TRACE_ARTIFACT_KEY}'. Run the RL precompile driver first."
            )
        self._traced = precompile_fx_trace_load(
            storage, expected_fingerprint=self._fingerprint(model)
        )

    def _ensure_traced(
        self,
        model: nn.Module,
        loss_fn,
        token_ids: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
        extra_kwargs: dict[str, Any],
    ) -> None:
        if self._traced is not None:
            return

        maybe_register_blockmask_pytree_node()

        if self.compile_config.precompile_artifact_dir:
            self._load_precompiled(model)
        else:
            self._traced = trace_rl_fwd_bwd(
                model,
                loss_fn,
                token_ids=token_ids,
                labels=labels,
                global_valid_tokens=global_valid_tokens,
                extra_kwargs=extra_kwargs,
                train_context=self.train_context,
            )

        if self.compile_config.enable_passes:
            passes = construct_default_graph_passes(
                self._traced, self._pass_config(), parallel_dims=self.parallel_dims
            )
            self._traced.gm = apply_graph_passes(
                self._traced.gm,
                self._traced.example_inputs,
                passes,
                compile_config=self.compile_config,
            )

    def run(
        self,
        model: nn.Module,
        loss_fn,
        *,
        token_ids: torch.Tensor,
        labels: torch.Tensor,
        positions: torch.Tensor,
        attention_masks: Any,
        generator_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        global_valid_tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Trace-once / run-many one fwd+loss+bwd microbatch.

        Writes grads into ``param.grad`` (assign when ``None``, else accumulate)
        and returns ``(loss, metrics)`` for the caller to reduce across ranks.

        Under cudagraph the returned ``loss``/``metrics`` tensors alias static
        replay buffers that the next ``run`` overwrites. The PolicyTrainer
        consumes them immediately (``reduce_forward_backward_metrics`` reduces to
        Python floats before the next microbatch), so this is safe there; any
        caller that keeps them across ``run`` calls must copy first.
        """
        # A 0-d float tensor so make_fx threads the denominator as a graph
        # input instead of baking the first step's value as a constant.
        gvt = torch.tensor(
            float(global_valid_tokens), dtype=torch.float32, device=device
        )
        extra_kwargs: dict[str, Any] = {
            "attention_masks": attention_masks,
            "positions": positions,
            "generator_logprobs": generator_logprobs,
            "advantages": advantages,
            "loss_mask": loss_mask,
        }

        self._ensure_traced(model, loss_fn, token_ids, labels, gvt, extra_kwargs)

        with self.train_context():
            outputs = run_traced(self._traced, module=model)(
                token_ids, labels, gvt, extra_kwargs
            )

        loss = outputs["loss"]
        metrics = outputs["metrics"]
        grads = outputs["grads"]

        params = [
            p
            for _, p in model.named_parameters(remove_duplicate=False)
            if p.requires_grad
        ]
        for param, grad in zip(params, grads, strict=True):
            grad = _maybe_materialize_grad_for_param_layout(param, grad)
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad

        return loss, metrics
