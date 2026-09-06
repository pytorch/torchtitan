# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from torchtitan.components.loss import ChunkedLossWrapper


class ChunkedLossWrapperWithParamGrads(ChunkedLossWrapper):
    """ChunkedLossWrapper variant that exposes sharded lm_head param grads as
    explicit autograd outputs of the returned loss tensor, so outer
    ``torch.autograd.grad(loss, [hidden_states, *lm_head.parameters()])``
    returns real grads instead of relying on ``param.grad`` side effects.

    Designed for graph_trainer, where the chunk loop's per-chunk
    ``param.grad`` side-effect writes don't survive the captured graph and
    replay therefore produces all-zero param grads. Compatible with both
    outer ``loss.backward()`` and ``torch.autograd.grad`` consumers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedLossWrapper.Config):
        pass

    @staticmethod
    def _gradient_backprop(
        hidden_states: tuple[torch.Tensor, ...],
        accumulated_grads: tuple[torch.Tensor, ...],
        total_loss: torch.Tensor,
        lm_head: nn.Module,
        fsdp_enabled: bool,
    ) -> torch.Tensor:
        return _ChunkedLossWrapperWithParamGrads.apply(
            len(hidden_states),
            *hidden_states,
            *accumulated_grads,
            total_loss,
            lm_head,
            fsdp_enabled,
            *lm_head.parameters(),
        )


class _ChunkedLossWrapperWithParamGrads(torch.autograd.Function):
    """Like ``_DecoderOutputGradientBackProp`` but also plumbs sharded grads
    for the lm_head parameters out as explicit autograd outputs, so outer
    ``torch.autograd.grad(loss, [hidden_states, *lm_head.parameters()])``
    returns correct grads instead of relying on ``param.grad`` side effects.

    Forward is invoked *after* the chunked ``chunk_loss.backward()`` loop has
    populated each lm_head param's sharded ``.grad`` (via FSDP's last-chunk
    reduce-scatter). Forward captures those grads, clears ``.grad``, and
    disables grad sync — so that outer ``loss.backward()`` consumers, whose
    AccumulateGrad would otherwise (a) double-add onto ``.grad`` and (b)
    re-fire FSDP's reduce-scatter on already-sharded data, get clean
    behavior. Backward queues a callback to restore grad sync after the
    engine drains the rest of the backward graph.

    Outer ``torch.autograd.grad`` consumers bypass AccumulateGrad entirely
    and just receive the saved sharded grads directly.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, num_predictions: int, *args: Any) -> torch.Tensor:
        expected_fixed_args = 2 * num_predictions + 3
        if len(args) < expected_fixed_args:
            raise ValueError(
                "Graph chunked-loss autograd bridge expected at least "
                f"{expected_fixed_args} arguments for {num_predictions} "
                f"predictions, got {len(args)}."
            )
        accumulated_h_grads = args[num_predictions : 2 * num_predictions]
        total_loss = args[2 * num_predictions]
        lm_head = args[2 * num_predictions + 1]
        fsdp_enabled = args[2 * num_predictions + 2]
        lm_params = args[2 * num_predictions + 3 :]
        # The chunk loop above already populated each lm_head param's
        # ``.grad`` with the correctly sharded value via the FSDP last-chunk
        # post-accumulate-grad hook (reduce-scatter). Capture those grads
        # into saved_tensors so backward ca route them as autograd outputs
        # for the lm_head param inputs of this Function. Additionally, we need
        # following changes:
        # 1. We need to clear ``.grad`` so a subsequent outer ``loss.backward()`` doesn't
        # double-add when AccumulateGrad fires on those params with our returned grads.
        # 2. We need to disable FSDP grad sync on lm_head: outer .backward() would
        # otherwise re-fire the post-accumulate-grad hook on already-sharded
        # data. The restore is queued in backward() below.
        sharded_param_grads = [p.grad.detach() for p in lm_params]
        for p in lm_params:
            p.grad = None
        if fsdp_enabled:
            lm_head.set_requires_gradient_sync(False, recurse=False)
        ctx.num_predictions = num_predictions
        ctx.save_for_backward(*accumulated_h_grads, *sharded_param_grads)
        ctx.lm_head = lm_head
        ctx.fsdp_enabled = fsdp_enabled
        return total_loss.detach().clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # pyrefly: ignore[bad-override]
        saved = ctx.saved_tensors
        accumulated_h_grads = saved[: ctx.num_predictions]
        param_grads = saved[ctx.num_predictions :]
        if ctx.fsdp_enabled:
            # Restore FSDP grad sync that forward() disabled. Use
            # queue_callback to defer the restore until the engine drains
            # the rest of the backward graph — including each lm_head
            # param's AccumulateGrad firing on the grads we return below.
            # If we restored here (synchronously, before returning), the
            # first AccumulateGrad would see sync=True and try to
            # reduce-scatter our already-sharded grad → wrong result.
            lm_head = ctx.lm_head
            torch.autograd.Variable._execution_engine.queue_callback(
                lambda: lm_head.set_requires_gradient_sync(True, recurse=False)
            )
        return (
            None,
            *accumulated_h_grads,
            *(None for _ in range(ctx.num_predictions)),
            None,
            None,
            None,
            *param_grads,
        )
