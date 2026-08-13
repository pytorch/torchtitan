# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Partial

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    disable_active_parametrization,
)


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

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute chunked loss with one final simple-FSDP gradient reduction.

        Graph trainer's simple FSDP expresses parameter all-gather as a DTensor
        redistribution. Its autograd formula would therefore reduce-scatter the
        parameter gradient from every ``chunk_loss.backward()`` call. Detach the
        replicated parameters, accumulate their per-chunk gradients locally in
        FP32, and redistribute the partial gradient once after the last chunk.
        """
        from torch.distributed._composable.fsdp import FSDPModule

        lm_head = self.lm_head
        assert lm_head is not None, "Set lm_head before calling ChunkedLossWrapper"
        assert not isinstance(lm_head, FSDPModule), (
            "ChunkedLossWrapperWithParamGrads does not support FSDPModule; "
            "graph trainer uses simple FSDP"
        )
        lm_head_params = tuple(lm_head.named_parameters())
        assert len(lm_head_params) == 1 and lm_head_params[0][0] == "weight", (
            "ChunkedLossWrapperWithParamGrads requires lm_head to have exactly "
            "one parameter named 'weight'"
        )
        lm_head_weight = lm_head_params[0][1]

        # Only simple FSDP needs the explicit deferred synchronization. Keep
        # graph trainer's non-distributed execution on the base implementation.
        if not pred.requires_grad or not isinstance(lm_head_weight, DTensor):
            return super().__call__(
                pred,
                labels,
                global_valid_tokens,
                **loss_inputs,
            )

        local_grad_lm_head = _LocalGradientAccumulationModule(lm_head)
        self.lm_head = local_grad_lm_head
        try:
            return super().__call__(
                pred,
                labels,
                global_valid_tokens,
                **loss_inputs,
            )
        finally:
            local_grad_lm_head.remove_grad_hook()
            self.lm_head = lm_head

    @staticmethod
    def _gradient_backprop(
        hidden_states: torch.Tensor,
        accumulated_grad: torch.Tensor,
        total_loss: torch.Tensor,
        lm_head: nn.Module,
        fsdp_enabled: bool,
    ) -> torch.Tensor:
        if isinstance(lm_head, _LocalGradientAccumulationModule):
            lm_head.reduce_accumulated_grads()
            lm_head = lm_head.wrapped_lm_head

        # Native FSDP2 is rejected in __call__. Graph trainer's simple-FSDP
        # gradient synchronization is completed explicitly above.
        assert not fsdp_enabled
        lm_head_weight = next(lm_head.parameters())
        return _ChunkedLossWrapperWithParamGrads.apply(
            hidden_states,
            accumulated_grad,
            total_loss,
            lm_head_weight,
        )


@dataclass(slots=True)
class _LocalWeightState:
    sharded: DTensor
    unsharded: torch.Tensor
    local: torch.Tensor
    accumulated_grad: torch.Tensor | None = None


class _LocalGradientAccumulationModule(nn.Module):
    """Run lm_head with a detached weight and reduce its gradient once."""

    def __init__(self, lm_head: nn.Module) -> None:
        super().__init__()
        self.wrapped_lm_head = lm_head

        sharded_weight = next(lm_head.parameters())
        assert isinstance(sharded_weight, DTensor)
        unsharded_weight = lm_head.weight  # pyrefly: ignore[missing-attribute]
        local_weight = unsharded_weight.detach().requires_grad_(
            sharded_weight.requires_grad
        )
        self._weight = _LocalWeightState(
            sharded=sharded_weight,
            unsharded=unsharded_weight,
            local=local_weight,
        )
        self._grad_hook_handle: torch.utils.hooks.RemovableHandle | None = (
            local_weight.register_post_accumulate_grad_hook(self._accumulate_grad)
        )

    def _accumulate_grad(self, weight: torch.Tensor) -> None:
        grad = weight.grad
        assert grad is not None
        if self._weight.accumulated_grad is None:
            self._weight.accumulated_grad = grad.to(torch.float32)
        else:
            self._weight.accumulated_grad.add_(grad)
        weight.grad = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # simple FSDP installs parameter accessors that normally all-gather on
        # every access. The local weight above is already replicated.
        with disable_active_parametrization():
            return torch.func.functional_call(
                self.wrapped_lm_head,
                {"weight": self._weight.local},
                args,
                kwargs,
                strict=False,
            )

    def reduce_accumulated_grads(self) -> None:
        """Reduce accumulated FP32 grads to the original parameter placements."""
        self.remove_grad_hook()
        accumulated_grad = self._weight.accumulated_grad
        assert accumulated_grad is not None
        sharded_weight = self._weight.sharded
        unsharded_weight = self._weight.unsharded

        if isinstance(unsharded_weight, DTensor):
            num_dp_axes = (
                sharded_weight.device_mesh.ndim - unsharded_weight.device_mesh.ndim
            )
            non_dp_placements = unsharded_weight.placements
        else:
            num_dp_axes = sharded_weight.device_mesh.ndim
            non_dp_placements = ()
        assert num_dp_axes > 0

        local_grad = (
            accumulated_grad.to_local()
            if isinstance(accumulated_grad, DTensor)
            else accumulated_grad
        )
        partial_grad = DTensor.from_local(
            local_grad,
            device_mesh=sharded_weight.device_mesh,
            placements=(Partial(reduce_op="sum"),) * num_dp_axes + non_dp_placements,
            run_check=False,
            shape=sharded_weight.shape,
            stride=sharded_weight.stride(),
        )
        sharded_weight.grad = partial_grad.redistribute(
            placements=sharded_weight.placements
        )

    def remove_grad_hook(self) -> None:
        if self._grad_hook_handle is not None:
            self._grad_hook_handle.remove()
            self._grad_hook_handle = None


class _ChunkedLossWrapperWithParamGrads(torch.autograd.Function):
    """Expose precomputed hidden-state and lm_head weight gradients.

    Forward captures and clears the weight's sharded ``.grad``. Backward
    returns it as an explicit autograd output, so both ``torch.autograd.grad``
    and outer ``loss.backward()`` consumers receive the precomputed gradient.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        accumulated_h_grad: torch.Tensor,
        total_loss: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        weight_grad = lm_head_weight.grad
        assert weight_grad is not None
        ctx.save_for_backward(accumulated_h_grad, weight_grad.detach())
        # Avoid double accumulation when outer loss.backward() reaches the weight.
        lm_head_weight.grad = None
        return total_loss.detach().clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # pyrefly: ignore[bad-override]
        saved = ctx.saved_tensors
        accumulated_h_grad = saved[0]
        weight_grad = saved[1]
        return (
            accumulated_h_grad,
            None,
            None,
            weight_grad,
        )
