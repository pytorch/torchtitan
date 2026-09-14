# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonEP token dispatch for the Kimi K3 latent MoE.

``MoonEPTokenDispatcher`` routes tokens through MoonEP's persistent ``Buffer``
(https://github.com/MoonshotAI/MoonEP); the expert side that consumes its
plan is ``moon_ep_experts.MoonEPGroupedExperts``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

import torch

from torchtitan.models.common.token_dispatcher import (
    BaseEPTokenDispatcher,
    LocalDispatchMetadata,
    LocalTokenDispatcher,
)

logger = logging.getLogger(__name__)


def _import_moonep():
    """Import MoonEP, or explain what is missing.

    Optional in the same sense as fla and DeepEP: absent on a machine that
    cannot run it, and the error names the package rather than surfacing as an
    AttributeError deep in dispatch.
    """
    try:
        import moonep  # pyrefly: ignore [missing-import]
    except ImportError as err:
        raise ImportError(
            "MoonEP is not installed. It is an optional dependency, like "
            "DeepEP: install from https://github.com/MoonshotAI/MoonEP, and "
            "note that it requires NVLink-connected GPUs. Use another "
            "comm_backend on hardware without that topology."
        ) from err
    return moonep


class _MoonEPDispatch(torch.autograd.Function):
    """``buffer.dispatch``; its backward is a combine of the grads.

    Routing weights ride along as a second output so their gradient reaches
    the router.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, buffer, plan_out, x_SH, weights_SK, ids_SK, counts_E):
        hidden_nvsh, weights_nvs, cu_seqlens, plan = buffer.dispatch(
            x_SH, weights_SK, ids_SK, counts_E, zero_copy=False
        )
        # The plan is not a tensor, so it leaves through the caller's box
        # rather than as an output.
        plan_out.append(plan)
        ctx.buffer = buffer
        ctx.plan = plan
        ctx.shape_nvsh = hidden_nvsh.shape
        return hidden_nvsh, weights_nvs, cu_seqlens

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_hidden_nvsh, grad_weights_nvs, _grad_cu):
        if grad_hidden_nvsh is None and grad_weights_nvs is None:
            return None, None, None, None, None, None
        if grad_hidden_nvsh is None:
            grad_hidden_nvsh = torch.zeros(
                ctx.shape_nvsh, dtype=torch.bfloat16, device=grad_weights_nvs.device
            )
        # One combine sums each token's K hidden-grad copies and, when handed
        # the weight grads, gathers them back to [S, K].
        grad_x_SH, grad_weights_SK, _ = ctx.buffer.combine(
            plan=ctx.plan,
            hidden_nvsh=grad_hidden_nvsh.to(torch.bfloat16).contiguous(),
            route_weights_nvs=(
                None
                if grad_weights_nvs is None
                else grad_weights_nvs.to(torch.float32).contiguous()
            ),
        )
        return None, None, grad_x_SH, grad_weights_SK, None, None


class _MoonEPCombine(torch.autograd.Function):
    """``buffer.combine``; its backward is a re-dispatch on the same plan."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, buffer, plan, hidden_nvsh):
        out_SH, _, _ = buffer.combine(plan=plan, hidden_nvsh=hidden_nvsh)
        ctx.buffer = buffer
        ctx.plan = plan
        return out_SH

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_out_SH):
        grad_hidden_nvsh, _, _, _ = ctx.buffer.dispatch(
            grad_out_SH.to(torch.bfloat16), plan=ctx.plan
        )
        return None, None, grad_hidden_nvsh


@dataclass
class MoonEPDispatchMetadata:
    """What ``combine`` needs to invert the routing."""

    plan: object
    weights_nvs: torch.Tensor
    input_dtype: torch.dtype


class MoonEPTokenDispatcher(BaseEPTokenDispatcher):
    """Balanced EP dispatch through MoonEP's kernels.

    With no EP mesh it falls back to local dispatch, so a flavor carrying this
    config still runs unsharded.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseEPTokenDispatcher.Config):
        static_token_capacity: ClassVar[bool] = True
        ep1_local_fallback: ClassVar[bool] = True

        hidden_dim: int | None = None
        """Feature width of the tokens entering dispatch (sizes the buffer)."""

        num_max_tokens_per_rank: int | None = None
        """MoonEP's ``S``: the exact per-rank token count of every dispatch,
        a static shape. Filled from the training config by core's
        ``update_ep_token_dispatcher_config``; never a guess."""

        num_prefetch_slots: int | None = None
        """MoonEP's ``B``; None is ``E // num_ep_ranks``, which training
        requires. The experts read it at attach."""

        num_sms: int = 32
        """SMs MoonEP's kernels may occupy (its default)."""

        token_padding: int = 128
        """MoonEP's internal alignment (its default)."""

    def __init__(self, config: "MoonEPTokenDispatcher.Config") -> None:
        super().__init__(config)
        self.hidden_dim = config.hidden_dim
        self.num_max_tokens_per_rank = config.num_max_tokens_per_rank
        self.num_prefetch_slots = config.num_prefetch_slots
        self.num_sms = config.num_sms
        self.token_padding = config.token_padding
        self._buffer = None
        self._current: tuple[object, torch.Tensor] | None = None

    def _buffer_factory(self, **kwargs):
        """``moonep.Buffer`` by default; the tests substitute their double."""
        return _import_moonep().Buffer(**kwargs)

    def current_plan(self) -> tuple[object, torch.Tensor]:
        """The plan and ``cu_seqlens`` of the dispatch in flight, for the
        expert side."""
        if self._current is None:
            raise RuntimeError("MoonEP experts ran before a dispatch in this step.")
        return self._current

    def init_buffer(self) -> None:
        """Allocate MoonEP's persistent buffer on the EP group, once, from
        ``wire_meshes`` (a collective)."""
        if self.ep_mesh is None:
            return
        if self.hidden_dim is None or self.num_max_tokens_per_rank is None:
            raise ValueError(
                "MoonEPTokenDispatcher.Config needs hidden_dim (the dispatched "
                "feature width) and num_max_tokens_per_rank (MoonEP's static "
                "S, the per-rank token count of every dispatch) before the "
                "buffer can be allocated."
            )
        ep_size = self.ep_mesh.size()
        self._buffer = self._buffer_factory(
            S=self.num_max_tokens_per_rank,
            H=self.hidden_dim,
            K=self.top_k,
            E=self.num_experts,
            num_ep_ranks=ep_size,
            num_sms=self.num_sms,
            token_padding=self.token_padding,
            B=self.num_prefetch_slots,
            group=self.ep_mesh.get_group(),
        )
        logger.info(
            "MoonEP dispatcher: buffer S=%d H=%d K=%d E=%d on an ep group of %d",
            self.num_max_tokens_per_rank,
            self.hidden_dim,
            self.top_k,
            self.num_experts,
            ep_size,
        )

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        """Route the padded local tokens through MoonEP's planner.

        Returns ``(routed_input_RD, num_tokens_per_row, metadata)``: the
        received tokens in MoonEP's row order and the token count of each of
        the ``E + B`` rows, which ``MoonEPGroupedExperts`` consumes.
        """
        if self.ep_mesh is None:
            return LocalTokenDispatcher.dispatch(
                self,
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
            )
        if self._buffer is None:
            raise RuntimeError(
                "MoonEP dispatcher used before wire_meshes(); the buffer is "
                "allocated collectively on the EP group first."
            )
        if x_TD.shape[0] != self.num_max_tokens_per_rank:
            raise ValueError(
                f"MoonEP's S is a static shape: the buffer was sized for "
                f"{self.num_max_tokens_per_rank} tokens per rank and this "
                f"dispatch carries {x_TD.shape[0]}. Set "
                "num_max_tokens_per_rank to the per-rank micro-batch token "
                "count."
            )
        # api.py asserts bf16 hidden; weights fp32, ids and counts int32.
        plan_box: list = []
        hidden_nvsh, weights_nvs, cu_seqlens = _MoonEPDispatch.apply(
            self._buffer,
            plan_box,
            x_TD.to(torch.bfloat16),
            topk_scores_TK.to(torch.float32),
            topk_expert_ids_TK.to(torch.int32),
            num_local_tokens_per_expert_E.to(torch.int32),
        )
        num_tokens_per_row = torch.diff(cu_seqlens, prepend=cu_seqlens.new_zeros(1))
        metadata = MoonEPDispatchMetadata(
            plan=plan_box[0],
            weights_nvs=weights_nvs,
            input_dtype=x_TD.dtype,
        )
        self._current = (plan_box[0], cu_seqlens)
        return hidden_nvsh, num_tokens_per_row, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: object,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Invert ``dispatch``: one weighted row per original token, in order.

        The routing weights are applied here, in autograd, exactly as the
        standard dispatcher does before its scatter-add; MoonEP's combine only
        sums the copies.
        """
        if self.ep_mesh is None:
            if not isinstance(metadata, LocalDispatchMetadata):
                raise TypeError(f"expected LocalDispatchMetadata, got {type(metadata)}")
            return LocalTokenDispatcher.combine(self, routed_output_RD, metadata, x_TD)
        if not isinstance(metadata, MoonEPDispatchMetadata):
            raise TypeError(f"expected MoonEPDispatchMetadata, got {type(metadata)}")
        weighted_nvsh = (
            routed_output_RD.to(torch.float32) * metadata.weights_nvs[:, None]
        ).to(torch.bfloat16)
        out_TD = _MoonEPCombine.apply(self._buffer, metadata.plan, weighted_nvsh)
        if out_TD.shape != x_TD.shape:
            raise RuntimeError(
                f"MoonEP combine returned {tuple(out_TD.shape)} for input "
                f"{tuple(x_TD.shape)}; token conservation is broken."
            )
        return out_TD.to(metadata.input_dtype)
