# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonEP token dispatch for the K3 MoE (report sec 5.2.1).

MoonshotAI released the transport as MoonEP
(https://github.com/MoonshotAI/MoonEP, package ``moonep``). This file is the
dispatch/combine half of an integration, written against ``moonep/api.py``:

* ``Buffer(S, H, K, E, num_ep_ranks, num_sms, token_padding, B, group)``
  preallocates once per EP group. ``S`` is a STATIC shape -- every dispatch
  input is exactly ``[S, H]`` bf16 -- so it is derived from the training config
  (tokens per micro-batch per rank after CP/TP), not treated as a bound.
* ``buffer.dispatch(hidden_sh, route_weights_sk, topk_experts_sk,
  tokens_per_expert)`` returns ``(hidden_nvsh, route_weights_nvs, cu_seqlens,
  plan)``: tokens in VM-group order over ``E + B`` rows (the ``E`` experts plus
  ``B`` prefetch slots), ``cu_seqlens[E + B]`` the padded end offset per row.
* The kernels do NOT carry autograd. The README gives the backward as a
  recipe -- dispatch's backward is ``combine(plan, grad)``, combine's backward
  is ``dispatch(grad, plan=plan)`` -- and the two ``autograd.Function``s below
  implement exactly that. Routing weights are applied on the torchtitan side,
  in autograd, so the router's gradient takes the same path it takes with the
  standard dispatcher; MoonEP's own combine-side weighting is not used.

The expert side -- the ``[E+B]`` weight tables, ``prefetch_weight`` before
the grouped GEMM and ``reduce_grad`` in the backward -- is
``moon_ep_experts.MoonEPGroupedExperts``; the MoE's parallelize attaches it
to this dispatcher, and it reads the plan of the dispatch in flight through
``current_plan``. The whole unit is exercised on CPU through the in-process
double in ``tests/moonep_fake.py``; what remains for NVLink hardware is the
table allocation over MoonEP's VMM primitives and the kernels themselves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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
        import moonep  # type: ignore[import-not-found]
    except ImportError as err:
        raise ImportError(
            "MoonEP is not installed. It is an optional dependency, like "
            "DeepEP: install from https://github.com/MoonshotAI/MoonEP, and "
            "note that it requires NVLink-connected GPUs. Use another "
            "comm_backend on hardware without that topology."
        ) from err
    return moonep


class _MoonEPDispatch(torch.autograd.Function):
    """``buffer.dispatch`` with the README's backward: a combine of the grads.

    Routing weights ride along as a second output so their gradient can be
    gathered back to ``[S, K]`` -- ``combine`` does that when handed
    ``route_weights_nvs`` -- which is what keeps the router trainable.
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
        buffer, plan = ctx.buffer, ctx.plan
        grad_x_SH = None
        if grad_hidden_nvsh is not None:
            grad_x_SH, _, _ = buffer.combine(
                plan=plan, hidden_nvsh=grad_hidden_nvsh.to(torch.bfloat16)
            )
        grad_weights_SK = None
        if grad_weights_nvs is not None:
            # combine gathers per-token weights back to [S, K]; the hidden
            # operand is required by the signature, its result is discarded.
            _, grad_weights_SK, _ = buffer.combine(
                plan=plan,
                hidden_nvsh=grad_hidden_nvsh.new_zeros(ctx.shape_nvsh)
                if grad_hidden_nvsh is not None
                else torch.zeros(
                    ctx.shape_nvsh,
                    dtype=torch.bfloat16,
                    device=grad_weights_nvs.device,
                ),
                route_weights_nvs=grad_weights_nvs.to(torch.float32),
            )
        return None, None, grad_x_SH, grad_weights_SK, None, None


class _MoonEPCombine(torch.autograd.Function):
    """``buffer.combine`` with the README's backward: a re-dispatch on the plan."""

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
    """Balanced EP dispatch (report sec 5.2.1), through MoonEP's kernels.

    Slots into the same place as ``AllToAllTokenDispatcher``. EP=1 falls back
    to local dispatch exactly as the standard dispatcher does, so a flavor
    carrying this config still runs unsharded.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseEPTokenDispatcher.Config):
        hidden_dim: int | None = None
        """Feature width of the tokens entering dispatch (sizes the buffer)."""

        num_max_tokens_per_rank: int | None = None
        """MoonEP's ``S``: the exact per-rank token count of every dispatch,
        a static shape. Filled from the training config by the model's
        ``update_from_config``; never a guess."""

        num_prefetch_slots: int | None = None
        """MoonEP's ``B``; None is its default, ``E // num_ep_ranks``, which
        training requires."""

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

    def wire_meshes(self, *, ep_mesh) -> None:
        # Unlike MinimalAsyncEP this does not REQUIRE an EP mesh: with EP off
        # the local fallback below runs and MoonEP is never imported.
        super().wire_meshes(ep_mesh=ep_mesh)

    def init_buffer(self) -> None:
        """Allocate MoonEP's persistent buffer on the EP group.

        Collective: every rank has to reach it the same number of times in
        the same order, so once, from ``wire_meshes``, never per step. The
        expert side (``MoonEPGroupedExperts``) is attached by the MoE's
        parallelize and reads the plan back through ``current_plan``.
        """
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
        received tokens in VM-group order and the token count of each of the
        ``E + B`` rows (from ``cu_seqlens``), which is what a MoonEP-aware
        grouped GEMM consumes. It is NOT the standard dispatcher's per-local-
        expert count; ``init_buffer`` keeps ``RoutedExperts`` from reaching
        this path until an expert-side unit that reads it exists.
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
