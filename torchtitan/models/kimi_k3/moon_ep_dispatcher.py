# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonEP token dispatch for the K3 MoE (report sec 5.2.1). DRAFT -- untested.

    Subclasses ``BaseEPTokenDispatcher``; both abstract methods raise. MoonEP needs
    8xNVLink to validate, which this box does not have.

    See ``phase13_k3like_48b_posttrain/MOONEP_DRAFT.md``.
    """

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.models.common.token_dispatcher import BaseEPTokenDispatcher
from torchtitan.tools.logging import logger


def _import_moon_ep():
    """Import MoonEP, or explain what is missing.

    Optional in the same sense as fla and DeepEP: absent on a machine that cannot run it,
    and the error names the package rather than surfacing as an AttributeError deep in
    dispatch.
    """
    try:
        import moon_ep  # type: ignore[import-not-found]
    except ImportError as err:
        raise ImportError(
            "MoonEP is not installed. It is an optional dependency, like DeepEP: "
            "pip install from https://github.com/MoonshotAI/MoonEP, and note that it "
            "requires NVLink-connected GPUs. Use the default AllToAllTokenDispatcher on "
            "hardware without that topology."
        ) from err
    return moon_ep


class MoonEPTokenDispatcher(BaseEPTokenDispatcher):
    """Balanced EP dispatch (report sec 5.2.1), through MoonEP's kernels. DRAFT.

    Slots into the same place as ``AllToAllTokenDispatcher``; the MoE module is unchanged.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseEPTokenDispatcher.Config):
        # Lifetime upper bound on tokens one rank can hold, for preallocating MoonEP's
        # buffers. The base class's docstring is explicit that this is a storage bound and
        # NOT the per-call token count -- MoE pads the sequence before routing, so the
        # per-call count is x_TD.shape[0] and is the same on every rank.
        num_max_tokens_per_rank: int = 8192
        # MoonEP overlaps dispatch with expert compute when asked. Off by default: it
        # changes when gradients become available, and the backward has not been verified
        # here at all.
        overlap_dispatch: bool = False

    def __init__(self, config: "MoonEPTokenDispatcher.Config") -> None:
        super().__init__(config)
        self._num_max_tokens_per_rank = config.num_max_tokens_per_rank
        self._overlap_dispatch = config.overlap_dispatch
        self._buffer = None

    def init_buffer(self) -> None:
        """Allocate MoonEP's persistent buffer once the EP mesh is known.

        Called from ``wire_meshes``, which is the only point where the mesh exists and
        before any dispatch. A per-step allocation would be wrong for the same reason the
        vision sub-CP groups are built up front: the buffer's creation is collective, so
        every rank has to reach it the same number of times in the same order.
        """
        if self.ep_mesh is None:
            return
        moon_ep = _import_moon_ep()
        group = self.ep_mesh.get_group()
        # DRAFT: buffer construction is the part most likely to differ from MoonEP's actual
        # API. Kept in one place so correcting it does not touch dispatch or combine.
        self._buffer = moon_ep.Buffer(
            group=group,
            num_max_tokens_per_rank=self._num_max_tokens_per_rank,
        )
        logger.info(
            "MoonEP dispatcher: buffer for %d tokens/rank on an ep mesh of %d",
            self._num_max_tokens_per_rank,
            self.ep_mesh.size(),
        )

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        """Route padded local tokens to their experts' ranks.

        Returns ``(routed_input_RD, routed_scores_R, metadata)`` to match the base class,
        where R is this rank's received token count and ``metadata`` is whatever ``combine``
        needs to invert the routing. Keeping the handle opaque is what lets a backend
        carry its own bookkeeping without the MoE module knowing.
        """
        if self._buffer is None:
            raise RuntimeError(
                "MoonEP dispatcher used before wire_meshes(); the EP mesh has to be "
                "installed first, and init_buffer allocates on that mesh collectively."
            )
        raise NotImplementedError(
            "DRAFT: MoonEP's dispatch call is not written. It needs the released API to "
            "map onto (routed_input, routed_scores, metadata), and the mapping is worth "
            "writing against the real signatures rather than guessed ones. The two "
            "properties to establish first are in this module's docstring."
        )

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: object,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Invert ``dispatch``: one row per original token, in original order.

        ``x_TD`` is passed so the output shape and dtype come from the input rather than
        from a recomputation -- the base class's contract, and what makes a zero-token
        expert a non-special case.
        """
        if self._buffer is None:
            raise RuntimeError("MoonEP dispatcher used before wire_meshes().")
        raise NotImplementedError(
            "DRAFT: see dispatch. combine must also be differentiable back into "
            "dispatch's input; if MoonEP's kernels do not carry that, an "
            "autograd.Function is required and a forward-only wrapper will look correct "
            "while silently dropping expert gradients."
        )
