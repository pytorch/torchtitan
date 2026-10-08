# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ungated (non-gated) MoE experts for Nemotron-H.

Nemotron-H's experts are *not* SwiGLU. Where a Llama/Mixtral expert computes
``w2(silu(w1(x)) * w3(x))`` -- three matrices, with ``w1`` gating ``w3`` -- a
Nemotron-H expert computes ``down(relu(up(x))**2)``: two matrices, no gate
branch. This is what ``has_gate: false`` means in NVIDIA's config, and it is
why a SwiGLU port of the 30B comes out ~15B parameters too large.

Only ``w1_EFD`` (the up projection) and ``w2_EDF`` (the down projection) exist
here; ``w3_EFD`` from the base class is removed. See ``kimi_k3/moe.py`` for the
same subclass-the-grouped-experts pattern.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.models.common.moe import GroupedExperts

# Shape suffixes:
# D = model dimension, E = experts, F = expert hidden dimension, R = routed tokens.


class NemotronGroupedExperts(GroupedExperts):
    """``common/moe.py::GroupedExperts`` without the SwiGLU gate branch.

    Computes ``down(relu(up(x))**2)`` per expert.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        # Deliberately no build() override. Module.Config.build is what attaches
        # _param_init and _sharding_config to the instance, so overriding it
        # here silently left the expert weights undistributed -- FSDP then hit
        # "Got plain tensor for parameter 'w1_EFD'". The base machinery already
        # resolves this Config to its enclosing class, exactly as kimi_k3 does.
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        # The base class allocates a gate projection; Nemotron-H has none.
        # Deleting it keeps the parameter count and the state dict honest
        # rather than carrying a tensor that never participates in forward.
        del self.w3_EFD

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w1_EFD, DTensor):
            w1_EFD = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)

        h_RF = self._grouped_mm(
            A=x_RD.bfloat16(), weight_EOI=w1_EFD, offs=offsets_E
        )
        # relu^2, matching NemotronHExperts' `relu2` activation.
        h_RF = torch.square(F.relu(h_RF))

        return self._grouped_mm(
            A=h_RF, weight_EOI=w2_EDF, offs=offsets_E
        ).type_as(x_RD)
