# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""K3 routed experts: SiTU-GLU instead of SwiGLU.

The released config sets ``hidden_act: "situ"`` globally, so the routed
experts -- which are the overwhelming majority of the model's FLOPs and
parameters -- use tech report Eq. 12, not SiLU. ``GroupedExperts`` in
``models/common`` hardcodes ``F.silu``, so this subclasses it the way
``GptOssGroupedExperts`` does for its clamped SwiGLU: same ``w1_EFD`` /
``w2_EDF`` / ``w3_EFD`` parameters (so the state-dict adapter, the expert
TP/EP layout, and the torchao MX/Float8 expert converters all keep working
unchanged), only the activation differs.

Shape suffixes follow ``models/common/moe.py``: R routed tokens on this
rank, D model dim, F expert hidden dim, E experts.
"""

from dataclasses import dataclass

import torch

from torchtitan.models.common.moe import GroupedExperts

from .model import situ_and_mul


class KimiSiTUGroupedExperts(GroupedExperts):
    """Grouped routed experts with K3's SiTU-GLU activation (Eq. 12).

    ``situ_linear_beta=None`` leaves the linear branch unclipped; K3 ships
    ``beta1=4`` on the gate branch and ``beta2=25`` on the linear branch,
    bounding the product at 100.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        situ_beta: float = 4.0
        situ_linear_beta: float | None = 25.0

    def __init__(self, config: Config):
        super().__init__(config)
        self.situ_beta = config.situ_beta
        self.situ_linear_beta = config.situ_linear_beta

    def gate_up_combine(
        self, gate_RF: torch.Tensor, up_RF: torch.Tensor
    ) -> torch.Tensor:
        """SiTU-GLU instead of the base class's SwiGLU (report Eq. 12)."""
        return situ_and_mul(gate_RF, up_RF, self.situ_beta, self.situ_linear_beta)
