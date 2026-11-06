# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP32 routing gates and released checkpoint buffers with the shared MoE stack.

T: tokens, E: experts, K: selected experts, R: routed tokens, F: expert features.
"""

from dataclasses import dataclass

import torch

from torchtitan.models.common.activation import LimitedSwiGLU
from torchtitan.models.common.moe import MoE, TokenChoiceTopKRouter


class Float32LimitedSwiGLU(LimitedSwiGLU):
    """Round once after the reference model's bounded FP32 activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(LimitedSwiGLU.Config):
        pass

    def _activation_fn(  # pyrefly: ignore [bad-param-name-override]
        self, gate_RF, up_RF
    ):
        return super()._activation_fn(gate_RF.float(), up_RF.float()).to(gate_RF.dtype)


class ModalityRouter(TokenChoiceTopKRouter):
    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.gate.float()


class ModalityMoE(MoE):
    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        router: ModalityRouter.Config  # pyrefly: ignore [bad-override]

        def __post_init__(self):
            super(ModalityMoE.Config, self).__post_init__()
            if self.load_balance_coeff is None:
                raise ValueError(
                    "V4.1 modality routing requires expert biases; use freeze_expert_bias to freeze updates."
                )

    def __init__(self, config: Config):
        super().__init__(config)
        self.router: ModalityRouter
        self.register_buffer(
            "expert_bias_vl_E",
            torch.zeros(config.num_experts, dtype=torch.float32),
            persistent=True,
        )

    def _init_self_buffers(self, *, buffer_device=None):
        super()._init_self_buffers(buffer_device=buffer_device)
        assert self.expert_bias_E is not None
        self.expert_bias_vl_E = torch.zeros_like(self.expert_bias_E)
