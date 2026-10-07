# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Activation checkpointing must not replay a nondeterministic DeepEP dispatch.

The DeepEP autograd nodes keep the dispatch layout from the original forward, so
a replayed dispatch has to return rows in the same order. Each policy below is
applied to a block holding a DeepEPTokenDispatcher; the guard must reject every
configuration that replays dispatch or combine with deterministic=False.
"""

import unittest

import pytest
import torch

pytest.importorskip("deep_ep")

# Registers torch.ops.deepep.{dispatch,combine}.
import torchtitan.distributed.deepep.deepep  # noqa: E402, F401
from torchtitan.distributed.activation_checkpoint import (  # noqa: E402
    FullAC,
    RegionAC,
    SelectiveAC,
)
from torchtitan.models.common.token_dispatcher import (  # noqa: E402
    DeepEPTokenDispatcher,
)
from torchtitan.protocols.module import Module, ModuleDict  # noqa: E402


class _Block(Module):
    def __init__(self, *, deterministic: bool | None):
        super().__init__()
        self.token_dispatcher = DeepEPTokenDispatcher.Config(
            num_experts=8,
            top_k=2,
            num_max_tokens_per_rank=16,
            hidden_dim=16,
            deterministic=deterministic,
        ).build()

    def forward(self, x):
        return x


class _Model(Module):
    def __init__(self, *, deterministic: bool | None):
        super().__init__()
        self.layers = ModuleDict({"0": _Block(deterministic=deterministic)})


def _selective_ac_saving(*ops: torch._ops.OpOverload) -> SelectiveAC:
    class _SelectiveAC(SelectiveAC):
        def get_save_ops(self) -> set:
            return set(ops)

    return _SelectiveAC(SelectiveAC.Config())


class TestDeepEPReplayGuard(unittest.TestCase):
    def setUp(self):
        self.dispatch = torch.ops.deepep.dispatch.default
        self.combine = torch.ops.deepep.combine.default

    def _replaying_policies(self):
        return {
            "full_ac": FullAC.Config().build(),
            "sac_saves_topk_only": _selective_ac_saving(torch.ops.aten.topk.default),
            "sac_saves_dispatch_only": _selective_ac_saving(self.dispatch),
            "sac_saves_combine_only": _selective_ac_saving(self.combine),
            "region_ac_saves_nothing": RegionAC.Config(save_regions=[]).build(),
        }

    def _saving_policies(self):
        return {
            "sac_default": SelectiveAC.Config().build(),
            "sac_saves_both": _selective_ac_saving(self.dispatch, self.combine),
            "region_ac_saves_ep_communication": RegionAC.Config(
                save_regions=["token_dispatcher.ep_communication"]
            ).build(),
        }

    def test_replay_with_nondeterministic_buffer_raises(self):
        for name, policy in self._replaying_policies().items():
            with self.subTest(policy=name):
                model = _Model(deterministic=False)
                with self.assertRaisesRegex(
                    ValueError, r"layers\.0\.token_dispatcher .*deterministic=True"
                ):
                    policy.apply(model)

    def test_replay_with_deterministic_buffer_is_allowed(self):
        for name, policy in self._replaying_policies().items():
            with self.subTest(policy=name):
                policy.apply(_Model(deterministic=True))

    def test_saving_deepep_ops_is_allowed(self):
        for name, policy in self._saving_policies().items():
            with self.subTest(policy=name):
                policy.apply(_Model(deterministic=False))

    def test_none_follows_torch_deterministic_algorithms(self):
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            deterministic_model = _Model(deterministic=None)
            torch.use_deterministic_algorithms(False)
            nondeterministic_model = _Model(deterministic=None)
        finally:
            torch.use_deterministic_algorithms(previous)

        FullAC.Config().build().apply(deterministic_model)
        with self.assertRaises(ValueError):
            FullAC.Config().build().apply(nondeterministic_model)


if __name__ == "__main__":
    unittest.main()
