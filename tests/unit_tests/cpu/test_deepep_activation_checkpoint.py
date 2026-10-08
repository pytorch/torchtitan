# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RegionAC must not replay a nondeterministic DeepEP dispatch.

The DeepEP autograd nodes keep the dispatch layout from the original forward, so
a replayed dispatch has to return rows in the same order. RegionAC replays
DeepEP's dispatch and combine unless ``ep_communication`` is saved, and must
reject that with ``deterministic=False``. FullAC always saves them, because the
ops are effectful.
"""

import unittest

import pytest
import torch

pytest.importorskip("deep_ep")

from torchtitan.distributed.activation_checkpoint import (  # noqa: E402
    ActivationCheckpointing,
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


def _replaying_policies() -> dict[str, ActivationCheckpointing]:
    return {
        "region_ac_saves_nothing": RegionAC.Config(save_regions=[]).build(),
        "region_ac_recomputes_ep_communication": RegionAC.Config(
            save_regions=["*"], recompute_regions=["*ep_communication"]
        ).build(),
    }


def _saving_policies() -> dict[str, ActivationCheckpointing]:
    return {
        "full_ac": FullAC.Config().build(),
        "selective_ac": SelectiveAC.Config().build(),
        "region_ac_saves_ep_communication": RegionAC.Config(
            save_regions=["token_dispatcher.ep_communication"]
        ).build(),
    }


class TestDeepEPReplayCheck(unittest.TestCase):
    def test_replay_with_nondeterministic_buffer_raises(self):
        for name, policy in _replaying_policies().items():
            with self.subTest(policy=name):
                with self.assertRaisesRegex(
                    ValueError,
                    r"layers\.0\.token_dispatcher, .*deterministic=True.*"
                    r"'token_dispatcher\.ep_communication'",
                ):
                    policy.apply(_Model(deterministic=False))

    def test_replay_with_deterministic_buffer_is_allowed(self):
        for name, policy in _replaying_policies().items():
            with self.subTest(policy=name):
                policy.apply(_Model(deterministic=True))

    def test_saving_deepep_communication_is_allowed(self):
        for name, policy in _saving_policies().items():
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

        policy = RegionAC.Config(save_regions=[]).build()
        policy.apply(deterministic_model)
        with self.assertRaises(ValueError):
            policy.apply(nondeterministic_model)


if __name__ == "__main__":
    unittest.main()
