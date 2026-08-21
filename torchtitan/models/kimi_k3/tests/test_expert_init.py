# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""init_weights must reach the routed experts.

torchtitan's flow is meta-build -> FSDP wrap -> to_empty -> init_weights, so
init_weights is the ONLY thing that gives the routed experts real values.
It used to dispatch on ``type(m).__name__ == "GroupedExperts"`` and init
parameters named ``("w1", "w2", "w3")``. Both went stale -- upstream renamed
the parameters to w1_EFD / w2_EDF / w3_EFD, and K3's experts are a
KimiSiTUGroupedExperts subclass -- so the routed experts stayed at to_empty
garbage. The model still trained to a plausible loss on the dense, shared and
latent paths, and the routed contribution was measurably nil: loss with real
experts and loss with every expert weight zeroed were bit-identical.

That is the failure mode these tests exist to make loud. A sentinel fill is
used instead of checking "not all zeros" because to_empty garbage is sometimes
nonzero, which would let the bug pass intermittently.
"""

from __future__ import annotations

import unittest

import torch

from torchtitan.models.common.moe import GroupedExperts

from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.models.kimi_k3.model_configs import build_kimi_linear_config

_SENTINEL = 1234.5


def _model() -> KimiK3Model:
    cfg = build_kimi_linear_config("k3mini", vocab_size=256)
    return KimiK3Model.make_config(cfg).build()


def _expert_params(model):
    for fqn, m in model.named_modules():
        if isinstance(m, GroupedExperts):
            for name, p in m._parameters.items():
                if p is not None:
                    yield f"{fqn}.{name}", p


class TestExpertInit(unittest.TestCase):
    def test_expert_init_is_not_silently_skipped(self):
        model = _model()
        params = dict(_expert_params(model))
        self.assertTrue(params, "k3mini must have routed expert params")
        with torch.no_grad():
            for p in params.values():
                p.fill_(_SENTINEL)

        model.init_weights()

        for name, p in params.items():
            self.assertFalse(
                torch.allclose(p, torch.full_like(p, _SENTINEL)),
                f"init_weights never touched {name}",
            )
            self.assertTrue(torch.isfinite(p).all(), name)
            self.assertGreater(p.abs().sum().item(), 0.0, name)

    def test_init_covers_the_shape_suffixed_names(self):
        # the specific stale-name trap: hardcoding ("w1","w2","w3") matches
        # nothing, and getattr returns None silently.
        model = _model()
        names = {n.rsplit(".", 1)[1] for n, _ in _expert_params(model)}
        self.assertEqual(names, {"w1_EFD", "w2_EDF", "w3_EFD"})

    def test_init_dispatches_on_type_not_class_name(self):
        # K3's experts are a subclass, so a class-name equality check misses
        # them entirely.
        model = _model()
        classes = {
            type(m).__name__
            for _, m in model.named_modules()
            if isinstance(m, GroupedExperts)
        }
        self.assertTrue(classes)
        self.assertNotIn(
            "GroupedExperts",
            classes,
            "k3mini experts must be a subclass -- otherwise this test cannot "
            "distinguish isinstance dispatch from a class-name check",
        )

    def test_packed_uint8_expert_bytes_are_left_for_the_checkpoint(self):
        from torchtitan.models.kimi_k3.lora import quantize_grouped_experts_mxfp4

        model = _model()
        model.init_weights()
        self.assertEqual(quantize_grouped_experts_mxfp4(model), 20)
        packed = {
            name: p.clone()
            for name, p in _expert_params(model)
            if p.dtype == torch.uint8
        }
        self.assertTrue(packed)
        # re-initializing must not scribble normal noise over packed bytes
        model.init_weights()
        for name, before in packed.items():
            after = dict(_expert_params(model))[name]
            self.assertTrue(torch.equal(before, after), f"{name} was re-inited")

    @unittest.skipUnless(torch.cuda.is_available(), "grouped_mm needs CUDA")
    def test_routed_experts_affect_the_output(self):
        """The end-to-end property the stale init silently violated."""
        torch.manual_seed(0)
        model = _model().cuda().bfloat16()
        model.init_weights(buffer_device="cuda")
        tokens = torch.randint(0, 256, (1, 128), device="cuda")
        with torch.no_grad():
            ref = model(tokens).float()
            for _, m in model.named_modules():
                if isinstance(m, GroupedExperts):
                    for p in m.parameters():
                        p.zero_()
            zeroed = model(tokens).float()
        rel = ((zeroed - ref).norm() / ref.norm()).item()
        self.assertGreater(
            rel, 1e-3, "zeroing every routed expert did not change the output"
        )


if __name__ == "__main__":
    unittest.main()
