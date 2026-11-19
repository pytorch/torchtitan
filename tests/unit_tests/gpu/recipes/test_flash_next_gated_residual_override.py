# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.config import derive
from torchtitan.models.qwen3_8_flash_next.flavors import _gated_residual
from torchtitan.models.qwen3_8_flash_next.norm import GroupedOffsetRMSNorm
from torchtitan_recipes.overrides.flash_next_gated_residual import FusedGatedResidual
from torchtitan_recipes.overrides.flash_next_grouped_rms_norm import (
    FusedGroupedOffsetRMSNorm,
)

# The override may round differently from eager, but its error against an
# fp64 golden must stay within this factor of eager's error (plus the rounding
# floor of the output dtype).
_FUDGE_FACTOR = 2.0
_NUM_STREAMS, _DIM, _RANK = 4, 256, 32


def _relative_error(value: torch.Tensor, golden: torch.Tensor) -> float:
    return ((value.double() - golden).norm() / golden.norm()).item()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFlashNextGatedResidualOverride(unittest.TestCase):
    def _assert_close_to_golden(self, name, reference, target, golden):
        floor = _relative_error(golden.to(target.dtype), golden)
        reference_error = _relative_error(reference, golden)
        target_error = _relative_error(target, golden)
        self.assertEqual(target.dtype, reference.dtype, name)
        self.assertLessEqual(
            target_error, _FUDGE_FACTOR * reference_error + floor, name
        )

    def _run(self, module, x, grad_outputs):
        x = x.detach().clone().requires_grad_()
        module.zero_grad(set_to_none=True)
        out = module(x)
        outputs = (out,) if isinstance(out, torch.Tensor) else (out[0], out[2])
        torch.autograd.backward(outputs, grad_outputs[: len(outputs)])
        params = [p.grad for p in module.parameters()]
        return [o.detach() for o in outputs], [x.grad, *params]

    def _check_gated_residual(self, num_tokens: int, *, use_combine: bool) -> None:
        torch.manual_seed(num_tokens)
        config = _gated_residual(
            dim=_DIM,
            num_streams=_NUM_STREAMS,
            rank=_RANK,
            enable_sp=False,
            use_combine=use_combine,
        )
        eager = config.build().cuda()
        with torch.no_grad():
            for param in eager.parameters():
                param.normal_(0, 0.05)
        golden = config.build().cuda().double()
        golden.load_state_dict(eager.state_dict())
        eager.bfloat16()
        fused = derive(config, FusedGatedResidual.Config).build().cuda().bfloat16()
        fused.load_state_dict(eager.state_dict())

        x = torch.randn(num_tokens, _NUM_STREAMS * _DIM, device="cuda") * 2
        grad_outputs = (
            torch.randn(num_tokens, _DIM, device="cuda"),
            torch.randn(num_tokens, _NUM_STREAMS, device="cuda"),
        )
        golden_outs, golden_grads = self._run(
            golden, x.double(), [g.double() for g in grad_outputs]
        )
        bf16_grads = [g.bfloat16() for g in grad_outputs]
        eager_outs, eager_grads = self._run(eager, x.bfloat16(), bf16_grads)
        fused_outs, fused_grads = self._run(fused, x.bfloat16(), bf16_grads)
        names = ["x"] + [name for name, _ in eager.named_parameters()]
        for i, (ref, tgt, gold) in enumerate(zip(eager_outs, fused_outs, golden_outs)):
            self._assert_close_to_golden(f"output {i}", ref, tgt, gold)
        for name, ref, tgt, gold in zip(names, eager_grads, fused_grads, golden_grads):
            self._assert_close_to_golden(f"grad {name}", ref, tgt, gold)

    def test_gated_residual_matches_eager(self) -> None:
        for num_tokens in (256, 257):
            for use_combine in (True, False):
                with self.subTest(num_tokens=num_tokens, use_combine=use_combine):
                    self._check_gated_residual(num_tokens, use_combine=use_combine)

    def test_grouped_offset_rms_norm_matches_eager(self) -> None:
        torch.manual_seed(0)
        config = GroupedOffsetRMSNorm.Config(
            dim=_NUM_STREAMS * _DIM, group_size=_DIM, eps=1e-6
        )
        weight = torch.randn(_NUM_STREAMS * _DIM, device="cuda") * 0.1
        modules = {
            "golden": config.build().cuda().double(),
            "eager": config.build().cuda().bfloat16(),
            "fused": derive(config, FusedGroupedOffsetRMSNorm.Config)
            .build()
            .cuda()
            .bfloat16(),
        }
        for module in modules.values():
            with torch.no_grad():
                module.weight.copy_(weight)
        # Leading dims and a token count that does not fill the last program.
        x = torch.randn(3, 101, _NUM_STREAMS * _DIM, device="cuda") * 3
        grad_y = torch.randn_like(x)
        results = {}
        for name, module in modules.items():
            dtype = torch.float64 if name == "golden" else torch.bfloat16
            inp = x.to(dtype).requires_grad_()
            y = module(inp)
            y.backward(grad_y.to(dtype))
            results[name] = (y.detach(), inp.grad, module.weight.grad)
        for label, ref, tgt, gold in zip(
            ("y", "dx", "dweight"),
            results["eager"],
            results["fused"],
            results["golden"],
        ):
            self._assert_close_to_golden(label, ref, tgt, gold)


if __name__ == "__main__":
    unittest.main()
