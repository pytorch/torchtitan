# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.config import derive
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.attention import QSAIndexerLoss
from torchtitan.models.qwen3_8_flash_next.norm import GroupedOffsetRMSNorm
from torchtitan_recipes.overrides.flash_next_gated_residual import FusedGatedResidual
from torchtitan_recipes.overrides.flash_next_grouped_rms_norm import (
    grouped_offset_rms_norm,
)

# The override may round differently from eager, but its error against an
# fp64 golden must stay within this factor of eager's error (plus the rounding
# floor of the output dtype).
_FUDGE_FACTOR = 2.0
# The debug model's gated residuals: 4 streams of width 256.
_NUM_STREAMS, _DIM = 4, 256


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
        model_config = build_model_config(
            "debugmodel",
            seq_len=64,
            parallelism=ParallelismConfig(),
            indexer_loss=QSAIndexerLoss.Config(coeff=1.0),
        )
        # Block read/write residuals have an inject projection; the final
        # mixer only reads.
        config = (
            model_config.layers[0].attn_hyper_connection
            if use_combine
            else model_config.hyper_connection_mixer
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
        # A group size that is not a power of two exercises the masked lanes
        # (the released model uses 2560 in a 4096-wide block).
        group_size = 320
        torch.manual_seed(0)
        config = GroupedOffsetRMSNorm.Config(
            dim=_NUM_STREAMS * group_size, group_size=group_size, eps=1e-6
        )
        weight = torch.randn(_NUM_STREAMS * group_size, device="cuda") * 0.1
        golden_norm = config.build().cuda().double()
        eager_norm = config.build().cuda().bfloat16()
        for module in (golden_norm, eager_norm):
            with torch.no_grad():
                module.weight.copy_(weight)
        # Leading dims and a token count that does not fill the last program.
        x = torch.randn(3, 101, _NUM_STREAMS * group_size, device="cuda") * 3
        grad_y = torch.randn_like(x)
        results = {}
        for name, dtype in (("golden", torch.float64), ("eager", torch.bfloat16)):
            module = golden_norm if name == "golden" else eager_norm
            inp = x.to(dtype).requires_grad_()
            y = module(inp)
            y.backward(grad_y.to(dtype))
            results[name] = (y.detach(), inp.grad, module.weight.grad)
        inp = x.bfloat16().requires_grad_()
        fused_weight = eager_norm.weight.detach().clone().requires_grad_()
        y = grouped_offset_rms_norm(
            inp, fused_weight, eps=config.eps, group_size=group_size
        )
        y.backward(grad_y.bfloat16())
        results["fused"] = (y.detach(), inp.grad, fused_weight.grad)
        for label, ref, tgt, gold in zip(
            ("y", "dx", "dweight"),
            results["eager"],
            results["fused"],
            results["golden"],
        ):
            self._assert_close_to_golden(label, ref, tgt, gold)


if __name__ == "__main__":
    unittest.main()
