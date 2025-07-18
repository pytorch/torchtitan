# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from torchtitan.config_manager import ActivationCheckpoint as ACConfig
from torchtitan.models.llama3.infra.parallelize import apply_ac


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({"0": TransformerBlock()})

    def forward(self, x):
        return self.layers["0"](x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = nn.Module()
        self.moe.router = nn.Module()
        self.moe.router.gate = nn.Linear(512, 512, bias=False)
        self.attention = nn.Module()
        self.attention.wq = nn.Linear(512, 512, bias=False)
        self.output = nn.Linear(512, 1024, bias=False)

    def forward(self, x):
        gate_out = self.moe.router.gate(x)
        wq_out = self.attention.wq(gate_out)
        final_out = self.output(wq_out)
        return final_out.sum()


class TestApplyAC(unittest.TestCase):
    def test_flops(self):
        def get_bw_flops(model_fn):
            x = torch.randn(512, 512, requires_grad=True)
            with torch.utils.checkpoint.set_checkpoint_early_stop(False):
                out = model_fn(x)
            out.backward()

            x = torch.randn(512, 512, requires_grad=True)
            with torch.utils.checkpoint.set_checkpoint_early_stop(False):
                out = model_fn(x)
            with FlopCounterMode(display=False) as mode:
                out.backward()
            return mode.get_total_flops() / (512**3 * 2)

        # 1. No AC
        model_no_ac = TestModule()
        flops_no_ac = get_bw_flops(model_no_ac)

        # 2. SAC
        # Per-op SAC's policy is to save every other mm
        model_selective_ac = TestModule()
        ac_config_no_force = ACConfig(
            mode="selective",
            selective_ac_option="op",
            per_op_sac_force_recompute_mm_shapes_by_fqns=[],  # Empty list
        )
        apply_ac(model_selective_ac, ac_config_no_force)
        flops_selective_ac = get_bw_flops(model_selective_ac)

        # 3. Per-op SAC with force recompute "moe.router.gate"
        # This leads to two mms being recomputed since they share the same shape!
        model_with_force_first = TestModule()
        ac_config_with_force_first = ACConfig(
            mode="selective",
            selective_ac_option="op",
            per_op_sac_force_recompute_mm_shapes_by_fqns=["moe.router.gate"],
        )
        apply_ac(model_with_force_first, ac_config_with_force_first)
        flops_with_force_first = get_bw_flops(model_with_force_first)

        # 4. Per-op SAC with force recompute "output"
        model_with_force_last = TestModule()
        ac_config_with_force_last = ACConfig(
            mode="selective",
            selective_ac_option="op",
            per_op_sac_force_recompute_mm_shapes_by_fqns=["output"],
        )
        apply_ac(model_with_force_last, ac_config_with_force_last)
        flops_with_force_last = get_bw_flops(model_with_force_last)

        # 5. Full AC
        model_with_full_ac = TestModule()
        ac_config_full_ac = ACConfig(
            mode="full",
        )
        apply_ac(model_with_full_ac, ac_config_full_ac)
        flops_full_ac = get_bw_flops(model_with_full_ac)

        self.assertEqual(flops_no_ac, 8.0)
        self.assertEqual(flops_selective_ac, 9.0)
        self.assertEqual(flops_with_force_first, 10.0)
        self.assertEqual(flops_with_force_last, 11.0)
        self.assertEqual(flops_full_ac, 12.0)

    def test_mem(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")

        def get_act_mem(model_fn):
            x = torch.randn(512, 512, requires_grad=True, device="cuda")
            out = model_fn(x)
            out.backward()
            start_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]

            out = model_fn(x)
            cur_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
            act_mem = (cur_mem - start_mem) / (1024 * 1024)  # → MB
            out.backward()
            return act_mem

        # 1. No AC
        model_no_ac = TestModule().cuda()
        mem_no_ac = get_act_mem(model_no_ac)

        # 2. SAC
        # Per-op SAC's policy is to save every other mm
        model_selective_ac = TestModule().cuda()
        ac_config_no_force = ACConfig(
            mode="selective",
            selective_ac_option="op",
            per_op_sac_force_recompute_mm_shapes_by_fqns=[],  # Empty list
        )
        apply_ac(model_selective_ac, ac_config_no_force)
        mem_selective_ac = get_act_mem(model_selective_ac)

        # 3. Per-op SAC with force recompute "moe.router.gate"
        # This leads to two mms being recomputed since they share the same shape!
        model_with_force_first = TestModule().cuda()
        ac_config_with_force_first = ACConfig(
            mode="selective",
            selective_ac_option="op",
            per_op_sac_force_recompute_mm_shapes_by_fqns=["moe.router.gate"],
        )
        apply_ac(model_with_force_first, ac_config_with_force_first)
        mem_with_force_first = get_act_mem(model_with_force_first)

        # 4. Per-op SAC with force recompute "output"
        model_with_force_last = TestModule().cuda()
        ac_config_with_force_last = ACConfig(
            mode="selective",
            selective_ac_option="op",
            per_op_sac_force_recompute_mm_shapes_by_fqns=["output"],
        )
        apply_ac(model_with_force_last, ac_config_with_force_last)
        mem_with_force_last = get_act_mem(model_with_force_last)

        # 5. Full AC
        model_with_full_ac = TestModule().cuda()
        ac_config_full_ac = ACConfig(
            mode="full",
        )
        apply_ac(model_with_full_ac, ac_config_full_ac)
        mem_full_ac = get_act_mem(model_with_full_ac)

        self.assertEqual(mem_no_ac, 2.0)
        self.assertEqual(mem_selective_ac, 3.0)
        self.assertEqual(mem_with_force_first, 2.0)
        self.assertEqual(mem_with_force_last, 1.0)
        self.assertEqual(mem_full_ac, 0.0)
        # Note: SAC > no-AC here because it unnecessarily saves "output"
        # even that is not needed for recomputaion and output is double
        # the size of the other two mms.

    def test_correctness(self):
        model_no_ac = TestModule()

        model_selective_ac = TestModule()
        model_selective_ac.load_state_dict(model_no_ac.state_dict())
        apply_ac(
            model_selective_ac,
            ACConfig(
                mode="selective",
                selective_ac_option="op",
                per_op_sac_force_recompute_mm_shapes_by_fqns=[],
            ),
        )
        model_force_first = TestModule()
        model_force_first.load_state_dict(model_no_ac.state_dict())
        apply_ac(
            model_force_first,
            ACConfig(
                mode="selective",
                selective_ac_option="op",
                per_op_sac_force_recompute_mm_shapes_by_fqns=["moe.router.gate"],
            ),
        )

        model_force_last = TestModule()
        model_force_last.load_state_dict(model_no_ac.state_dict())
        apply_ac(
            model_force_last,
            ACConfig(
                mode="selective",
                selective_ac_option="op",
                per_op_sac_force_recompute_mm_shapes_by_fqns=["output"],
            ),
        )

        def run_fwd_bwd(model, batch):
            model.zero_grad(set_to_none=True)
            xin = batch.clone().detach().requires_grad_(True)
            out = model(xin)  # scalar
            out.backward()

            grad_in = xin.grad.detach().clone()
            grad_params = [
                p.grad.detach().clone() if isinstance(p.grad, torch.Tensor) else None
                for p in model.parameters()
            ]
            return out.detach(), grad_in, grad_params

        batch = torch.randn(64, 512)

        out_ref, gin_ref, gparams_ref = run_fwd_bwd(model_no_ac, batch)
        out_sel, gin_sel, gparams_sel = run_fwd_bwd(model_selective_ac, batch)
        out_f1, gin_f1, gparams_f1 = run_fwd_bwd(model_force_first, batch)
        out_fl, gin_fl, gparams_fl = run_fwd_bwd(model_force_last, batch)

        for other_out in (out_sel, out_f1, out_fl):
            torch.testing.assert_close(out_ref, other_out)

        for other_gin in (gin_sel, gin_f1, gin_fl):
            torch.testing.assert_close(gin_ref, other_gin)

        for g_ref, g_sel, g_f1, g_fl in zip(
            gparams_ref, gparams_sel, gparams_f1, gparams_fl
        ):
            # Skip wrapper / missing grads
            if not (
                torch.is_tensor(g_ref)
                and torch.is_tensor(g_sel)
                and torch.is_tensor(g_f1)
                and torch.is_tensor(g_fl)
            ):
                continue

            torch.testing.assert_close(g_ref, g_sel)
            torch.testing.assert_close(g_ref, g_f1)
            torch.testing.assert_close(g_ref, g_fl)


if __name__ == "__main__":
    unittest.main()
