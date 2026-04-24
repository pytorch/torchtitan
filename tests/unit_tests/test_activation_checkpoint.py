# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from copy import deepcopy

import torch
from torch.utils.flop_counter import FlopCounterMode
from torchtitan.config import ActivationCheckpointConfig as ACConfig
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module, ModuleDict


class ToyModule(Module):
    def __init__(self):
        super().__init__()
        self.layers = ModuleDict({"0": TransformerBlock()})

    def forward(self, x):
        return self.layers["0"](x)


class TransformerBlock(Module):
    def __init__(self):
        super().__init__()
        linear_config = Linear.Config(in_features=512, out_features=512, bias=False)
        self.moe = Module()
        self.moe.router = Module()
        self.moe.router.gate = linear_config.build()
        self.attention = Module()
        self.attention.wq = linear_config.build()
        output_config = deepcopy(linear_config)
        output_config.out_features = 1024
        self.output = output_config.build()

    def forward(self, x):
        gate_out = self.moe.router.gate(x)
        wq_out = self.attention.wq(gate_out)
        final_out = self.output(wq_out)
        return final_out.sum()


class TestApplyAC(unittest.TestCase):
    def test_flops(self):
        def get_bw_flops(model_fn):
            x = torch.randn(512, 512, requires_grad=True)
            out = model_fn(x)
            out.backward()

            x = torch.randn(512, 512, requires_grad=True)
            out = model_fn(x)
            with FlopCounterMode(display=False) as mode:
                out.backward()
            return mode.get_total_flops() / (512**3 * 2)

        # 1. No AC
        model_no_ac = ToyModule()
        flops_no_ac = get_bw_flops(model_no_ac)

        # 2. SAC
        # Per-op SAC's policy is to save every other mm
        model_selective_ac = ToyModule()
        ac_config_no_force = ACConfig(
            mode="selective",
            per_op_sac_force_recompute_mm_shapes_by_fqns=[],  # Empty list
            early_stop=False,
        )
        apply_ac(
            model_selective_ac,
            ac_config_no_force,
            model_compile_enabled=False,
        )
        flops_selective_ac = get_bw_flops(model_selective_ac)

        # 3. Per-op SAC with force recompute "moe.router.gate"
        # This leads to two mms being recomputed since they share the same shape!
        model_with_force_first = ToyModule()
        ac_config_with_force_first = ACConfig(
            mode="selective",
            per_op_sac_force_recompute_mm_shapes_by_fqns=["moe.router.gate"],
            early_stop=False,
        )
        apply_ac(
            model_with_force_first,
            ac_config_with_force_first,
            model_compile_enabled=False,
        )
        flops_with_force_first = get_bw_flops(model_with_force_first)

        # 4. Per-op SAC with force recompute "output"
        model_with_force_last = ToyModule()
        ac_config_with_force_last = ACConfig(
            mode="selective",
            per_op_sac_force_recompute_mm_shapes_by_fqns=["output"],
            early_stop=False,
        )
        apply_ac(
            model_with_force_last,
            ac_config_with_force_last,
            model_compile_enabled=False,
        )
        flops_with_force_last = get_bw_flops(model_with_force_last)

        # 5. Full AC
        model_with_full_ac = ToyModule()
        ac_config_full_ac = ACConfig(
            mode="full",
            early_stop=False,
        )
        apply_ac(
            model_with_full_ac,
            ac_config_full_ac,
            model_compile_enabled=False,
        )
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
        model_no_ac = ToyModule().cuda()
        mem_no_ac = get_act_mem(model_no_ac)

        # 2. SAC
        # Per-op SAC's policy is to save every other mm
        model_selective_ac = ToyModule().cuda()
        ac_config_no_force = ACConfig(
            mode="selective",
            per_op_sac_force_recompute_mm_shapes_by_fqns=[],  # Empty list
        )
        apply_ac(
            model_selective_ac,
            ac_config_no_force,
            model_compile_enabled=False,
        )
        mem_selective_ac = get_act_mem(model_selective_ac)

        # 3. Per-op SAC with force recompute "moe.router.gate"
        # This leads to two mms being recomputed since they share the same shape!
        model_with_force_first = ToyModule().cuda()
        ac_config_with_force_first = ACConfig(
            mode="selective",
            per_op_sac_force_recompute_mm_shapes_by_fqns=["moe.router.gate"],
        )
        apply_ac(
            model_with_force_first,
            ac_config_with_force_first,
            model_compile_enabled=False,
        )
        mem_with_force_first = get_act_mem(model_with_force_first)

        # 4. Per-op SAC with force recompute "output"
        model_with_force_last = ToyModule().cuda()
        ac_config_with_force_last = ACConfig(
            mode="selective",
            per_op_sac_force_recompute_mm_shapes_by_fqns=["output"],
        )
        apply_ac(
            model_with_force_last,
            ac_config_with_force_last,
            model_compile_enabled=False,
        )
        mem_with_force_last = get_act_mem(model_with_force_last)

        # 5. Full AC
        model_with_full_ac = ToyModule().cuda()
        ac_config_full_ac = ACConfig(
            mode="full",
        )
        apply_ac(
            model_with_full_ac,
            ac_config_full_ac,
            model_compile_enabled=False,
        )
        mem_full_ac = get_act_mem(model_with_full_ac)

        self.assertEqual(mem_no_ac, 2.0)
        self.assertEqual(mem_selective_ac, 3.0)
        self.assertEqual(mem_with_force_first, 2.0)
        self.assertEqual(mem_with_force_last, 1.0)
        self.assertEqual(mem_full_ac, 0.0)
        # Note: SAC > no-AC here because it unnecessarily saves "output"
        # even that is not needed for recomputation and output is double
        # the size of the other two mms.

    def test_correctness(self):
        model_no_ac = ToyModule()

        model_selective_ac = ToyModule()
        model_selective_ac.load_state_dict(model_no_ac.state_dict())
        apply_ac(
            model_selective_ac,
            ACConfig(
                mode="selective",
                per_op_sac_force_recompute_mm_shapes_by_fqns=[],
            ),
            model_compile_enabled=False,
        )
        model_force_first = ToyModule()
        model_force_first.load_state_dict(model_no_ac.state_dict())
        apply_ac(
            model_force_first,
            ACConfig(
                mode="selective",
                per_op_sac_force_recompute_mm_shapes_by_fqns=["moe.router.gate"],
            ),
            model_compile_enabled=False,
        )

        model_force_last = ToyModule()
        model_force_last.load_state_dict(model_no_ac.state_dict())
        apply_ac(
            model_force_last,
            ACConfig(
                mode="selective",
                per_op_sac_force_recompute_mm_shapes_by_fqns=["output"],
            ),
            model_compile_enabled=False,
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

    def test_force_recompute_mm_fqns(self):
        """Test that per_op_sac_force_recompute_mm_shapes_by_fqns controls
        exactly which matmuls are recomputed vs stored during backward.

        Approach: during backward, count aten.mm calls per weight tensor.
        count=1 means stored (gradient mm only), count=2 means recomputed
        (gradient mm + recomputed forward mm).
        """
        from torch.utils._python_dispatch import TorchDispatchMode

        class MmWeightTracker(TorchDispatchMode):
            def __init__(self, ptrs):
                super().__init__()
                self._ptrs = ptrs
                self.counts = {n: 0 for n in ptrs.values()}

            def __torch_dispatch__(self, func, types, args, kwargs=None):
                if func == torch.ops.aten.mm.default:
                    for arg in args:
                        name = self._ptrs.get(arg.data_ptr())
                        if name is not None:
                            self.counts[name] += 1
                            break
                return func(*args, **(kwargs or {}))

        def get_recomputed(force_recompute_fqns):
            m = ToyModule()
            apply_ac(
                m,
                ACConfig(
                    mode="selective",
                    per_op_sac_force_recompute_mm_shapes_by_fqns=force_recompute_fqns,
                    early_stop=False,
                ),
                model_compile_enabled=False,
            )
            ptr_to_name = {
                mod.weight.data_ptr(): fqn.rsplit(".", 1)[-1]
                for fqn, mod in m.named_modules()
                if isinstance(mod, Linear)
            }
            x = torch.randn(64, 512, requires_grad=True)
            out = m(x)
            tracker = MmWeightTracker(ptr_to_name)
            with tracker:
                out.backward()
            return {n for n, c in tracker.counts.items() if c == 2}

        # No force recompute: alternating pattern recomputes every 2nd mm
        self.assertEqual(get_recomputed([]), {"wq"})
        # force_recompute="moe.router.gate": shape (512,512) also matches wq,
        # so both are force-recomputed; output is 1st in alternation → saved
        self.assertEqual(get_recomputed(["moe.router.gate"]), {"gate", "wq"})
        # force_recompute="output": shape (512,1024) is unique to output,
        # gate and wq still alternate (gate saved, wq recomputed)
        self.assertEqual(get_recomputed(["output"]), {"wq", "output"})


if __name__ == "__main__":
    unittest.main()
