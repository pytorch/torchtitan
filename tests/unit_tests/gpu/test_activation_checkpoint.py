# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from copy import deepcopy
from functools import partial

import torch
from torch.nn import init
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.models.common.config_utils import (
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts, TokenChoiceTopKRouter
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
            with FlopCounterMode(display=False) as fwd_mode:
                out = model_fn(x)
            with FlopCounterMode(display=False) as bwd_mode:
                out.backward()
            return bwd_mode.get_total_flops() / (512**3 * 2)

        # 1. No AC
        model_no_ac = ToyModule()
        flops_no_ac = get_bw_flops(model_no_ac)

        # 2. SAC
        # Per-op SAC's policy is to save every other mm
        model_selective_ac = ToyModule()
        SelectiveAC.Config(
            force_recompute_mm_shapes_by_fqns=[],  # Empty list
        ).build().apply(model_selective_ac)
        flops_selective_ac = get_bw_flops(model_selective_ac)

        # 3. Per-op SAC with force recompute "moe.router.gate"
        # This leads to two mms being recomputed since they share the same shape!
        model_with_force_first = ToyModule()
        SelectiveAC.Config(
            force_recompute_mm_shapes_by_fqns=["moe.router.gate"],
        ).build().apply(model_with_force_first)
        flops_with_force_first = get_bw_flops(model_with_force_first)

        # 4. Per-op SAC with force recompute "output"
        model_with_force_last = ToyModule()
        SelectiveAC.Config(
            force_recompute_mm_shapes_by_fqns=["output"],
        ).build().apply(model_with_force_last)
        flops_with_force_last = get_bw_flops(model_with_force_last)

        # 5. Full AC
        model_with_full_ac = ToyModule()
        FullAC.Config().build().apply(model_with_full_ac)
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
        SelectiveAC.Config(
            force_recompute_mm_shapes_by_fqns=[],  # Empty list
        ).build().apply(model_selective_ac)
        mem_selective_ac = get_act_mem(model_selective_ac)

        # 3. Per-op SAC with force recompute "moe.router.gate"
        # This leads to two mms being recomputed since they share the same shape!
        model_with_force_first = ToyModule().cuda()
        SelectiveAC.Config(
            force_recompute_mm_shapes_by_fqns=["moe.router.gate"],
        ).build().apply(model_with_force_first)
        mem_with_force_first = get_act_mem(model_with_force_first)

        # 4. Per-op SAC with force recompute "output"
        model_with_force_last = ToyModule().cuda()
        SelectiveAC.Config(
            force_recompute_mm_shapes_by_fqns=["output"],
        ).build().apply(model_with_force_last)
        mem_with_force_last = get_act_mem(model_with_force_last)

        # 5. Full AC
        model_with_full_ac = ToyModule().cuda()
        FullAC.Config().build().apply(model_with_full_ac)
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
        SelectiveAC.Config(
            force_recompute_mm_shapes_by_fqns=[],
        ).build().apply(model_selective_ac)

        model_force_first = ToyModule()
        model_force_first.load_state_dict(model_no_ac.state_dict())
        SelectiveAC.Config(
            force_recompute_mm_shapes_by_fqns=["moe.router.gate"],
        ).build().apply(model_force_first)

        model_force_last = ToyModule()
        model_force_last.load_state_dict(model_no_ac.state_dict())
        SelectiveAC.Config(
            force_recompute_mm_shapes_by_fqns=["output"],
        ).build().apply(model_force_last)

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
        """Test that force_recompute_mm_shapes_by_fqns controls
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
            SelectiveAC.Config(
                force_recompute_mm_shapes_by_fqns=force_recompute_fqns,
            ).build().apply(m)
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


# Mini MoE AC coverage. The tests above use a Linear named like a router
# gate; these wrap a real TokenChoiceTopKRouter + GroupedExperts so SAC
# and FullAC actually recompute grouped-mm experts.
#
# Shape suffix legend:
#   T = num tokens, D = model dim, E = num experts, K = top-k
_MOE_DIM = 16
_MOE_HIDDEN = 32
_MOE_E = 4
_MOE_K = 2
_MOE_T = 8


def _mini_moe_config():
    cfg = make_moe_config(
        num_experts=_MOE_E,
        router=make_router_config(
            dim=_MOE_DIM,
            num_experts=_MOE_E,
            gate_param_init={"weight": partial(init.trunc_normal_, std=0.02)},
            top_k=_MOE_K,
        ),
        routed_experts=make_routed_experts_config(
            dim=_MOE_DIM,
            hidden_dim=_MOE_HIDDEN,
            num_experts=_MOE_E,
            top_k=_MOE_K,
            param_init={
                "w1_EFD": partial(init.trunc_normal_, std=0.02),
                "w2_EDF": partial(init.trunc_normal_, std=0.02),
                "w3_EFD": partial(init.trunc_normal_, std=0.02),
            },
            comm_backend="standard",
        ),
    )
    # FullAC recomputes topk (not in the SAC save set). Keep expert
    # assignment stable between the original forward and rematerialization.
    cfg.router._debug_force_load_balance = True
    return cfg


class MiniMoEBlock(Module):
    def __init__(self):
        super().__init__()
        self.moe = _mini_moe_config().build()

    def forward(self, x_TD):
        return self.moe(x_TD).sum()


class MiniMoEModel(Module):
    def __init__(self):
        super().__init__()
        self.layers = ModuleDict({"0": MiniMoEBlock()})

    def forward(self, x_TD):
        return self.layers["0"](x_TD)


def _build_mini_moe_model(device: str = "cpu") -> MiniMoEModel:
    model = MiniMoEModel()
    model.init_states()
    return model.to(device)


def _clone_mini_moe_model(src: MiniMoEModel, device: str) -> MiniMoEModel:
    model = _build_mini_moe_model(device)
    model.load_state_dict(src.state_dict())
    return model


def _grouped_mm_supported(device: str) -> bool:
    if not hasattr(torch, "_grouped_mm"):
        return False
    try:
        x_TD = torch.randn(4, 8, dtype=torch.bfloat16, device=device)
        w_EFD = torch.randn(2, 8, 8, dtype=torch.bfloat16, device=device)
        offs_E = torch.tensor([2, 4], dtype=torch.int32, device=device)
        torch._grouped_mm(x_TD, w_EFD.transpose(-2, -1), offs=offs_E)
        return True
    except (RuntimeError, NotImplementedError):
        return False


def _iter_moe_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    ran = False
    for device in devices:
        if _grouped_mm_supported(device):
            ran = True
            yield device
    if not ran:
        raise unittest.SkipTest("torch._grouped_mm is unavailable on cpu/cuda")


def _run_moe_fwd_bwd(model, batch_TD):
    model.zero_grad(set_to_none=True)
    x_TD = batch_TD.clone().detach().requires_grad_(True)
    out = model(x_TD)
    out.backward()
    grad_in_TD = x_TD.grad.detach().clone()
    grad_params = [
        p.grad.detach().clone() if isinstance(p.grad, torch.Tensor) else None
        for p in model.parameters()
    ]
    return out.detach(), grad_in_TD, grad_params


class _MmWeightTracker(TorchDispatchMode):
    """Count aten.mm / aten.linear uses of selected weight tensors."""

    def __init__(self, ptrs):
        super().__init__()
        self._ptrs = ptrs
        self.counts = {n: 0 for n in ptrs.values()}

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        if func in (torch.ops.aten.mm.default, torch.ops.aten.linear.default):
            for arg in args:
                if torch.is_tensor(arg):
                    name = self._ptrs.get(arg.data_ptr())
                    if name is not None:
                        self.counts[name] += 1
                        break
        return func(*args, **(kwargs or {}))


class TestMiniMoEAC(unittest.TestCase):
    """AC over a real mini MoE (router top-k + GroupedExperts)."""

    def test_wraps_real_router_and_grouped_experts(self):
        model = _build_mini_moe_model()
        moe = model.layers["0"].moe
        self.assertIsInstance(moe.router, TokenChoiceTopKRouter)
        self.assertIsInstance(moe.routed_experts.inner_experts, GroupedExperts)
        self.assertIsInstance(moe.router.gate, Linear)

        # FQN must resolve to nn.Linear or SelectiveAC.apply raises.
        SelectiveAC.Config(
            force_recompute_mm_shapes_by_fqns=["moe.router.gate"],
        ).build().apply(model)

    def test_ac_matches_no_ac(self):
        # Experts run grouped_mm in bf16; rematerialized grads should still
        # match. Do not assert flop integers -- grouped-mm counts vary by
        # backend.
        for device in _iter_moe_devices():
            with self.subTest(device=device):
                torch.manual_seed(0)
                model_no_ac = _build_mini_moe_model(device)

                model_full_ac = _clone_mini_moe_model(model_no_ac, device)
                FullAC.Config().build().apply(model_full_ac)

                model_selective_ac = _clone_mini_moe_model(model_no_ac, device)
                SelectiveAC.Config(
                    force_recompute_mm_shapes_by_fqns=[],
                ).build().apply(model_selective_ac)

                model_force_gate = _clone_mini_moe_model(model_no_ac, device)
                SelectiveAC.Config(
                    force_recompute_mm_shapes_by_fqns=["moe.router.gate"],
                ).build().apply(model_force_gate)

                batch_TD = torch.randn(_MOE_T, _MOE_DIM, device=device)
                out_ref, gin_ref, gparams_ref = _run_moe_fwd_bwd(
                    model_no_ac, batch_TD
                )
                for model in (model_full_ac, model_selective_ac, model_force_gate):
                    out, gin, gparams = _run_moe_fwd_bwd(model, batch_TD)
                    torch.testing.assert_close(out_ref, out, atol=1e-2, rtol=1e-2)
                    torch.testing.assert_close(gin_ref, gin, atol=1e-2, rtol=1e-2)
                    for g_ref, g_other in zip(gparams_ref, gparams):
                        if not (
                            torch.is_tensor(g_ref) and torch.is_tensor(g_other)
                        ):
                            continue
                        torch.testing.assert_close(
                            g_ref, g_other, atol=1e-2, rtol=1e-2
                        )

    def test_force_recompute_gate_increases_recompute(self):
        # count=1: activation stored (grad mm only). count=2: rematerialized.
        # Gate is the first (and only) nn.Linear, so default SAC saves it.
        for device in _iter_moe_devices():
            with self.subTest(device=device):
                torch.manual_seed(0)

                def gate_mm_count(force_recompute_fqns):
                    model = _build_mini_moe_model(device)
                    SelectiveAC.Config(
                        force_recompute_mm_shapes_by_fqns=force_recompute_fqns,
                    ).build().apply(model)
                    ptr_to_name = {
                        mod.weight.data_ptr(): "gate"
                        for fqn, mod in model.named_modules()
                        if isinstance(mod, Linear) and fqn.endswith("gate")
                    }
                    self.assertIn("gate", ptr_to_name.values())
                    x_TD = torch.randn(
                        _MOE_T, _MOE_DIM, device=device, requires_grad=True
                    )
                    out = model(x_TD)
                    tracker = _MmWeightTracker(ptr_to_name)
                    with tracker:
                        out.backward()
                    return tracker.counts["gate"]

                default_count = gate_mm_count([])
                force_count = gate_mm_count(["moe.router.gate"])
                self.assertEqual(default_count, 1)
                self.assertGreater(force_count, default_count)
                self.assertEqual(force_count, 2)


if __name__ == "__main__":
    unittest.main()
