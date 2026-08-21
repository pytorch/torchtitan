# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA merge (checkpoint export) tests.

merge_lora_state_dict folds adapters into base weights so a trained
LoRA can be saved back to HF (the raw adapter drops lora_* keys). Tested
at the tensor level -- W_merged == W_base + scaling*(B@A) -- which is the
merge contract; full-model forward fidelity is subject to bf16 (the
standard LoRA merge-and-unload property).
"""

import unittest

import torch


@unittest.skipIf(not torch.cuda.is_available(), "build needs CUDA (fla)")
class TestLoRAMerge(unittest.TestCase):
    def _lora_model(self, quantize=None, mla_only=False):
        # Real QLoRA order: build+init a plain backbone, THEN wrap/quantize
        # (quantizing a loaded weight, not init'ing over an NF4 tensor).
        from torchtitan.models.kimi_k3.lora import apply_lora

        spec = self._spec_plain(mla_only=mla_only)
        with torch.device("cuda"):
            m = spec.build()
            m.init_weights()
        m = m.to(torch.bfloat16)
        apply_lora(m, rank=8, alpha=16, quantize_base=quantize)
        for n, p in m.named_parameters():
            if n.endswith("lora_b"):
                p.data.normal_(0, 0.02)  # trained-like adapter
        return m

    def _graft_lora_model(self):
        # AttnRes-graft (alpha gate) + LoRA -- the 48B post-training flavor:
        # base frozen, LoRA adapters + graft params (alpha-fullparam
        # exception) train. This is what the real 48B LoRA run exercised.
        from torchtitan.models.kimi_k3 import config_registry
        from torchtitan.models.kimi_k3.lora import apply_lora
        from torchtitan.models.kimi_k3.model import KimiK3Spec

        kc = config_registry.kimi_k3_debugmodel().model_spec.model.kimi_config
        spec = KimiK3Spec(kimi_config=kc, num_blocks=2, attn_res_gated=True)
        with torch.device("cuda"):
            m = spec.build()
            m.init_weights()
        m = m.to(torch.bfloat16)
        apply_lora(m, rank=8, alpha=16)
        for n, p in m.named_parameters():
            if n.endswith("lora_b"):
                p.data.normal_(0, 0.02)
        return spec, m

    def _spec_plain(self, mla_only=False):
        import dataclasses

        from torchtitan.models.kimi_k3 import config_registry
        from torchtitan.models.kimi_k3.model import KimiK3Spec

        kc = config_registry.kimi_k3_debugmodel().model_spec.model.kimi_config
        if mla_only:
            # All-MLA: KDA kernels are nondeterministic at debug scale and
            # can NaN under accumulated cross-test GPU state; forward-
            # executing tests use the deterministic MLA path.
            n = kc.num_hidden_layers
            kc = dataclasses.replace(
                kc, kda_layers=[], full_attn_layers=list(range(1, n + 1))
            )
        return KimiK3Spec(kimi_config=kc, num_blocks=None)

    def test_merge_tensor_math_and_key_space(self):
        from torchtitan.models.kimi_k3.lora import KimiLoRALinear, merge_lora_state_dict

        m = self._lora_model()
        merged = merge_lora_state_dict(m)
        self.assertFalse(any("lora_a" in k or "lora_b" in k for k in merged))
        self.assertFalse(any(".base." in k for k in merged))
        checked = 0
        for mod_name, module in m.named_modules():
            if isinstance(module, KimiLoRALinear):
                expect = (
                    module.base.weight.float()
                    + module._lora_scaling
                    * (module.lora_b.float() @ module.lora_a.float())
                ).to(module.base.weight.dtype)
                got = merged[f"{mod_name}.weight"]
                self.assertEqual(got.shape, module.base.weight.shape)
                self.assertLess((got.float() - expect.float()).abs().max().item(), 1e-2)
                checked += 1
        self.assertGreater(checked, 0)

    def test_merge_exports_to_hf(self):
        from torchtitan.models.kimi_k3.lora import merge_lora_state_dict
        from torchtitan.models.kimi_k3.state_dict_adapter import (
            KimiLinearStateDictAdapter,
        )

        spec = self._spec_plain()
        m = self._lora_model()
        merged = merge_lora_state_dict(m)
        adapter = KimiLinearStateDictAdapter(spec, hf_assets_path=None)
        hf = adapter.to_hf(merged)
        self.assertTrue(hf)
        self.assertFalse(any("lora" in k or ".base." in k for k in hf))

    def test_merge_dequantizes_nf4_base(self):
        from torchao.dtypes.nf4tensor import NF4Tensor

        from torchtitan.models.kimi_k3.lora import merge_lora_state_dict

        m = self._lora_model(quantize="nf4")
        merged = merge_lora_state_dict(m)
        self.assertFalse(any(isinstance(v, NF4Tensor) for v in merged.values()))

    def test_post_load_quantize_hook(self):
        # The trainer order: build+load bf16, THEN quantize (not at
        # build over init noise / meta storage).
        from torchao.dtypes.nf4tensor import NF4Tensor

        from torchtitan.models.kimi_k3.lora import KimiLoRALinear, quantize_lora_bases

        # all-MLA: this test runs a forward (deterministic MLA path)
        m = self._lora_model(mla_only=True)  # bf16 bases, loaded-like
        # a reference: one alignable base weight, pre-quantization
        ref = None
        for module in m.modules():
            if (
                isinstance(module, KimiLoRALinear)
                and module.base.weight.numel() % 16384 == 0
            ):
                ref = (module, module.base.weight.detach().float().clone())
                break
        self.assertIsNotNone(ref, "need >=1 NF4-alignable base for this test")

        packed = quantize_lora_bases(m, experts=False)
        self.assertGreater(packed, 0)
        module, ref_w = ref
        self.assertIsInstance(module.base.weight, NF4Tensor)
        # dequant tracks the loaded weight within NF4 error (not init noise)
        deq = module.base.weight.get_original_weight().float()
        self.assertLess((deq - ref_w).norm().item() / ref_w.norm().item(), 0.15)
        # idempotent: second call packs nothing new, no error
        self.assertEqual(quantize_lora_bases(m, experts=False), packed)
        # forward still runs through the NF4 base path
        tok = torch.randint(0, 2016, (1, 96), device="cuda")
        with torch.no_grad():
            m(tok)

    def test_mxfp4_base_merge_and_export(self):
        # MXFP4 (K3's native FP4 weight format) base + LoRA: merge must
        # dequant the split-storage MXTensor and leave NO qdata/scale in
        # the exported HF dict.
        from torchtitan.models.kimi_k3.lora import merge_lora_state_dict
        from torchtitan.models.kimi_k3.state_dict_adapter import (
            KimiLinearStateDictAdapter,
        )

        m = self._lora_model(quantize="mxfp4")
        merged = merge_lora_state_dict(m)
        self.assertFalse(
            any("qdata" in k or "scale" in k or ".base." in k for k in merged)
        )
        spec = self._spec_plain()
        hf = KimiLinearStateDictAdapter(spec, hf_assets_path=None).to_hf(merged)
        self.assertTrue(hf)
        self.assertFalse(any("qdata" in k or "lora" in k for k in hf))

    def test_post_load_quantize_mxfp4(self):
        # Trainer order: build+load bf16, THEN MXFP4-pack (not at build).
        from torchtitan.models.kimi_k3.lora import KimiLoRALinear, quantize_lora_bases

        m = self._lora_model(mla_only=True)  # forward-running -> MLA path
        ref = None
        for mod in m.modules():
            if isinstance(mod, KimiLoRALinear) and (
                mod.base._parameters.get("weight") is not None
                and mod.base.weight.shape[-1] % 32 == 0
            ):
                ref = (mod, mod.base.weight.detach().float().clone())
                break
        self.assertIsNotNone(ref)
        packed = quantize_lora_bases(m, mode="mxfp4", experts=False)
        self.assertGreater(packed, 0)
        mod, ref_w = ref
        # split-storage present, bf16 base weight gone
        param_names = {n for n, _ in mod.named_parameters()}
        self.assertIn("base_qdata", param_names)
        self.assertIn("base_scale", param_names)
        self.assertNotIn("weight", mod.base._parameters)
        # dequant tracks the loaded weight within MXFP4 error (~10-13%)
        deq = mod._dequant_base_mxfp4().float()
        self.assertLess((deq - ref_w).norm().item() / ref_w.norm().item(), 0.15)
        # idempotent + forward runs through the MXFP4 base path
        self.assertEqual(quantize_lora_bases(m, mode="mxfp4", experts=False), packed)
        tok = torch.randint(0, 2016, (1, 96), device="cuda")
        with torch.no_grad():
            m(tok)

    def test_graft_lora_compose_merge_and_export(self):
        # The 48B post-training composition: AttnRes graft + LoRA. Locks
        # (1) trainable set = LoRA + graft, base frozen; (2) merge folds
        # LoRA and CARRIES THE GRAFT params through unchanged; (3) to_hf
        # drops both graft and lora keys, leaving a clean base HF export.
        from torchtitan.models.kimi_k3.lora import KimiLoRALinear, merge_lora_state_dict
        from torchtitan.models.kimi_k3.state_dict_adapter import (
            KimiLinearStateDictAdapter,
        )

        spec, m = self._graft_lora_model()
        graft = "attention_res", "ffn_res", "output_res"
        train = {n for n, p in m.named_parameters() if p.requires_grad}
        # every trainable is either a LoRA adapter or a graft param
        for n in train:
            self.assertTrue(
                n.endswith("lora_a")
                or n.endswith("lora_b")
                or any(g in n for g in graft),
                f"unexpected trainable param {n}",
            )
        self.assertTrue(any(any(g in n for g in graft) for n in train))
        self.assertTrue(any(n.endswith("lora_b") for n in train))

        merged = merge_lora_state_dict(m)
        self.assertFalse(any("lora_a" in k or "lora_b" in k for k in merged))
        self.assertFalse(any(".base." in k for k in merged))
        # graft params survive the merge unchanged (carried as non-LoRA)
        graft_keys = [n for n in train if any(g in n for g in graft)]
        for gk in graft_keys:
            self.assertIn(gk, merged)
        # one wrapped linear merged correctly
        for mod_name, module in m.named_modules():
            if isinstance(module, KimiLoRALinear):
                expect = (
                    module.base.weight.float()
                    + module._lora_scaling
                    * (module.lora_b.float() @ module.lora_a.float())
                ).to(module.base.weight.dtype)
                got = merged[f"{mod_name}.weight"]
                self.assertLess((got.float() - expect.float()).abs().max().item(), 1e-2)
                break

        # HF export drops graft + lora, keeps the base backbone
        adapter = KimiLinearStateDictAdapter(spec, hf_assets_path=None)
        hf = adapter.to_hf(merged)
        self.assertTrue(hf)
        self.assertFalse(
            any("lora" in k or ".base." in k or any(g in k for g in graft) for k in hf)
        )


class TestMergeMaterializesShardedAdapters(unittest.TestCase):
    """The merge must not mix a full base with sharded adapters.

    Real DTensors need a process group, so this stands in a tensor that reports
    a ``full_tensor()`` -- which is the only thing the merge path keys on -- and
    checks the merge used the full value rather than the local one.
    """

    def test_an_adapter_offering_full_tensor_is_materialized(self):
        from torchtitan.models.kimi_k3.lora import _materialize

        class _Sharded(torch.Tensor):
            @staticmethod
            def __new__(cls, local, full):
                obj = torch.Tensor._make_subclass(cls, local, False)
                obj._full = full
                return obj

            def full_tensor(self):
                return self._full

        local = torch.zeros(2, 2)
        full = torch.ones(2, 2)
        out = _materialize(_Sharded(local, full))
        self.assertTrue(torch.equal(out, full))

    def test_a_plain_tensor_passes_through_untouched(self):
        from torchtitan.models.kimi_k3.lora import _materialize

        t = torch.randn(3, 3)
        self.assertIs(_materialize(t), t)


class TestMergeUnderModuleWrappers(unittest.TestCase):
    """Activation checkpointing renames the module path but not the state dict.

    ``named_modules()`` reports ``...._checkpoint_wrapped_module.feed_forward.gate_proj`` while
    ``state_dict()`` strips the segment back out. Composing merged keys from the module
    path wrote a name nothing recognises and left the adapter keys in place, since the
    pops missed as well. It surfaced from a GRPO weight sync as
    ``Unmapped tt key: 'layers.0._checkpoint_wrapped_module.feed_forward.gate_proj.weight'``.
    """

    def _wrapped_lora_model(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
        )

        from torchtitan.models.kimi_k3 import config_registry as cr

        model = cr.kimi_k3_debugmodel_gated_lora().model_spec.model.build()
        model.init_weights()
        # Wrap one real decoder layer, the way apply_ac does.
        key = next(iter(model.layers))
        model.layers[key] = checkpoint_wrapper(model.layers[key])
        return model

    def test_the_wrapper_segment_is_absent_from_state_dict_keys(self):
        """The premise. If this ever fails, the fix below is solving nothing."""
        model = self._wrapped_lora_model()
        paths = [n for n, _ in model.named_modules()]
        self.assertTrue(any("_checkpoint_wrapped_module" in n for n in paths))
        self.assertFalse(
            any("_checkpoint_wrapped_module" in k for k in model.state_dict())
        )

    def test_merged_keys_use_state_dict_names(self):
        from torchtitan.models.kimi_k3.lora import merge_lora_state_dict

        merged = merge_lora_state_dict(self._wrapped_lora_model())
        self.assertFalse(
            [k for k in merged if "_checkpoint_wrapped_module" in k],
            "merged key carries a wrapper segment no loader recognises",
        )
        self.assertFalse(
            [k for k in merged if k.endswith((".base.weight", ".lora_a", ".lora_b"))],
            "adapter keys survived the merge, so the pops missed",
        )

    def test_an_unrecognised_wrapper_raises(self):
        """Guessing a name here ships weights nothing can load, in silence."""
        from torchtitan.models.kimi_k3.lora import _state_dict_prefix

        with self.assertRaises(KeyError):
            _state_dict_prefix("layers.0._made_up_wrapper.feed_forward.gate_proj", {})


if __name__ == "__main__":
    unittest.main()


class TestLoRAWrapperTransparency(unittest.TestCase):
    """The wrapper must look enough like an nn.Linear for callers that
    inspect it. init_weights reads ``attn_gate_proj.bias`` to detect the
    near-identity graft gate, and attn_gate_proj is a LoRA target, so a
    missing passthrough crashes model construction rather than degrading
    quietly."""

    def _wrapped(self, bias: bool):
        import torch.nn as nn

        from torchtitan.models.kimi_k3.lora import KimiLoRALinear

        base = nn.Linear(32, 16, bias=bias)
        return base, KimiLoRALinear(base, rank=4, alpha=8.0)

    def test_bias_and_weight_passthrough(self):
        for bias in (True, False):
            base, w = self._wrapped(bias)
            self.assertIs(w.weight, base.weight)
            if bias:
                self.assertIs(w.bias, base.bias)
            else:
                self.assertIsNone(w.bias)
            self.assertEqual(w.in_features, 32)
            self.assertEqual(w.out_features, 16)

    def test_weight_is_none_when_base_is_packed(self):
        base, w = self._wrapped(False)
        w.quantize_base_mxfp4()
        # packed bases have no base.weight; None is the signal init_weights
        # already uses to skip them
        self.assertIsNone(w.weight)

    def test_k3_lora_targets_cover_the_compressed_q_and_latent_paths(self):
        from torchtitan.models.kimi_k3.lora import (
            apply_lora,
            DEFAULT_LORA_TARGETS,
            KimiLoRALinear,
        )
        from torchtitan.models.kimi_k3.model import KimiK3Model
        from torchtitan.models.kimi_k3.model_configs import build_kimi_linear_config

        with torch.device("meta"):
            model = KimiK3Model.make_config(build_kimi_linear_config("k3mini", vocab_size=256)).build()
        apply_lora(model, rank=8, alpha=16.0)
        leaves = {
            fqn.rsplit(".", 1)[1]
            for fqn, m in model.named_modules()
            if isinstance(m, KimiLoRALinear)
        }
        # the modules that did not exist when the target list was written
        for name in ("q_a_proj", "q_b_proj", "attn_gate_proj", "down", "up"):
            self.assertIn(name, leaves, f"{name} not adapted")
        # dotted entries must match a qualified suffix, not a bare leaf name
        self.assertIn("latent.down", DEFAULT_LORA_TARGETS)
        latent_wrapped = [
            fqn
            for fqn, m in model.named_modules()
            if isinstance(m, KimiLoRALinear) and fqn.endswith(".latent.down")
        ]
        self.assertTrue(latent_wrapped)
